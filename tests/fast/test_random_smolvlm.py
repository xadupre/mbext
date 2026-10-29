# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np
import torch
from transformers import LlamaConfig, SmolVLMConfig, SmolVLMForConditionalGeneration
from transformers.models.smolvlm.configuration_smolvlm import SmolVLMVisionConfig

from modelbuilder.builder import create_model
from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

MODEL_NAME = "HuggingFaceTB/SmolVLM-256M-Instruct"


class TestRandomSmolVLM(ExtTestCase):
    @staticmethod
    def _make_model(scale_factor=4):
        torch.manual_seed(42)
        config = SmolVLMConfig(
            architectures=["SmolVLMForConditionalGeneration"],
            image_token_id=127,
            scale_factor=scale_factor,
            text_config=LlamaConfig(
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
                vocab_size=128,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
                tie_word_embeddings=True,
            ),
            vision_config=SmolVLMVisionConfig(
                hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, image_size=32, patch_size=4
            ),
            tie_word_embeddings=True,
        )
        return SmolVLMForConditionalGeneration(config).eval()

    def _export(self, model, precision="fp32", provider="cpu", multimodal=False, generation_config=True):
        basename = f"{self.id().split('.')[-1]}_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)
        model.save_pretrained(model_dir)
        if not generation_config:
            os.remove(os.path.join(model_dir, "generation_config.json"))
        self.make_word_level_tokenizer().save_pretrained(model_dir)
        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            multimodal=multimodal,
        )
        return output_dir

    def _text_discrepancy(self, precision, provider):
        model = self._make_model().to(provider)
        config = model.config.text_config
        self.run_random_weights_test(
            model=model,
            tokenizer=self.make_word_level_tokenizer(),
            model_name=MODEL_NAME,
            basename=f"test_smolvlm_text_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
        )

    @hide_stdout()
    def test_text_cpu(self):
        for precision in ("fp32", "fp16", "int4"):
            with self.subTest(precision=precision):
                self._text_discrepancy(precision, "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_text_cuda(self):
        for precision in ("fp16", "int4"):
            with self.subTest(precision=precision):
                self._text_discrepancy(precision, "cuda")

    @hide_stdout()
    @requires_genai()
    def test_text_genai_generation(self):
        model = self._make_model()
        output_dir = self._export(model)
        self.run_genai_generation_test(output_dir, model, 128, 2, prompt_ids=torch.tensor([[1, 4, 5, 6]]))

    @hide_stdout()
    def test_text_onnx_light_generation(self):
        model = self._make_model()
        output_dir = self._export(model)
        self.run_onnx_light_generation_test(output_dir, model, 128, 2, prompt_ids=torch.tensor([[1, 4, 5, 6]]))

    def _multimodal_discrepancy(self, precision, provider):
        model = self._make_model()
        output_dir = self._export(model, precision, provider, multimodal=True)
        self.check_phi3v_genai_config(output_dir, spatial_merge_size=4)
        vision = self.check_ort(os.path.join(output_dir, "vision_encoder.onnx"), provider)
        embedding = self.check_ort(os.path.join(output_dir, "embedding.onnx"), provider)
        decoder = self.check_ort(os.path.join(output_dir, "model.onnx"), provider)
        dtype = self.get_input_np_dtype(precision)
        pixels = torch.randn(1, 2, 3, 32, 32)
        with torch.no_grad():
            pt_features = model.model.connector(model.model.vision_model(pixels.flatten(0, 1)).last_hidden_state).flatten(0, 1)
        ort_features = np.concatenate([vision.run(None, {"pixel_values": crop.numpy().astype(dtype)})[0] for crop in pixels[0].split(1)])
        atol = {"fp32": 2e-5, "fp16": 3e-3, "int4": 0.15}[precision]
        np.testing.assert_allclose(pt_features.numpy(), ort_features, atol=atol, rtol=atol)
        # Two images interleaved with text catch connector ordering and scatter errors.
        ids = torch.tensor([[1, 127, 127, 127, 127, 4, 127, 127, 127, 127, 5]])
        with torch.no_grad():
            pt_embeds = model.get_input_embeddings()(ids)
            pt_embeds[ids == 127] = pt_features
        ort_embeds = embedding.run(None, {"input_ids": ids.numpy(), "image_features": ort_features})[0]
        np.testing.assert_allclose(pt_embeds.numpy(), ort_embeds, atol=atol, rtol=atol)
        tc = model.config.text_config
        feed = {
            "inputs_embeds": ort_embeds,
            "past_key_values.0.key": np.zeros((1, 2, 0, 32), dtype=dtype),
            "past_key_values.0.value": np.zeros((1, 2, 0, 32), dtype=dtype),
        }
        pt_cache = None
        pt_tokens, ort_tokens = [], []
        total_length = ids.shape[1]
        input_names = {value.name for value in decoder.get_inputs()}
        for step in range(4):
            seq_len = feed["inputs_embeds"].shape[1]
            feed["attention_mask"] = np.ones((1, total_length), dtype=np.int64)
            feed["position_ids"] = np.arange(total_length - seq_len, total_length, dtype=np.int64)[None]
            outputs = dict(
                zip([value.name for value in decoder.get_outputs()], decoder.run(None, {k: v for k, v in feed.items() if k in input_names}))
            )
            with torch.no_grad():
                pt_output = (
                    model(input_ids=ids, pixel_values=pixels, use_cache=True)
                    if step == 0
                    else model(input_ids=ids, past_key_values=pt_cache, use_cache=True)
                )
            pt_cache = pt_output.past_key_values
            np.testing.assert_allclose(
                pt_output.logits.numpy(), outputs["logits"], atol={"fp32": 1e-4, "fp16": 1e-2, "int4": 0.15}[precision], rtol=1e-2
            )
            pt_token = int(pt_output.logits[0, -1].argmax())
            ort_token = int(outputs["logits"][0, -1].argmax())
            pt_tokens.append(pt_token)
            ort_tokens.append(ort_token)
            ids = torch.tensor([[pt_token]])
            feed["inputs_embeds"] = embedding.run(
                None, {"input_ids": np.array([[ort_token]], dtype=np.int64), "image_features": np.empty((0, tc.hidden_size), dtype=dtype)}
            )[0]
            for kind in ("key", "value"):
                feed[f"past_key_values.0.{kind}"] = outputs[f"present.0.{kind}"]
            total_length += 1
        self.assertEqual(pt_tokens, ort_tokens)

    @hide_stdout()
    def test_multimodal_cpu(self):
        for precision in ("fp32", "fp16", "int4"):
            with self.subTest(precision=precision):
                self._multimodal_discrepancy(precision, "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_multimodal_cuda(self):
        for precision in ("fp16", "int4"):
            with self.subTest(precision=precision):
                self._multimodal_discrepancy(precision, "cuda")

    @hide_stdout()
    @requires_genai()
    def test_multimodal_genai_generation(self):
        import onnxruntime_genai as og

        model = self._make_model()
        output_dir = self._export(model, multimodal=True)
        pixels = torch.randn(1, 1, 3, 32, 32)
        ids = torch.tensor([[1, 127, 127, 127, 127, 4, 5]])
        with torch.no_grad():
            expected = model.generate(input_ids=ids, pixel_values=pixels, max_new_tokens=4, do_sample=False, pad_token_id=0)[0].tolist()
        og_model = og.Model(output_dir)
        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=ids.shape[1] + 4)
        generator = og.Generator(og_model, params)
        inputs = og.NamedTensors()
        inputs["pixel_values"] = pixels[0].numpy()
        inputs["num_image_tokens"] = np.array([4], dtype=np.int64)
        generator.set_inputs(inputs)
        generator.append_tokens(ids.numpy())
        actual = ids[0].tolist()
        while not generator.is_done():
            generator.generate_next_token()
            actual.append(int(generator.get_next_tokens()[0]))
        self.assertEqual(expected, actual)

    @hide_stdout()
    def test_multimodal_onnx_light_generation(self):
        model = self._make_model()
        output_dir = self._export(model, multimodal=True)
        # Multi-graph inputs_embeds generation is not yet implemented:
        # https://github.com/xadupre/onnx-light/issues/5264
        self.run_onnx_light_generation_test(output_dir, model, 128, 2, prompt_ids=torch.tensor([[1, 4, 5, 6]]))

    @hide_stdout()
    def test_invalid_pixel_shuffle_grid(self):
        from onnx_light.onnx import TensorProto

        from modelbuilder.builders.smollm import SmolVLMVisionEncoderModel

        config = self._make_model().config
        config.scale_factor = 3
        with self.assertRaisesRegex(ValueError, "divisible"):
            SmolVLMVisionEncoderModel(config, TensorProto.FLOAT, TensorProto.FLOAT, "cpu", "", {})

    @hide_stdout()
    def test_nested_special_token_ids(self):
        model = self._make_model()
        output_dir = self._export(model, generation_config=False)
        with open(os.path.join(output_dir, "genai_config.json")) as stream:
            config = json.load(stream)["model"]
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            self.assertEqual(config[name], getattr(model.config.text_config, name))

    @hide_stdout()
    def test_unsupported_export_options(self):
        from onnx_light.onnx import TensorProto

        from modelbuilder.builders.smollm import SmolVLMTextModel

        config = self._make_model().config
        for name in ("adapter_path", "reuse_downloaded_weights"):
            with self.subTest(option=name):
                with self.assertRaisesRegex(NotImplementedError, name):
                    SmolVLMTextModel(config, TensorProto.FLOAT, TensorProto.FLOAT, "cpu", "", {name: True})


if __name__ == "__main__":
    unittest.main(verbosity=2)
