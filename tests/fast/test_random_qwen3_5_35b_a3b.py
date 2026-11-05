# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, requires_transformers

MODEL_NAME = "Qwen/Qwen3.5-35B-A3B"


def _make_config(text_only=False):
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig, Qwen3_5MoeVisionConfig

    text = Qwen3_5MoeTextConfig(
        bos_token_id=1,
        eos_token_id=2,
        head_dim=64,
        hidden_size=128,
        layer_types=["linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_value_head_dim=16,
        max_position_embeddings=256,
        moe_intermediate_size=64,
        num_attention_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_key_value_heads=1,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000_000, "partial_rotary_factor": 0.25, "mrope_section": [2, 3, 3]},
        shared_expert_intermediate_size=96,
        vocab_size=128,
    )
    if text_only:
        text.architectures = ["Qwen3_5MoeForCausalLM"]
        return text
    vision = Qwen3_5MoeVisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_position_embeddings=16,
        out_hidden_size=text.hidden_size,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    config = Qwen3_5MoeConfig(
        image_token_id=120, text_config=text, video_token_id=121, vision_config=vision, vision_start_token_id=122, vision_end_token_id=123
    )
    config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
    return config


@requires_transformers("5")
class TestRandomQwen3_5_35BA3B(ExtTestCase):
    @staticmethod
    def _quantize_fp8(weight):
        import torch

        rows, cols = weight.shape
        padded = torch.nn.functional.pad(weight.float(), (0, -cols % 128, 0, -rows % 128))
        blocks = padded.reshape(padded.shape[0] // 128, 128, padded.shape[1] // 128, 128)
        scale = (blocks.abs().amax(dim=(1, 3)) / 448).clamp_min(1e-8).to(torch.bfloat16)
        expanded = scale.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:rows, :cols]
        quantized = (weight.float() / expanded).to(torch.float8_e4m3fn)
        return quantized, scale, quantized.float() * expanded

    def _build(self, fp8=False, text_only=False, multimodal=False, precision="fp32", provider="cpu", **extra_options):
        import torch
        from safetensors.torch import load_file, save_file
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            Qwen2VLImageProcessor,
            Qwen3VLProcessor,
            Qwen3VLVideoProcessor,
        )

        from modelbuilder.builder import create_model

        prefix = f"{self._testMethodName}_{fp8}_{text_only}_{multimodal}_{precision}_{provider}"
        model_dir = self.get_model_dir(prefix)
        output_dir, cache_dir = self.get_dirs(prefix)
        config = _make_config(text_only)
        model_class = AutoModelForCausalLM if text_only else AutoModelForImageTextToText
        torch.manual_seed(42)
        model = model_class.from_config(config).eval()
        model.save_pretrained(model_dir)
        if fp8:
            path = os.path.join(model_dir, "model.safetensors")
            state = load_file(path)
            quantized_state = {}
            for name, value in state.items():
                quantize = (
                    value.ndim == 2
                    and name.endswith(".weight")
                    and any(
                        part in name
                        for part in (
                            ".q_proj.",
                            ".k_proj.",
                            ".v_proj.",
                            ".o_proj.",
                            ".in_proj_qkv.",
                            ".in_proj_z.",
                            ".out_proj.",
                            ".gate_proj.",
                            ".up_proj.",
                            ".down_proj.",
                        )
                    )
                )
                if quantize:
                    quantized, scale, dequantized = self._quantize_fp8(value)
                    quantized_state[name] = quantized
                    quantized_state[name.removesuffix(".weight") + ".weight_scale_inv"] = scale
                    state[name] = dequantized
                else:
                    quantized_state[name] = value
            # Load the independently dequantized checkpoint as the reference,
            # including HF's per-expert packing into gate_up_proj.
            save_file(state, path, metadata={"format": "pt"})
            model = model_class.from_pretrained(model_dir).eval()
            save_file(quantized_state, path, metadata={"format": "pt"})
            config.quantization_config = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head"],
            }
            config.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        if multimodal:
            vision = config.vision_config
            processor = Qwen3VLProcessor(
                image_processor=Qwen2VLImageProcessor(
                    patch_size=vision.patch_size, temporal_patch_size=vision.temporal_patch_size, merge_size=vision.spatial_merge_size
                ),
                tokenizer=tokenizer,
                video_processor=Qwen3VLVideoProcessor(
                    patch_size=vision.patch_size, temporal_patch_size=vision.temporal_patch_size, merge_size=vision.spatial_merge_size
                ),
            )
            processor.save_pretrained(model_dir)
        else:
            tokenizer.save_pretrained(model_dir)
        create_model(
            model_name=MODEL_NAME + ("-FP8" if fp8 else ""),
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            **({"multimodal": True} if multimodal else {"exclude_embeds": False}),
            **extra_options,
        )
        return config, model, output_dir

    def _check_decoder(self, config, model, output_dir, precision="fp32", provider="cpu", inputs_embeds=None):
        import torch

        tc = getattr(config, "text_config", config)
        session = self.check_ort(os.path.join(output_dir, "model.onnx"), provider=provider)
        dtype = self.get_input_np_dtype(precision)
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.int64)
        cache = {}
        conv_width = tc.linear_num_key_heads * tc.linear_key_head_dim * 2 + tc.linear_num_value_heads * tc.linear_value_head_dim
        for i, layer_type in enumerate(tc.layer_types):
            shapes = (
                {"key": (1, tc.num_key_value_heads, 0, tc.head_dim), "value": (1, tc.num_key_value_heads, 0, tc.head_dim)}
                if layer_type == "full_attention"
                else {
                    "conv_state": (1, conv_width, tc.linear_conv_kernel_dim - 1),
                    "recurrent_state": (1, tc.linear_num_value_heads, tc.linear_key_head_dim, tc.linear_value_head_dim),
                }
            )
            cache.update({f"past_key_values.{i}.{name}": np.zeros(shape, dtype=dtype) for name, shape in shapes.items()})

        past = None
        past_length = 0
        for step in range(3):
            length = input_ids.shape[1] if inputs_embeds is None else inputs_embeds.shape[1]
            positions = np.broadcast_to(np.arange(past_length, past_length + length, dtype=np.int64)[None, None, :], (3, 1, length)).copy()
            feed = {"attention_mask": np.ones((1, past_length + length), dtype=np.int64), "position_ids": positions, **cache}
            kwargs = {"input_ids": input_ids}
            if inputs_embeds is not None:
                kwargs = {"inputs_embeds": torch.from_numpy(inputs_embeds).float()}
                feed["inputs_embeds"] = inputs_embeds.astype(dtype)
            else:
                feed["input_ids"] = input_ids.numpy()
            with torch.no_grad():
                expected = model(
                    **kwargs,
                    attention_mask=torch.from_numpy(feed["attention_mask"]),
                    position_ids=torch.from_numpy(positions),
                    past_key_values=past,
                    use_cache=True,
                )
            past = expected.past_key_values
            actual = dict(zip([value.name for value in session.get_outputs()], session.run(None, feed)))
            tolerance = 0.15 if precision == "int4" else (2e-3 if precision == "fp32" else 3e-2)
            np.testing.assert_allclose(actual["logits"], expected.logits.numpy(), atol=tolerance, rtol=tolerance)
            if precision != "int4":
                np.testing.assert_array_equal(actual["logits"][:, -1].argmax(-1), expected.logits[:, -1].argmax(-1).numpy())
            cache = {
                name.replace("present.", "past_key_values.", 1): value for name, value in actual.items() if name.startswith("present.")
            }
            past_length += length
            # Free-running token parity is tested separately.
            input_ids = torch.tensor([[8 + step]], dtype=torch.int64)
            if inputs_embeds is not None:
                with torch.no_grad():
                    inputs_embeds = model.model.language_model.embed_tokens(input_ids).numpy()

    @hide_stdout()
    def test_unquantized_checkpoint_fp32_cpu(self):
        self._check_decoder(*self._build())

    @hide_stdout()
    def test_fp8_checkpoint_fp32_cpu(self):
        self._check_decoder(*self._build(fp8=True))

    @hide_stdout()
    def test_causallm_fp32_cpu(self):
        self._check_decoder(*self._build(text_only=True))

    @hide_stdout()
    def test_fp8_causallm_fp32_cpu(self):
        self._check_decoder(*self._build(fp8=True, text_only=True))

    @hide_stdout()
    def test_fp8_fp16_cpu(self):
        self._check_decoder(*self._build(fp8=True, precision="fp16"), precision="fp16")

    @hide_stdout()
    def test_fp8_to_int4_qdq(self):
        import onnx_light.onnx as onnx

        config, model, directory = self._build(fp8=True, precision="int4", use_qdq=True)
        graph = onnx.load(os.path.join(directory, "model.onnx"), load_external_data=False).graph
        self.assertIn("DequantizeLinear", {node.op_type for node in graph.node})
        self.assertIn("QMoE", {node.op_type for node in graph.node})
        self._check_decoder(config, model, directory, precision="int4")

    @hide_stdout()
    def test_int4_cpu(self):
        self._check_decoder(*self._build(precision="int4"), precision="int4")

    @hide_stdout()
    def test_fp8_int4_cpu(self):
        self._check_decoder(*self._build(fp8=True, precision="int4"), precision="int4")

    @hide_stdout()
    def test_fp16_cpu(self):
        self._check_decoder(*self._build(precision="fp16"), precision="fp16")

    @hide_stdout()
    @requires_cuda()
    def test_fp16_cuda(self):
        self._check_decoder(*self._build(precision="fp16", provider="cuda"), precision="fp16", provider="cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fp8_fp16_cuda(self):
        self._check_decoder(*self._build(fp8=True, precision="fp16", provider="cuda"), precision="fp16", provider="cuda")

    def _check_multimodal(self, fp8=False, precision="fp32"):
        import torch

        config, model, output_dir = self._build(fp8=fp8, multimodal=True, precision=precision)
        with open(os.path.join(output_dir, "genai_config.json")) as stream:
            self.assertEqual(json.load(stream)["model"]["type"], "qwen3_5_moe")
        vision_session = self.check_ort(os.path.join(output_dir, "vision.onnx"))
        embedding_session = self.check_ort(os.path.join(output_dir, "embedding.onnx"))
        vision_tolerance = {"fp32": 2e-4, "fp16": 3e-2, "int4": 0.15}[precision]
        embedding_tolerance = {"fp32": 2e-4, "fp16": 3e-2, "int4": 2e-4}[precision]
        vc = config.vision_config
        for temporal in (1, 2):
            grid = torch.tensor([[temporal, 2, 2]], dtype=torch.int64)
            torch.manual_seed(temporal)
            pixels = torch.randn(temporal * 4, vc.in_channels * vc.temporal_patch_size * vc.patch_size**2)
            with torch.no_grad():
                expected = model.model.visual(pixels, grid).pooler_output.numpy()
            (features,) = vision_session.run(None, {"pixel_values": pixels.numpy(), "image_grid_thw": grid.numpy()})
            np.testing.assert_allclose(features, expected, atol=vision_tolerance, rtol=vision_tolerance)
            token = config.image_token_id if temporal == 1 else config.video_token_id
            ids = np.array([[5] + [token] * temporal + [6]], dtype=np.int64)
            (embeds,) = embedding_session.run(None, {"input_ids": ids, "image_features": features})
            with torch.no_grad():
                input_ids = torch.from_numpy(ids)
                expected_embeds = model.model.language_model.embed_tokens(input_ids).numpy()
            expected_embeds[0, 1:-1] = features
            np.testing.assert_allclose(embeds, expected_embeds, atol=embedding_tolerance, rtol=embedding_tolerance)
            self._check_decoder(config, model, output_dir, precision=precision, inputs_embeds=embeds)
        return config, model, output_dir

    @hide_stdout()
    def test_multimodal(self):
        self._check_multimodal()

    @hide_stdout()
    def test_fp8_multimodal(self):
        self._check_multimodal(fp8=True)

    @hide_stdout()
    def test_fp16_multimodal(self):
        self._check_multimodal(precision="fp16")

    @hide_stdout()
    def test_fp8_fp16_multimodal(self):
        self._check_multimodal(fp8=True, precision="fp16")

    @hide_stdout()
    def test_int4_multimodal(self):
        self._check_multimodal(precision="int4")

    @hide_stdout()
    def test_fp8_int4_multimodal(self):
        self._check_multimodal(fp8=True, precision="int4")

    def _check_generation(self, generation_test, fp8=False, text_only=False, multimodal=False):
        config, model, output_dir = self._build(fp8=fp8, text_only=text_only, multimodal=multimodal)
        tc = getattr(config, "text_config", config)
        generation_test(output_dir, model, tc.vocab_size, tc.eos_token_id)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_genai_generation(self):
        self._check_generation(self.run_genai_generation_test)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_genai_generation(self):
        self._check_generation(self.run_genai_generation_test, fp8=True)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_causallm_genai_generation(self):
        self._check_generation(self.run_genai_generation_test, text_only=True)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_causallm_genai_generation(self):
        self._check_generation(self.run_genai_generation_test, fp8=True, text_only=True)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_multimodal_text_genai_generation(self):
        self._check_generation(self.run_genai_generation_test, multimodal=True)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_multimodal_text_genai_generation(self):
        self._check_generation(self.run_genai_generation_test, fp8=True, multimodal=True)

    def _check_multimodal_generation(self, fp8=False, video=False, precision="fp32"):
        import onnxruntime_genai as og
        import torch

        config, model, directory = self._build(fp8=fp8, multimodal=True, precision=precision)
        token = config.video_token_id if video else config.image_token_id
        temporal, height, width = (2, 2, 2) if video else (1, 4, 4)
        tokens_per_frame = height * width // config.vision_config.spatial_merge_size**2
        frame = [config.vision_start_token_id] + [token] * tokens_per_frame + [config.vision_end_token_id]
        prompt_ids = [5]
        for frame_index in range(temporal):
            prompt_ids.extend(frame + [6 + frame_index])
        prompt = torch.tensor([prompt_ids])
        token_types = torch.where(prompt == token, 2 if video else 1, 0)
        patch_width = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size**2
        patch_count = temporal * height * width
        pixels = torch.linspace(-0.5, 0.5, patch_count * patch_width).reshape(patch_count, patch_width)
        grid = torch.tensor([[temporal, height, width]], dtype=torch.int64)
        media = {"pixel_values_videos": pixels, "video_grid_thw": grid} if video else {"pixel_values": pixels, "image_grid_thw": grid}
        with torch.no_grad():
            expected = model.generate(
                input_ids=prompt, mm_token_type_ids=token_types, **media, max_new_tokens=3, do_sample=False, pad_token_id=2
            )[0].tolist()
        genai_model = og.Model(directory)
        params = og.GeneratorParams(genai_model)
        params.set_search_options(do_sample=False, max_length=prompt.shape[1] + 3, temperature=1.0, top_k=1)
        generator = og.Generator(genai_model, params)
        inputs = og.NamedTensors()
        inputs["pixel_values"] = pixels.numpy()
        inputs["image_grid_thw"] = grid.numpy()
        inputs["num_image_tokens"] = np.array([temporal * tokens_per_frame], dtype=np.int64)
        if video:
            inputs["video_grid_thw"] = np.tile([1, height, width], (temporal, 1)).astype(np.int64)
        generator.set_inputs(inputs)
        generator.append_tokens(prompt.numpy())
        actual = prompt[0].tolist()
        while not generator.is_done():
            generator.generate_next_token()
            actual.append(int(generator.get_next_tokens()[0]))
        self.assertEqual(expected, actual)
        self.assertEqual(len(actual), prompt.shape[1] + 3)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_multimodal_genai_generation(self):
        self._check_multimodal_generation()

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_multimodal_genai_generation(self):
        self._check_multimodal_generation(fp8=True)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_video_genai_generation(self):
        self._check_multimodal_generation(video=True)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_video_genai_generation(self):
        self._check_multimodal_generation(fp8=True, video=True)

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp16_multimodal_genai_generation(self):
        self._check_multimodal_generation(precision="fp16")

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_fp16_multimodal_genai_generation(self):
        self._check_multimodal_generation(fp8=True, precision="fp16")

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp16_video_genai_generation(self):
        self._check_multimodal_generation(video=True, precision="fp16")

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_fp16_video_genai_generation(self):
        self._check_multimodal_generation(fp8=True, video=True, precision="fp16")

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_int4_multimodal_genai_generation(self):
        self._check_multimodal_generation(precision="int4")

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_int4_multimodal_genai_generation(self):
        self._check_multimodal_generation(fp8=True, precision="int4")

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_int4_video_genai_generation(self):
        self._check_multimodal_generation(video=True, precision="int4")

    @hide_stdout()
    @requires_genai("0.17.1")
    def test_fp8_int4_video_genai_generation(self):
        self._check_multimodal_generation(fp8=True, video=True, precision="int4")

    @hide_stdout()
    def test_onnx_light_generation(self):
        self._check_generation(self.run_onnx_light_generation_test)

    @hide_stdout()
    def test_fp8_onnx_light_generation(self):
        self._check_generation(self.run_onnx_light_generation_test, fp8=True)

    @hide_stdout()
    def test_causallm_onnx_light_generation(self):
        self._check_generation(self.run_onnx_light_generation_test, text_only=True)

    @hide_stdout()
    def test_fp8_causallm_onnx_light_generation(self):
        self._check_generation(self.run_onnx_light_generation_test, fp8=True, text_only=True)

    @hide_stdout()
    def test_multimodal_onnx_light_generation(self):
        # Text-only split-pipeline generation works natively. Image/video
        # generation still needs xadupre/onnx-light-cpu#1111 (vision Gelu) and
        # xadupre/onnx-light#5801 (prefill-only embedding features).
        config, model, directory = self._build(multimodal=True)
        self.run_onnx_light_generation_test(directory, model, config.text_config.vocab_size, config.text_config.eos_token_id)

    @hide_stdout()
    def test_fp8_multimodal_onnx_light_generation(self):
        config, model, directory = self._build(fp8=True, multimodal=True)
        self.run_onnx_light_generation_test(directory, model, config.text_config.vocab_size, config.text_config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
