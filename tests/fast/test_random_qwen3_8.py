# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_genai, requires_transformers

QWEN3_8_MODEL_NAME = "Qwen/Qwen3.8-27B"
TERNARY_BONSAI_MODEL_NAME = "prism-ml/Ternary-Bonsai-27B-unpacked"


def _make_qwen3_8_config():
    """Create a tiny offline configuration with the Qwen3.8 architecture."""
    from transformers import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

    rope_parameters = {
        "mrope_interleaved": True,
        "mrope_section": [2, 3, 3],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000,
        "rope_type": "default",
    }
    text_config = Qwen3_5TextConfig(
        attention_bias=False,
        attn_output_gate=True,
        bos_token_id=1,
        eos_token_id=2,
        full_attention_interval=4,
        head_dim=64,
        hidden_size=128,
        intermediate_size=256,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=16,
        linear_num_key_heads=1,
        linear_num_value_heads=3,
        linear_value_head_dim=16,
        max_position_embeddings=256,
        num_attention_heads=2,
        num_hidden_layers=4,
        num_key_value_heads=1,
        output_gate_type="swish",
        partial_rotary_factor=0.25,
        rms_norm_eps=1e-6,
        rope_parameters=rope_parameters,
        vocab_size=128,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_act="gelu_pytorch_tanh",
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        num_position_embeddings=16,
        out_hidden_size=128,
        patch_size=4,
        spatial_merge_size=2,
        temporal_patch_size=2,
    )
    config = Qwen3_5Config(
        image_token_id=120,
        text_config=text_config,
        tie_word_embeddings=False,
        video_token_id=121,
        vision_config=vision_config,
        vision_end_token_id=123,
        vision_start_token_id=122,
    )
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    return config


@requires_transformers("5")
class TestRandomQwen3_8(ExtTestCase):
    @staticmethod
    def _ternarize_language_weights(model):
        import torch

        with torch.no_grad():
            for name, module in model.named_modules():
                if name.startswith("model.visual") or not isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                    continue
                weight = module.weight
                padded_width = ((weight.shape[1] + 127) // 128) * 128
                padded = torch.nn.functional.pad(weight, (0, padded_width - weight.shape[1]))
                grouped = padded.reshape(weight.shape[0], -1, 128)
                scales = grouped.abs().amax(dim=-1, keepdim=True)
                safe_scales = torch.where(scales == 0, 1, scales)
                ternary = (grouped / safe_scales).round().clamp(-1, 1) * scales
                weight.copy_(ternary.reshape(weight.shape[0], -1)[:, : weight.shape[1]])

    def _build_multimodal_model(self, ternary=False):
        import torch
        from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor, Qwen3_5ForConditionalGeneration

        from modelbuilder.builder import create_model

        config = _make_qwen3_8_config()
        prefix = "test_random_ternary_bonsai_multimodal" if ternary else "test_random_qwen3_8_multimodal"
        model_dir = self.get_model_dir(prefix)
        output_dir, cache_dir = self.get_dirs(prefix)

        torch.manual_seed(42)
        model = Qwen3_5ForConditionalGeneration(config)
        model.eval()
        if ternary:
            self._ternarize_language_weights(model)
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        processor = Qwen3VLProcessor(
            image_processor=Qwen2VLImageProcessor(
                patch_size=config.vision_config.patch_size,
                temporal_patch_size=config.vision_config.temporal_patch_size,
                merge_size=config.vision_config.spatial_merge_size,
            ),
            tokenizer=tokenizer,
            video_processor=Qwen3VLVideoProcessor(
                patch_size=config.vision_config.patch_size,
                temporal_patch_size=config.vision_config.temporal_patch_size,
                merge_size=config.vision_config.spatial_merge_size,
            ),
        )
        processor.save_pretrained(model_dir)

        create_model(
            model_name=TERNARY_BONSAI_MODEL_NAME if ternary else QWEN3_8_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="int2" if ternary else "fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            multimodal=True,
            int4_algo_config="ternary" if ternary else "default",
        )
        return config, model, output_dir

    @hide_stdout()
    def test_qwen3_8_dispatches_to_qwen3_5_builder(self):
        from onnx_light.onnx import TensorProto

        from modelbuilder.builders.qwen import Qwen35TextModel

        config = _make_qwen3_8_config()
        model = Qwen35TextModel(config, TensorProto.FLOAT, TensorProto.FLOAT, "cpu", "", {})
        self.assertIsInstance(model, Qwen35TextModel)
        self.assertEqual(model.layer_types.count("linear_attention"), 3)
        self.assertEqual(model.layer_types.count("full_attention"), 1)

    @hide_stdout()
    def test_qwen3_8_multimodal_conversion_and_parity(self):
        import torch

        config, model, output_dir = self._build_multimodal_model()
        expected_files = {"embedding.onnx", "genai_config.json", "model.onnx", "processor_config.json", "vision.onnx"}
        self.assertTrue(expected_files.issubset(set(os.listdir(output_dir))), os.listdir(output_dir))

        with open(os.path.join(output_dir, "genai_config.json")) as file:
            genai_config = json.load(file)
        self.assertEqual(genai_config["model"]["type"], "qwen3_5")
        self.assertEqual(genai_config["model"]["vision"]["filename"], "vision.onnx")
        self.assertEqual(genai_config["model"]["vision"]["inputs"], {"image_grid_thw": "image_grid_thw", "pixel_values": "pixel_values"})
        self.assertEqual(genai_config["model"]["embedding"]["filename"], "embedding.onnx")
        self.assertEqual(genai_config["model"]["image_token_id"], 120)
        self.assertEqual(genai_config["model"]["video_token_id"], 121)
        decoder = genai_config["model"]["decoder"]
        self.assertEqual(decoder["num_hidden_layers"], 1)
        self.assertEqual(decoder["inputs"]["past_conv_names"], "past_key_values.%d.conv_state")
        self.assertEqual(decoder["inputs"]["past_recurrent_names"], "past_key_values.%d.recurrent_state")
        self.assertEqual(decoder["outputs"]["present_conv_names"], "present.%d.conv_state")
        self.assertEqual(decoder["outputs"]["present_recurrent_names"], "present.%d.recurrent_state")

        vision_session = self.check_ort(os.path.join(output_dir, "vision.onnx"))
        embedding_session = self.check_ort(os.path.join(output_dir, "embedding.onnx"))
        vision_config = config.vision_config
        input_width = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2

        for temporal, height, width in ((1, 4, 4), (1, 4, 6), (2, 4, 4)):
            grid = torch.tensor([[temporal, height, width]], dtype=torch.int64)
            torch.manual_seed(temporal * 100 + height * 10 + width)
            pixel_values = torch.randn(temporal * height * width, input_width)
            with torch.no_grad():
                expected = model.model.visual(pixel_values, grid).pooler_output.numpy()
            (actual,) = vision_session.run(None, {"image_grid_thw": grid.numpy(), "pixel_values": pixel_values.numpy()})
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)

        image_features = actual
        input_ids = np.array([[5] + [config.image_token_id] * image_features.shape[0] + [6]], dtype=np.int64)
        (inputs_embeds,) = embedding_session.run(None, {"image_features": image_features, "input_ids": input_ids})
        with torch.no_grad():
            expected_embeds = model.model.language_model.embed_tokens(torch.from_numpy(input_ids)).numpy()
        expected_embeds[0, 1:-1] = image_features
        np.testing.assert_allclose(inputs_embeds, expected_embeds, atol=1e-6, rtol=1e-6)

    @hide_stdout()
    def test_ternary_bonsai_int2_onnxruntime(self):
        import onnx_light.onnx as onnx
        import torch

        config, model, output_dir = self._build_multimodal_model(ternary=True)
        text_proto = onnx.load(os.path.join(output_dir, "model.onnx"), load_external_data=False)
        vision_proto = onnx.load(os.path.join(output_dir, "vision.onnx"), load_external_data=False)
        embedding_proto = onnx.load(os.path.join(output_dir, "embedding.onnx"), load_external_data=False)

        text_bits = {
            attribute.i
            for node in text_proto.graph.node
            if node.op_type == "MatMulNBits"
            for attribute in node.attribute
            if attribute.name == "bits"
        }
        vision_bits = {
            attribute.i
            for node in vision_proto.graph.node
            if node.op_type == "MatMulNBits"
            for attribute in node.attribute
            if attribute.name == "bits"
        }
        embedding_nodes = [node for node in embedding_proto.graph.node if node.op_type == "GatherBlockQuantized"]
        self.assertEqual(text_bits, {2})
        self.assertEqual(vision_bits, {4})
        self.assertEqual(len(embedding_nodes), 1)
        embedding_attributes = {attribute.name: attribute.i for attribute in embedding_nodes[0].attribute}
        self.assertEqual(embedding_attributes["bits"], 2)
        self.assertEqual(embedding_attributes["block_size"], 128)

        text_session = self.check_ort(os.path.join(output_dir, "model.onnx"))
        self.check_ort(os.path.join(output_dir, "vision.onnx"))

        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.int64)
        with torch.no_grad():
            inputs_embeds = model.model.language_model.embed_tokens(input_ids).numpy()
            expected_logits = model(input_ids=input_ids, use_cache=False).logits.numpy()
        sequence_length = input_ids.shape[1]
        positions = np.arange(sequence_length, dtype=np.int64)
        text_inputs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((1, sequence_length), dtype=np.int64),
            "position_ids": np.stack([positions, positions, positions], axis=0)[:, np.newaxis, :],
        }
        text_config = config.text_config
        conv_width = (
            text_config.linear_num_key_heads * text_config.linear_key_head_dim * 2
            + text_config.linear_num_value_heads * text_config.linear_value_head_dim
        )
        for layer_id, layer_type in enumerate(text_config.layer_types):
            if layer_type == "full_attention":
                text_inputs[f"past_key_values.{layer_id}.key"] = np.zeros(
                    (1, text_config.num_key_value_heads, 0, text_config.head_dim), dtype=np.float32
                )
                text_inputs[f"past_key_values.{layer_id}.value"] = np.zeros(
                    (1, text_config.num_key_value_heads, 0, text_config.head_dim), dtype=np.float32
                )
            else:
                text_inputs[f"past_key_values.{layer_id}.conv_state"] = np.zeros(
                    (1, conv_width, text_config.linear_conv_kernel_dim - 1), dtype=np.float32
                )
                text_inputs[f"past_key_values.{layer_id}.recurrent_state"] = np.zeros(
                    (1, text_config.linear_num_value_heads, text_config.linear_key_head_dim, text_config.linear_value_head_dim),
                    dtype=np.float32,
                )
        text_input_names = {value.name for value in text_session.get_inputs()}
        actual_logits, *_ = text_session.run(None, {name: value for name, value in text_inputs.items() if name in text_input_names})
        np.testing.assert_allclose(actual_logits, expected_logits, rtol=2e-2, atol=2e-2)

        embedding_session = self.check_ort(os.path.join(output_dir, "embedding.onnx"))
        input_ids = np.array([[5, config.image_token_id, 6]], dtype=np.int64)
        image_features = np.arange(config.text_config.hidden_size, dtype=np.float32).reshape(1, -1)
        (actual,) = embedding_session.run(None, {"image_features": image_features, "input_ids": input_ids})
        with torch.no_grad():
            expected = model.model.language_model.embed_tokens(torch.from_numpy(input_ids)).numpy()
        expected[0, 1] = image_features[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    @hide_stdout()
    @requires_genai("0.16", "Qwen3.5 hybrid states require ONNX Runtime GenAI 0.16 or newer.")
    def test_ternary_bonsai_int2_genai_generation(self):
        import onnxruntime_genai as og
        import torch

        _, model, output_dir = self._build_multimodal_model(ternary=True)
        prompt = np.array([5, 6, 7], dtype=np.int64)
        with torch.no_grad():
            expected_token = int(model(input_ids=torch.from_numpy(prompt[np.newaxis, :]), use_cache=False).logits[0, -1].argmax())

        genai_model = og.Model(output_dir)
        params = og.GeneratorParams(genai_model)
        params.set_search_options(do_sample=False, max_length=len(prompt) + 1, temperature=1.0, top_k=1)
        generator = og.Generator(genai_model, params)
        generator.append_tokens(prompt)
        generator.generate_next_token()
        self.assertEqual(int(generator.get_next_tokens()[0]), expected_token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
