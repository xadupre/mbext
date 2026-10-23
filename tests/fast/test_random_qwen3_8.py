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
    def _quantize_fp8(weight):
        import torch

        scale = weight.abs().amax(dim=1, keepdim=True).float() / torch.finfo(torch.float8_e4m3fn).max
        scale = scale.clamp_min(torch.finfo(torch.float8_e4m3fn).tiny)
        quantized = (weight.float() / scale).clamp(min=torch.finfo(torch.float8_e4m3fn).min, max=torch.finfo(torch.float8_e4m3fn).max)
        quantized = quantized.to(torch.float8_e4m3fn)
        return quantized, scale

    @staticmethod
    def _quantize_nvfp4(weight):
        import torch

        if weight.shape[1] % 16:
            raise ValueError(f"NVFP4 requires an input dimension divisible by 16, got {weight.shape}.")
        levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
        blocks = weight.float().reshape(weight.shape[0], -1, 16)
        scale = (blocks.abs().amax(dim=-1) / 6).clamp_min(torch.finfo(torch.float8_e4m3fn).tiny)
        scale = scale.to(torch.float8_e4m3fn)
        normalized = blocks / scale.float().unsqueeze(-1)
        distances = (normalized.abs().unsqueeze(-1) - levels.reshape(1, 1, 1, -1)).abs()
        codes = distances.argmin(dim=-1).to(torch.uint8)
        codes |= (normalized < 0).to(torch.uint8) << 3
        codes = codes.reshape(weight.shape)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        return packed.contiguous(), scale.contiguous()

    def _make_native_nvfp4_checkpoint(self):
        import torch
        from safetensors.torch import save_file
        from transformers import Qwen3_5ForConditionalGeneration

        config = _make_qwen3_8_config()
        prefix = "test_random_qwen3_8_native_nvfp4"
        model_dir = self.get_model_dir(prefix)

        torch.manual_seed(42)
        model = Qwen3_5ForConditionalGeneration(config)
        model.eval()
        linear_names = {name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
        state = {}
        for name, value in model.state_dict().items():
            module_name, separator, parameter_name = name.rpartition(".")
            if separator and parameter_name == "weight" and module_name in linear_names and ".mlp." in module_name:
                packed, scale = self._quantize_nvfp4(value)
                state[f"{module_name}.weight_packed"] = packed
                state[f"{module_name}.weight_scale"] = scale
                state[f"{module_name}.weight_global_scale"] = torch.ones(1, dtype=torch.float32)
            elif separator and parameter_name == "weight" and module_name in linear_names:
                quantized, scale = self._quantize_fp8(value)
                state[name] = quantized
                state[f"{module_name}.weight_scale"] = scale
            else:
                state[name] = value

        config.quantization_config = {"quant_method": "compressed-tensors", "format": "pack-quantized"}
        config.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)
        save_file(state, os.path.join(model_dir, "model.safetensors"))
        return config, model_dir

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

    def _build_multimodal_model(self, ternary=False, precision=None):
        import torch
        from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor, Qwen3_5ForConditionalGeneration

        from modelbuilder.builder import create_model

        config = _make_qwen3_8_config()
        if precision is None:
            precision = "int2" if ternary else "fp32"
        prefix = "test_random_ternary_bonsai_multimodal" if ternary else "test_random_qwen3_8_multimodal"
        prefix = f"{prefix}_{precision}"
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
            precision=precision,
            execution_provider="cpu",
            cache_dir=cache_dir,
            multimodal=True,
            int4_algo_config="ternary" if ternary and precision == "int2" else "default",
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

    def _check_quantized_multimodal_model(self, precision, ternary=False):
        import onnx_light.onnx as onnx
        import torch

        config, model, output_dir = self._build_multimodal_model(ternary=ternary, precision=precision)
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
        exact_ternary = ternary and precision == "int2"
        bits = int(precision[3:])
        self.assertEqual(text_bits, {2} if exact_ternary else {bits, 8})
        self.assertEqual(vision_bits, {4} if exact_ternary else {bits})
        self.assertEqual(len(embedding_nodes), 1 if exact_ternary else 0)
        if exact_ternary:
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
        atol = 2e-2 if exact_ternary else {"int8": 0.05, "int4": 0.15 if ternary else 0.1, "int2": 0.4}[precision]
        np.testing.assert_allclose(actual_logits, expected_logits, rtol=2e-2, atol=atol)

        embedding_session = self.check_ort(os.path.join(output_dir, "embedding.onnx"))
        input_ids = np.array([[5, config.image_token_id, 6]], dtype=np.int64)
        image_features = np.arange(config.text_config.hidden_size, dtype=np.float32).reshape(1, -1)
        (actual,) = embedding_session.run(None, {"image_features": image_features, "input_ids": input_ids})
        with torch.no_grad():
            expected = model.model.language_model.embed_tokens(torch.from_numpy(input_ids)).numpy()
        expected[0, 1] = image_features[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)

    @hide_stdout()
    def test_qwen3_8_int4_onnxruntime(self):
        self._check_quantized_multimodal_model("int4")

    @hide_stdout()
    def test_qwen3_8_int2_onnxruntime(self):
        self._check_quantized_multimodal_model("int2")

    @hide_stdout()
    def test_qwen3_8_int8_onnxruntime(self):
        self._check_quantized_multimodal_model("int8")

    @hide_stdout()
    def test_qwen3_8_fp16_onnxruntime(self):
        import torch

        config, model, output_dir = self._build_multimodal_model(precision="fp16")
        session = self.check_ort(os.path.join(output_dir, "model.onnx"))
        input_ids = torch.tensor([[5, 6, 7]], dtype=torch.int64)
        with torch.no_grad():
            inputs_embeds = model.model.language_model.embed_tokens(input_ids)
            expected = model(input_ids=input_ids, use_cache=False).logits.numpy()
        sequence_length = input_ids.shape[1]
        positions = np.arange(sequence_length, dtype=np.int64)
        feed = {
            "inputs_embeds": inputs_embeds.numpy().astype(np.float16),
            "attention_mask": np.ones((1, sequence_length), dtype=np.int64),
            "position_ids": np.stack([positions, positions, positions], axis=0)[:, np.newaxis, :],
        }
        conv_width = (
            config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim * 2
            + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim
        )
        for layer_id, layer_type in enumerate(config.text_config.layer_types):
            if layer_type == "full_attention":
                feed[f"past_key_values.{layer_id}.key"] = np.zeros(
                    (1, config.text_config.num_key_value_heads, 0, config.text_config.head_dim), dtype=np.float16
                )
                feed[f"past_key_values.{layer_id}.value"] = np.zeros(
                    (1, config.text_config.num_key_value_heads, 0, config.text_config.head_dim), dtype=np.float16
                )
            else:
                feed[f"past_key_values.{layer_id}.conv_state"] = np.zeros(
                    (1, conv_width, config.text_config.linear_conv_kernel_dim - 1), dtype=np.float16
                )
                feed[f"past_key_values.{layer_id}.recurrent_state"] = np.zeros(
                    (
                        1,
                        config.text_config.linear_num_value_heads,
                        config.text_config.linear_key_head_dim,
                        config.text_config.linear_value_head_dim,
                    ),
                    dtype=np.float16,
                )
        input_names = {value.name for value in session.get_inputs()}
        actual, *_ = session.run(None, {name: value for name, value in feed.items() if name in input_names})
        np.testing.assert_allclose(actual, expected, rtol=3e-2, atol=3e-2)

    @hide_stdout()
    @requires_cuda()
    def test_qwen3_8_bf16_cuda_build(self):
        _, _, output_dir = self._build_multimodal_model(precision="bf16")
        session = self.check_ort(os.path.join(output_dir, "model.onnx"), provider="cuda")
        input_types = {value.name: value.type for value in session.get_inputs()}
        self.assertEqual(input_types["inputs_embeds"], "tensor(bfloat16)")

    @hide_stdout()
    def test_qwen3_8_native_nvfp4_checkpoint(self):
        import onnx_light.onnx as onnx

        from modelbuilder.builder import create_model

        _, model_dir = self._make_native_nvfp4_checkpoint()
        output_dir, cache_dir = self.get_dirs("test_random_qwen3_8_native_nvfp4")
        create_model(
            model_name=QWEN3_8_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="bf16",
            execution_provider="cuda",
            cache_dir=cache_dir,
        )
        model = onnx.load(os.path.join(output_dir, "model.onnx"), load_external_data=False)
        op_types = {node.op_type for node in model.graph.node}
        self.assertIn("MatMulBlockQuantizedFp4Weight", op_types)
        self.assertIn("MatMulBlockQuantizedFp8Weight", op_types)

    @hide_stdout()
    def test_ternary_bonsai_int4_onnxruntime(self):
        self._check_quantized_multimodal_model("int4", ternary=True)

    @hide_stdout()
    def test_ternary_bonsai_int2_onnxruntime(self):
        self._check_quantized_multimodal_model("int2", ternary=True)

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

    @hide_stdout()
    @requires_genai("0.16", "Qwen3.5 hybrid states require ONNX Runtime GenAI 0.16 or newer.")
    def test_qwen3_8_genai_generation(self):
        import onnxruntime_genai as og
        import torch

        _, model, output_dir = self._build_multimodal_model()
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

    @hide_stdout()
    @requires_genai("0.16", "Qwen3.5 hybrid states require ONNX Runtime GenAI 0.16 or newer.")
    def test_qwen3_8_onnx_light_cpu_genai_generation(self):
        import onnx_light_cpu

        onnx_light_cpu.register_kernels()
        self.test_qwen3_8_genai_generation()


if __name__ == "__main__":
    unittest.main(verbosity=2)
