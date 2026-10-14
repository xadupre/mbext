# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Offline checks for MLX groupwise checkpoint conversion."""

import os
import tempfile

import numpy as np
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as helper
import onnx_light.onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import torch
from onnx_light.onnx import TensorProto
from safetensors.numpy import save_file

from modelbuilder.ext_test_case import ExtTestCase, requires_transformers
from modelbuilder.helpers.mlx import MLXCheckpoint, repack_mlx


class TestMLXCheckpoint(ExtTestCase):
    def _group(self, bits, group_size=128):
        rows, columns = group_size * 2, 3
        levels = np.arange(rows * columns, dtype=np.uint8).reshape(columns, rows) % (1 << bits)
        words = np.zeros((columns, rows * bits // 32), dtype=np.uint32)
        for index in range(32 // bits):
            words |= levels[:, index :: 32 // bits].astype(np.uint32) << (index * bits)
        scales = np.array([[0.25, 0.5], [0.75, 1.0], [1.25, 1.5]], dtype=np.float32)
        biases = np.array([[-0.125, -0.5], [-0.25, -0.75], [0.25, -1.0]], dtype=np.float32)
        return levels, words, scales, biases

    def test_packing_and_affine_cpu(self):
        for bits in (2, 4):
            with self.subTest(bits=bits):
                levels, words, scales, biases = self._group(bits)
                packed, result_scales, zeros, residual = repack_mlx(words, scales, biases, bits=bits)
                self.assertEqual(packed.shape, (3, 2, 128 * bits // 8))
                np.testing.assert_array_equal(result_scales, scales)
                zp = np.array([(zeros[:, 0] >> (index * bits)) & ((1 << bits) - 1) for index in range(2)]).T
                reconstructed = (levels.reshape(3, 2, 128).astype(np.float32) - zp[:, :, None]) * scales[:, :, None] + residual[:, :, None]
                reference = levels.reshape(3, 2, 128) * scales[:, :, None] + biases[:, :, None]
                np.testing.assert_allclose(reconstructed, reference, atol=1e-6)

                x = np.arange(256, dtype=np.float32).reshape(1, 256) / 256
                graph = helper.make_graph(
                    [
                        helper.make_node(
                            "MatMulNBits",
                            ["x", "packed", "scales", "zeros"],
                            ["quantized"],
                            domain="com.microsoft",
                            K=256,
                            N=3,
                            bits=bits,
                            block_size=128,
                        ),
                        helper.make_node("Reshape", ["x", "group_shape"], ["grouped"]),
                        helper.make_node("ReduceSum", ["grouped", "axes"], ["summed"], keepdims=0),
                        helper.make_node("MatMul", ["summed", "residual"], ["correction"]),
                        helper.make_node("Add", ["quantized", "correction"], ["y"]),
                    ],
                    "mlx",
                    [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 256])],
                    [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])],
                    [
                        numpy_helper.from_array(value, name=name)
                        for name, value in (
                            ("packed", packed),
                            ("scales", scales),
                            ("zeros", zeros),
                            ("group_shape", np.array([1, 2, 128], np.int64)),
                            ("axes", np.array([2], np.int64)),
                            ("residual", residual.T.astype(np.float32)),
                        )
                    ],
                )
                model = helper.make_model(
                    graph, ir_version=10, opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)]
                )
                result = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": x})[0]
                np.testing.assert_allclose(result, x @ reference.reshape(3, 256).T, rtol=1e-5, atol=1e-4)

    def test_invalid_metadata(self):
        _, words, scales, biases = self._group(2)
        for weight, scale, bias, message in (
            (words.astype(np.int32), scales, biases, "uint32"),
            (words, scales[:, :1], biases, "group shapes"),
            (words, scales, np.full_like(biases, np.nan), "finite"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    repack_mlx(weight, scale, bias)
        with self.assertRaisesRegex(ValueError, "2 or 4 bits"):
            repack_mlx(words, scales, biases, bits=3)

    def test_zero_scale_constant_group(self):
        _, words, scales, biases = self._group(2)
        scales[0, 0] = 0
        biases[0, 0] = 0.25
        _, _, zeros, residual = repack_mlx(words, scales, biases)
        self.assertEqual(int(zeros[0, 0] & 3), 0)
        self.assertEqual(residual[0, 0], 0.25)

    def test_safetensors_index(self):
        _, words, scales, biases = self._group(2)
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {
                    "model.language_model.layers.0.mlp.gate_proj.weight": words,
                    "model.language_model.layers.0.mlp.gate_proj.scales": scales,
                    "model.language_model.layers.0.mlp.gate_proj.biases": biases,
                },
                f"{directory}/model.safetensors",
            )
            checkpoint = MLXCheckpoint(directory)
            np.testing.assert_array_equal(checkpoint.tensor("model.language_model.layers.0.mlp.gate_proj.weight").numpy(), words)
        with tempfile.TemporaryDirectory() as directory:
            save_file({"layer.weight": words, "layer.scales": scales}, f"{directory}/model.safetensors")
            with self.assertRaisesRegex(ValueError, "Incomplete MLX"):
                MLXCheckpoint(directory)

    @staticmethod
    def _mlx_name(name):
        if name.startswith("model.language_model."):
            return "language_model.model." + name[len("model.language_model.") :]
        if name.startswith("lm_head."):
            return "language_model.lm_head." + name[len("lm_head.") :]
        return name

    def _decoder_step(self, session, embeddings, positions, attention_mask, past=None):
        feeds = {}
        for entry in session.get_inputs():
            if entry.name == "inputs_embeds":
                feeds[entry.name] = embeddings
            elif entry.name == "position_ids":
                feeds[entry.name] = positions
            elif entry.name == "attention_mask":
                feeds[entry.name] = attention_mask
            elif entry.name.startswith("past_key_values."):
                output_name = "present." + entry.name[len("past_key_values.") :]
                if past is not None:
                    feeds[entry.name] = past[output_name]
                else:
                    shape = [1 if size == "batch_size" else (0 if size == "past_sequence_length" else size) for size in entry.shape]
                    feeds[entry.name] = np.zeros(shape, dtype=np.float32)
        return dict(zip((output.name for output in session.get_outputs()), session.run(None, feeds)))

    @requires_transformers("5")
    def test_tiny_qwen_components(self):
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig

        from modelbuilder.builders.qwen import Qwen35EmbeddingModel, Qwen35TextModel, Qwen35VisionEncoderModel

        rope = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3], "rope_theta": 10000.0, "partial_rotary_factor": 0.25}
        text = Qwen3_5TextConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=256,
            vocab_size=64,
            layer_types=["full_attention", "linear_attention"],
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_conv_kernel_dim=4,
        )
        text.rope_parameters = rope
        text.rope_scaling = rope
        vision = Qwen3_5VisionConfig(
            hidden_size=128,
            intermediate_size=256,
            depth=1,
            num_heads=4,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=256,
        )
        config = Qwen3_5Config(text_config=text, vision_config=vision, bos_token_id=1, eos_token_id=2)
        config.architectures = ["Qwen3_5ForConditionalGeneration"]
        text.architectures = config.architectures
        with torch.device("meta"):
            skeleton = Qwen3_5ForConditionalGeneration(config)
        with tempfile.TemporaryDirectory() as directory:
            rng = np.random.default_rng(42)
            tensors = {}
            reference_state = {}
            for name, value in skeleton.state_dict().items():
                module = skeleton.get_submodule(name.rsplit(".", 1)[0])
                width = value.numel() // value.shape[0] if value.ndim else 0
                bits = 4 if name.startswith("model.visual.") else 2
                group = 32 if bits == 4 else 128
                if (
                    name.endswith(".weight")
                    and isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
                    and width >= group
                    and width % group == 0
                ):
                    outputs = value.shape[0]
                    levels = np.arange(outputs * width, dtype=np.uint32).reshape(outputs, width) % (1 << bits)
                    words = np.zeros((outputs, width * bits // 32), dtype=np.uint32)
                    for index in range(32 // bits):
                        words |= levels[:, index :: 32 // bits] << (index * bits)
                    source_name = self._mlx_name(name)
                    scales = np.full((outputs, width // group), 0.125, dtype=np.float32)
                    biases = np.full((outputs, width // group), -0.0625, dtype=np.float32)
                    tensors[source_name] = words
                    tensors[source_name[:-7] + ".scales"] = scales
                    tensors[source_name[:-7] + ".biases"] = biases
                    reference_array = levels.astype(np.float32) * scales.repeat(group, axis=1) + biases.repeat(group, axis=1)
                    reference_state[name] = torch.from_numpy(reference_array).to(dtype=value.dtype)
                else:
                    source_name = self._mlx_name(name)
                    if name.endswith("linear_attn.conv1d.weight"):
                        reference_array = np.arange(value.numel(), dtype=np.float16).reshape(value.shape)
                        tensors[source_name] = reference_array.transpose(0, 2, 1).copy()
                    elif name.endswith("linear_attn.norm.weight"):
                        reference_array = np.full(value.shape, 0.25, dtype=np.float16)
                        tensors[source_name] = reference_array
                    elif name.endswith(
                        ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight")
                    ) and source_name.startswith("language_model.model."):
                        reference_array = np.zeros(value.shape, dtype=np.float16)
                        tensors[source_name] = reference_array + 1
                    elif name == "model.language_model.norm.weight":
                        reference_array = np.zeros(value.shape, dtype=np.float16)
                        tensors[source_name] = reference_array + 1
                    elif name.startswith("model.visual.") and value.is_floating_point():
                        reference_array = rng.normal(0, 0.05, size=value.shape).astype(np.float32)
                        if name.endswith((".norm1.weight", ".norm2.weight")):
                            reference_array += 1
                        tensors[source_name] = reference_array
                    else:
                        reference_array = np.zeros(value.shape, dtype=np.float16)
                        tensors[source_name] = reference_array
                    reference_state[name] = torch.from_numpy(reference_array.copy()).to(dtype=value.dtype)
            save_file(tensors, f"{directory}/model.safetensors")
            reference_model = Qwen3_5ForConditionalGeneration(config)
            reference_model.load_state_dict(reference_state)
            reference_model.eval()
            reference_model_directory = f"{directory}/reference_model"
            reference_model.save_pretrained(reference_model_directory)
            options = {"_mlx_checkpoint": directory, "disable_qkv_fusion": True, "int4_bits": 2}
            checkpoint = MLXCheckpoint(directory)
            loaded = checkpoint.load_model(config, "text")
            self.assertTrue(hasattr(loaded.model.language_model.layers[0].mlp.gate_proj, "qweight"))
            self.assertTrue(loaded.model.language_model.layers[0].mlp.gate_proj.weight.is_meta)
            self.assertEqual(loaded.model.language_model.layers[0].input_layernorm.weight.dtype, torch.float16)
            self.assertTrue(torch.all(loaded.model.language_model.layers[0].input_layernorm.weight == 0))
            self.assertTrue(torch.all(loaded.model.language_model.layers[0].self_attn.q_norm.weight == 0))
            self.assertTrue(torch.all(loaded.model.language_model.norm.weight == 0))
            self.assertTrue(torch.all(loaded.model.language_model.layers[1].linear_attn.norm.weight == 0.25))
            np.testing.assert_array_equal(
                loaded.model.language_model.layers[1].linear_attn.conv1d.weight.cpu().numpy(),
                np.arange(loaded.model.language_model.layers[1].linear_attn.conv1d.weight.numel(), dtype=np.float16).reshape(
                    loaded.model.language_model.layers[1].linear_attn.conv1d.weight.shape
                ),
            )
            hf_tensors = {}
            for source_name, value in tensors.items():
                if source_name.startswith("language_model.model."):
                    name = "model.language_model." + source_name[len("language_model.model.") :]
                elif source_name.startswith("language_model.lm_head."):
                    name = "lm_head." + source_name[len("language_model.lm_head.") :]
                else:
                    name = source_name
                if name.endswith(
                    (
                        "input_layernorm.weight",
                        "post_attention_layernorm.weight",
                        "self_attn.q_norm.weight",
                        "self_attn.k_norm.weight",
                        "model.language_model.norm.weight",
                    )
                ):
                    value = reference_state[name].numpy()
                elif name.endswith("linear_attn.conv1d.weight"):
                    value = reference_state[name].numpy()
                hf_tensors[name] = value
            hf_directory = f"{directory}/hf_style"
            os.mkdir(hf_directory)
            save_file(hf_tensors, f"{hf_directory}/model.safetensors")
            hf_loaded = MLXCheckpoint(hf_directory).load_model(config, "text")
            self.assertTrue(torch.all(hf_loaded.model.language_model.layers[0].input_layernorm.weight == 0))
            np.testing.assert_array_equal(
                hf_loaded.model.language_model.layers[1].linear_attn.conv1d.weight.cpu().numpy(),
                reference_state[
                    "model.language_model.layers.1.linear_attn.conv1d.weight"
                ].numpy(),
            )
            loaded_vision = checkpoint.load_model(config, "vision")
            self.assertTrue(
                torch.count_nonzero(
                    loaded_vision.model.visual.pos_embed.mlx_residual
                )
            )

            for builder_type, name, builder_config, extra in (
                (Qwen35VisionEncoderModel, "vision.onnx", config, options),
                (Qwen35EmbeddingModel, "embedding.onnx", text, {**options, "_mlx_config": config, "image_token_id": config.image_token_id}),
                (Qwen35TextModel, "model.onnx", config, options),
            ):
                builder = builder_type(
                    builder_config,
                    TensorProto.FLOAT,
                    (TensorProto.FLOAT if name == "embedding.onnx" else TensorProto.UINT4),
                    "cpu",
                    directory,
                    dict(extra),
                )
                builder.make_model(directory)
                builder.save_model(directory)
                graph = onnx.load_model(f"{directory}/{name}", load_external_data=False).graph
                quantized_nodes = [node for node in graph.node if node.op_type in ("MatMulNBits", "GatherBlockQuantized")]
                self.assertTrue(quantized_nodes)
                expected_bits = 4 if name == "vision.onnx" else 2
                self.assertTrue(
                    any(
                        attribute.i == expected_bits for node in quantized_nodes for attribute in node.attribute if attribute.name == "bits"
                    )
                )
                if name == "model.onnx":
                    self.assertNotIn("model.layers.0.mlp.gate_proj.MatMul.weight", {initializer.name for initializer in graph.initializer})
                session = ort.InferenceSession(f"{directory}/{name}", providers=["CPUExecutionProvider"])
                self.assertTrue(session.get_outputs())
                if name == "embedding.onnx":
                    embeddings = session.run(
                        None, {"input_ids": np.array([[2, 3]], dtype=np.int64), "image_features": np.empty((0, 256), dtype=np.float32)}
                    )[0]
                    self.assertEqual(embeddings.shape, (1, 2, 256))
                    levels = np.arange(256, dtype=np.float32) % 4
                    np.testing.assert_allclose(embeddings[0, 0], levels * 0.125 - 0.0625, atol=1e-6)
                elif name == "vision.onnx":
                    pixel_values = rng.normal(0, 0.25, size=(4, 24)).astype(np.float32)
                    image_grid_thw = np.array([[1, 2, 2]], dtype=np.int64)
                    images = session.run(None, {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw})[0]
                    self.assertEqual(images.shape, (1, 256))
                    with torch.no_grad():
                        expected_images = reference_model.model.visual(
                            torch.from_numpy(pixel_values), torch.from_numpy(image_grid_thw)
                        ).pooler_output.numpy()
                    np.testing.assert_allclose(images, expected_images, rtol=2e-4, atol=2e-4)
                else:
                    feeds = {
                        "inputs_embeds": np.zeros((1, 2, 256), dtype=np.float32),
                        "attention_mask": np.ones((1, 2), dtype=np.int64),
                        "position_ids": np.zeros((3, 1, 2), dtype=np.int64),
                    }
                    for entry in session.get_inputs():
                        if entry.name.startswith("past_key_values."):
                            feeds[entry.name] = np.zeros(
                                [(1 if size == "batch_size" else (0 if size == "past_sequence_length" else size)) for size in entry.shape],
                                dtype=np.float32,
                            )
                    logits = session.run(None, feeds)[0]
                    self.assertEqual(logits.shape, (1, 2, 64))
                    reference_directory = f"{directory}/reference_onnx"
                    os.mkdir(reference_directory)
                    reference_config = config.__class__.from_dict(config.to_dict())
                    reference_config._name_or_path = reference_model_directory
                    reference_builder = Qwen35TextModel(
                        reference_config, TensorProto.FLOAT, TensorProto.FLOAT, "cpu", reference_directory, {"disable_qkv_fusion": True}
                    )
                    reference_builder.make_model(reference_model_directory)
                    reference_builder.save_model(reference_directory)
                    reference_session = ort.InferenceSession(f"{reference_directory}/model.onnx", providers=["CPUExecutionProvider"])
                    embeddings = np.arange(3 * 256, dtype=np.float32).reshape(1, 3, 256) / 768
                    positions = np.broadcast_to(np.arange(3, dtype=np.int64), (3, 1, 3)).copy()
                    mask = np.ones((1, 3), dtype=np.int64)
                    mlx_prefill = self._decoder_step(session, embeddings, positions, mask)
                    dense_prefill = self._decoder_step(reference_session, embeddings, positions, mask)
                    np.testing.assert_allclose(mlx_prefill["logits"], dense_prefill["logits"], rtol=1e-3, atol=2e-3)
                    decode_positions = np.full((3, 1, 1), 3, dtype=np.int64)
                    decode_embeddings = embeddings[:, :1, :]
                    decode_mask = np.ones((1, 4), dtype=np.int64)
                    mlx_decode = self._decoder_step(session, decode_embeddings, decode_positions, decode_mask, mlx_prefill)
                    dense_decode = self._decoder_step(reference_session, decode_embeddings, decode_positions, decode_mask, dense_prefill)
                    np.testing.assert_allclose(mlx_decode["logits"], dense_decode["logits"], rtol=1e-3, atol=2e-3)
