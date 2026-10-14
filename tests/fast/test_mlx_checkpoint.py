# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Offline checks for MLX groupwise checkpoint conversion."""

import tempfile

import numpy as np
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as helper
import onnx_light.onnx.numpy_helper as numpy_helper
import onnxruntime as ort
import torch
from onnx_light.onnx import TensorProto
from safetensors.numpy import save_file

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.helpers.mlx import MLXCheckpoint, repack_mlx


class TestMLXCheckpoint(ExtTestCase):
    def _group(self, bits, group_size=128):
        rows, columns = group_size * 2, 3
        levels = np.arange(rows * columns, dtype=np.uint8).reshape(columns, rows) % (
            1 << bits
        )
        words = np.zeros((columns, rows * bits // 32), dtype=np.uint32)
        for index in range(32 // bits):
            words |= levels[:, index :: 32 // bits].astype(np.uint32) << (index * bits)
        scales = np.array([[0.25, 0.5], [0.75, 1.0], [1.25, 1.5]], dtype=np.float32)
        biases = np.array(
            [[-0.125, -0.5], [-0.25, -0.75], [0.25, -1.0]], dtype=np.float32
        )
        return levels, words, scales, biases

    def test_packing_and_affine_cpu(self):
        for bits in (2, 4):
            with self.subTest(bits=bits):
                levels, words, scales, biases = self._group(bits)
                packed, result_scales, zeros, residual = repack_mlx(
                    words, scales, biases, bits=bits
                )
                self.assertEqual(packed.shape, (3, 2, 128 * bits // 8))
                np.testing.assert_array_equal(result_scales, scales)
                zp = np.array(
                    [
                        (zeros[:, 0] >> (index * bits)) & ((1 << bits) - 1)
                        for index in range(2)
                    ]
                ).T
                reconstructed = (
                    levels.reshape(3, 2, 128).astype(np.float32) - zp[:, :, None]
                ) * scales[:, :, None] + residual[:, :, None]
                reference = (
                    levels.reshape(3, 2, 128) * scales[:, :, None] + biases[:, :, None]
                )
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
                        helper.make_node(
                            "ReduceSum", ["grouped", "axes"], ["summed"], keepdims=0
                        ),
                        helper.make_node(
                            "MatMul", ["summed", "residual"], ["correction"]
                        ),
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
                    graph,
                    ir_version=10,
                    opset_imports=[
                        helper.make_opsetid("", 21),
                        helper.make_opsetid("com.microsoft", 1),
                    ],
                )
                result = ort.InferenceSession(
                    model.SerializeToString(), providers=["CPUExecutionProvider"]
                ).run(None, {"x": x})[0]
                np.testing.assert_allclose(
                    result, x @ reference.reshape(3, 256).T, rtol=1e-5, atol=1e-4
                )

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
            np.testing.assert_array_equal(
                checkpoint.tensor(
                    "model.language_model.layers.0.mlp.gate_proj.weight"
                ).numpy(),
                words,
            )
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"layer.weight": words, "layer.scales": scales},
                f"{directory}/model.safetensors",
            )
            with self.assertRaisesRegex(ValueError, "Incomplete MLX"):
                MLXCheckpoint(directory)

    def test_tiny_qwen_components(self):
        from transformers import Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5.configuration_qwen3_5 import (
            Qwen3_5Config,
            Qwen3_5TextConfig,
            Qwen3_5VisionConfig,
        )

        from modelbuilder.builders.qwen import (
            Qwen35EmbeddingModel,
            Qwen35TextModel,
            Qwen35VisionEncoderModel,
        )

        rope = {
            "type": "mrope",
            "rope_type": "default",
            "mrope_section": [2, 3, 3],
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        }
        text = Qwen3_5TextConfig(
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            max_position_embeddings=256,
            vocab_size=64,
            layer_types=["full_attention"],
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
        config = Qwen3_5Config(
            text_config=text, vision_config=vision, bos_token_id=1, eos_token_id=2
        )
        config.architectures = ["Qwen3_5ForConditionalGeneration"]
        text.architectures = config.architectures
        with torch.device("meta"):
            skeleton = Qwen3_5ForConditionalGeneration(config)
        with tempfile.TemporaryDirectory() as directory:
            tensors = {}
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
                    levels = np.arange(outputs * width, dtype=np.uint32).reshape(
                        outputs, width
                    ) % (1 << bits)
                    words = np.zeros((outputs, width * bits // 32), dtype=np.uint32)
                    for index in range(32 // bits):
                        words |= levels[:, index :: 32 // bits] << (index * bits)
                    tensors[name] = words
                    tensors[name[:-7] + ".scales"] = np.full(
                        (outputs, width // group), 0.125, dtype=np.float32
                    )
                    tensors[name[:-7] + ".biases"] = np.full(
                        (outputs, width // group), -0.0625, dtype=np.float32
                    )
                else:
                    tensors[name] = np.zeros(value.shape, dtype=np.float16)
            save_file(tensors, f"{directory}/model.safetensors")
            options = {
                "_mlx_checkpoint": directory,
                "disable_qkv_fusion": True,
                "int4_bits": 2,
            }
            checkpoint = MLXCheckpoint(directory)
            loaded = checkpoint.load_model(config, "text")
            self.assertTrue(
                hasattr(loaded.model.language_model.layers[0].mlp.gate_proj, "qweight")
            )
            self.assertTrue(
                loaded.model.language_model.layers[0].mlp.gate_proj.weight.is_meta
            )
            self.assertEqual(
                loaded.model.language_model.layers[0].input_layernorm.weight.dtype,
                torch.float16,
            )
            for builder_type, name, builder_config, extra in (
                (Qwen35VisionEncoderModel, "vision.onnx", config, options),
                (
                    Qwen35EmbeddingModel,
                    "embedding.onnx",
                    text,
                    {
                        **options,
                        "_mlx_config": config,
                        "image_token_id": config.image_token_id,
                    },
                ),
                (Qwen35TextModel, "model.onnx", config, options),
            ):
                builder = builder_type(
                    builder_config,
                    TensorProto.FLOAT,
                    (
                        TensorProto.FLOAT
                        if name == "embedding.onnx"
                        else TensorProto.UINT4
                    ),
                    "cpu",
                    directory,
                    dict(extra),
                )
                builder.make_model(directory)
                builder.save_model(directory)
                graph = onnx.load_model(
                    f"{directory}/{name}", load_external_data=False
                ).graph
                quantized_nodes = [
                    node
                    for node in graph.node
                    if node.op_type in ("MatMulNBits", "GatherBlockQuantized")
                ]
                self.assertTrue(quantized_nodes)
                expected_bits = 4 if name == "vision.onnx" else 2
                self.assertTrue(
                    any(
                        attribute.i == expected_bits
                        for node in quantized_nodes
                        for attribute in node.attribute
                        if attribute.name == "bits"
                    )
                )
                if name == "model.onnx":
                    self.assertNotIn(
                        "model.layers.0.mlp.gate_proj.MatMul.weight",
                        {initializer.name for initializer in graph.initializer},
                    )
                session = ort.InferenceSession(
                    f"{directory}/{name}", providers=["CPUExecutionProvider"]
                )
                self.assertTrue(session.get_outputs())
                if name == "embedding.onnx":
                    embeddings = session.run(
                        None,
                        {
                            "input_ids": np.array([[2, 3]], dtype=np.int64),
                            "image_features": np.empty((0, 256), dtype=np.float32),
                        },
                    )[0]
                    self.assertEqual(embeddings.shape, (1, 2, 256))
                    levels = np.arange(256, dtype=np.float32) % 4
                    np.testing.assert_allclose(
                        embeddings[0, 0], levels * 0.125 - 0.0625, atol=1e-6
                    )
                elif name == "vision.onnx":
                    images = session.run(
                        None,
                        {
                            "pixel_values": np.zeros((4, 24), dtype=np.float32),
                            "image_grid_thw": np.array([[1, 2, 2]], dtype=np.int64),
                        },
                    )[0]
                    self.assertEqual(images.shape, (1, 256))
                else:
                    feeds = {
                        "inputs_embeds": np.zeros((1, 2, 256), dtype=np.float32),
                        "attention_mask": np.ones((1, 2), dtype=np.int64),
                        "position_ids": np.zeros((3, 1, 2), dtype=np.int64),
                    }
                    for entry in session.get_inputs():
                        if entry.name.startswith("past_key_values."):
                            feeds[entry.name] = np.zeros(
                                [
                                    (
                                        1
                                        if size == "batch_size"
                                        else (
                                            0
                                            if size == "past_sequence_length"
                                            else size
                                        )
                                    )
                                    for size in entry.shape
                                ],
                                dtype=np.float32,
                            )
                    logits = session.run(None, feeds)[0]
                    self.assertEqual(logits.shape, (1, 2, 64))
