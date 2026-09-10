# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for the dtype/token helpers in :mod:`modelbuilder.builder`.
"""

import os
import unittest

import torch
from onnx_light.onnx import load
from onnx_light.onnx.numpy_helper import to_array
from onnx_light.onnx import TensorProto as ir

from modelbuilder.builder import create_model, parse_hf_token, set_io_dtype, set_onnx_dtype
from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase


class TestParseHfToken(ExtTestCase):
    def test_disabled_variants_return_none(self):
        for value in ("false", "False", "0"):
            with self.subTest(value=value):
                self.assertIsNone(parse_hf_token(value))

    def test_enabled_variants_return_true(self):
        for value in ("true", "True", "1"):
            with self.subTest(value=value):
                self.assertIs(parse_hf_token(value), True)

    def test_user_token_returned_as_is(self):
        self.assertEqual(parse_hf_token("hf_secret_token"), "hf_secret_token")


class TestSetIoDtype(ExtTestCase):
    def test_fp32_and_int_cpu_are_float(self):
        self.assertEqual(set_io_dtype("fp32", "cpu", {}), ir.DataType.FLOAT)
        for precision in ("int2", "int4", "int8"):
            with self.subTest(precision=precision):
                self.assertEqual(set_io_dtype(precision, "cpu", {}), ir.DataType.FLOAT)

    def test_int4_cpu_is_float(self):
        self.assertEqual(set_io_dtype("int4", "cpu", {}), ir.DataType.FLOAT)

    def test_webgpu_fp32_option(self):
        self.assertEqual(set_io_dtype("int4", "webgpu", {"use_webgpu_fp32": True}), ir.DataType.FLOAT)
        self.assertEqual(set_io_dtype("int4", "webgpu", {}), ir.DataType.FLOAT16)

    def test_bf16_precision(self):
        self.assertEqual(set_io_dtype("bf16", "cpu", {}), ir.DataType.BFLOAT16)

    def test_int4_cuda_bf16_option(self):
        self.assertEqual(set_io_dtype("int4", "cuda", {"use_cuda_bf16": True}), ir.DataType.BFLOAT16)
        self.assertEqual(set_io_dtype("int4", "trt-rtx", {"use_cuda_bf16": True}), ir.DataType.BFLOAT16)

    def test_default_is_fp16(self):
        self.assertEqual(set_io_dtype("fp16", "cuda", {}), ir.DataType.FLOAT16)
        self.assertEqual(set_io_dtype("int4", "cuda", {}), ir.DataType.FLOAT16)


class TestSetOnnxDtype(ExtTestCase):
    def test_int4_symmetric_default(self):
        self.assertEqual(set_onnx_dtype("int4", {}), ir.DataType.INT4)

    def test_int_precisions_use_nbits_container(self):
        for precision in ("int2", "int4", "int8"):
            with self.subTest(precision=precision):
                self.assertEqual(set_onnx_dtype(precision, {}), ir.DataType.INT4)
                self.assertEqual(set_onnx_dtype(precision, {"int4_is_symmetric": False}), ir.DataType.UINT4)

    def test_int4_asymmetric(self):
        self.assertEqual(set_onnx_dtype("int4", {"int4_is_symmetric": False}), ir.DataType.UINT4)

    def test_float_precisions(self):
        self.assertEqual(set_onnx_dtype("fp32", {}), ir.DataType.FLOAT)
        self.assertEqual(set_onnx_dtype("fp16", {}), ir.DataType.FLOAT16)
        self.assertEqual(set_onnx_dtype("bf16", {}), ir.DataType.BFLOAT16)

    def test_unknown_precision_raises(self):
        with self.assertRaises(KeyError):
            set_onnx_dtype("int3", {})


class TestSourceWeightRelease(ExtTestCase):
    def test_releases_unique_module_parameters(self):
        holder = Model.__new__(Model)
        holder.weights = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))
        holder._initialize_source_tensor_tracking()

        first = holder.weights[0]
        second = holder.weights[1]
        holder._release_source_module(first)

        self.assertEqual(first.weight.numel(), 0)
        self.assertEqual(first.bias.numel(), 0)
        self.assertGreater(second.weight.numel(), 0)

    def test_keeps_tied_parameter_until_final_alias(self):
        holder = Model.__new__(Model)
        holder.weights = torch.nn.Module()
        holder.weights.embedding = torch.nn.Embedding(8, 4)
        holder.weights.lm_head = torch.nn.Linear(4, 8, bias=False)
        holder.weights.lm_head.weight = holder.weights.embedding.weight
        holder._initialize_source_tensor_tracking()

        tied_weight = holder.weights.embedding.weight
        holder._release_source_module(holder.weights.embedding)
        self.assertEqual(tied_weight.numel(), 32)

        holder._release_source_module(holder.weights.lm_head)
        self.assertEqual(tied_weight.numel(), 0)


class TestReuseDownloadedWeights(ExtTestCase):
    def test_reuses_safetensors_checkpoint(self):
        from transformers import AutoModelForCausalLM, LlamaConfig

        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=16,
            intermediate_size=24,
            max_position_embeddings=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=1,
            tie_word_embeddings=True,
            vocab_size=32,
        )
        model = AutoModelForCausalLM.from_config(config)
        output_dir, cache_dir = self.get_dirs("test_reuse_downloaded_weights")
        model_dir = os.path.join(output_dir, ".weights")
        os.makedirs(model_dir)
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)

        create_model(
            model_name=None,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp16",
            execution_provider="cpu",
            cache_dir=cache_dir,
            reuse_downloaded_weights=True,
        )

        model_proto = load(os.path.join(output_dir, "model.onnx"), load_external_data=False)
        external_locations = {
            entry.value for initializer in model_proto.graph.initializer for entry in initializer.external_data if entry.key == "location"
        }
        self.assertIn(os.path.join(".weights", "model.safetensors"), external_locations)
        self.assertFalse(os.path.exists(os.path.join(output_dir, "model.safetensors")))
        model_with_weights = load(os.path.join(output_dir, "model.onnx"))
        embedding = next(
            initializer for initializer in model_with_weights.graph.initializer if initializer.name == "model.embed_tokens.weight.source"
        )
        self.assertEqualArray(to_array(embedding, output_dir), model.model.embed_tokens.weight.detach().numpy())
        self.assertIn(
            "model.embed_tokens.weight",
            {output for node in model_with_weights.graph.node if node.op_type == "Cast" for output in node.output},
        )
        self._check_with_ort(os.path.join(output_dir, "model.onnx"), cpu=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
