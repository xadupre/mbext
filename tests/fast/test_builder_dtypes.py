# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for the dtype/token helpers in :mod:`modelbuilder.builder`.
"""

import unittest

import torch
from onnx_light.onnx import TensorProto as ir

from modelbuilder.builder import parse_hf_token, set_io_dtype, set_onnx_dtype
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
