# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for the dtype/token helpers in :mod:`modelbuilder.builder`.
"""

import unittest

import onnx_ir as ir

from modelbuilder.builder import parse_hf_token, set_io_dtype, set_onnx_dtype
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
    def test_fp32_and_int8_are_float(self):
        self.assertEqual(set_io_dtype("fp32", "cpu", {}), ir.DataType.FLOAT)
        self.assertEqual(set_io_dtype("int8", "cuda", {}), ir.DataType.FLOAT)

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

    def test_int4_asymmetric(self):
        self.assertEqual(set_onnx_dtype("int4", {"int4_is_symmetric": False}), ir.DataType.UINT4)

    def test_float_precisions(self):
        self.assertEqual(set_onnx_dtype("fp32", {}), ir.DataType.FLOAT)
        self.assertEqual(set_onnx_dtype("fp16", {}), ir.DataType.FLOAT16)
        self.assertEqual(set_onnx_dtype("bf16", {}), ir.DataType.BFLOAT16)

    def test_unknown_precision_raises(self):
        with self.assertRaises(KeyError):
            set_onnx_dtype("int8", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
