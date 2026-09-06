# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for :mod:`modelbuilder.helpers.onnx_helper`.
"""

import unittest
from unittest import mock

import numpy as np
import torch
from onnx_light.onnx import TensorProto

from modelbuilder.ext_test_case import ExtTestCase


class TestOnnxHelper(ExtTestCase):
    def test_onnx_is_onnx_light(self):
        import onnx_light.onnx as onnx_light_onnx

        from modelbuilder.helpers import onnx_helper

        self.assertIs(onnx_helper.onnx, onnx_light_onnx)

    def test_submodules_are_available(self):
        from modelbuilder.helpers.onnx_helper import onnx

        for name in ("checker", "external_data_helper", "helper", "numpy_helper", "reference", "shape_inference"):
            self.assertTrue(hasattr(onnx, name), f"onnx.{name} is missing")

    def test_reference_evaluator_is_available(self):
        from onnx_light.onnx.reference import ReferenceEvaluator

        self.assertTrue(callable(ReferenceEvaluator))

    def test_torch_tensor_conversion(self):
        from modelbuilder.helpers.onnx_helper import from_torch_dtype, torch_tensor_to_numpy

        tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
        self.assertEqual(from_torch_dtype(tensor.dtype), TensorProto.FLOAT)
        self.assertEqualArray(torch_tensor_to_numpy(tensor), np.arange(4, dtype=np.float32).reshape(2, 2))

    def test_get_default_onnx_opset_returns_positive_int(self):
        from modelbuilder.helpers import onnx_helper

        opset = onnx_helper.get_default_onnx_opset()
        self.assertIsInstance(opset, int)
        self.assertGreater(opset, 0)

    def test_get_default_onnx_opset_fallback_without_onnxruntime(self):
        from modelbuilder.helpers import onnx_helper

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("onnxruntime"):
                raise ImportError("onnxruntime is not available")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(onnx_helper.get_default_onnx_opset(), onnx_helper.DEFAULT_ONNX_OPSET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
