# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Tests for :mod:`modelbuilder.builders.local_functions`."""

import sys
import unittest
from unittest import mock

import numpy as np
import onnx_light.onnx.checker as checker
import onnx_light.onnx.helper as onnx_helper
from onnx_light.onnx import FunctionProto, TensorProto
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder

from modelbuilder.builders.local_functions import LocalFunctionsMixin
from modelbuilder.ext_test_case import ExtTestCase


def _empty_holder() -> LocalFunctionsMixin:
    class _Holder(LocalFunctionsMixin):
        def __init__(self) -> None:
            self.graph = GraphBuilder("main", schema_lookup=None)
            self.graph.set_opset_version("", 21)
            self.graph.set_opset_version("com.microsoft", 1)
            self.io_dtype = TensorProto.FLOAT
            self.local_functions = {}

    return _Holder()


def _make_function_model(function, inputs, outputs):
    node = onnx_helper.make_node(function.name, [value.name for value in inputs], [value.name for value in outputs], domain=function.domain)
    graph = onnx_helper.make_graph([node], "main", inputs, outputs)
    model = onnx_helper.make_model(
        graph, ir_version=10, opset_imports=[onnx_helper.make_opsetid("", 21), onnx_helper.make_opsetid("com.microsoft", 1)]
    )
    model.functions.extend([function])
    return model


class TestLocalFunctionsOrtVersion(ExtTestCase):
    def test_ort_version_returns_int_tuple(self):
        version = LocalFunctionsMixin._ort_version()
        self.assertIsInstance(version, tuple)
        self.assertTrue(all(isinstance(x, int) for x in version))

    def test_ort_version_parses_installed_version(self):
        with mock.patch("onnxruntime.__version__", "1.24.4"):
            self.assertEqual(LocalFunctionsMixin._ort_version(), (1, 24, 4))

    def test_ort_version_non_numeric_suffix_falls_back(self):
        with mock.patch("onnxruntime.__version__", "1.26.0dev"):
            self.assertEqual(LocalFunctionsMixin._ort_version(), (99, 99, 0))

    def test_ort_version_missing_onnxruntime_falls_back(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            self.assertEqual(LocalFunctionsMixin._ort_version(), (99, 99, 0))


class TestCausalConvLocalFunction(ExtTestCase):
    def test_structure(self):
        function = LocalFunctionsMixin._make_causal_conv_local_function(4, TensorProto.FLOAT)
        self.assertIsInstance(function, FunctionProto)
        self.assertEqual(function.domain, "com.microsoft")
        self.assertEqual(function.name, "CausalConvWithState")
        self.assertEqual(list(function.input), ["X", "W", "bias", "past_state"])
        self.assertEqual(list(function.output), ["Y", "present_state"])
        self.assertEqual({item.domain: item.version for item in function.opset_import}, {"": 21})

    def _build_model(self, kernel_size: int, batch: int, channels: int, sequence: int):
        function = LocalFunctionsMixin._make_causal_conv_local_function(kernel_size, TensorProto.FLOAT)
        inputs = [
            onnx_helper.make_tensor_value_info("X", TensorProto.FLOAT, [batch, channels, sequence]),
            onnx_helper.make_tensor_value_info("W", TensorProto.FLOAT, [channels, 1, kernel_size]),
            onnx_helper.make_tensor_value_info("bias", TensorProto.FLOAT, [channels]),
            onnx_helper.make_tensor_value_info("past", TensorProto.FLOAT, [batch, channels, kernel_size - 1]),
        ]
        outputs = [
            onnx_helper.make_tensor_value_info("Y", TensorProto.FLOAT, [batch, channels, sequence]),
            onnx_helper.make_tensor_value_info("present", TensorProto.FLOAT, [batch, channels, kernel_size - 1]),
        ]
        return _make_function_model(function, inputs, outputs)

    def test_onnx_check(self):
        checker.check_model(self._build_model(kernel_size=4, batch=1, channels=3, sequence=5))

    def test_numeric_matches_reference(self):
        rng = np.random.default_rng(0)
        for kernel_size, batch, channels, sequence in [(2, 1, 3, 5), (4, 2, 4, 6), (3, 1, 1, 4)]:
            with self.subTest(kernel_size=kernel_size, batch=batch, channels=channels, sequence=sequence):
                proto = self._build_model(kernel_size, batch, channels, sequence)
                x = rng.standard_normal((batch, channels, sequence)).astype(np.float32)
                weight = rng.standard_normal((channels, 1, kernel_size)).astype(np.float32)
                bias = rng.standard_normal((channels,)).astype(np.float32)
                past = rng.standard_normal((batch, channels, kernel_size - 1)).astype(np.float32)
                actual, actual_present = ReferenceEvaluator(proto).run(None, {"X": x, "W": weight, "bias": bias, "past": past})

                padded = np.concatenate([past, x], axis=2)
                expected = np.zeros((batch, channels, sequence), dtype=np.float32)
                for index in range(kernel_size):
                    expected += weight[:, 0, index][None, :, None] * padded[:, :, index : index + sequence]
                expected += bias[None, :, None]
                expected = expected * (1.0 / (1.0 + np.exp(-expected)))

                self.assertEqualArray(expected, actual.astype(np.float32), atol=1e-5)
                self.assertEqualArray(padded[:, :, sequence:], actual_present.astype(np.float32), atol=1e-5)


class TestLinearAttentionLocalFunction(ExtTestCase):
    def test_structure(self):
        function = LocalFunctionsMixin.make_linear_attention_local_function(4, 2, 3, 3, TensorProto.FLOAT)
        self.assertIsInstance(function, FunctionProto)
        self.assertEqual(function.domain, "com.microsoft")
        self.assertEqual(function.name, "LinearAttention")
        self.assertEqual(list(function.input), ["Q", "K", "V", "past_state", "decay", "beta"])
        self.assertEqual(list(function.output), ["output", "present_state"])
        self.assertEqual({item.domain: item.version for item in function.opset_import}, {"": 21})
        self.assertIn("Loop", [node.op_type for node in function.node])

    def _run_function(self, nq, nkv, hk, hv, batch, sequence, feeds):
        import onnxruntime as ort

        function = LocalFunctionsMixin.make_linear_attention_local_function(nq, nkv, hk, hv, TensorProto.FLOAT)
        shapes = {
            "Q": [batch, sequence, nq * hk],
            "K": [batch, sequence, nkv * hk],
            "V": [batch, sequence, nkv * hv],
            "past_state": [batch, nkv, hk, hv],
            "decay": [batch, sequence, nkv],
            "beta": [batch, sequence, nkv],
        }
        inputs = [onnx_helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in shapes.items()]
        outputs = [
            onnx_helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, sequence, nq * hv]),
            onnx_helper.make_tensor_value_info("present_state", TensorProto.FLOAT, [batch, nkv, hk, hv]),
        ]
        graph = onnx_helper.make_graph(list(function.node), "linear_attention", inputs, outputs)
        model = onnx_helper.make_model(graph, ir_version=10, opset_imports=[onnx_helper.make_opsetid("", 21)])
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        return session.run(None, feeds)

    @staticmethod
    def _reference(nq, nkv, hk, hv, batch, sequence, feeds):
        nq_per_kv = nq // nkv
        state = feeds["past_state"].copy()
        outputs = []
        for index in range(sequence):
            key = feeds["K"][:, index, :].reshape(batch, nkv, hk)
            value = feeds["V"][:, index, :].reshape(batch, nkv, hv)
            query = feeds["Q"][:, index, :].reshape(batch, nkv, nq_per_kv, hk)
            decay = feeds["decay"][:, index, :].reshape(batch, nkv, 1, 1)
            beta = feeds["beta"][:, index, :].reshape(batch, nkv, 1, 1)
            projected = np.matmul(key[:, :, None, :], state)[:, :, 0, :]
            difference = value - projected
            state = decay * state + beta * (key[:, :, :, None] * difference[:, :, None, :])
            outputs.append(np.matmul(query, state).reshape(batch, nq * hv))
        return np.stack(outputs, axis=1).astype(np.float32), state.astype(np.float32)

    def test_numeric_matches_reference(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime is not installed")

        rng = np.random.default_rng(1)
        for nq, nkv, hk, hv, batch, sequence in [(2, 2, 3, 3, 2, 5), (4, 2, 3, 3, 1, 4), (4, 2, 2, 3, 2, 3)]:
            with self.subTest(nq=nq, nkv=nkv, hk=hk, hv=hv, batch=batch, sequence=sequence):
                feeds = {
                    "Q": rng.standard_normal((batch, sequence, nq * hk)).astype(np.float32),
                    "K": rng.standard_normal((batch, sequence, nkv * hk)).astype(np.float32),
                    "V": rng.standard_normal((batch, sequence, nkv * hv)).astype(np.float32),
                    "past_state": rng.standard_normal((batch, nkv, hk, hv)).astype(np.float32),
                    "decay": rng.standard_normal((batch, sequence, nkv)).astype(np.float32),
                    "beta": rng.standard_normal((batch, sequence, nkv)).astype(np.float32),
                }
                actual, actual_present = self._run_function(nq, nkv, hk, hv, batch, sequence, feeds)
                expected, expected_present = self._reference(nq, nkv, hk, hv, batch, sequence, feeds)
                self.assertEqualArray(expected, actual.astype(np.float32), atol=1e-4)
                self.assertEqualArray(expected_present, actual_present.astype(np.float32), atol=1e-4)


class TestLocalFunctionRegistration(ExtTestCase):
    def test_register_causal_conv_old_ort(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 25, 0)
        holder._register_causal_conv_local_function(4)
        self.assertIn(("com.microsoft", "CausalConvWithState", ""), holder.local_functions)
        holder._register_causal_conv_local_function(4)
        self.assertEqual(len(holder.local_functions), 1)

    def test_register_causal_conv_recent_ort_skips(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 26, 0)
        holder._register_causal_conv_local_function(4)
        self.assertNotIn(("com.microsoft", "CausalConvWithState", ""), holder.local_functions)

    def test_register_linear_attention_old_ort(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 25, 0)
        holder.register_linear_attention_local_function(4, 2, 3, 3)
        self.assertIn(("com.microsoft", "LinearAttention", ""), holder.local_functions)
        holder.register_linear_attention_local_function(4, 2, 3, 3)
        self.assertEqual(len(holder.local_functions), 1)

    def test_register_linear_attention_recent_ort_skips(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 26, 0)
        holder.register_linear_attention_local_function(4, 2, 3, 3)
        self.assertNotIn(("com.microsoft", "LinearAttention", ""), holder.local_functions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
