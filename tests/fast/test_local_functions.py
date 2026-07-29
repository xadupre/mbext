# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for :mod:`modelbuilder.builders.local_functions`.
"""

import sys
import unittest
from unittest import mock

import numpy as np
import onnx
import onnx_ir as ir
from onnx.reference import ReferenceEvaluator

from modelbuilder.builders.local_functions import LocalFunctionsMixin
from modelbuilder.ext_test_case import ExtTestCase


def _empty_holder() -> LocalFunctionsMixin:
    """Return a minimal object exposing ``model`` / ``io_dtype`` like ``Model``."""

    class _Holder(LocalFunctionsMixin):
        def __init__(self) -> None:
            graph = ir.Graph([], [], nodes=[], opset_imports={"": 21}, name="main")
            self.model = ir.Model(graph, ir_version=10)
            self.io_dtype = ir.DataType.FLOAT

    return _Holder()


def _mkv(name: str, shape: list[int]) -> ir.Value:
    v = ir.Value(name=name)
    v.dtype = ir.DataType.FLOAT
    v.shape = ir.Shape(shape)
    return v


class TestLocalFunctionsOrtVersion(ExtTestCase):
    def test_ort_version_returns_int_tuple(self):
        version = LocalFunctionsMixin._ort_version()
        self.assertIsInstance(version, tuple)
        self.assertTrue(all(isinstance(x, int) for x in version))

    def test_ort_version_parses_installed_version(self):
        with mock.patch("onnxruntime.__version__", "1.24.4"):
            self.assertEqual(LocalFunctionsMixin._ort_version(), (1, 24, 4))

    def test_ort_version_non_numeric_suffix_falls_back(self):
        # A dev build such as "1.26.0dev" cannot be parsed as ints and is
        # treated as "current enough" -> (99, 99, 0).
        with mock.patch("onnxruntime.__version__", "1.26.0dev"):
            self.assertEqual(LocalFunctionsMixin._ort_version(), (99, 99, 0))

    def test_ort_version_missing_onnxruntime_falls_back(self):
        with mock.patch.dict(sys.modules, {"onnxruntime": None}):
            self.assertEqual(LocalFunctionsMixin._ort_version(), (99, 99, 0))


class TestCausalConvLocalFunction(ExtTestCase):
    def test_structure(self):
        func = LocalFunctionsMixin._make_causal_conv_local_function(4, ir.DataType.FLOAT)
        self.assertIsInstance(func, ir.Function)
        self.assertEqual(func.domain, "com.microsoft")
        self.assertEqual(func.name, "CausalConvWithState")
        self.assertEqual([i.name for i in func.inputs], ["X", "W", "bias", "past_state"])
        self.assertEqual([o.name for o in func.outputs], ["Y", "present_state"])
        self.assertEqual(dict(func.opset_imports), {"": 21})

    def _build_model(self, K: int, B: int, C: int, S: int) -> onnx.ModelProto:
        func = LocalFunctionsMixin._make_causal_conv_local_function(K, ir.DataType.FLOAT)
        X = _mkv("X", [B, C, S])
        W = _mkv("W", [C, 1, K])
        bias = _mkv("bias", [C])
        past = _mkv("past", [B, C, K - 1])
        Y = _mkv("Y", [B, C, S])
        present = _mkv("present", [B, C, K - 1])
        node = ir.node("CausalConvWithState", inputs=[X, W, bias, past], outputs=[Y, present], domain="com.microsoft")
        graph = ir.Graph(
            inputs=[X, W, bias, past], outputs=[Y, present], nodes=[node], opset_imports={"": 21, "com.microsoft": 1}, name="main"
        )
        model = ir.Model(graph, ir_version=10, functions=[func])
        return ir.serde.serialize_model(model)

    def test_onnx_check(self):
        proto = self._build_model(K=4, B=1, C=3, S=5)
        onnx.checker.check_model(proto)

    def test_numeric_matches_reference(self):
        rng = np.random.default_rng(0)
        for K, B, C, S in [(2, 1, 3, 5), (4, 2, 4, 6), (3, 1, 1, 4)]:
            with self.subTest(K=K, B=B, C=C, S=S):
                proto = self._build_model(K=K, B=B, C=C, S=S)
                Xv = rng.standard_normal((B, C, S)).astype(np.float32)
                Wv = rng.standard_normal((C, 1, K)).astype(np.float32)
                bv = rng.standard_normal((C,)).astype(np.float32)
                pv = rng.standard_normal((B, C, K - 1)).astype(np.float32)

                sess = ReferenceEvaluator(proto)
                Yv, prv = sess.run(None, {"X": Xv, "W": Wv, "bias": bv, "past": pv})

                padded = np.concatenate([pv, Xv], axis=2)
                out = np.zeros((B, C, S), dtype=np.float32)
                for k in range(K):
                    out += Wv[:, 0, k][None, :, None] * padded[:, :, k : k + S]
                out += bv[None, :, None]
                ref_Y = out * (1.0 / (1.0 + np.exp(-out)))
                ref_present = padded[:, :, S:]

                self.assertEqualArray(ref_Y, Yv.astype(np.float32), atol=1e-5)
                self.assertEqualArray(ref_present, prv.astype(np.float32), atol=1e-5)


class TestLinearAttentionLocalFunction(ExtTestCase):
    def test_structure(self):
        func = LocalFunctionsMixin.make_linear_attention_local_function(4, 2, 3, 3, ir.DataType.FLOAT)
        self.assertIsInstance(func, ir.Function)
        self.assertEqual(func.domain, "com.microsoft")
        self.assertEqual(func.name, "LinearAttention")
        self.assertEqual([i.name for i in func.inputs], ["Q", "K", "V", "past_state", "decay", "beta"])
        self.assertEqual([o.name for o in func.outputs], ["output", "present_state"])
        self.assertEqual(dict(func.opset_imports), {"": 21})
        # The GatedDeltaNet recurrence is implemented with an ONNX Loop.
        self.assertIn("Loop", [n.op_type for n in func.graph])

    def _run_function(self, nq, nkv, hk, hv, B, S, feeds):
        import onnxruntime as ort

        func = LocalFunctionsMixin.make_linear_attention_local_function(nq, nkv, hk, hv, ir.DataType.FLOAT)
        body = func.graph
        shapes = {
            "Q": [B, S, nq * hk],
            "K": [B, S, nkv * hk],
            "V": [B, S, nkv * hv],
            "past_state": [B, nkv, hk, hv],
            "decay": [B, S, nkv],
            "beta": [B, S, nkv],
        }
        for v in body.inputs:
            v.shape = ir.Shape(shapes[v.name])
        # Give the Loop body's formal inputs explicit shapes so the body graph
        # can run stand-alone (in production the function is inlined into a
        # larger model where shape inference provides these).
        for n in body:
            if n.op_type == "Loop":
                lb = n.attributes["body"].as_graph()
                lb.inputs[0].shape = ir.Shape([])
                lb.inputs[0].dtype = ir.DataType.INT64
                lb.inputs[1].shape = ir.Shape([])
                lb.inputs[1].dtype = ir.DataType.BOOL
                lb.inputs[2].shape = ir.Shape([B, nkv, hk, hv])
                lb.inputs[2].dtype = ir.DataType.FLOAT
        proto = ir.serde.serialize_model(ir.Model(body, ir_version=10))
        sess = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
        return sess.run(None, feeds)

    @staticmethod
    def _reference(nq, nkv, hk, hv, B, S, feeds):
        nq_per_kv = nq // nkv
        Qv, Kv, Vv = feeds["Q"], feeds["K"], feeds["V"]
        dv, bv = feeds["decay"], feeds["beta"]
        state = feeds["past_state"].copy()
        outs = []
        for t in range(S):
            k_t = Kv[:, t, :].reshape(B, nkv, hk)
            v_t = Vv[:, t, :].reshape(B, nkv, hv)
            q_t = Qv[:, t, :].reshape(B, nkv, nq_per_kv, hk)
            g_t = dv[:, t, :].reshape(B, nkv, 1, 1)
            beta_t = bv[:, t, :].reshape(B, nkv, 1, 1)
            kS = np.matmul(k_t[:, :, None, :], state)[:, :, 0, :]
            v_prime = v_t - kS
            outer = k_t[:, :, :, None] * v_prime[:, :, None, :]
            state = g_t * state + beta_t * outer
            y_t = np.matmul(q_t, state).reshape(B, nq, hv)
            outs.append(y_t.reshape(B, nq * hv))
        return np.stack(outs, axis=1).astype(np.float32), state.astype(np.float32)

    def test_numeric_matches_reference(self):
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            self.skipTest("onnxruntime is not installed")

        rng = np.random.default_rng(1)
        # (nq, nkv) covers both multi-head (nq == nkv) and grouped-query (nq > nkv).
        for nq, nkv, hk, hv, B, S in [(2, 2, 3, 3, 2, 5), (4, 2, 3, 3, 1, 4), (4, 2, 2, 3, 2, 3)]:
            with self.subTest(nq=nq, nkv=nkv, hk=hk, hv=hv, B=B, S=S):
                feeds = {
                    "Q": rng.standard_normal((B, S, nq * hk)).astype(np.float32),
                    "K": rng.standard_normal((B, S, nkv * hk)).astype(np.float32),
                    "V": rng.standard_normal((B, S, nkv * hv)).astype(np.float32),
                    "past_state": rng.standard_normal((B, nkv, hk, hv)).astype(np.float32),
                    "decay": rng.standard_normal((B, S, nkv)).astype(np.float32),
                    "beta": rng.standard_normal((B, S, nkv)).astype(np.float32),
                }
                out, present = self._run_function(nq, nkv, hk, hv, B, S, feeds)
                ref_out, ref_present = self._reference(nq, nkv, hk, hv, B, S, feeds)
                self.assertEqualArray(ref_out, out.astype(np.float32), atol=1e-4)
                self.assertEqualArray(ref_present, present.astype(np.float32), atol=1e-4)


class TestLocalFunctionRegistration(ExtTestCase):
    CAUSAL_KEY = ("com.microsoft", "CausalConvWithState", "")
    LINEAR_KEY = ("com.microsoft", "LinearAttention", "")

    def test_register_causal_conv_old_ort(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 25, 0)
        holder._register_causal_conv_local_function(4)
        self.assertIn(self.CAUSAL_KEY, holder.model.functions)
        # Registration is idempotent.
        holder._register_causal_conv_local_function(4)
        self.assertEqual(len(holder.model.functions), 1)

    def test_register_causal_conv_recent_ort_skips(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 26, 0)
        holder._register_causal_conv_local_function(4)
        self.assertNotIn(self.CAUSAL_KEY, holder.model.functions)
        self.assertEqual(len(holder.model.functions), 0)

    def test_register_linear_attention_old_ort(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 25, 0)
        holder.register_linear_attention_local_function(4, 2, 3, 3)
        self.assertIn(self.LINEAR_KEY, holder.model.functions)
        # Registration is idempotent.
        holder.register_linear_attention_local_function(4, 2, 3, 3)
        self.assertEqual(len(holder.model.functions), 1)

    def test_register_linear_attention_recent_ort_skips(self):
        holder = _empty_holder()
        holder._ort_version = lambda: (1, 26, 0)
        holder.register_linear_attention_local_function(4, 2, 3, 3)
        self.assertNotIn(self.LINEAR_KEY, holder.model.functions)
        self.assertEqual(len(holder.model.functions), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
