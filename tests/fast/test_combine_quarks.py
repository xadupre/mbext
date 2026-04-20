# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest
from types import SimpleNamespace

import onnx_ir as ir
import torch

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout


def _make_gptoss_model():
    """Instantiate a minimal GPTOSSModel for unit-testing helper methods."""
    from transformers import GptOssConfig

    from modelbuilder.builders.gptoss import GPTOSSModel

    config = GptOssConfig(
        architectures=["GptOssForCausalLM"],
        hidden_act="silu",
        hidden_size=64,
        intermediate_size=64,
        head_dim=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        num_local_experts=4,
        num_experts_per_tok=2,
        rms_norm_eps=1e-5,
        sliding_window=32,
        vocab_size=256,
    )
    return GPTOSSModel(config, io_dtype=ir.DataType.FLOAT, onnx_dtype=ir.DataType.FLOAT, ep="cpu", cache_dir=None, extra_options={})


class _MockQuarkExperts:
    """Mock experts container that satisfies has_quark_experts checks."""

    def __init__(self, experts_dict):
        self.fc1_weights = torch.zeros(1)
        self.fc2_weights = torch.zeros(1)
        self.fc1_scales = torch.zeros(1)
        self.fc2_scales = torch.zeros(1)
        self.fc1_zero_points = torch.zeros(1)
        self.fc2_zero_points = torch.zeros(1)
        self._experts = experts_dict

    def keys(self):
        return self._experts.keys()

    def __getitem__(self, key):
        return self._experts[key]


def _fused_expert(out_dim, bias=None):
    """Build a mock expert with a fused gate_up_proj."""
    qweight = torch.zeros(out_dim, 1)
    gate_up_proj = SimpleNamespace(qweight=qweight, bias=bias)
    down_proj = SimpleNamespace(qweight=torch.zeros(out_dim, 1), bias=torch.zeros(out_dim))
    return SimpleNamespace(gate_up_proj=gate_up_proj, down_proj=down_proj)


def _separate_expert(gate_dim, up_dim, gate_bias=None, up_bias=None, down_bias=None):
    """Build a mock expert with separate gate_proj, up_proj, down_proj."""
    gate_proj = SimpleNamespace(qweight=torch.zeros(gate_dim, 1), bias=gate_bias)
    up_proj = SimpleNamespace(qweight=torch.zeros(up_dim, 1), bias=up_bias)
    # gate_up_proj.qweight is None → triggers the separate-projection path
    gate_up_proj = SimpleNamespace(qweight=None)
    down_proj = SimpleNamespace(qweight=torch.zeros(gate_dim, 1), bias=down_bias)
    return SimpleNamespace(gate_up_proj=gate_up_proj, gate_proj=gate_proj, up_proj=up_proj, down_proj=down_proj)


class TestCombineQuarks(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _make_gptoss_model()

    # ------------------------------------------------------------------
    # has_quark_experts
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_has_quark_experts_true(self):
        experts = SimpleNamespace(fc1_weights=torch.zeros(1), fc2_weights=torch.zeros(1))
        self.assertTrue(self.model.has_quark_experts(experts))

    @hide_stdout()
    def test_has_quark_experts_false_no_attrs(self):
        experts = SimpleNamespace()
        self.assertFalse(self.model.has_quark_experts(experts))

    @hide_stdout()
    def test_has_quark_experts_false_missing_fc2(self):
        experts = SimpleNamespace(fc1_weights=torch.zeros(1))
        self.assertFalse(self.model.has_quark_experts(experts))

    @hide_stdout()
    def test_has_quark_experts_false_missing_fc1(self):
        experts = SimpleNamespace(fc2_weights=torch.zeros(1))
        self.assertFalse(self.model.has_quark_experts(experts))

    # ------------------------------------------------------------------
    # combine_quark_gate_up_biases_from_experts – fused projection path
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_combine_gate_up_biases_fused_with_bias(self):
        """Fused gate_up_proj with an explicit bias tensor."""
        out_dim = 8
        bias0 = torch.arange(out_dim, dtype=torch.float32)
        bias1 = torch.arange(out_dim, dtype=torch.float32) * 2.0
        experts = _MockQuarkExperts({0: _fused_expert(out_dim, bias=bias0), 1: _fused_expert(out_dim, bias=bias1)})

        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        self.assertEqual(result.shape, (2, out_dim))
        torch.testing.assert_close(result[0], bias0)
        torch.testing.assert_close(result[1], bias1)

    @hide_stdout()
    def test_combine_gate_up_biases_fused_no_bias(self):
        """Fused gate_up_proj without a bias – falls back to zero tensor."""
        out_dim = 8
        experts = _MockQuarkExperts({0: _fused_expert(out_dim, bias=None), 1: _fused_expert(out_dim, bias=None)})

        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        self.assertEqual(result.shape, (2, out_dim))
        torch.testing.assert_close(result, torch.zeros(2, out_dim))

    @hide_stdout()
    def test_combine_gate_up_biases_fused_expert_ordering(self):
        """Experts must be stacked in sorted key order."""
        out_dim = 4
        bias_for_1 = torch.ones(out_dim)
        bias_for_0 = torch.zeros(out_dim)
        # Insert in reverse order to confirm sorted() is applied
        experts = _MockQuarkExperts({1: _fused_expert(out_dim, bias=bias_for_1), 0: _fused_expert(out_dim, bias=bias_for_0)})

        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        torch.testing.assert_close(result[0], bias_for_0)
        torch.testing.assert_close(result[1], bias_for_1)

    # ------------------------------------------------------------------
    # combine_quark_gate_up_biases_from_experts – separate projection path
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_combine_gate_up_biases_separate_with_bias(self):
        """Separate gate/up projections with explicit biases (interleaved)."""
        n = 4  # gate_out_dim == up_out_dim == n
        gate_bias = torch.ones(n)
        up_bias = torch.full((n,), 2.0)

        experts = _MockQuarkExperts({0: _separate_expert(n, n, gate_bias=gate_bias, up_bias=up_bias)})

        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        # Shape: (num_experts, gate_out_dim + up_out_dim)
        self.assertEqual(result.shape, (1, n + n))

        # Even indices = gate, odd indices = up
        torch.testing.assert_close(result[0, ::2], gate_bias)
        torch.testing.assert_close(result[0, 1::2], up_bias)

    @hide_stdout()
    def test_combine_gate_up_biases_separate_no_bias(self):
        """Separate gate/up projections without biases – falls back to zeros."""
        n = 4
        experts = _MockQuarkExperts({0: _separate_expert(n, n, gate_bias=None, up_bias=None)})

        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        self.assertEqual(result.shape, (1, n + n))
        torch.testing.assert_close(result, torch.zeros(1, n + n))

    @hide_stdout()
    def test_combine_gate_up_biases_separate_multiple_experts(self):
        """Multiple experts with separate projections are stacked correctly."""
        n = 4
        gate0 = torch.arange(n, dtype=torch.float32)
        up0 = torch.full((n,), 5.0)
        gate1 = torch.full((n,), 9.0)
        up1 = torch.arange(n, dtype=torch.float32) * 3.0

        experts = _MockQuarkExperts(
            {0: _separate_expert(n, n, gate_bias=gate0, up_bias=up0), 1: _separate_expert(n, n, gate_bias=gate1, up_bias=up1)}
        )

        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        self.assertEqual(result.shape, (2, n + n))
        torch.testing.assert_close(result[0, ::2], gate0)
        torch.testing.assert_close(result[0, 1::2], up0)
        torch.testing.assert_close(result[1, ::2], gate1)
        torch.testing.assert_close(result[1, 1::2], up1)

    # ------------------------------------------------------------------
    # combine_quark_down_biases_from_experts
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_combine_down_biases_with_bias(self):
        """down_proj with explicit bias tensors."""
        out_dim = 8
        down_bias0 = torch.arange(out_dim, dtype=torch.float32)
        down_bias1 = torch.arange(out_dim, dtype=torch.float32) * 3.0

        experts = _MockQuarkExperts({0: _fused_expert(out_dim, bias=None), 1: _fused_expert(out_dim, bias=None)})
        # Override down_proj biases on the individual experts
        experts._experts[0].down_proj.bias = down_bias0
        experts._experts[1].down_proj.bias = down_bias1

        result = self.model.combine_quark_down_biases_from_experts(experts)

        self.assertEqual(result.shape, (2, out_dim))
        torch.testing.assert_close(result[0], down_bias0)
        torch.testing.assert_close(result[1], down_bias1)

    @hide_stdout()
    def test_combine_down_biases_no_bias(self):
        """down_proj without a bias – falls back to zero tensor."""
        out_dim = 8
        # _fused_expert gives a non-None down_proj.bias; override to None
        e0 = _fused_expert(out_dim, bias=None)
        e0.down_proj.bias = None
        e1 = _fused_expert(out_dim, bias=None)
        e1.down_proj.bias = None

        experts = _MockQuarkExperts({0: e0, 1: e1})

        result = self.model.combine_quark_down_biases_from_experts(experts)

        self.assertEqual(result.shape, (2, out_dim))
        torch.testing.assert_close(result, torch.zeros(2, out_dim))

    @hide_stdout()
    def test_combine_down_biases_expert_ordering(self):
        """Down biases are stacked in sorted key order."""
        out_dim = 4
        bias_for_1 = torch.ones(out_dim)
        bias_for_0 = torch.zeros(out_dim)

        e0 = _fused_expert(out_dim, bias=None)
        e0.down_proj.bias = bias_for_0
        e1 = _fused_expert(out_dim, bias=None)
        e1.down_proj.bias = bias_for_1

        # Insert in reverse order
        experts = _MockQuarkExperts({1: e1, 0: e0})

        result = self.model.combine_quark_down_biases_from_experts(experts)

        torch.testing.assert_close(result[0], bias_for_0)
        torch.testing.assert_close(result[1], bias_for_1)

    # ------------------------------------------------------------------
    # Discrepancy checks – independently-built reference vs method output
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_discrepancy_combine_gate_up_biases_fused(self):
        """No discrepancy between method output and a hand-built reference (fused path)."""
        n_experts, out_dim = 3, 6
        torch.manual_seed(0)
        biases = [torch.randn(out_dim) for _ in range(n_experts)]

        experts = _MockQuarkExperts({i: _fused_expert(out_dim, bias=biases[i]) for i in range(n_experts)})
        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        # Reference: simply stack the bias tensors in sorted expert order.
        reference = torch.stack(biases, dim=0)

        disc = self.get_pytorch_discrepancy(result, reference)
        self.assertEqual(disc["max_abs_err"], 0.0, f"Unexpected discrepancy: {disc}")

    @hide_stdout()
    def test_discrepancy_combine_gate_up_biases_separate(self):
        """No discrepancy between method output and a hand-built reference (separate path)."""
        n_experts, n = 4, 4
        torch.manual_seed(1)
        gate_biases = [torch.randn(n) for _ in range(n_experts)]
        up_biases = [torch.randn(n) for _ in range(n_experts)]

        experts = _MockQuarkExperts({i: _separate_expert(n, n, gate_bias=gate_biases[i], up_bias=up_biases[i]) for i in range(n_experts)})
        result = self.model.combine_quark_gate_up_biases_from_experts(experts)

        # Reference: manually interleave gate (even) and up (odd) for each expert.
        reference = torch.zeros(n_experts, n + n)
        for i in range(n_experts):
            reference[i, ::2] = gate_biases[i]
            reference[i, 1::2] = up_biases[i]

        disc = self.get_pytorch_discrepancy(result, reference)
        self.assertEqual(disc["max_abs_err"], 0.0, f"Unexpected discrepancy: {disc}")

    @hide_stdout()
    def test_discrepancy_combine_down_biases(self):
        """No discrepancy between method output and a hand-built reference."""
        n_experts, out_dim = 4, 8
        torch.manual_seed(2)
        down_biases = [torch.randn(out_dim) for _ in range(n_experts)]

        experts = _MockQuarkExperts({i: _fused_expert(out_dim, bias=None) for i in range(n_experts)})
        for i in range(n_experts):
            experts._experts[i].down_proj.bias = down_biases[i]

        result = self.model.combine_quark_down_biases_from_experts(experts)

        # Reference: simply stack the per-expert down biases in sorted order.
        reference = torch.stack(down_biases, dim=0)

        disc = self.get_pytorch_discrepancy(result, reference)
        self.assertEqual(disc["max_abs_err"], 0.0, f"Unexpected discrepancy: {disc}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
