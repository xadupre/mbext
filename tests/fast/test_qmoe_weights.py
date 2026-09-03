# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for CUDA QMoE expert-weight encoding."""

import unittest

import torch

from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase


class _QMoEModel:
    make_qmoe_weights = Model.make_qmoe_weights
    _matmulnbits_blockwise_quantize = Model._matmulnbits_blockwise_quantize

    def __init__(self, block_size=32, bits=4):
        self.ep = "cuda"
        self.qmoe_block_size = block_size
        self.moe_attrs = {"expert_weight_bits": bits, "weights_prepacked": 0}
        self.quant_attrs = {"int4": {"qmoe_block_size": block_size}}


class TestQMoEWeights(ExtTestCase):
    def test_cuda_qmoe_uses_raw_runtime_prepack_layout(self):
        model = _QMoEModel()
        weights = torch.zeros((8, 32), dtype=torch.float32)

        qweight, scales = model.make_qmoe_weights(weights)

        self.assertEqual(tuple(qweight.shape), (8, 16))
        self.assertEqual(tuple(scales.shape), (8, 1))
        self.assertEqual(model.moe_attrs["block_size"], 32)
        self.assertEqual(model.moe_attrs["weights_prepacked"], 0)

    def test_cuda_qmoe_preserves_signed_block_scales(self):
        model = _QMoEModel()
        weights = torch.zeros((2, 32), dtype=torch.float32)
        weights[0, 0] = -8.0
        weights[0, 1] = 7.0
        weights[1, 0] = 8.0
        weights[1, 1] = 7.0

        qweight, scales = model.make_qmoe_weights(weights)

        self.assertEqual(scales.tolist(), [[1.0], [-1.0]])
        self.assertEqual(qweight[0, 0].item(), 0xF0)
        self.assertEqual(qweight[1, 0].item(), 0x10)

    def test_cuda_qmoe_rejects_unsupported_block_size(self):
        model = _QMoEModel(block_size=16)

        with self.assertRaisesRegex(ValueError, "32, 64, or 128"):
            model.make_qmoe_weights(torch.zeros((8, 32), dtype=torch.float32))

    def test_cuda_qmoe_rejects_partial_blocks(self):
        model = _QMoEModel()

        with self.assertRaisesRegex(RuntimeError, "must be divisible"):
            model.make_qmoe_weights(torch.zeros((8, 48), dtype=torch.float32))

    def test_cuda_qmoe_int8_uses_unsigned_offset_storage(self):
        model = _QMoEModel(bits=8)
        weights = torch.zeros((1, 32), dtype=torch.float32)
        weights[0, 0] = -128.0
        weights[0, 1] = 127.0

        qweight, scales = model.make_qmoe_weights(weights)

        self.assertEqual(scales.tolist(), [[1.0]])
        self.assertEqual(qweight[0, :2].tolist(), [0, 255])


if __name__ == "__main__":
    unittest.main()
