# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.helpers.deepseek_v41 import DeepseekV41Checkpoint, dequantize_v41_weight


class TestDeepseekV41Checkpoint(ExtTestCase):
    def test_packed_fp4_nibble_order_and_row_scales(self):
        packed = torch.tensor([[0x21, 0xFE], [0x43, 0xA9]], dtype=torch.uint8)
        actual = dequantize_v41_weight(packed, torch.tensor([[2.0, 0.5], [4.0, 8.0]]), (2, 4), block_size=(2, 2))
        torch.testing.assert_close(actual, torch.tensor([[1.0, 2.0, -2.0, -3.0], [6.0, 8.0, -4.0, -8.0]]))

    def test_fp8_partial_blocks(self):
        weight = torch.arange(15).reshape(3, 5).to(torch.float8_e4m3fn)
        scale = torch.tensor([[1.0, 2.0, 4.0], [8.0, 16.0, 32.0]])
        expected = torch.arange(15).reshape(3, 5).float() * torch.tensor([[1, 1, 2, 2, 4], [1, 1, 2, 2, 4], [8, 8, 16, 16, 32]])
        actual = dequantize_v41_weight(weight, scale, (3, 5), block_size=(2, 2))
        torch.testing.assert_close(actual, expected)

    def test_invalid_quantization_metadata(self):
        weight = torch.ones(3, 4, dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError, "scale shape"):
            dequantize_v41_weight(weight, torch.ones(1, 1), (3, 4), block_size=(2, 2))
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    dequantize_v41_weight(weight, torch.full((2, 2), value), (3, 4), block_size=(2, 2))
        with self.assertRaisesRegex(ValueError, "packed FP4 shape"):
            dequantize_v41_weight(torch.zeros(3, 4, dtype=torch.int8), torch.ones(3, 1), (3, 4))

    def test_assign_meta_parameters_and_missing_scales(self):
        with tempfile.TemporaryDirectory() as directory:
            save_file(
                {"model.weight": torch.ones(4, 8).to(torch.float8_e4m3fn), "model.scale": torch.tensor([[2.0]])},
                os.path.join(directory, "model.safetensors"),
            )
            checkpoint = DeepseekV41Checkpoint(directory)
            with torch.device("meta"):
                model = torch.nn.Linear(8, 4, bias=False)
            checkpoint.load(model)
            self.assertEqual(model.weight.device.type, "cpu")
            torch.testing.assert_close(model(torch.ones(1, 8)), torch.full((1, 4), 16.0))
            with self.assertRaisesRegex(ValueError, "Missing.*tensor"):
                checkpoint.tensor("missing.weight", (4, 8))
            save_file({"weight": torch.ones(4, 8).to(torch.float8_e4m3fn)}, os.path.join(directory, "model.safetensors"))
            with self.assertRaisesRegex(ValueError, "Missing.*scale"):
                DeepseekV41Checkpoint(directory).tensor("weight", (4, 8))

    @torch.no_grad()
    def test_compressed_engram_retains_requested_activation_dtype(self):
        from modelbuilder.models.deepseek_v41 import Engram

        config = SimpleNamespace(
            hidden_size=32, hc_mult=2, rms_norm_eps=1e-5, engram_head_dim=32, engram_max_ngram_size=2, engram_n_heads=1
        )
        with torch.device("meta"):
            model = torch.nn.Module()
            model.layers = torch.nn.ModuleList([torch.nn.Module()])
            model.layers[0].engram = Engram(config, 2)
        tensors = {name: torch.ones(parameter.shape) for name, parameter in model.named_parameters()}
        name = "layers.0.engram.embed"
        tensors[name + ".weight"] = tensors[name + ".weight"].to(torch.float8_e4m3fn)
        tensors[name + ".scale"] = torch.ones(2, 1)
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, os.path.join(directory, "model.safetensors"))
            DeepseekV41Checkpoint(directory).load(model, dtype=torch.float16)
        engram = model.layers[0].engram
        self.assertEqual(engram.embed.weight.dtype, torch.uint8)
        self.assertEqual(engram.embed.scale.shape, (2, 1))
        result = engram(
            torch.ones(1, 1, 2, 32, dtype=torch.float16), torch.zeros(1, 1, 1, dtype=torch.int64), torch.ones(1, 1, dtype=torch.bool)
        )
        self.assertEqual(result.dtype, torch.float16)
        self.assertTrue(torch.isfinite(result).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
