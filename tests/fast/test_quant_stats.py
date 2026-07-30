# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import tempfile
import unittest

import numpy as np
import onnx_ir as ir

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.quant_stats import compute_weight_statistics, save_weight_statistics


def _make_model_with_matmul(weight: np.ndarray, node_name: str = "/mm/MatMul", op_type: str = "MatMul") -> ir.Model:
    weight_value = ir.Value(name="weight", const_value=ir.tensor(weight, name="weight"))
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, weight.shape[0]]))
    y = ir.Value(name="y")
    node = ir.Node("", op_type, inputs=[x, weight_value], outputs=[y], name=node_name)
    graph = ir.Graph(inputs=[x], outputs=[y], nodes=[node], initializers=[weight_value], opset_imports={"": 21}, name="g")
    return ir.Model(graph, ir_version=10)


class TestQuantStats(ExtTestCase):
    def test_statistics_keys_and_values(self):
        rng = np.random.default_rng(0)
        weight = rng.standard_normal((64, 32)).astype(np.float32)
        model = _make_model_with_matmul(weight)

        stats_list = compute_weight_statistics(model)
        self.assertEqual(len(stats_list), 1)
        stats = stats_list[0]

        for key in ("name", "shape", "size", "min", "max", "mean", "median", "std", "quantiles", "normal_distance"):
            self.assertIn(key, stats)

        self.assertEqual(stats["name"], "weight")
        self.assertEqual(stats["shape"], [64, 32])
        self.assertEqual(stats["size"], weight.size)
        self.assertAlmostEqual(stats["min"], float(weight.min()), atol=1e-4)
        self.assertAlmostEqual(stats["max"], float(weight.max()), atol=1e-4)
        self.assertAlmostEqual(stats["mean"], float(weight.mean()), atol=1e-4)
        self.assertAlmostEqual(stats["median"], float(np.median(weight)), atol=1e-4)
        # A standard-normal tensor should be very close to a normal distribution.
        self.assertLess(stats["normal_distance"], 0.1)
        self.assertEqual(set(stats["quantiles"]), {"0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99"})

    def test_normal_distance_larger_for_non_normal(self):
        rng = np.random.default_rng(0)
        normal = rng.standard_normal((256, 64)).astype(np.float32)
        uniform = rng.uniform(-1.0, 1.0, size=(256, 64)).astype(np.float32)

        normal_stats = compute_weight_statistics(_make_model_with_matmul(normal))[0]
        uniform_stats = compute_weight_statistics(_make_model_with_matmul(uniform))[0]

        self.assertLess(normal_stats["normal_distance"], uniform_stats["normal_distance"])

    def test_op_types_filter(self):
        weight = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
        model = _make_model_with_matmul(weight, op_type="Gemm")

        self.assertEqual(compute_weight_statistics(model, op_types=("MatMul",)), [])
        self.assertEqual(len(compute_weight_statistics(model, op_types=("Gemm",))), 1)

    def test_nodes_to_exclude(self):
        weight = np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32)
        model = _make_model_with_matmul(weight, node_name="/mm/MatMul")

        self.assertEqual(compute_weight_statistics(model, nodes_to_exclude=["/mm/MatMul"]), [])
        self.assertEqual(len(compute_weight_statistics(model)), 1)

    def test_save_weight_statistics_writes_json(self):
        weight = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
        model = _make_model_with_matmul(weight)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.onnx.weight_stats.json")
            returned = save_weight_statistics(model, path)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.assertEqual(loaded, returned)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["name"], "weight")


if __name__ == "__main__":
    unittest.main(verbosity=2)
