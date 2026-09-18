# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Tests for native onnx-light weight-only quantization."""

import unittest

import ml_dtypes
import numpy as np
import onnx_light.onnx.helper as onnx_helper
import onnx_light.onnx.numpy_helper as numpy_helper
from onnx_light.onnx import ModelProto, TensorProto

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.helpers.quantization import quantize_matmul_nbits


def _make_matmul_model(rows=32, columns=4):
    weight = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns) / (rows * columns) - 0.5
    graph = onnx_helper.make_graph(
        [onnx_helper.make_node("MatMul", ["x", "weight"], ["y"], name="/MatMul")],
        "quantize",
        [onnx_helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, rows])],
        [onnx_helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, columns])],
        [numpy_helper.from_array(weight, name="weight")],
    )
    model = onnx_helper.make_model(graph, ir_version=10, opset_imports=[onnx_helper.make_opsetid("", 21)])
    return model, weight


def _make_shared_matmul_model(rows=64, columns=4):
    weight_name = "lm_head.MatMul.weight"
    weight = np.arange(rows * columns, dtype=np.float32).reshape(rows, columns) / (rows * columns) - 0.5
    graph = onnx_helper.make_graph(
        [
            onnx_helper.make_node("MatMul", ["x1", weight_name], ["y1"], name="/shared/MatMul"),
            onnx_helper.make_node("MatMul", ["x2", weight_name], ["y2"], name="/lm_head/MatMul"),
        ],
        "shared_quantize",
        [
            onnx_helper.make_tensor_value_info("x1", TensorProto.FLOAT, [2, rows]),
            onnx_helper.make_tensor_value_info("x2", TensorProto.FLOAT, [2, rows]),
        ],
        [
            onnx_helper.make_tensor_value_info("y1", TensorProto.FLOAT, [2, columns]),
            onnx_helper.make_tensor_value_info("y2", TensorProto.FLOAT, [2, columns]),
        ],
        [numpy_helper.from_array(weight, name=weight_name)],
    )
    return onnx_helper.make_model(graph, ir_version=10, opset_imports=[onnx_helper.make_opsetid("", 21)])


class TestQuantization(ExtTestCase):
    def _quantize(self, *, use_qdq=False, algorithm="default", rows=32):
        model, weight = _make_matmul_model(rows=rows)
        quantize_matmul_nbits(
            model,
            bits=4,
            block_size=32,
            is_symmetric=True,
            accuracy_level=0,
            nodes_to_exclude=[],
            op_types_to_quantize=["MatMul"],
            use_qdq=use_qdq,
            algorithm_config={"algorithm": algorithm, "customized_weight_config": {}},
        )
        return model, weight

    def _check_runtime(self, model, weight):
        import onnxruntime as ort

        values = np.arange(2 * weight.shape[0], dtype=np.float32).reshape(2, weight.shape[0]) / (2 * weight.shape[0])
        actual = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": values})[0]
        self.assertEqual(actual.shape, (2, 4))
        self.assertLess(np.max(np.abs(actual - values @ weight)), 2.0)

    def test_default_qoperator(self):
        model, weight = self._quantize()
        self.assertEqual([node.op_type for node in model.graph.node], ["MatMulNBits"])
        self._check_runtime(model, weight)

    def test_qdq(self):
        model, weight = self._quantize(use_qdq=True)
        self.assertEqual([node.op_type for node in model.graph.node], ["DequantizeLinear", "MatMul"])
        self._check_runtime(model, weight)

    def test_rtn_and_k_quant(self):
        for algorithm in ("rtn", "k_quant"):
            with self.subTest(algorithm=algorithm):
                model, weight = self._quantize(algorithm=algorithm)
                self.assertEqual([node.op_type for node in model.graph.node], ["MatMulNBits"])
                self.assertIn("G32", model.graph.initializer[0].name)
                self._check_runtime(model, weight)

    def test_ternary_int2_is_exact(self):
        import onnxruntime as ort

        rows, columns, block_size = 256, 4, 128
        model, _ = _make_matmul_model(rows=rows, columns=columns)
        levels = (np.arange(columns * rows, dtype=np.int64).reshape(columns, rows) % 3) - 1
        group_scales = np.array([[0.25, 0.5], [0.75, 1.0], [1.25, 1.5], [1.75, 2.0]], dtype=np.float32)
        weight = (levels.reshape(columns, 2, block_size) * group_scales[:, :, np.newaxis]).reshape(columns, rows).T.astype(np.float32)
        model.graph.initializer.clear()
        model.graph.initializer.append(numpy_helper.from_array(weight, name="weight"))

        quantize_matmul_nbits(
            model,
            bits=2,
            block_size=block_size,
            is_symmetric=True,
            accuracy_level=0,
            nodes_to_exclude=[],
            op_types_to_quantize=["MatMul"],
            use_qdq=False,
            algorithm_config={"algorithm": "ternary", "customized_weight_config": {}},
        )

        node = model.graph.node[0]
        self.assertEqual(node.op_type, "MatMulNBits")
        self.assertEqual(len(node.input), 4)
        attributes = {attribute.name: attribute.i for attribute in node.attribute}
        self.assertEqual(attributes["bits"], 2)
        self.assertEqual(attributes["block_size"], block_size)
        values = np.arange(2 * rows, dtype=np.float32).reshape(2, rows) / rows
        actual = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, {"x": values})[0]
        np.testing.assert_allclose(actual, values @ weight, rtol=1e-6, atol=1e-5)

    def test_ternary_rejects_non_ternary_weights(self):
        model, _ = _make_matmul_model(rows=128)
        with self.assertRaisesRegex(ValueError, "only.*-scale, 0, \\+scale"):
            quantize_matmul_nbits(
                model,
                bits=2,
                block_size=128,
                is_symmetric=True,
                accuracy_level=0,
                nodes_to_exclude=[],
                op_types_to_quantize=["MatMul"],
                use_qdq=False,
                algorithm_config={"algorithm": "ternary", "customized_weight_config": {}},
            )

    def test_rtn_and_k_quant_multi_block_scale_shape(self):
        for algorithm in ("rtn", "k_quant"):
            with self.subTest(algorithm=algorithm):
                model, weight = self._quantize(algorithm=algorithm, rows=70)
                scales = next(initializer for initializer in model.graph.initializer if "_scale" in initializer.name)
                self.assertEqual(list(scales.dims), [4, 3])
                self._check_runtime(model, weight)

    def test_bfloat16_scales_preserve_weight_dtype(self):
        model, _ = _make_matmul_model(rows=64)
        weight = next(initializer for initializer in model.graph.initializer if initializer.name == "weight")
        bfloat16_weight = numpy_helper.from_array(numpy_helper.to_array(weight).astype(ml_dtypes.bfloat16), name=weight.name)
        model.graph.initializer.clear()
        model.graph.initializer.append(bfloat16_weight)

        for algorithm, use_qdq in (("default", False), ("rtn", False), ("k_quant", False), ("default", True)):
            with self.subTest(algorithm=algorithm, use_qdq=use_qdq):
                quantized = ModelProto()
                quantized.ParseFromString(model.SerializeToString())
                quantize_matmul_nbits(
                    quantized,
                    bits=4,
                    block_size=32,
                    is_symmetric=True,
                    accuracy_level=0,
                    nodes_to_exclude=[],
                    op_types_to_quantize=["MatMul"],
                    use_qdq=use_qdq,
                    algorithm_config={"algorithm": algorithm, "customized_weight_config": {}},
                )
                scales = next(initializer for initializer in quantized.graph.initializer if "scale" in initializer.name)
                self.assertEqual(scales.data_type, TensorProto.BFLOAT16)

    def test_shared_weight_matching_config_reuses_initializers(self):
        import onnxruntime as ort

        for algorithm, expected_initializers in (("default", 2), ("rtn", 2), ("k_quant", 3)):
            with self.subTest(algorithm=algorithm):
                model = _make_shared_matmul_model()
                quantize_matmul_nbits(
                    model,
                    bits=4,
                    block_size=32,
                    is_symmetric=True,
                    accuracy_level=0,
                    nodes_to_exclude=[],
                    op_types_to_quantize=["MatMul"],
                    use_qdq=False,
                    algorithm_config={"algorithm": algorithm, "customized_weight_config": {}},
                )
                self.assertEqual(len(model.graph.initializer), expected_initializers)
                self.assertEqual(list(model.graph.node[0].input)[1:], list(model.graph.node[1].input)[1:])
                names = [initializer.name for initializer in model.graph.initializer]
                self.assertEqual(len(names), len(set(names)))
                values = np.arange(128, dtype=np.float32).reshape(2, 64) / 128
                outputs = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"]).run(
                    None, {"x1": values, "x2": values}
                )
                self.assertEqual([output.shape for output in outputs], [(2, 4), (2, 4)])

    def test_shared_weight_different_config_uses_unique_initializers(self):
        model = _make_shared_matmul_model()
        quantize_matmul_nbits(
            model,
            bits=4,
            block_size=32,
            is_symmetric=True,
            accuracy_level=0,
            nodes_to_exclude=[],
            op_types_to_quantize=["MatMul"],
            use_qdq=False,
            algorithm_config={"algorithm": "rtn", "customized_weight_config": {"/lm_head/MatMul": {"bits": 8}}},
        )
        names = [initializer.name for initializer in model.graph.initializer]
        self.assertEqual(len(names), 4)
        self.assertEqual(len(names), len(set(names)))
        shared_node, lm_head_node = model.graph.node
        self.assertNotEqual(list(shared_node.input)[1:], list(lm_head_node.input)[1:])
        self.assertEqual(lm_head_node.input[1], "lm_head.MatMul.weight_Q8G32")
        self.assertEqual(lm_head_node.input[2], "lm_head.MatMul.weight_scale")

    def test_shared_weight_qdq_reuses_initializers_with_unique_values(self):
        model = _make_shared_matmul_model()
        quantize_matmul_nbits(
            model,
            bits=4,
            block_size=32,
            is_symmetric=True,
            accuracy_level=0,
            nodes_to_exclude=[],
            op_types_to_quantize=["MatMul"],
            use_qdq=True,
            algorithm_config={"algorithm": "default", "customized_weight_config": {}},
        )
        self.assertEqual(len(model.graph.initializer), 2)
        dequantize_nodes = [node for node in model.graph.node if node.op_type == "DequantizeLinear"]
        self.assertEqual(len(dequantize_nodes), 2)
        self.assertEqual(list(dequantize_nodes[0].input), list(dequantize_nodes[1].input))
        self.assertNotEqual(dequantize_nodes[0].output[0], dequantize_nodes[1].output[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
