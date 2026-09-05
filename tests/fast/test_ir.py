# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for :mod:`modelbuilder.ir`, the graph IR built on top of ``onnx-light``.
"""

import os
import tempfile
import unittest

import numpy as np
from modelbuilder import ir

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.helpers.onnx_helper import onnx


def _make_model(weight: np.ndarray) -> ir.Model:
    weight_value = ir.Value(name="weight", const_value=ir.tensor(weight, name="weight"))
    x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(["batch", weight.shape[0]]))
    mm = ir.Value(name="mm")
    y = ir.Value(name="y", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape(["batch", weight.shape[1]]))
    nodes = [ir.Node("", "MatMul", inputs=[x, weight_value], outputs=[mm], name="mm"), ir.node("Relu", [mm], outputs=[y], name="relu")]
    graph = ir.Graph(inputs=[x], outputs=[y], nodes=nodes, initializers=[weight_value], opset_imports={"": 21}, name="g")
    return ir.Model(graph, ir_version=10, producer_name="mbext")


class TestIr(ExtTestCase):
    def test_datatype(self):
        self.assertEqual(ir.DataType.FLOAT, 1)
        self.assertEqual(ir.DataType["FLOAT16"], ir.DataType.FLOAT16)
        self.assertEqual(ir.DataType.FLOAT.numpy(), np.float32)
        self.assertEqual(ir.DataType.from_numpy(np.dtype("int64")), ir.DataType.INT64)
        self.assertEqual(ir.DataType.BFLOAT16.short_name(), "bf16")
        self.assertEqual(ir.DataType.INT4.bitwidth, 4)

    def test_tensor(self):
        value = np.arange(6).reshape((2, 3)).astype(np.float32)
        tensor = ir.tensor(value, name="t")
        self.assertEqual(tensor.dtype, ir.DataType.FLOAT)
        self.assertEqual(tuple(tensor.shape), (2, 3))
        self.assertEqualArray(value, tensor.numpy())
        self.assertEqual(tensor.tobytes(), value.tobytes())

    def test_serialize_deserialize(self):
        weight = np.random.default_rng(0).standard_normal((4, 3)).astype(np.float32)
        model = _make_model(weight)
        proto = ir.serde.serialize_model(model)
        self.assertEqual(proto.producer_name, "mbext")
        onnx.checker.check_model(proto)

        restored = ir.from_proto(proto)
        self.assertEqual(restored.graph.name, "g")
        self.assertEqual([n.op_type for n in restored.graph], ["MatMul", "Relu"])
        self.assertEqual([v.name for v in restored.graph.inputs], ["x"])
        self.assertEqual([v.name for v in restored.graph.outputs], ["y"])
        self.assertEqualArray(weight, restored.graph.initializers["weight"].const_value.numpy())
        self.assertEqual(tuple(restored.graph.inputs[0].shape), ("batch", 4))

    def test_scalar_shape_is_preserved(self):
        value = ir.Value(name="s", type=ir.TensorType(ir.DataType.INT64), shape=ir.Shape([]))
        proto = onnx.ValueInfoProto()
        ir.serde.serialize_value_into(proto, value)
        restored = ir.serde.deserialize_value_info_proto(proto)
        self.assertIsNotNone(restored.shape)
        self.assertEqual(len(restored.shape), 0)

    def test_attributes(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        y = ir.Value(name="y")
        node = ir.node("Cast", [x], outputs=[y], attributes={"to": ir.DataType.FLOAT16}, name="cast")
        self.assertEqual(node.attributes["to"].as_int(), int(ir.DataType.FLOAT16))
        node2 = ir.node("Concat", [x, x], attributes={"axis": 0}, name="concat")
        self.assertEqual(node2.attributes["axis"].as_int(), 0)

    def test_sort(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        mm = ir.Value(name="neg")
        y = ir.Value(name="y")
        neg = ir.node("Neg", [x], outputs=[mm], name="neg")
        relu = ir.node("Relu", [mm], outputs=[y], name="relu")
        # The nodes are given in the reverse topological order.
        graph = ir.Graph(inputs=[x], outputs=[y], nodes=[relu, neg], opset_imports={"": 21}, name="g")
        graph.sort()
        self.assertEqual([n.op_type for n in graph], ["Neg", "Relu"])

    def test_save_load_external_data(self):
        weight = np.random.default_rng(1).standard_normal((32, 16)).astype(np.float32)
        model = _make_model(weight)
        with tempfile.TemporaryDirectory(dir=os.path.dirname(__file__)) as root:
            path = os.path.join(root, "model.onnx")
            seen = []

            def callback(tensor, metadata):
                seen.append((tensor.name, metadata.total))

            ir.save(model, path, external_data="model.onnx.data", size_threshold_bytes=0, callback=callback)
            self.assertExists(path)
            self.assertExists(path + ".data")
            self.assertEqual([name for name, _ in seen], ["weight"])

            restored = ir.load(path)
            self.assertEqualArray(weight, restored.graph.initializers["weight"].const_value.numpy())

    def test_function(self):
        x = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([2]))
        y = ir.Value(name="y")
        body = ir.Graph(inputs=[x], outputs=[y], nodes=[ir.node("Neg", [x], outputs=[y])], opset_imports={"": 21}, name="f")
        func = ir.Function(domain="local", name="Neg1", overload="", graph=body, attributes=[])
        model = _make_model(np.zeros((4, 3), dtype=np.float32))
        model.functions[func.domain, func.name, func.overload] = func
        proto = ir.serde.serialize_model(model)
        self.assertEqual(len(proto.functions), 1)
        self.assertEqual(proto.functions[0].name, "Neg1")
        restored = ir.from_proto(proto)
        self.assertIn(("local", "Neg1", ""), restored.functions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
