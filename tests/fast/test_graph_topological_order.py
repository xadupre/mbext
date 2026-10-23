# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
from onnx_light.onnx import TensorProto, checker
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder

from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase


class TestGraphTopologicalOrder(ExtTestCase):
    def test_deferred_consumer_before_producer(self):
        builder = Model.__new__(Model)
        builder.graph = GraphBuilder("test")
        builder.graph.set_opset_version("", 21)
        builder.values = {}
        builder.node_names = set()
        builder._deferred_nodes = {}
        builder._deferred_outputs = set()
        builder._node_order = []
        builder._graph_outputs = set()
        builder.local_functions = {}

        builder.make_graph_input("source", TensorProto.FLOAT, [1])
        builder.make_graph_output("final", TensorProto.FLOAT, [1])
        builder.make_node("Identity", ["later"], ["result"], name="consumer")
        builder.make_node("Identity", ["source"], ["other"], name="independent")
        builder.make_node("Identity", ["source"], ["later"], name="producer")
        builder.make_node("Identity", ["later"], ["native_use"], name="native_use")
        builder.make_node("Add", ["result", "later"], ["final"], name="last")

        model = builder.to_model_proto()
        self.assertEqual([node.name for node in model.graph.node], ["independent", "producer", "consumer", "native_use", "last"])
        checker.check_model(model)
        result = ReferenceEvaluator(model).run(None, {"source": np.array([2], dtype=np.float32)})
        self.assertEqualArray(np.array([4], dtype=np.float32), result[0])
