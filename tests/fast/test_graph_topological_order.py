# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import threading

import numpy as np
import onnx_light.onnx.helper as oh
import torch
from onnx_light.onnx import TensorProto, checker
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder

from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase


class TestGraphTopologicalOrder(ExtTestCase):
    def make_builder(self, cls=None, graph=None):
        if cls is None:
            cls = Model
        builder = cls.__new__(cls)
        builder.graph = GraphBuilder("test") if graph is None else graph
        builder.graph.set_opset_version("", 21)
        builder.graph.set_opset_version("com.microsoft", 1)
        builder._initialize_graph_state()
        builder.local_functions = {}
        builder.reuse_downloaded_weights = False
        return builder

    def test_consumer_before_producer_uses_complete_native_graph(self):
        builder = self.make_builder()
        builder.make_graph_input("source", TensorProto.FLOAT, [1])
        builder.make_graph_output("final", TensorProto.FLOAT, [1])
        builder.make_node("Identity", ["later"], ["result"], name="consumer")
        builder.make_node("Identity", ["source"], ["other"], name="independent")
        builder.make_node("Identity", ["source"], ["later"], name="producer")
        builder.make_node("Identity", ["later"], ["native_use"], name="native_use")
        builder.make_node("Add", ["result", "later"], ["final"], name="last")

        model = builder.to_model_proto()
        native = builder.graph.to_model(ir_version=10)
        self.assertEqual([node.name for node in native.graph.node], ["independent", "producer", "consumer", "native_use", "last"])
        self.assertEqual([node.name for node in model.graph.node], ["independent", "producer", "consumer", "native_use", "last"])
        self.assertEqual(model.graph.SerializeToString(), builder.graph.to_model(ir_version=10).graph.SerializeToString())
        checker.check_model(model)
        result = ReferenceEvaluator(model).run(None, {"source": np.array([2], dtype=np.float32)})
        self.assertEqualArray(np.array([4], dtype=np.float32), result[0])

    def test_normalization_and_contrib_nodes_use_native_builder(self):
        builder = self.make_builder()
        builder.make_graph_input("source", TensorProto.FLOAT, [2, 3])
        builder.make_graph_output("result", TensorProto.FLOAT, [2, 3])
        builder.make_initializer(np.ones(3, dtype=np.float32), "scale")
        builder.make_node("LayerNormalization", ["source", "scale"], ["normalized"], name="norm", axis=-1)
        builder.make_node("SimplifiedLayerNormalization", ["normalized", "scale"], ["simplified"], name="rms", axis=-1)
        builder.make_node("Gelu", ["simplified"], ["result"], name="gelu", domain="com.microsoft", optional_attribute=None)
        model = builder.to_model_proto()
        self.assertEqual([node.name for node in builder.graph.to_model().graph.node], ["norm", "rms", "gelu"])
        self.assertEqual(model.graph.node[-1].domain, "com.microsoft")
        self.assertFalse(model.graph.node[-1].attribute)
        self.assertEqual(model.graph.SerializeToString(), builder.graph.to_model(ir_version=10).graph.SerializeToString())
        self.assertEqual(model.SerializeToString(), builder.to_model_proto().SerializeToString())

    def test_pending_initializer_registers_nodes_on_the_calling_thread(self):
        main_thread = threading.get_ident()
        workers = []
        graph_threads = []
        started = threading.Event()
        proceed = threading.Event()

        class RecordingGraph:
            def __init__(self, name):
                self.native = GraphBuilder(name)

            def __getattr__(self, name):
                return getattr(self.native, name)

            def make_node(self, *args, **kwargs):
                graph_threads.append(threading.get_ident())
                return self.native.make_node(*args, **kwargs)

            def make_initializer_move(self, *args, **kwargs):
                graph_threads.append(threading.get_ident())
                return self.native.make_initializer_move(*args, **kwargs)

        class DelayedModel(Model):
            def _prepare_initializer(self, tensor, name, to):
                workers.append(threading.get_ident())
                started.set()
                if not proceed.wait(10):
                    raise TimeoutError("Initializer preparation was not released.")
                return super()._prepare_initializer(tensor, name, to)

        builder = self.make_builder(DelayedModel, RecordingGraph("test"))
        builder.make_graph_input("source", TensorProto.FLOAT, [2, 4])
        builder.make_graph_output("result", TensorProto.FLOAT, [2, 6])
        with builder._parallel_initializers():
            try:
                builder.make_initializer(torch.ones(4, 6), "weight")
                self.assertTrue(started.wait(10))
                builder.make_node("MatMul", ["source", "weight"], ["result"], name="matmul")
                self.assertFalse(builder.graph.has_name("result"))
            finally:
                proceed.set()
            model = builder.to_model_proto()
        self.assertTrue(workers)
        self.assertTrue(all(thread != main_thread for thread in workers))
        self.assertEqual(graph_threads, [main_thread, main_thread])
        checker.check_model(model)
        actual = ReferenceEvaluator(model).run(None, {"source": np.ones((2, 4), dtype=np.float32)})[0]
        self.assertEqualArray(np.full((2, 6), 4, dtype=np.float32), actual)

    def test_subgraph_capture_waits_for_its_producer(self):
        builder = self.make_builder()
        builder.make_graph_input("source", TensorProto.FLOAT, [1])
        builder.make_graph_input("condition", TensorProto.BOOL, [])
        builder.make_graph_output("result", TensorProto.FLOAT, [1])
        branch = oh.make_graph(
            [oh.make_node("Identity", ["later"], ["branch_result"])],
            "branch",
            [],
            [oh.make_tensor_value_info("branch_result", TensorProto.FLOAT, [1])],
        )
        builder.make_node("If", ["condition"], ["result"], name="consumer", then_branch=branch, else_branch=branch)
        self.assertFalse(builder.graph.has_name("result"))
        builder.make_node("Identity", ["source"], ["later"], name="producer")
        model = builder.to_model_proto()
        self.assertEqual([node.name for node in model.graph.node], ["producer", "consumer"])
        checker.check_model(model)
        actual = ReferenceEvaluator(model).run(None, {"source": np.array([2], dtype=np.float32), "condition": np.array(True)})[0]
        self.assertEqualArray(np.array([2], dtype=np.float32), actual)

    def test_unresolved_values_and_cycles_are_rejected(self):
        for cycle in (False, True):
            with self.subTest(cycle=cycle):
                builder = self.make_builder()
                builder.make_node("Identity", ["missing"], ["result"], name="first")
                if cycle:
                    builder.make_node("Identity", ["result"], ["missing"], name="second")
                with self.assertRaisesRegex(ValueError, "cycle" if cycle else "unresolved inputs: missing"):
                    builder.to_model_proto()
        builder = self.make_builder()
        builder.make_graph_output("missing", TensorProto.FLOAT, [1])
        with self.assertRaisesRegex(ValueError, "unresolved output"):
            builder.to_model_proto()

    def test_pending_producer_cannot_be_shadowed(self):
        for source in ("node", "initializer", "input"):
            with self.subTest(source=source):
                builder = self.make_builder()
                builder.make_node("Identity", ["missing"], ["value"], name="pending")
                with self.assertRaisesRegex(ValueError, "redefine value|pending node producer"):
                    if source == "node":
                        builder.make_node("Identity", ["other"], ["value"], name="duplicate")
                    elif source == "initializer":
                        builder.make_initializer(np.ones(1, dtype=np.float32), "value")
                    else:
                        builder.make_graph_input("value", TensorProto.FLOAT, [1])

    def test_reverse_chain_drains_iteratively(self):
        builder = self.make_builder()
        length = 1100
        builder.make_graph_output("value_0", TensorProto.FLOAT, [1])
        for index in range(length):
            builder.make_node("Identity", [f"value_{index + 1}"], [f"value_{index}"], name=f"node_{index}")
        builder.make_graph_input(f"value_{length}", TensorProto.FLOAT, [1])
        model = builder.to_model_proto()
        self.assertEqual(len(model.graph.node), length)
        self.assertEqual(model.graph.node[0].name, f"node_{length - 1}")
        self.assertEqual(model.graph.node[-1].name, "node_0")
