# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import os
import tempfile
import threading

import numpy as np
import onnx_light.onnx.helper as oh
import torch
from onnx_light.onnx import TensorProto, checker, save_model
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

    def test_omitted_optional_node_outputs_survive_serialization(self):
        builder = self.make_builder()
        builder.make_graph_input("source", TensorProto.FLOAT, [1, 2, 4])
        builder.make_graph_input("skip", TensorProto.FLOAT, [1, 2, 4])
        builder.make_graph_input("scale", TensorProto.FLOAT, [4])
        builder.make_graph_input("bias", TensorProto.FLOAT, [4])
        builder.make_graph_output("normalized", TensorProto.FLOAT, [1, 2, 4])
        builder.make_graph_output("residual", TensorProto.FLOAT, [1, 2, 4])
        builder.make_graph_output("attention", TensorProto.FLOAT, [1, 2, 4])
        builder.make_node(
            "SkipLayerNormalization",
            ["source", "skip", "scale", "bias"],
            ["normalized", "", "", "residual"],
            name="skip_norm",
            domain="com.microsoft",
            epsilon=1e-5,
        )
        builder.make_node(
            "MultiHeadAttention",
            ["source", "source", "source", "", "", "", "", ""],
            ["attention", "", ""],
            name="cross_attention",
            domain="com.microsoft",
            num_heads=1,
        )

        model = builder.to_model_proto()
        serialized = type(model)()
        serialized.ParseFromString(model.SerializeToString())
        outputs_by_name = {node.name: list(node.output) for node in serialized.graph.node}
        self.assertEqual(outputs_by_name["skip_norm"], ["normalized", "", "", "residual"])
        self.assertEqual(outputs_by_name["cross_attention"], ["attention", "", ""])

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
                graph_threads.append(("make_node", threading.get_ident()))
                return self.native.make_node(*args, **kwargs)

            def make_initializer_move(self, *args, **kwargs):
                graph_threads.append(("make_initializer_move", threading.get_ident()))
                return self.native.make_initializer_move(*args, **kwargs)

            def init(self, *args, **kwargs):
                graph_threads.append(("init", threading.get_ident()))
                return self.native.init(*args, **kwargs)

        class DelayedModel(Model):
            def _prepare_initializer(self, tensor, name, to):
                if name == "weight":
                    workers.append(threading.get_ident())
                    started.set()
                    if not proceed.wait(10):
                        raise TimeoutError("Initializer preparation was not released.")
                return super()._prepare_initializer(tensor, name, to)

        builder = self.make_builder(DelayedModel, RecordingGraph("test"))
        self.assertFalse(builder._pending_initializers)
        builder.make_graph_input("source", TensorProto.FLOAT, [2, 4])
        builder.make_graph_output("result", TensorProto.FLOAT, [2, 6])
        with builder._parallel_initializers():
            try:
                builder.make_initializer(torch.ones(4, 6), "weight")
                self.assertTrue(started.wait(10))
                builder.make_initializer(np.ones(6, dtype=np.float32), "bias")
                builder.make_node("MatMul", ["source", "weight"], ["product"], name="matmul")
                builder.make_node("Add", ["product", "bias"], ["result"], name="add")
                self.assertFalse(builder.graph.has_name("product"))
            finally:
                proceed.set()
            model = builder.to_model_proto()
        self.assertTrue(workers)
        self.assertTrue(all(thread != main_thread for thread in workers))
        self.assertEqual(sorted(operation for operation, _ in graph_threads), ["init", "make_initializer_move", "make_node", "make_node"])
        self.assertTrue(all(thread == main_thread for _, thread in graph_threads))
        checker.check_model(model)
        actual = ReferenceEvaluator(model).run(None, {"source": np.ones((2, 4), dtype=np.float32)})[0]
        self.assertEqualArray(np.full((2, 6), 5, dtype=np.float32), actual)

    def test_omitted_lstm_outputs_keep_native_shapes_and_positions(self):
        import onnxruntime as ort

        shapes = {"sequence": [2, 1, 3, 4], "hidden": [1, 3, 4], "cell": [1, 3, 4]}
        for outputs in (["", "hidden", "cell"], ["sequence", "", "cell"]):
            with self.subTest(outputs=outputs):
                builder = self.make_builder()
                builder.make_graph_input("source", TensorProto.FLOAT, [2, 3, 5])
                builder.make_initializer(np.zeros((1, 16, 5), dtype=np.float32), "weight")
                builder.make_initializer(np.zeros((1, 16, 4), dtype=np.float32), "recurrence")
                builder.make_node("LSTM", ["source", "weight", "recurrence"], outputs, name="lstm", hidden_size=4)
                for output in outputs:
                    if output:
                        builder.make_graph_output(output, TensorProto.FLOAT, shapes[output])
                forwarded = next(output for output in outputs if output)
                builder.make_node("Identity", [forwarded], ["result"], name="consumer")
                builder.make_graph_output("result", TensorProto.FLOAT, shapes[forwarded])

                model = builder.to_model_proto()
                native = builder.graph.to_model(ir_version=10)
                self.assertEqual(list(model.graph.node[0].output), outputs)
                for index, output in enumerate(outputs):
                    if output:
                        self.assertEqual(native.graph.node[0].output[index], output)
                for output in model.graph.output:
                    expected_shape = shapes[forwarded] if output.name == "result" else shapes[output.name]
                    self.assertEqual([dim.dim_value for dim in output.type.tensor_type.shape.dim], expected_shape)
                checker.check_model(model)
                with tempfile.TemporaryDirectory() as directory:
                    path = os.path.join(directory, "lstm.onnx")
                    save_model(model, path)
                    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                    actual = session.run(None, {"source": np.ones((2, 3, 5), dtype=np.float32)})
                    for metadata, value in zip(session.get_outputs(), actual):
                        expected_shape = shapes[forwarded] if metadata.name == "result" else shapes[metadata.name]
                        self.assertEqualArray(np.zeros(expected_shape, dtype=np.float32), value)

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

    def test_existing_source_values_cannot_be_redefined(self):
        builder = self.make_builder()
        builder.make_graph_input("source", TensorProto.FLOAT, [1])
        with self.assertRaisesRegex(ValueError, "already available"):
            builder.make_initializer(np.ones(1, dtype=np.float32), "source")

        builder = self.make_builder()
        builder.make_initializer(np.ones(1, dtype=np.float32), "weight")
        with self.assertRaisesRegex(ValueError, "already available"):
            builder.make_graph_input("weight", TensorProto.FLOAT, [1])

    def test_duplicate_node_name_is_idempotent_only_for_same_node(self):
        builder = self.make_builder()
        builder.make_graph_input("source", TensorProto.FLOAT, [1])
        builder.make_node("Identity", ["source"], ["result"], name="identity")
        builder.make_node("Identity", ["source"], ["result"], name="identity")
        with self.assertRaisesRegex(ValueError, "Node name 'identity' is already used"):
            builder.make_node("Identity", ["source"], ["other"], name="identity")

        model = builder.to_model_proto()
        self.assertEqual([node.name for node in model.graph.node], ["identity"])

    def test_duplicate_node_attributes_are_compared_by_value(self):
        builder = self.make_builder()
        builder.make_graph_input("source", TensorProto.FLOAT, [2, 3])
        builder.make_graph_output("result", TensorProto.FLOAT, [3, 2])
        builder.make_node("Transpose", ["source"], ["result"], name="transpose", perm=[1, 0])
        builder.make_node("Transpose", ["source"], ["result"], name="transpose", perm=[1, 0])
        with self.assertRaisesRegex(ValueError, "Node name 'transpose' is already used"):
            builder.make_node("Transpose", ["source"], ["result"], name="transpose", perm=[0, 1])
        model = builder.to_model_proto()
        self.assertEqual(len(model.graph.node), 1)
        inputs = np.arange(6, dtype=np.float32).reshape(2, 3)
        self.assertEqualArray(ReferenceEvaluator(model).run(None, {"source": inputs})[0], inputs.T)

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
