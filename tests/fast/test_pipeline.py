# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import contextlib
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest

import ml_dtypes
import numpy as np
from onnx_light.onnx import PipelineProto, TensorProto, helper, load_model, save_model
from onnx_light.onnx.external_data_helper import load_external_data_for_model
from onnx_light.onnx.numpy_helper import from_array, to_array
from onnx_light.onnx_proto.verify import verify_pipeline

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout
from modelbuilder.pipeline import PIPELINE_FILENAME, save_pipeline


class TestPipeline(ExtTestCase):
    @staticmethod
    def _model(nodes, inputs, outputs, initializers=()):
        return helper.make_model(
            helper.make_graph(nodes, "component", inputs, outputs, initializers),
            opset_imports=[helper.make_opsetid("", 18)],
            ir_version=10,
        )

    @staticmethod
    def _value(name, dtype=TensorProto.FLOAT, shape=None):
        return helper.make_tensor_value_info(
            name, dtype, [1, "sequence", 16] if shape is None else shape
        )

    def _export(self, folder, separate=False, reused=False, sharded=False):
        rng = np.random.default_rng(42)
        table = rng.normal(size=(32, 16)).astype(np.float32)
        weight = rng.normal(size=(16, 32)).astype(np.float32)
        embedding = self._model(
            [helper.make_node("Gather", ["table", "tokens"], ["embeds"])],
            [self._value("tokens", TensorProto.INT64, [1, "sequence"])],
            [self._value("embeds")],
            [from_array(table, name="table")],
        )
        decoder = self._model(
            [
                helper.make_node("Add", ["embeds", "past"], ["hidden"]),
                helper.make_node("MatMul", ["hidden", "weight"], ["scores"]),
                helper.make_node(
                    "ReduceMean", ["hidden", "axes"], ["present"], keepdims=0
                ),
            ],
            [self._value("embeds"), self._value("past", shape=[1, 16])],
            [
                self._value("scores", shape=[1, "sequence", 32]),
                self._value("present", shape=[1, 16]),
            ],
            [
                from_array(weight, name="weight"),
                from_array(np.array([1], dtype=np.int64), name="axes"),
                # Packed quantized storage must survive without decoding/repacking.
                from_array(
                    np.arange(512).astype(ml_dtypes.uint4), name="packed_weight"
                ),
            ],
        )
        for name, model in (("embedding", embedding), ("prefill", decoder)):
            location = f"{name}.onnx.data"
            if reused:
                os.makedirs(os.path.join(folder, ".weights"), exist_ok=True)
                location = f".weights/{name}.data"
            save_model(
                model,
                os.path.join(folder, f"{name}.onnx"),
                save_as_external_data=True,
                location=location,
                size_threshold=0,
                max_external_file_size=2048 if sharded else 0,
            )
        decoder_config = {
            "filename": "prefill.onnx",
            "inputs": {"inputs_embeds": "embeds", "past_key_names": ["past"]},
            "outputs": {"logits": "scores", "present_key_names": ["present"]},
        }
        config = {
            "model": {
                "embedding": {
                    "filename": "embedding.onnx",
                    "inputs": {"input_ids": "tokens"},
                    "outputs": {"inputs_embeds": "embeds"},
                },
                "decoder": decoder_config,
                "eos_token_id": 31,
                "pad_token_id": 0,
            },
            "search": {
                "do_sample": False,
                "temperature": 1.0,
                "top_k": 4,
                "top_p": 0.9,
                "max_new_tokens": 4,
                "random_seed": 42,
            },
        }
        if separate:
            # Reuse all prefill shards for decode, as an exporter with shared
            # weights would. Native serialization must not externalize again.
            saved = load_model(
                os.path.join(folder, "prefill.onnx"), load_external_data=False
            )
            saved.graph.persistent_bindings.add(
                input_name="past", output_name="present"
            )
            save_model(saved, os.path.join(folder, "decode.onnx"))
            config["model"]["prefill"] = dict(decoder_config)
            config["model"]["decoder"] = {**decoder_config, "filename": "decode.onnx"}
        self._write_config(folder, config)
        return config, table, weight

    @staticmethod
    def _write_config(folder, config):
        with open(os.path.join(folder, "genai_config.json"), "w") as stream:
            json.dump(config, stream)

    @staticmethod
    def _snapshot(folder):
        result = {}
        for root, _, files in os.walk(folder):
            for filename in files:
                path = os.path.join(root, filename)
                with open(path, "rb") as stream:
                    result[os.path.relpath(path, folder)] = hashlib.sha256(
                        stream.read()
                    ).hexdigest()
        return result

    def _check_components(self, folder, pipeline, separate):
        verify_pipeline(pipeline)
        self.assertTrue(pipeline.has_embedding())
        self.assertEqual(pipeline.has_decode(), separate)
        self.assertEqual(pipeline.input_ids_name, "tokens")
        self.assertEqual(pipeline.logits_name, "scores")
        self.assertEqual(pipeline.temperature, 0.0)
        self.assertEqual(pipeline.max_new_tokens, 4)
        self.assertEqual(pipeline.seed, 42)
        self.assertEqual(pipeline.eos_token_id, 31)
        self.assertEqual(pipeline.pad_token_id, 0)
        self.assertEqual(pipeline.top_k, 4)
        self.assertAlmostEqual(pipeline.top_p, 0.9, atol=1e-7)
        for role in (
            ("embedding", "prefill", "decode") if separate else ("embedding", "prefill")
        ):
            standalone = load_model(
                os.path.join(folder, f"{role}.onnx"), load_external_data=False
            )
            component = getattr(pipeline, role)
            self.assertEqual(
                [value.name for value in component.graph.input],
                [value.name for value in standalone.graph.input],
            )
            for expected, actual in zip(
                standalone.graph.initializer, component.graph.initializer
            ):
                self.assertEqual(
                    expected.SerializeToString(), actual.SerializeToString()
                )
                self.assertFalse(actual.raw_data)
            if role != "embedding":
                self.assertEqual(
                    [
                        (binding.input_name, binding.output_name)
                        for binding in component.graph.persistent_bindings
                    ],
                    [("past", "present")],
                )
            load_external_data_for_model(component, folder)
            for actual, expected in zip(
                component.graph.initializer,
                load_model(os.path.join(folder, f"{role}.onnx")).graph.initializer,
            ):
                self.assertEqualArray(
                    to_array(actual, folder), to_array(expected, folder)
                )

    @hide_stdout()
    def test_shared_weights_shards_relocation_overwrite(self):
        for separate in (False, True):
            for reused, sharded in ((False, False), (False, True), (True, True)):
                with (
                    self.subTest(separate=separate, reused=reused, sharded=sharded),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    folder = os.path.join(tmp, "export")
                    os.mkdir(folder)
                    self._export(folder, separate, reused, sharded)
                    before = self._snapshot(folder)
                    path = save_pipeline(folder)
                    self.assertEqual(path, os.path.join(folder, PIPELINE_FILENAME))
                    after = self._snapshot(folder)
                    self.assertEqual(set(after) - set(before), {PIPELINE_FILENAME})
                    self.assertEqual(before, {key: after[key] for key in before})
                    save_pipeline(folder)
                    self.assertEqual(after, self._snapshot(folder))
                    moved = os.path.join(tmp, "relocated")
                    shutil.move(folder, moved)
                    pipeline = PipelineProto()
                    pipeline.ParseFromFile(os.path.join(moved, PIPELINE_FILENAME))
                    self._check_components(moved, pipeline, separate)
                    # A fresh export may rewrite shards; overwrite the pipeline
                    # too, so it never keeps old offsets or stale components.
                    self._export(moved, separate, reused, sharded)
                    save_pipeline(moved)
                    restored = PipelineProto()
                    restored.ParseFromFile(os.path.join(moved, PIPELINE_FILENAME))
                    self._check_components(moved, restored, separate)

    @hide_stdout()
    def test_generation_parity(self):
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as folder:
            self._export(folder, separate=True)
            save_pipeline(folder)
            pipeline = PipelineProto()
            pipeline.ParseFromFile(os.path.join(folder, PIPELINE_FILENAME))
            results = []
            for bundled in (False, True):
                sessions = {}
                for role in ("embedding", "prefill", "decode"):
                    model = (
                        getattr(pipeline, role)
                        if bundled
                        else load_model(
                            os.path.join(folder, f"{role}.onnx"),
                            load_external_data=False,
                        )
                    )
                    load_external_data_for_model(model, folder)
                    sessions[role] = ort.InferenceSession(
                        model.SerializeToString(), providers=["CPUExecutionProvider"]
                    )
                tokens = np.array([[1, 2]], dtype=np.int64)
                state = np.zeros((1, 16), dtype=np.float32)
                generated, logits = [], []
                for step in range(4):
                    embeds = sessions["embedding"].run(None, {"tokens": tokens})[0]
                    scores, state = sessions["prefill" if step == 0 else "decode"].run(
                        None, {"embeds": embeds, "past": state}
                    )
                    logits.append(scores[:, -1].copy())
                    tokens = scores[:, -1].argmax(axis=-1).reshape(1, 1)
                    generated.append(int(tokens[0, 0]))
                results.append((generated, logits, state))
            self.assertEqual(results[0][0], results[1][0])
            self.assertEqualArray(np.array(results[0][1]), np.array(results[1][1]))
            self.assertEqualArray(results[0][2], results[1][2])

    def test_unsupported_multimodal_is_explicit(self):
        for roles in (("vision",), ("vision", "speech"), ("encoder",)):
            with self.subTest(roles=roles), tempfile.TemporaryDirectory() as folder:
                config, _, _ = self._export(folder)
                for role in roles:
                    config["model"][role] = {"filename": f"{role}.onnx"}
                self._write_config(folder, config)
                # A stale bundle must not be reported as the new pipeline.
                PipelineProto().SerializeToFile(os.path.join(folder, PIPELINE_FILENAME))
                before = self._snapshot(folder)
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertIsNone(save_pipeline(folder))
                self.assertIn("schema extensions", output.getvalue())
                for role in roles:
                    self.assertIn(role, output.getvalue())
                self.assertFalse(
                    os.path.exists(os.path.join(folder, PIPELINE_FILENAME))
                )
                after = self._snapshot(folder)
                self.assertEqual(
                    after,
                    {
                        key: value
                        for key, value in before.items()
                        if key != PIPELINE_FILENAME
                    },
                )

    def test_single_model_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = self._export(folder)
            del config["model"]["embedding"]
            self._write_config(folder, config)
            before = self._snapshot(folder)
            self.assertIsNone(save_pipeline(folder))
            self.assertEqual(before, self._snapshot(folder))

    def test_interface_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self._export(folder)
            model = load_model(
                os.path.join(folder, "embedding.onnx"), load_external_data=False
            )
            model.graph.output[0].name = "renamed"
            model.graph.node[0].output.clear()
            model.graph.node[0].output.append("renamed")
            save_model(model, os.path.join(folder, "embedding.onnx"))
            with self.assertRaisesRegex(
                ValueError, "does not support interface renaming"
            ):
                save_pipeline(folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
