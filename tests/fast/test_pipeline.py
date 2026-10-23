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
import subprocess
import sys
import tempfile
import textwrap
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
    @hide_stdout()
    def test_cli_split_llama_reuses_checkpoint(self):
        from transformers import AutoModelForCausalLM, LlamaConfig

        # A real private exporter splits the saved Llama Gather from the decoder,
        # keeping its external initializer reference, not a second weight copy.
        converter = textwrap.dedent("""\
            import os
            from onnx_light.onnx import helper, load_model, save_model
            from modelbuilder.builders.llama import LlamaModel

            class SplitLlamaModel(LlamaModel):
                def save_model(self, out_dir):
                    super().save_model(out_dir)
                    path = os.path.join(out_dir, self.filename)
                    model = load_model(path, load_external_data=False)
                    gather = next(n for n in model.graph.node
                                  if n.op_type == "Gather" and "input_ids" in n.input)
                    self.embedding_output = str(gather.output[0])
                    value = helper.make_tensor_value_info(
                        self.embedding_output, self.io_dtype,
                        ["batch_size", "sequence_length", self.hidden_size])
                    embedding = helper.make_model(helper.make_graph(
                        [gather], "embedding",
                        [v for v in model.graph.input if v.name == "input_ids"],
                        [value],
                        [v for v in model.graph.initializer
                         if v.name == gather.input[0]]),
                        opset_imports=list(model.opset_import),
                        ir_version=model.ir_version)
                    nodes = [n for n in model.graph.node if n.name != gather.name]
                    weights = [v for v in model.graph.initializer
                               if v.name != gather.input[0]]
                    inputs = [v for v in model.graph.input if v.name != "input_ids"]
                    model.graph.node.clear()
                    model.graph.node.extend(nodes)
                    model.graph.initializer.clear()
                    model.graph.initializer.extend(weights)
                    model.graph.input.clear()
                    model.graph.input.extend(inputs + [value])
                    save_model(model, path)
                    save_model(embedding, os.path.join(out_dir, "embedding.onnx"))

                def create_genai_config(self, *args):
                    config = super().create_genai_config(*args)
                    inputs = config["model"]["decoder"]["inputs"]
                    del inputs["input_ids"]
                    inputs["inputs_embeds"] = self.embedding_output
                    config["model"]["embedding"] = {
                        "filename": "embedding.onnx",
                        "inputs": {"input_ids": "input_ids"},
                        "outputs": {"inputs_embeds": self.embedding_output}}
                    return config
            """)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "export")
            source_dir = os.path.join(output_dir, ".weights")
            config = LlamaConfig(
                architectures=["LlamaForCausalLM"],
                hidden_size=16,
                intermediate_size=24,
                max_position_embeddings=16,
                num_attention_heads=2,
                num_hidden_layers=1,
                num_key_value_heads=1,
                vocab_size=32,
            )
            AutoModelForCausalLM.from_config(config).save_pretrained(source_dir)
            self.make_word_level_tokenizer().save_pretrained(source_dir)
            convert_path = os.path.join(tmp, "convert.py")
            with open(convert_path, "w") as stream:
                stream.write(converter)
            command = [
                sys.executable,
                "-m",
                "modelbuilder.builder",
                "-i",
                source_dir,
                "-o",
                output_dir,
                "-p",
                "fp32",
                "-e",
                "cpu",
                "-c",
                os.path.join(tmp, "cache"),
                "--reuse-weights",
                "--private",
                f";{convert_path};",
            ]
            for _ in range(2):
                completed = subprocess.run(command, check=False, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                self.assertIn("Saving PipelineProto", completed.stdout)
                self.assertIn(PIPELINE_FILENAME, completed.stdout)
                pipeline = PipelineProto()
                pipeline.ParseFromFile(os.path.join(output_dir, PIPELINE_FILENAME))
                verify_pipeline(pipeline)
                self.assertFalse(pipeline.has_decode())
                for role, filename in (("embedding", "embedding.onnx"), ("prefill", "model.onnx")):
                    standalone = load_model(os.path.join(output_dir, filename), load_external_data=False)
                    self.assertEqual(
                        [t.SerializeToString() for t in standalone.graph.initializer],
                        [t.SerializeToString() for t in getattr(pipeline, role).graph.initializer],
                    )
                locations = {
                    entry.value
                    for tensor in pipeline.embedding.graph.initializer
                    for entry in tensor.external_data
                    if entry.key == "location"
                }
                self.assertEqual(locations, {".weights/model.safetensors"})
                self.assertFalse(any(name.startswith("pipeline.onnx.") for name in os.listdir(output_dir)))

    @staticmethod
    def _model(nodes, inputs, outputs, initializers=()):
        return helper.make_model(
            helper.make_graph(nodes, "component", inputs, outputs, initializers), opset_imports=[helper.make_opsetid("", 18)], ir_version=10
        )

    @staticmethod
    def _value(name, dtype=TensorProto.FLOAT, shape=None):
        return helper.make_tensor_value_info(name, dtype, [1, "sequence", 16] if shape is None else shape)

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
                helper.make_node("ReduceMean", ["hidden", "axes"], ["present"], keepdims=0),
            ],
            [self._value("embeds"), self._value("past", shape=[1, 16])],
            [self._value("scores", shape=[1, "sequence", 32]), self._value("present", shape=[1, 16])],
            [
                from_array(weight, name="weight"),
                from_array(np.array([1], dtype=np.int64), name="axes"),
                # Packed quantized storage must survive without decoding/repacking.
                from_array(np.arange(512).astype(ml_dtypes.uint4), name="packed_weight"),
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
                "embedding": {"filename": "embedding.onnx", "inputs": {"input_ids": "tokens"}, "outputs": {"inputs_embeds": "embeds"}},
                "decoder": decoder_config,
                "eos_token_id": 31,
                "pad_token_id": 0,
            },
            "search": {"do_sample": False, "temperature": 1.0, "top_k": 4, "top_p": 0.9, "max_new_tokens": 4, "random_seed": 42},
        }
        if separate:
            # Reuse all prefill shards for decode, as an exporter with shared
            # weights would. Native serialization must not externalize again.
            saved = load_model(os.path.join(folder, "prefill.onnx"), load_external_data=False)
            saved.graph.persistent_bindings.add(input_name="past", output_name="present")
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
                    result[os.path.relpath(path, folder)] = hashlib.sha256(stream.read()).hexdigest()
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
        for role in (("embedding", "prefill", "decode") if separate else ("embedding", "prefill")):
            standalone = load_model(os.path.join(folder, f"{role}.onnx"), load_external_data=False)
            component = getattr(pipeline, role)
            self.assertEqual([value.name for value in component.graph.input], [value.name for value in standalone.graph.input])
            for expected, actual in zip(standalone.graph.initializer, component.graph.initializer):
                self.assertEqual(expected.SerializeToString(), actual.SerializeToString())
                self.assertFalse(actual.raw_data)
            if role != "embedding":
                self.assertEqual(
                    [(binding.input_name, binding.output_name) for binding in component.graph.persistent_bindings], [("past", "present")]
                )
            load_external_data_for_model(component, folder)
            for actual, expected in zip(component.graph.initializer, load_model(os.path.join(folder, f"{role}.onnx")).graph.initializer):
                self.assertEqualArray(to_array(actual, folder), to_array(expected, folder))

    @hide_stdout()
    def test_shared_weights_shards_relocation_overwrite(self):
        for separate in (False, True):
            for reused, sharded in ((False, False), (False, True), (True, True)):
                with self.subTest(separate=separate, reused=reused, sharded=sharded), tempfile.TemporaryDirectory() as tmp:
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
                        getattr(pipeline, role) if bundled else load_model(os.path.join(folder, f"{role}.onnx"), load_external_data=False)
                    )
                    load_external_data_for_model(model, folder)
                    sessions[role] = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
                tokens = np.array([[1, 2]], dtype=np.int64)
                state = np.zeros((1, 16), dtype=np.float32)
                generated, logits = [], []
                for step in range(4):
                    embeds = sessions["embedding"].run(None, {"tokens": tokens})[0]
                    scores, state = sessions["prefill" if step == 0 else "decode"].run(None, {"embeds": embeds, "past": state})
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
                self.assertFalse(os.path.exists(os.path.join(folder, PIPELINE_FILENAME)))
                after = self._snapshot(folder)
                self.assertEqual(after, {key: value for key, value in before.items() if key != PIPELINE_FILENAME})

    def test_single_model_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = self._export(folder)
            del config["model"]["embedding"]
            self._write_config(folder, config)
            before = self._snapshot(folder)
            self.assertIsNone(save_pipeline(folder))
            self.assertEqual(before, self._snapshot(folder))

    @hide_stdout()
    def test_prefill_decode_without_separate_embedding(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = self._export(folder, separate=True)
            embedding = load_model(os.path.join(folder, "embedding.onnx"), load_external_data=False)
            del config["model"]["embedding"]
            for role, filename in (("prefill", "prefill.onnx"), ("decoder", "decode.onnx")):
                model = load_model(os.path.join(folder, filename), load_external_data=False)
                nodes = list(embedding.graph.node) + list(model.graph.node)
                inputs = [value for value in model.graph.input if value.name != "embeds"]
                model.graph.node.clear()
                model.graph.node.extend(nodes)
                model.graph.input.clear()
                model.graph.input.extend(list(embedding.graph.input) + inputs)
                model.graph.initializer.extend(embedding.graph.initializer)
                if role == "prefill":
                    # The supported prefill contract also allows state-only outputs.
                    outputs = [value for value in model.graph.output if value.name != "scores"]
                    model.graph.output.clear()
                    model.graph.output.extend(outputs)
                save_model(model, os.path.join(folder, filename))
                config["model"][role]["inputs"] = {"input_ids": "tokens", "past_key_names": "past"}
            config["model"]["prefill"]["outputs"] = {"present_key_names": "present"}
            self._write_config(folder, config)
            save_pipeline(folder)
            pipeline = PipelineProto()
            pipeline.ParseFromFile(os.path.join(folder, PIPELINE_FILENAME))
            verify_pipeline(pipeline)
            self.assertFalse(pipeline.has_embedding())
            self.assertTrue(pipeline.has_decode())
            self.assertEqual(pipeline.input_ids_name, "tokens")
            self.assertEqual(pipeline.logits_name, "scores")
            self.assertEqual(len(pipeline.prefill.graph.persistent_bindings), 1)

    def test_sampling_metadata_and_shared_decoder(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = self._export(folder)
            config["model"]["prefill"] = dict(config["model"]["decoder"])
            config["model"]["decoder"]["filename"] = "./prefill.onnx"
            config["model"]["eos_token_id"] = [2, 3]
            config["search"].update(do_sample=True, temperature=0.75)
            self._write_config(folder, config)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                save_pipeline(folder)
            pipeline = PipelineProto()
            pipeline.ParseFromFile(os.path.join(folder, PIPELINE_FILENAME))
            self.assertFalse(pipeline.has_decode())
            self.assertEqual(pipeline.temperature, 0.75)
            self.assertEqual(pipeline.eos_token_id, 2)
            self.assertIn("Multiple stop IDs", output.getvalue())

    def test_phase_interface_names_must_match(self):
        for kind, name in (("inputs", "attention_mask"), ("inputs", "position_ids"), ("outputs", "logits")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                config, _, _ = self._export(folder, separate=True)
                config["model"]["prefill"][kind] = {**config["model"]["prefill"][kind], name: "prompt_name"}
                config["model"]["decoder"][kind] = {**config["model"]["decoder"][kind], name: "decode_name"}
                self._write_config(folder, config)
                with self.assertRaisesRegex(ValueError, "identical prefill/decode"):
                    save_pipeline(folder)

    def test_conflicting_state_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self._export(folder)
            path = os.path.join(folder, "prefill.onnx")
            model = load_model(path, load_external_data=False)
            model.graph.persistent_bindings.add(input_name="past", output_name="scores")
            save_model(model, path)
            with self.assertRaisesRegex(ValueError, "Conflicting state binding"):
                save_pipeline(folder)

    def test_reserved_filename_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            config, _, _ = self._export(folder)
            config["model"]["embedding"]["filename"] = "./pipeline.onnx"
            self._write_config(folder, config)
            with self.assertRaisesRegex(ValueError, "reserved"):
                save_pipeline(folder)

    def test_interface_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            self._export(folder)
            model = load_model(os.path.join(folder, "embedding.onnx"), load_external_data=False)
            model.graph.output[0].name = "renamed"
            model.graph.node[0].output.clear()
            model.graph.node[0].output.append("renamed")
            save_model(model, os.path.join(folder, "embedding.onnx"))
            with self.assertRaisesRegex(ValueError, "does not support interface renaming"):
                save_pipeline(folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
