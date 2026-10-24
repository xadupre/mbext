# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import hashlib
import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as helper
import onnxruntime as ort
from onnx_light.onnx.numpy_helper import from_array, to_array
from safetensors.torch import save_file
import torch

from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase, hide_stdout
from modelbuilder.helpers.weight_reuse import WeightReuse
from modelbuilder.pipeline import save_pipeline


class StorageModel(Model):
    def to_model_proto(self):
        return self.proto


class TestWeightRelocation(ExtTestCase):
    def checkpoint(self, directory):
        os.makedirs(directory)
        paths = []
        for i in range(2):
            path = Path(directory) / f"model-0000{i + 1}-of-00002.safetensors"
            save_file({f"weight{i}": torch.arange(512).reshape(16, 32).float()}, path)
            paths.append(path)
        (Path(directory) / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"weight{i}": path.name for i, path in enumerate(paths)}})
        )
        return paths

    @hide_stdout()
    def test_shards_move_unchanged_and_repeat_in_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "export"
            output.mkdir()
            paths = self.checkpoint(source)
            original = {p.name: p.read_bytes() for p in paths}
            unrelated = source / "optimizer.safetensors"
            unrelated.write_bytes(b"unrelated")
            reuse = WeightReuse(source, output)
            self.assertIs(copy.deepcopy(reuse), reuse)
            reuse.relocate()
            for path in paths:
                self.assertFalse(path.exists())
                self.assertEqual((output / ".weights" / path.name).read_bytes(), original[path.name])
            self.assertEqual(unrelated.read_bytes(), b"unrelated")
            self.assertTrue((source / "model.safetensors.index.json").exists())
            shutil.copyfile(source / "model.safetensors.index.json", output / ".weights" / "model.safetensors.index.json")
            WeightReuse(output / ".weights", output).relocate()
            for name, data in original.items():
                self.assertEqual((output / ".weights" / name).read_bytes(), data)

    def test_collision_is_preflighted_without_moving_any_shard(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "export"
            paths = self.checkpoint(source)
            (output / ".weights").mkdir(parents=True)
            collision = output / ".weights" / paths[-1].name
            collision.write_bytes(b"unrelated")
            with self.assertRaises(FileExistsError):
                WeightReuse(source, output)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(collision.read_bytes(), b"unrelated")

    @hide_stdout()
    def test_collision_after_preflight_preserves_valid_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "export"
            paths = self.checkpoint(source)
            output.mkdir()
            original = paths[0].read_bytes()
            reuse = WeightReuse(source, output)
            (output / ".weights").mkdir()
            collision = output / ".weights" / paths[0].name
            collision.write_bytes(b"unrelated")
            with self.assertRaises(FileExistsError):
                reuse.relocate()
            self.assertEqual(paths[0].read_bytes(), original)
            self.assertEqual(collision.read_bytes(), b"unrelated")

    def test_symlink_destinations_and_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "export"
            paths = self.checkpoint(source)
            output.mkdir()
            (output / ".weights").symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "escapes"):
                WeightReuse(source, output)
            (output / ".weights").unlink()
            real = paths[0].with_suffix(".original")
            paths[0].rename(real)
            paths[0].symlink_to(real)
            with self.assertRaisesRegex(ValueError, "regular"):
                WeightReuse(source, output)

    @hide_stdout()
    def test_cross_device_move(self):
        if not os.path.isdir("/dev/shm") or os.stat("/tmp").st_dev == os.stat("/dev/shm").st_dev:
            self.skipTest("Requires two writable filesystems.")
        with tempfile.TemporaryDirectory(dir="/dev/shm") as source_tmp, tempfile.TemporaryDirectory() as output:
            source = Path(source_tmp) / "source"
            paths = self.checkpoint(source)
            original = {path.name: path.read_bytes() for path in paths}
            WeightReuse(source, output).relocate()
            self.assertFalse(any(path.exists() for path in paths))
            for name, data in original.items():
                self.assertEqual((Path(output) / ".weights" / name).read_bytes(), data)
            self.assertEqual(sorted(path.name for path in (Path(output) / ".weights").iterdir()), sorted(original))


class TestSharedGeneratedWeights(ExtTestCase):
    def component(self, weight):
        model = onnx.ModelProto()
        model.ir_version = 10
        model.opset_import.add(domain="", version=21)
        model.graph.name = "test"
        model.graph.input.append(helper.make_tensor_value_info("input_ids", onnx.TensorProto.FLOAT, [32, 32]))
        model.graph.output.append(helper.make_tensor_value_info("logits", onnx.TensorProto.FLOAT, [32, 32]))
        model.graph.initializer.append(from_array(weight, name="weight"))
        model.graph.initializer.append(from_array(weight, name="alias"))
        model.graph.initializer.append(from_array(np.array([1], dtype=np.float32), name="small"))
        model.graph.node.append(helper.make_node("Add", ["input_ids", "weight"], ["logits"]))
        return model

    @hide_stdout()
    def test_shared_storage_shards_pipeline_and_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            tmp = os.path.join(root, "export")
            os.mkdir(tmp)
            source = Path(root) / "source"
            source.mkdir()
            a = np.arange(1024, dtype=np.float32).reshape(32, 32)
            b = a + 1
            save_file({"original": torch.from_numpy(a)}, source / "model.safetensors")
            checkpoint = (source / "model.safetensors").read_bytes()
            offset = 8 + struct.unpack("<Q", checkpoint[:8])[0]
            reuse = WeightReuse(source, tmp)
            paths = [os.path.join(tmp, name) for name in ("prefill.onnx", "decode.onnx")]
            models = [self.component(a), self.component(b)]
            # One identical tensor under different names and in both components.
            models[1].graph.initializer[1].CopyFrom(from_array(a.reshape(16, 64), name="shared"))
            for model, path in zip(models, paths):
                original = model.graph.initializer.add()
                original.CopyFrom(from_array(a, name="original"))
                original.raw_data = b""
                original.data_location = onnx.TensorProto.EXTERNAL
                for key, value in (("location", ".weights/model.safetensors"), ("offset", str(offset)), ("length", str(a.nbytes))):
                    entry = original.external_data.add()
                    entry.key, entry.value = key, value
                model.graph.node[0].output.clear()
                model.graph.node[0].output.append("sum")
                model.graph.node.append(helper.make_node("Add", ["sum", "original"], ["logits"]))
                builder = StorageModel.__new__(StorageModel)
                builder.proto = model
                builder.filename = os.path.basename(path)
                builder.cache_dir = os.path.join(root, "cache")
                builder.quant_type = None
                builder.onnx_dtype = onnx.TensorProto.FLOAT
                builder.extra_options = {"_weight_reuse": copy.deepcopy(reuse), "max_shard_size": 4096}
                builder.reuse_downloaded_weights = True
                builder._used_source_weight_files = set(reuse.paths)
                builder.save_model(tmp)
                staged = onnx.load_model(path, load_external_data=False)
                self.assertTrue(all(not tensor.raw_data for tensor in staged.graph.initializer if tensor.name != "small"))
                self.assertLess(os.path.getsize(path), a.nbytes)
            self.assertEqual(sum(path.stat().st_size for path in Path(tmp).glob("*.data*")), a.nbytes + b.nbytes)
            self.assertTrue((source / "model.safetensors").exists())
            self.assertTrue(os.path.exists(paths[0]))
            reuse.relocate()
            self.assertFalse((source / "model.safetensors").exists())
            self.assertEqual((Path(tmp) / ".weights" / "model.safetensors").read_bytes(), checkpoint)
            exports = reuse.collect_exports()
            Model._save_with_reused_weights(*exports[0], exports=exports)
            files = sorted(name for name in os.listdir(tmp) if ".data" in name)
            self.assertEqual(files, ["decode.onnx.data", "prefill.onnx.data"])
            self.assertEqual(sum(os.path.getsize(os.path.join(tmp, name)) for name in files), a.nbytes + b.nbytes)
            saved = [onnx.load_model(path, load_external_data=False) for path in paths]
            refs = [[{e.key: e.value for e in tensor.external_data} for tensor in model.graph.initializer] for model in saved]
            self.assertEqual(refs[0][0], refs[0][1])
            self.assertEqual(refs[0][0], refs[1][1])
            self.assertEqual(refs[0][2], {})
            self.assertEqual(refs[0][3], refs[1][3])
            config = {"model": {role: {"filename": os.path.basename(path)} for role, path in zip(("prefill", "decode"), paths)}}
            with open(os.path.join(tmp, "genai_config.json"), "w") as stream:
                json.dump(config, stream)
            hashes = {name: hashlib.sha256(Path(tmp, name).read_bytes()).digest() for name in files}
            pipeline_path = save_pipeline(tmp)
            pipeline = onnx.PipelineProto()
            pipeline.ParseFromFile(pipeline_path)
            self.assertEqual({e.key: e.value for e in pipeline.decode.graph.initializer[1].external_data}, refs[0][0])
            self.assertEqual(hashes, {name: hashlib.sha256(Path(tmp, name).read_bytes()).digest() for name in files})
            for path, expected in zip(paths, (a, b)):
                self.assertEqualArray(to_array(onnx.load_model(path).graph.initializer[0], tmp), expected)
                self.assertEqualArray(to_array(onnx.load_model(path).graph.initializer[3], tmp), a)
            Model._save_with_reused_weights(*exports[0], exports=exports)
            self.assertEqual(files, sorted(name for name in os.listdir(tmp) if ".data" in name))
            self.assertEqual(hashes, {name: hashlib.sha256(Path(tmp, name).read_bytes()).digest() for name in files})
            relocated = os.path.join(root, "relocated")
            shutil.move(tmp, relocated)
            shutil.rmtree(source)
            pipeline.ParseFromFile(os.path.join(relocated, "pipeline.onnx"))
            for role, path, weight in zip(("prefill", "decode"), paths, (a, b)):
                component = getattr(pipeline, role)
                self.assertEqualArray(to_array(component.graph.initializer[3], relocated), a)
                session = ort.InferenceSession(os.path.join(relocated, os.path.basename(path)), providers=["CPUExecutionProvider"])
                result = session.run(None, {"input_ids": np.ones((32, 32), dtype=np.float32)})
                self.assertEqualArray(result[0], 1 + weight + a)

    def test_no_generated_file_for_inline_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.component(np.ones((1, 1), dtype=np.float32))
            path = os.path.join(tmp, "model.onnx")
            Model._save_with_reused_weights(model, path, path + ".data", None)
            self.assertEqual(os.listdir(tmp), ["model.onnx"])
            self.assertTrue(all(t.data_location != onnx.TensorProto.EXTERNAL for t in onnx.load_model(path).graph.initializer))

    def test_component_with_only_shared_payloads_creates_no_data_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            payloads = {}
            weight = np.arange(1024, dtype=np.float32).reshape(32, 32)
            paths = [os.path.join(tmp, name + ".onnx") for name in ("first", "second")]
            for path in paths:
                Model._save_with_reused_weights(self.component(weight), path, path + ".data", 4096, payloads=payloads)
            self.assertEqual(sorted(Path(tmp).glob("*.data*")), [Path(paths[0] + ".data")])
            self.assertEqual(os.path.getsize(paths[0] + ".data"), weight.nbytes)
            for path in paths:
                self.assertEqualArray(to_array(onnx.load_model(path).graph.initializer[0], tmp), weight)


if __name__ == "__main__":
    unittest.main(verbosity=2)
