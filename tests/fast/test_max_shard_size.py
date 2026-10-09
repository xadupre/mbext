# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import argparse
import os
import tempfile
import unittest

from onnx_light.onnx import ModelProto, TensorProto, load
from onnx_light.onnx.numpy_helper import to_array

from modelbuilder.builder import parse_shard_size
from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase


class SmallModel(Model):
    def to_model_proto(self):
        model = ModelProto()
        model.graph.name = "test"
        for name, size in (("first", 1200), ("second", 1024), ("third", 1200), ("inline", 20)):
            initializer = model.graph.initializer.add()
            initializer.name = name
            initializer.data_type = TensorProto.UINT8
            initializer.dims.append(size)
            initializer.raw_data = bytes([size % 251]) * size
        if self.reuse_downloaded_weights:
            first = model.graph.initializer[0]
            first.raw_data = b""
            first.data_location = TensorProto.EXTERNAL
            for key, value in (("location", ".weights/weights.bin"), ("offset", "0"), ("length", "1200")):
                entry = first.external_data.add()
                entry.key = key
                entry.value = value
        return model


class TestMaxShardSize(ExtTestCase):
    def make_model(self, tmp, max_shard_size=None):
        model = SmallModel.__new__(SmallModel)
        model.filename = "model.onnx"
        model.cache_dir = os.path.join(tmp, "cache")
        model.quant_type = None
        model.onnx_dtype = TensorProto.FLOAT
        model.extra_options = {"max_shard_size": max_shard_size}
        model.reuse_downloaded_weights = False
        model._used_source_weight_files = set()
        return model

    def test_parse_size(self):
        self.assertEqual(parse_shard_size("2048"), 2048)
        self.assertEqual(parse_shard_size("2GB"), 2_000_000_000)
        self.assertEqual(parse_shard_size("512MiB"), 512 * 1024**2)
        for value in ("0", "-1", "1.5GB", "hello"):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                parse_shard_size(value)

    def test_shards_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = self.make_model(tmp, 2224)
            model.save_model(tmp)
            first = os.path.join(tmp, "model.onnx.data")
            second = first + ".1"
            self.assertEqual(os.path.getsize(first), 2224)
            self.assertEqual(os.path.getsize(second), 1200)

            saved = load(os.path.join(tmp, "model.onnx"), load_external_data=False)
            locations = [{entry.key: entry.value for entry in init.external_data} for init in saved.graph.initializer]
            self.assertEqual(
                locations[:3],
                [
                    {"location": "model.onnx.data", "offset": "0", "length": "1200"},
                    {"location": "model.onnx.data", "offset": "1200", "length": "1024"},
                    {"location": "model.onnx.data.1", "offset": "0", "length": "1200"},
                ],
            )
            self.assertEqual(locations[3], {})
            loaded = load(os.path.join(tmp, "model.onnx"))
            for initializer in loaded.graph.initializer:
                self.assertEqual(to_array(initializer, tmp).tobytes(), bytes([initializer.dims[0] % 251]) * initializer.dims[0])

            self.make_model(tmp).save_model(tmp)
            self.assertFalse(os.path.exists(second))
            self.assertEqual(os.path.getsize(first), 3424)

    def test_initializer_larger_than_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "first.*exceeds max_shard_size"):
                self.make_model(tmp, 1024).save_model(tmp)
            self.assertFalse(os.path.exists(os.path.join(tmp, "model.onnx")))

    def test_reused_weights_are_not_sharded(self):
        with tempfile.TemporaryDirectory() as tmp:
            weights_dir = os.path.join(tmp, ".weights")
            os.mkdir(weights_dir)
            weights_path = os.path.join(weights_dir, "weights.bin")
            with open(weights_path, "wb") as stream:
                stream.write(bytes([1200 % 251]) * 1200)
            model = self.make_model(tmp, 1200)
            model.reuse_downloaded_weights = True
            model._used_source_weight_files.add(weights_path)
            model.save_model(tmp)
            saved = load(os.path.join(tmp, "model.onnx"), load_external_data=False)
            locations = [
                next(entry.value for entry in init.external_data if entry.key == "location")
                for index, init in enumerate(saved.graph.initializer)
                if index < 3
            ]
            self.assertEqual(locations, [".weights/weights.bin", "model.onnx.data", "model.onnx.data.1"])
            self.assertEqual(os.path.getsize(weights_path), 1200)
            self.assertEqual(os.path.getsize(os.path.join(tmp, "model.onnx.data")), 1024)


if __name__ == "__main__":
    unittest.main(verbosity=2)
