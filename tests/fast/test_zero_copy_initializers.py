# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""In-memory initializer ownership and serialization."""

import gc
import os
import tempfile
import unittest
import weakref

import numpy as np
import torch
from onnx_light.onnx import ModelProto, TensorProto, load, save_model
from onnx_light.onnx.numpy_helper import to_array
from onnx_light.onnx.reference import ReferenceEvaluator
from onnx_light.onnx_core.graph_builder import GraphBuilder

from examples.benchmark_zero_copy_initializers import peak_rss_mib
from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase


def make_builder():
    model = Model.__new__(Model)
    model.graph = GraphBuilder("weights")
    model.reuse_downloaded_weights = False
    model._initializer_names = set()
    model.values = {}
    return model


class TestZeroCopyInitializers(ExtTestCase):
    def test_peak_rss_mib_platform_units(self):
        for kib, expected in ((0, 0.0), (256 * 1024, 256.0), (256 * 1024 + 256, 256.2)):
            with self.subTest(kib=kib):
                self.assertEqual(peak_rss_mib(kib, "linux"), expected)
                self.assertEqual(peak_rss_mib(kib * 1024, "darwin"), expected)

    def test_numpy_owner_survives_builder_and_serialization(self):
        model = make_builder()
        array = np.arange(12, dtype=np.float32).reshape(3, 4)
        expected = array.copy()
        owner = weakref.ref(array)
        model.make_initializer(array, "weight", to=TensorProto.FLOAT)
        model.graph.inp("input", TensorProto.FLOAT, [3, 4])
        model.graph.out(model.graph.op.Add("input", "weight"))
        built = model.graph.to_model()
        initializer = built.graph.initializer[0]
        self.assertEqual(np.from_dlpack(initializer).ctypes.data, array.ctypes.data)
        self.assertEqual(model.values["weight"], {"dtype": TensorProto.FLOAT, "shape": (3, 4)})
        del array, model
        gc.collect()
        self.assertIsNotNone(owner())
        serialized = built.SerializeToString()
        restored = ModelProto()
        restored.ParseFromString(serialized)
        self.assertEqual(initializer.name, "weight")
        self.assertEqual(list(initializer.dims), [3, 4])
        np.testing.assert_array_equal(to_array(restored.graph.initializer[0]), expected)
        result = ReferenceEvaluator(restored).run(None, {"input": np.ones((3, 4), dtype=np.float32)})[0]
        np.testing.assert_array_equal(result, expected + 1)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "weights.onnx")
            save_model(built, path, save_as_external_data=True, location="weights.bin", size_threshold=0)
            np.testing.assert_array_equal(to_array(load(path).graph.initializer[0], directory), expected)
        del initializer, built
        gc.collect()
        self.assertIsNone(owner())

    def test_torch_storage_survives_source_release(self):
        model = make_builder()
        source = torch.nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        pointer = source.data_ptr()
        model.make_initializer(source, "weight")
        source.data = torch.empty(0)
        del source
        built = model.graph.to_model()
        del model
        gc.collect()
        initializer = built.graph.initializer[0]
        self.assertEqual(np.from_dlpack(initializer).ctypes.data, pointer)
        np.testing.assert_array_equal(to_array(initializer), np.arange(12, dtype=np.float32).reshape(3, 4))
        restored = ModelProto()
        restored.ParseFromString(built.SerializeToString())
        np.testing.assert_array_equal(to_array(restored.graph.initializer[0]), to_array(initializer))

    def test_fallback_for_strides_and_special_dtype(self):
        for weight in (
            np.arange(12, dtype=np.float32).reshape(3, 4).T,
            torch.arange(12, dtype=torch.float32).reshape(3, 4).T,
            torch.arange(4, dtype=torch.bfloat16),
        ):
            with self.subTest(dtype=weight.dtype):
                model = make_builder()
                model.make_initializer(weight, "weight")
                initializer = model.graph.to_model().graph.initializer[0]
                self.assertEqual(initializer.name, "weight")
                self.assertEqual(list(initializer.dims), list(weight.shape))
                expected = weight.detach().float().numpy() if isinstance(weight, torch.Tensor) else weight
                actual = to_array(initializer)
                np.testing.assert_array_equal(actual.astype(np.float32), expected)

    def test_shape_constant_values_feed_native_inference(self):
        for shape in (np.array([2, 3], dtype=np.int64), torch.tensor([2, 3], dtype=torch.int64)):
            with self.subTest(source=type(shape).__name__):
                model = make_builder()
                model.make_initializer(shape, "shape")
                model.graph.inp("input", TensorProto.FLOAT, [6])
                reshaped = model.graph.op.Reshape("input", "shape")
                model.graph.out(model.graph.op.Transpose(reshaped, perm=[1, 0]))
                built = model.graph.to_model()
                self.assertEqual([dimension.dim_value for dimension in built.graph.output[0].type.tensor_type.shape.dim], [3, 2])
                values = np.arange(6, dtype=np.float32)
                actual = ReferenceEvaluator(built).run(None, {"input": values})[0]
                np.testing.assert_array_equal(actual, values.reshape(2, 3).T)

    def test_torch_metadata_views_preserve_values(self):
        for weight in (torch.tensor([1 + 2j, 3 - 4j]).conj(), torch.tensor([1.0, 2.0])._neg_view()):
            with self.subTest(conjugate=weight.is_conj(), negative=weight.is_neg()):
                model = make_builder()
                model.make_initializer(weight, "weight")
                initializer = model.graph.to_model().graph.initializer[0]
                np.testing.assert_array_equal(to_array(initializer), weight.numpy(force=True))

    def test_numpy_owner_released_exactly_once(self):
        model = make_builder()
        array = np.arange(32, dtype=np.int64)
        released = []
        weakref.finalize(array, released.append, "released")
        model.make_initializer(array, "weight")
        built = model.graph.to_model()
        del array, model
        gc.collect()
        self.assertEqual(released, [])
        del built
        gc.collect()
        self.assertEqual(released, ["released"])

    def test_rejects_non_native_endian_storage(self):
        model = make_builder()
        with self.assertRaisesRegex(ValueError, "native-endian"):
            model.make_initializer(np.array([1, 2], dtype=np.dtype("i4").newbyteorder("S")), "weight")
        self.assertNotIn("weight", model._initializer_names)

    def test_rejects_unsupported_torch_layout_and_device(self):
        for weight in (torch.ones(2).to_sparse(), torch.empty(2, device="meta")):
            with self.subTest(layout=weight.layout, device=weight.device):
                model = make_builder()
                with self.assertRaisesRegex(ValueError, "accessible storage"):
                    model.make_initializer(weight, "weight")
                self.assertNotIn("weight", model._initializer_names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
