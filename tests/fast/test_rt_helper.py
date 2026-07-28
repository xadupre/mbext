# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for the internal helpers of :mod:`modelbuilder.helpers.rt_helper`.
"""

import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.helpers.rt_helper import _get_dim, _make_empty_cache, _ort_type_to_numpy_dtype


class TestRtHelperInternals(ExtTestCase):
    def test_ort_type_to_numpy_dtype_known(self):
        cases = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(uint8)": np.uint8,
            "tensor(bool)": np.bool_,
        }
        for ort_type, expected in cases.items():
            with self.subTest(ort_type=ort_type):
                self.assertIs(_ort_type_to_numpy_dtype(ort_type), expected)

    def test_ort_type_to_numpy_dtype_bfloat16(self):
        try:
            import ml_dtypes
        except ImportError:
            self.skipTest("ml_dtypes not installed")
        self.assertIs(_ort_type_to_numpy_dtype("tensor(bfloat16)"), ml_dtypes.bfloat16)

    def test_ort_type_to_numpy_dtype_unknown_raises(self):
        with self.assertRaises(ValueError):
            _ort_type_to_numpy_dtype("tensor(unknown)")

    def test_get_dim_integer_passthrough(self):
        self.assertEqual(_get_dim(0, 7), 7)
        self.assertEqual(_get_dim(3, 5), 5)

    def test_get_dim_batch_dimension(self):
        # Symbolic/None values at position 0 resolve to the batch size.
        self.assertEqual(_get_dim(0, "batch", batch=4), 4)
        self.assertEqual(_get_dim(0, None, batch=2), 2)

    def test_get_dim_non_batch_symbolic_is_zero(self):
        # Symbolic/None values at any other position start empty.
        self.assertEqual(_get_dim(1, "past_seq_len"), 0)
        self.assertEqual(_get_dim(2, None), 0)

    def test_make_empty_cache_shapes_and_dtype(self):
        cache_names = ["past_key_values.0.key", "past_key_values.0.value"]
        cache_shapes = [["batch", 4, "past", 8], ["batch", 4, "past", 8]]
        cache_types = ["tensor(float)", "tensor(float16)"]

        feeds = _make_empty_cache(2, cache_names, cache_shapes, cache_types)

        self.assertEqual(set(feeds), set(cache_names))
        for name in cache_names:
            # batch=2, symbolic head/sequence dims collapse to 0.
            self.assertEqual(feeds[name].shape, (2, 4, 0, 8))
            self.assertTrue(np.all(feeds[name] == 0))
        self.assertEqual(feeds[cache_names[0]].dtype, np.float32)
        self.assertEqual(feeds[cache_names[1]].dtype, np.float16)

    def test_make_empty_cache_empty_inputs(self):
        self.assertEqual(_make_empty_cache(1, [], [], []), {})

    def test_make_empty_cache_null_batch_raises(self):
        with self.assertRaises(ValueError):
            _make_empty_cache(0, ["k"], [["batch", 2]], ["tensor(float)"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
