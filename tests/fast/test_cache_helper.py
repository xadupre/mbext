# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for :mod:`modelbuilder.helpers.cache_helper`.
"""

import unittest

import torch
import transformers

from modelbuilder.ext_test_case import ExtTestCase
from modelbuilder.helpers.cache_helper import _flatten_key_value_cache, _flatten_with_keys_cache, _unflatten_cache, registers_dynamic_cache


def _make_cache(num_layers: int) -> transformers.cache_utils.DynamicCache:
    cache = transformers.cache_utils.DynamicCache()
    for i in range(num_layers):
        key = torch.full((1, 2, 1, 4), float(i) + 0.1)
        value = torch.full((1, 2, 1, 4), float(i) + 0.5)
        cache.update(key, value, layer_idx=i)
    return cache


class TestCacheHelper(ExtTestCase):
    def test_flatten_key_value_cache(self):
        cache = _make_cache(2)
        flat, context = _flatten_key_value_cache(cache)

        self.assertEqual(context, ["key_0", "value_0", "key_1", "value_1"])
        self.assertEqual(len(flat), 4)
        # Interleaved key/value order.
        self.assertTrue(torch.equal(flat[0], cache.layers[0].keys))
        self.assertTrue(torch.equal(flat[1], cache.layers[0].values))
        self.assertTrue(torch.equal(flat[2], cache.layers[1].keys))
        self.assertTrue(torch.equal(flat[3], cache.layers[1].values))

    def test_flatten_with_keys_cache(self):
        cache = _make_cache(2)
        pairs, context = _flatten_with_keys_cache(cache)

        self.assertEqual(context, ["key_0", "value_0", "key_1", "value_1"])
        self.assertEqual([k.key for k, _ in pairs], context)
        self.assertTrue(torch.equal(pairs[0][1], cache.layers[0].keys))
        self.assertTrue(torch.equal(pairs[1][1], cache.layers[0].values))

    def test_unflatten_cache_roundtrip(self):
        for num_layers in (1, 2, 3):
            with self.subTest(num_layers=num_layers):
                cache = _make_cache(num_layers)
                flat, context = _flatten_key_value_cache(cache)
                restored = _unflatten_cache(flat, context)

                self.assertIsInstance(restored, transformers.cache_utils.DynamicCache)
                self.assertEqual(len(restored.layers), num_layers)
                for i in range(num_layers):
                    self.assertTrue(torch.equal(restored.layers[i].keys, cache.layers[i].keys))
                    self.assertTrue(torch.equal(restored.layers[i].values, cache.layers[i].values))

    def test_unflatten_cache_output_type(self):
        cache = _make_cache(2)
        flat, context = _flatten_key_value_cache(cache)
        restored = _unflatten_cache(flat, context, output_type=transformers.cache_utils.DynamicCache)
        self.assertIsInstance(restored, transformers.cache_utils.DynamicCache)

    def test_unflatten_cache_bad_context_raises(self):
        cache = _make_cache(2)
        flat, _ = _flatten_key_value_cache(cache)
        with self.assertRaises(AssertionError):
            _unflatten_cache(flat, ["not", "a", "dynamic", "cache"])

    def test_registers_dynamic_cache_pytree_roundtrip(self):
        try:
            registers_dynamic_cache()
        except ValueError as e:
            # DynamicCache may already be registered by transformers or by
            # another test in the same session.
            if "already registered" not in str(e):
                raise
        cache = _make_cache(3)

        flat, spec = torch.utils._pytree.tree_flatten(cache)
        self.assertEqual(len(flat), 6)
        restored = torch.utils._pytree.tree_unflatten(flat, spec)

        self.assertIsInstance(restored, transformers.cache_utils.DynamicCache)
        for i in range(3):
            self.assertTrue(torch.equal(restored.layers[i].keys, cache.layers[i].keys))
            self.assertTrue(torch.equal(restored.layers[i].values, cache.layers[i].values))


if __name__ == "__main__":
    unittest.main(verbosity=2)
