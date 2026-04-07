# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from typing import Any, List, Tuple
import itertools
import torch
import transformers


def _flatten_key_value_cache(
    cache: transformers.cache_utils.DynamicCache,
) -> Tuple[List[Any], torch.utils._pytree.Context]:
    keys = [lay.keys for lay in cache.layers]
    values = [lay.values for lay in cache.layers]
    flat = list(itertools.chain.from_iterable(zip(keys, values)))
    unique = set(type(lay) for lay in cache.layers)
    assert unique == {
        transformers.cache_utils.DynamicLayer
    }, f"Not implemented for layers type {unique}"
    keys = list(
        itertools.chain.from_iterable(
            (f"key_{i}", f"value_{i}") for i in range(len(cache.layers))
        )
    )
    return flat, keys


def _flatten_with_keys_cache(
    cache: transformers.cache_utils.DynamicCache,
) -> Tuple[List[Tuple[torch.utils._pytree.MappingKey, Any]], torch.utils._pytree.Context]:
    values, context = _flatten_key_value_cache(cache)
    return [(torch.utils._pytree.MappingKey(k), v) for k, v in zip(context, values)], context


def _unflatten_cache(
    values: List[Any],
    context: torch.utils._pytree.Context,
    output_type=None,
) -> transformers.cache_utils.DynamicCache:
    """Restores a cache from python objects."""
    expected = list(
        itertools.chain.from_iterable((f"key_{i}", f"value_{i}") for i in range(len(values) // 2))
    )
    assert expected == context, f"Does not seem to be a dynamic cache {expected} != {context}"
    res = transformers.cache_utils.DynamicCache()
    for i in range(len(values) // 2):
        res.update(values[i * 2], values[i * 2 + i], layer_idx=i)
    assert output_type is None or isinstance(
        res, output_type
    ), f"Type mismatch between {output_type} (expected) and {type(res)}"
    return res


def registers_dynamic_cache():
    cls = transformers.cache_utils.DynamicCache
    torch.utils._pytree.register_pytree_node(
        cls,
        _flatten_key_value_cache,
        _unflatten_cache,
        serialized_type_name=f"{cls.__module__}.{cls.__name__}",
        flatten_with_keys_fn=_flatten_with_keys_cache,
    )
