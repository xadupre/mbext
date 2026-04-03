# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import struct

from .base import Model


class MistralModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class MistralNeMoModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Mistral-NeMo uses head_dim=128 even though
        # hidden_size / num_attention_heads = 160.  When the model's config.json
        # omits ``head_dim``, transformers defaults it to 160 (wrong).
        #
        # Fix config.head_dim BEFORE calling super().__init__() so that
        # self.head_size, self.input_shapes, and self.output_shapes are all
        # derived from the correct value from the very start.  This only
        # works when the model is already available as a local directory;
        # otherwise make_model() handles the correction after loading weights.
        _fix_config_head_dim(config)
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_model(self, input_path):
        # Pre-load the weights to read k_proj.weight.shape and correct
        # self.head_size when _fix_config_head_dim() could not do so during
        # __init__ (e.g. the model was not yet a local directory at that
        # point).  The load_weights() override below ensures the weights are
        # only loaded once.
        self._preloaded_weights = self.load_weights(input_path)
        for module in self._preloaded_weights.modules():
            k_proj = getattr(module, "k_proj", None)
            if k_proj is None:
                continue
            weight = getattr(k_proj, "weight", None)
            if weight is None:
                continue
            # k_proj.weight shape: [num_kv_heads * head_dim, hidden_size]
            actual_head_size = weight.shape[0] // self.num_kv_heads
            if actual_head_size != self.head_size:
                self.head_size = actual_head_size
                # Patch the KV-cache shape lists baked in by __init__.
                # Each list is [batch, num_kv_heads, seq, head_size].
                for key in ("past_key_values.key", "past_key_values.value"):
                    self.input_shapes[key][3] = actual_head_size
                for key in ("present.key", "present.value"):
                    self.output_shapes[key][3] = actual_head_size
            break
        super().make_model(input_path)

    def load_weights(self, input_path):
        # Return the pre-loaded model cached by make_model() so the full
        # model is loaded only once.
        if hasattr(self, "_preloaded_weights"):
            weights = self._preloaded_weights
            del self._preloaded_weights
            return weights
        return super().load_weights(input_path)


def _fix_config_head_dim(config):
    """
    Patch ``config.head_dim`` by reading the k_proj.weight shape from the
    safetensors header when the model is already available as a local
    directory (``config._name_or_path`` is an existing directory).

    When ``config._name_or_path`` is a Hub repo ID (not a local directory)
    this function returns immediately without modifying ``config``; the
    correction is deferred to ``MistralNeMoModel.make_model``.

    Args:
        config: A HuggingFace ``PretrainedConfig`` whose ``_name_or_path``
            points to the local model directory or a Hub repo ID.
    """
    if not os.path.isdir(config._name_or_path):
        return

    num_kv_heads = (
        config.num_key_value_heads
        if hasattr(config, "num_key_value_heads")
        else config.num_attention_heads
    )
    current_head_dim = (
        config.head_dim
        if hasattr(config, "head_dim") and config.head_dim is not None
        else config.hidden_size // config.num_attention_heads
    )

    actual_head_dim = _read_head_dim_from_safetensors(config._name_or_path, num_kv_heads)
    if actual_head_dim is not None and actual_head_dim != current_head_dim:
        config.head_dim = actual_head_dim


def _read_head_dim_from_safetensors(model_path, num_kv_heads):
    """
    Return the actual head_dim (int) derived from k_proj.weight in the
    safetensors header, or ``None`` when no suitable file is found.

    Only the compact JSON header (a few KB) is read — no tensor data is loaded.

    Args:
        model_path: Path to a local directory containing ``model.safetensors``
            or ``model.safetensors.index.json``.
        num_kv_heads: Number of key/value attention heads used to compute
            ``head_dim = k_proj_rows // num_kv_heads``.
    """
    # Sharded model: locate the shard that holds k_proj via the index file.
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        sf_basename = next(
            (weight_map[k] for k in weight_map if "k_proj.weight" in k),
            None,
        )
        if sf_basename is None:
            return None
        sf_path = os.path.join(model_path, sf_basename)
    else:
        sf_path = os.path.join(model_path, "model.safetensors")

    if not os.path.exists(sf_path):
        return None

    # Safetensors format: 8-byte LE uint64 header length, then JSON bytes.
    with open(sf_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

    k_proj_key = next((k for k in header if "k_proj.weight" in k), None)
    if k_proj_key is None:
        return None

    shape = header[k_proj_key]["shape"]
    return shape[0] // num_kv_heads
