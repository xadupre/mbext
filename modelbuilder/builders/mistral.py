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
        # derived from the correct value from the very start.  This covers
        # every code path — including ``config_only=True`` — where
        # make_model() is never called.
        _fix_config_head_dim(config, cache_dir)
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_model(self, input_path):
        # Fallback for models not yet downloaded locally: self.head_size may
        # still be wrong (160) if _fix_config_head_dim() could not find a
        # local safetensors file.  Pre-load the weights here to get the real
        # head_dim from the k_proj weight, then patch self.head_size and the
        # KV-cache shape lists before super().make_model() calls
        # make_inputs_and_outputs().  The load_weights() override below
        # returns the cached model so the weights are loaded only once.
        try:
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
        except Exception:  # noqa: BLE001
            if hasattr(self, "_preloaded_weights"):
                del self._preloaded_weights

        super().make_model(input_path)

    def load_weights(self, input_path):
        # Return the pre-loaded model cached by make_model() so the full
        # model is loaded only once.
        if hasattr(self, "_preloaded_weights"):
            weights = self._preloaded_weights
            del self._preloaded_weights
            return weights
        return super().load_weights(input_path)


def _fix_config_head_dim(config, cache_dir):
    """
    Read the k_proj weight shape from the safetensors header (a few KB,
    no tensor data) and patch ``config.head_dim`` to the actual value.

    Args:
        config: A HuggingFace ``PretrainedConfig`` whose ``_name_or_path``
            points to the local model directory or a HuggingFace Hub repo ID.
        cache_dir: Directory used as the HuggingFace Hub cache.  May be
            ``None`` to use the default cache location.

    This is a best-effort operation: any failure is silently ignored and
    the caller proceeds with the config-derived head_dim.
    """
    try:
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

        # Locate the model directory.
        model_path = config._name_or_path
        if not os.path.isdir(model_path):
            # Not a local directory — try the HuggingFace Hub cache.
            try:
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(
                    model_path,
                    cache_dir=cache_dir,
                    local_files_only=True,
                )
            except Exception:  # noqa: BLE001
                return

        # Find the shard that contains the k_proj weight.
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
                return
            sf_path = os.path.join(model_path, sf_basename)
        else:
            sf_path = os.path.join(model_path, "model.safetensors")

        if not os.path.exists(sf_path):
            return

        # Read only the JSON header of the safetensors file (shape metadata).
        # Format: 8-byte LE uint64 giving the header length, then JSON bytes.
        with open(sf_path, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_size))

        k_proj_key = next((k for k in header if "k_proj.weight" in k), None)
        if k_proj_key is None:
            return

        shape = header[k_proj_key]["shape"]
        actual_head_dim = shape[0] // num_kv_heads

        if actual_head_dim != current_head_dim:
            config.head_dim = actual_head_dim

    except Exception:  # noqa: BLE001
        pass
