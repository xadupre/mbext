# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os

from .base import Model


class MistralModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class MistralNeMoModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_model(self, input_path):
        # Mistral-Nemo-Instruct uses head_dim=128 even though
        # hidden_size / num_attention_heads = 160.  When the model config.json
        # does not carry an explicit ``head_dim`` field the base class falls
        # back to the division and produces the wrong value.  Correct it here
        # by reading the actual K-projection weight shape from the on-disk
        # safetensors shard before any ONNX nodes are emitted.
        self._fix_head_size_from_weights(input_path)
        super().make_model(input_path)

    def _fix_head_size_from_weights(self, input_path):
        """Infer ``head_size`` from the K-projection weight dimensions.

        The K-projection weight has shape
        ``[num_kv_heads * head_dim, hidden_size]``, so
        ``head_dim = weight.shape[0] // num_kv_heads``.

        This check is performed without loading the full tensors into memory
        by using the safetensors metadata API (``get_shape()``).  If no
        safetensors file is found, or if the inferred value matches the
        current ``self.head_size``, the method is a no-op.
        """
        if not os.path.isdir(input_path):
            return

        try:
            from safetensors import safe_open
        except ImportError:
            return

        for filename in sorted(os.listdir(input_path)):
            if not filename.endswith(".safetensors"):
                continue
            filepath = os.path.join(input_path, filename)
            try:
                with safe_open(filepath, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.endswith(".k_proj.weight"):
                            shape = f.get_slice(key).get_shape()
                            # shape: [num_kv_heads * head_dim, hidden_size]
                            actual_head_size = shape[0] // self.num_kv_heads
                            if actual_head_size != self.head_size:
                                self.head_size = actual_head_size
                            return
            except Exception:
                continue
