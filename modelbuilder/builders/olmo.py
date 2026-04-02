# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import torch

from .base import Model


class OLMoModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        layernorm.weight = torch.ones(self.hidden_size)
        layernorm.bias = torch.zeros(self.hidden_size)
        super().make_layernorm(layer_id, layernorm, skip, simple, location)


class OLMo2Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()


class OLMo3Model(OLMo2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layer_types = config.layer_types

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # OLMo3 uses per-layer sliding window attention controlled by layer_types.
        # We temporarily override window_size for full-attention layers, then restore it.
        original_window_size = self.window_size
        if self.layer_types[layer_id] == "full_attention":
            self.window_size = -1
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size
