# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""ONNX builder for GPT-2 style models (e.g. ``codeparrot/codeparrot-small``).

GPT-2 differs from the RoPE-based decoder models handled by :class:`Model` in a
few ways:

* it uses learned **absolute** position embeddings (``wpe``) added to the token
  embeddings instead of rotary position embeddings;
* every LayerNorm has a learnable bias (``LayerNorm`` rather than
  ``SimplifiedLayerNorm``);
* attention and MLP projections are stored as ``transformers`` ``Conv1D``
  layers whose weight matrices are the transpose of the equivalent
  ``torch.nn.Linear`` weight, and the attention QKV projection is fused into a
  single ``c_attn`` layer.

This builder adapts those modules to the shapes expected by the generic base
builder and adds the position-embedding lookup.
"""

import types

import torch

from .base import Model


def _conv1d_to_linear(conv):
    """Convert a ``transformers`` ``Conv1D`` layer to an equivalent ``Linear``.

    ``Conv1D`` stores its weight as ``(in_features, out_features)`` whereas
    :class:`torch.nn.Linear` expects ``(out_features, in_features)``.
    """
    in_features, out_features = conv.weight.shape
    linear = torch.nn.Linear(in_features, out_features, bias=conv.bias is not None)
    linear.weight = torch.nn.Parameter(conv.weight.detach().T.contiguous(), requires_grad=False)
    if conv.bias is not None:
        linear.bias = torch.nn.Parameter(conv.bias.detach().clone(), requires_grad=False)
    return linear


class GPT2Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # GPT-2 uses LayerNorm (with bias), not SimplifiedLayerNorm.
        self.layernorm_attrs["simple"] = False

        # GPT-2 uses learned absolute position embeddings, not RoPE.
        self.attention_attrs["rope"] = False

        # GPT-2 MLP is a plain two-layer fully-connected block (FC1 -> act -> FC2),
        # not the gated GateProj/UpProj/DownProj style.
        self.mlp_attrs["use_proj"] = False
        self.mlp_attrs["use_fc"] = True

    def make_attention_init(self):
        super().make_attention_init()
        # GPT-2 has no rotary embeddings, so keep ``position_ids`` as a model
        # input (used for the learned absolute position embeddings) and make sure
        # the attention op does not apply RoPE internally.
        self.attention_attrs["use_rope_in_attn"] = False
        if "position_ids" not in self.input_names:
            self.input_names["position_ids"] = "position_ids"

    def is_layer(self, module):
        return module.__class__.__name__ == "GPT2Block"

    def has_final_norm(self, module, orig_model):
        return hasattr(orig_model, "transformer") and hasattr(orig_model.transformer, "ln_f") and module is orig_model.transformer.ln_f

    def make_embedding(self, embedding):
        # Token embeddings: wte(input_ids)
        super().make_embedding(embedding)

        # Position embeddings: wpe(position_ids), added to the token embeddings.
        token_embeds = self.layernorm_attrs["root_input"]

        pos_weight = "model.wpe.weight"
        self.make_initializer(self.weights.transformer.wpe.weight, pos_weight, to=self.io_dtype)

        gather_name = "/model/wpe/Gather"
        gather_output = f"{gather_name}/output_0"
        self.make_node("Gather", inputs=[pos_weight, self.input_names["position_ids"]], outputs=[gather_output], name=gather_name)
        self.make_value(gather_output, self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size])

        add_name = "/model/wpe/Add"
        self.make_add(
            add_name, [token_embeds, gather_output], dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size]
        )
        add_output = f"{add_name}/output_0"

        self.layernorm_attrs["root_input"] = add_output
        self.layernorm_attrs["skip_input"] = add_output

    def make_layer(self, layer_id, layer):
        # Adapt the GPT-2 block (ln_1 -> attn -> ln_2 -> mlp) to the generic
        # decoder-layer structure expected by the base builder.
        attention = types.SimpleNamespace(query_key_value=_conv1d_to_linear(layer.attn.c_attn), o_proj=_conv1d_to_linear(layer.attn.c_proj))
        mlp = types.SimpleNamespace(fc1=_conv1d_to_linear(layer.mlp.c_fc), fc2=_conv1d_to_linear(layer.mlp.c_proj))
        adapted = types.SimpleNamespace(input_layernorm=layer.ln_1, self_attn=attention, post_attention_layernorm=layer.ln_2, mlp=mlp)
        super().make_layer(layer_id, adapted)
