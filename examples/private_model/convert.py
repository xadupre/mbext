# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Converter file for the ``private_model`` example.

This is the ``convert-file`` of the ``--private`` option. It defines the ONNX
builder used to convert the tiny private MoE model to ONNX. That model is
implemented from scratch in ``modeling.py`` as ``PrivateModelForCausalLM`` (a
stack of ``PrivateDecoderLayer`` blocks wrapped by ``PrivateModel``).

The example shows how to implement a **custom decoder layer with a
Mixture-of-Experts (MoE) MLP**. Each :class:`~modeling.PrivateDecoderLayer`
mirrors the Mistral attention stack (RMSNorm + rotary GQA attention), so the
builder reuses :class:`modelbuilder.builders.mistral.MistralModel` for the
attention and overrides the layer/MLP to build a sparse MoE instead of the dense
MLP. It reads the modules exposed by :class:`~modeling.PrivateDecoderLayer`:

* ``self_attn.{q,k,v,o}_proj`` — the attention projections,
* ``mlp.gate`` — the router that sends each token to the top-``num_experts_per_tok``
  of ``num_local_experts`` experts, and
* ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj`` — the packed
  per-expert SwiGLU weights.

The MoE subgraph itself is emitted by the shared
:meth:`~modelbuilder.builders.base.Model.make_fused_moe` helper, which produces a
single ``com.microsoft:MoE`` op (or ``QMoE`` when quantizing). The routing
weights are normalized over the selected experts, so ``normalize_routing_weights``
is enabled and the SwiGLU activation is fused (``swiglu_fusion=1``).

:func:`modelbuilder.builder.load_private_model_builder` selects the builder from
the module-level ``MODEL_BUILDER`` attribute if present, otherwise it uses the
single :class:`~modelbuilder.builders.Model` subclass defined in this file.
"""

from modelbuilder.builders.mistral import MistralModel


class PrivateMoEModel(MistralModel):
    """Custom builder for ``PrivateModelForCausalLM``: Mistral attention + a MoE MLP."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # The experts use a SwiGLU MLP; the gate/up projections are fused and the
        # top-k routing weights are renormalized over the selected experts.
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["swiglu_fusion"] = 1
        self.moe_attrs["normalize_routing_weights"] = True
        # The MoE layers are baked into the exported ONNX graph, so for the
        # onnxruntime-genai runtime the model is a standard Mistral-family
        # decoder. Expose it as such so ``genai_config.json`` carries a
        # model type the genai runtime recognizes.
        self.model_type = "MistralForCausalLM"

    def make_layer(self, layer_id, layer):
        # Each decoder layer is defined as:
        # input_layernorm --> attention --> post_attention_layernorm --> MoE
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(
            layer_id, layer.post_attention_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="post_attention"
        )
        self.make_moe(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_moe(self, layer_id, mlp, root_input):
        """Build the sparse MoE subgraph for one decoder layer.

        This model has no shared expert and no expert biases, so it simply
        delegates to the shared fused-MoE builder and wires the MoE output as the
        residual (skip) input of the next layernorm.
        """
        moe_name = self.make_fused_moe(layer_id, mlp, root_input)
        self.layernorm_attrs["skip_input"] = f"{moe_name}/output_0"


MODEL_BUILDER = PrivateMoEModel
