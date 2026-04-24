# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Builder for DeepSeek-V3 / DeepSeek-V4-Flash (DeepseekV3ForCausalLM / DeepseekV4ForCausalLM).

Both architectures share the same MLA+MoE design and the same HuggingFace config
fields, so a single builder handles both.

Architecture highlights:
* Multi-head Latent Attention (MLA): compressed KV via low-rank projection.
  - Q path:  hidden → q_a_proj → q_a_layernorm → q_b_proj → split(nope|rope) → RoPE on rope part → concat
  - KV path: hidden → kv_a_proj_with_mqa → split(k_pass|k_pe)
             k_pass → kv_a_layernorm → kv_b_proj → split(k_nope|v)
             k_pe  → RoPE (single head) → expand to all heads
  - V is padded to qk_head_dim for GQA compatibility, then sliced after attention.
* MoE FFN for layers with index >= first_k_dense_replace (with shared expert).
* Dense MLP for the first first_k_dense_replace layers.
* Interleaved RoPE applied only to the qk_rope_head_dim dimensions.

Note on fp8/fp4: DeepSeek's proprietary inference/model.py uses fp8/fp4 quantized
kernels for their in-house CUDA inference engine.  The HuggingFace checkpoints
store weights as standard bfloat16, and the transformers library exposes them as
ordinary PyTorch tensors.  This builder exports ONNX from those float weights;
fp8/fp4 quantization is not required and is not applied here.

Note: The ORT MoE fused kernel uses softmax-based routing, which is an
approximation of DeepSeek's sigmoid-based grouped-topk routing.  The model
structure and weight layout are correct; numerical routing accuracy may differ.
"""

import numpy as np

from .base import Model


class DeepSeekV3Model(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # The base Model class reads config.head_dim if present.  For DeepSeek V3,
        # config.__post_init__ sets config.head_dim = qk_rope_head_dim (e.g., 64).
        # We call super() first so it can set up all standard attrs; then we
        # override head_size to qk_head_dim (e.g., 192) for the KV-cache layout.
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # After super(): self.head_size == config.head_dim == qk_rope_head_dim (64)

        # ── DeepSeek-specific attention dimensions ──────────────────────────
        self.qk_nope_head_dim = config.qk_nope_head_dim  # e.g., 128
        self.qk_rope_head_dim = config.qk_rope_head_dim  # e.g.,  64
        self.v_head_dim = config.v_head_dim  # e.g., 128
        self.q_lora_rank = config.q_lora_rank  # e.g., 1536 (None = direct proj)
        self.kv_lora_rank = config.kv_lora_rank  # e.g., 512
        qk_head_dim = config.qk_head_dim  # e.g., 192 = qk_nope + qk_rope

        # ── MoE config ────────────────────────────────────────────────────
        self.first_k_dense_replace = config.first_k_dense_replace  # dense MLP layers
        self.n_shared_experts = getattr(config, "n_shared_experts", 1)

        # ── Override head_size to qk_head_dim for KV-cache layout ─────────
        # K cache:   [B, num_heads, S, qk_head_dim]
        # V cache:   [B, num_heads, S, qk_head_dim]  (V padded from v_head_dim)
        self.head_size = qk_head_dim
        self.input_shapes["past_key_values.key"] = ["batch_size", self.num_kv_heads, "past_sequence_length", qk_head_dim]
        self.input_shapes["past_key_values.value"] = ["batch_size", self.num_kv_heads, "past_sequence_length", qk_head_dim]
        self.output_shapes["present.key"] = ["batch_size", self.num_kv_heads, "total_sequence_length", qk_head_dim]
        self.output_shapes["present.value"] = ["batch_size", self.num_kv_heads, "total_sequence_length", qk_head_dim]

        # ── RoPE: partial rotation on qk_rope_head_dim dims only ──────────
        # After super(), head_size was qk_rope_head_dim (64), so the base rope
        # cache computation dim = partial_rotary_factor * head_size must yield
        # qk_rope_head_dim.  Now that head_size = qk_head_dim (192), we set
        # partial_rotary_factor = qk_rope_head_dim / qk_head_dim and
        # rotary_embedding_dim = qk_rope_head_dim so the cache is shaped
        # [context_length, qk_rope_head_dim // 2].
        self.rope_attrs["partial_rotary_factor"] = self.qk_rope_head_dim / qk_head_dim
        self.rope_attrs["rotary_embedding_dim"] = self.qk_rope_head_dim
        # Interleaved format (rope_interleave=True by default in DeepSeek V3)
        self.rope_attrs["interleaved"] = 1 if getattr(config, "rope_interleave", True) else 0

        # ── Attention scale ───────────────────────────────────────────────
        # Scale is based on qk_head_dim, with optional YaRN mscale.
        scale = 1.0 / np.sqrt(qk_head_dim)
        rope_params = getattr(config, "rope_parameters", {})
        if rope_params and rope_params.get("rope_type", "default") != "default":
            mscale_all_dim = rope_params.get("mscale_all_dim", 0)
            factor = rope_params.get("factor", 1.0)
            if mscale_all_dim and factor > 1:
                mscale = 0.1 * np.log(factor) + 1.0
                scale = scale * mscale * mscale
        self.attention_attrs["scale"] = scale

        # ── MoE kernel settings ───────────────────────────────────────────
        self.moe_attrs["activation_type"] = "swiglu"
        self.moe_attrs["normalize_routing_weights"] = True
        self.moe_attrs["swiglu_fusion"] = 1

    def make_attention_init(self):
        super().make_attention_init()
        # MLA applies RoPE manually (only to a subset of Q/K dims), so we must
        # disable the GQA-internal RoPE and keep position_ids as a model input.
        if self.attention_attrs.get("use_rope_in_attn", False):
            self.attention_attrs["use_rope_in_attn"] = False
            # The base class deleted position_ids from input_names; restore it.
            if "position_ids" not in self.input_names:
                new_names = {}
                for k, v in self.input_names.items():
                    if k == "past_key_values.key":
                        new_names["position_ids"] = "position_ids"
                    new_names[k] = v
                self.input_names = new_names
                # types/shapes for position_ids still exist from __init__, so no
                # need to re-add them.

    # ------------------------------------------------------------------
    # Layer dispatch: dense MLP for first N layers, MoE for the rest
    # ------------------------------------------------------------------

    def make_layer(self, layer_id, layer):
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
        if layer_id < self.first_k_dense_replace:
            self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])
        else:
            self.make_moe_layer(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    # ------------------------------------------------------------------
    # MLA Attention
    # ------------------------------------------------------------------

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Multi-head Latent Attention (MLA).

        Q path (LoRA decomposition):
          hidden → q_a_proj → q_a_layernorm → q_b_proj
            → split: [q_nope | q_pe]
            → RoPE on q_pe (all heads, qk_rope_head_dim dims)
            → concat(q_nope, q_pe_rotated) per head

        KV path (shared compressed KV):
          hidden → kv_a_proj_with_mqa → split: [k_pass | k_pe]
          k_pass → kv_a_layernorm → kv_b_proj → split: [k_nope | v]
          k_pe  → RoPE (single head) → expand to all heads
          concat(k_nope, k_pe_expanded) per head

        V is zero-padded to qk_head_dim for GQA, sliced back after attention.
        """
        b = f"/model/layers.{layer_id}/attn"
        h = self.num_attn_heads
        hq = self.head_size  # = qk_head_dim
        hn = self.qk_nope_head_dim
        hr = self.qk_rope_head_dim
        hv = self.v_head_dim
        kv_r = self.kv_lora_rank
        seq = "sequence_length"
        bs = "batch_size"

        # ── Q path ────────────────────────────────────────────────────────
        if self.q_lora_rank is not None:
            # Q LoRA: hidden → q_a_proj → q_a_layernorm → q_b_proj
            q_a_name = self._make_mla_matmul(b, "q_a_proj", attention.q_a_proj, root_input, [bs, seq, self.q_lora_rank])
            q_a_norm_weight = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
            self.make_initializer(attention.q_a_layernorm.weight, q_a_norm_weight, to=self.io_dtype)
            q_a_norm_out = f"{b}/q_a_layernorm/output_0"
            self._make_simplified_layer_norm(
                f"{b}/q_a_layernorm",
                root_input=q_a_name,
                weight_name=q_a_norm_weight,
                output_0=q_a_norm_out,
                io_dtype=self.io_dtype,
                shape=[bs, seq, self.q_lora_rank],
            )
            q_full = self._make_mla_matmul(b, "q_b_proj", attention.q_b_proj, q_a_norm_out, [bs, seq, h * hq])
        else:
            q_full = self._make_mla_matmul(b, "q_proj", attention.q_proj, root_input, [bs, seq, h * hq])

        # Reshape Q to [B, S, h, hq], split into nope and rope parts
        q_4d = self._reshape_3d_to_4d(f"{b}/q/Reshape_4d", q_full, h, hq, [bs, seq, h, hq])
        q_nope = self._slice_last_dim(f"{b}/q_nope/Slice", q_4d, 0, hn, [bs, seq, h, hn])
        q_pe = self._slice_last_dim(f"{b}/q_pe/Slice", q_4d, hn, hn + hr, [bs, seq, h, hr])

        # Flatten q_pe for RotaryEmbedding: [B, S, h, hr] → [B, S, h*hr]
        q_pe_3d = self._reshape_4d_to_3d(f"{b}/q_pe/Reshape_3d", q_pe, h * hr, [bs, seq, h * hr])
        q_pe_rotated_3d = self._apply_rope(f"{b}/q_pe/RotaryEmbedding", q_pe_3d, num_heads=h, shape=[bs, seq, h * hr], **kwargs)
        q_pe_rotated_4d = self._reshape_3d_to_4d(f"{b}/q_pe/Reshape_4d_back", q_pe_rotated_3d, h, hr, [bs, seq, h, hr])

        # Concat q_nope and q_pe_rotated along last dim → [B, S, h, hq]
        q_combined_4d = self._concat_last_dim(f"{b}/q_combined/Concat", q_nope, q_pe_rotated_4d, [bs, seq, h, hq])
        # Reshape to GQA-ready 3D: [B, S, h*hq]
        q_3d = self._reshape_4d_to_3d(f"{b}/q/Reshape_3d", q_combined_4d, h * hq, [bs, seq, h * hq])

        # ── KV path ───────────────────────────────────────────────────────
        # kv_a_proj_with_mqa: hidden → [B, S, kv_lora_rank + qk_rope_head_dim]
        kv_a = self._make_mla_matmul(b, "kv_a_proj_with_mqa", attention.kv_a_proj_with_mqa, root_input, [bs, seq, kv_r + hr])
        k_pass_raw = self._slice_last_dim(f"{b}/k_pass_raw/Slice", kv_a, 0, kv_r, [bs, seq, kv_r])
        k_pe_raw = self._slice_last_dim(f"{b}/k_pe_raw/Slice", kv_a, kv_r, kv_r + hr, [bs, seq, hr])

        # k_pass → kv_a_layernorm → kv_b_proj → [B, S, h*(hn+hv)]
        kv_a_norm_weight = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
        self.make_initializer(attention.kv_a_layernorm.weight, kv_a_norm_weight, to=self.io_dtype)
        k_pass_norm_out = f"{b}/kv_a_layernorm/output_0"
        self._make_simplified_layer_norm(
            f"{b}/kv_a_layernorm",
            root_input=k_pass_raw,
            weight_name=kv_a_norm_weight,
            output_0=k_pass_norm_out,
            io_dtype=self.io_dtype,
            shape=[bs, seq, kv_r],
        )
        kv_b = self._make_mla_matmul(b, "kv_b_proj", attention.kv_b_proj, k_pass_norm_out, [bs, seq, h * (hn + hv)])

        # Reshape kv_b to [B, S, h, hn+hv], split into k_nope and v
        kv_4d = self._reshape_3d_to_4d(f"{b}/kv/Reshape_4d", kv_b, h, hn + hv, [bs, seq, h, hn + hv])
        k_nope = self._slice_last_dim(f"{b}/k_nope/Slice", kv_4d, 0, hn, [bs, seq, h, hn])
        v_4d = self._slice_last_dim(f"{b}/v/Slice", kv_4d, hn, hn + hv, [bs, seq, h, hv])

        # k_pe (single head) → RoPE: [B, S, hr] → [B, S, hr]
        k_pe_rotated = self._apply_rope(f"{b}/k_pe/RotaryEmbedding", k_pe_raw, num_heads=1, shape=[bs, seq, hr], **kwargs)

        # Expand k_pe_rotated to all heads: [B, S, hr] → [B, S, h, hr]
        k_pe_4d = self._expand_to_heads(f"{b}/k_pe_expand", k_pe_rotated, h, hr, k_nope)

        # Concat k_nope and k_pe_expanded along last dim → [B, S, h, hq]
        k_combined_4d = self._concat_last_dim(f"{b}/k_combined/Concat", k_nope, k_pe_4d, [bs, seq, h, hq])
        k_3d = self._reshape_4d_to_3d(f"{b}/k/Reshape_3d", k_combined_4d, h * hq, [bs, seq, h * hq])

        # ── V padding ─────────────────────────────────────────────────────
        # Pad V from hv to hq along the head_dim axis, then reshape to 3D.
        v_padded_4d = self._pad_v(f"{b}/v_pad/Pad", v_4d, hv, hq, h)
        v_3d = self._reshape_4d_to_3d(f"{b}/v/Reshape_3d", v_padded_4d, h * hq, [bs, seq, h * hq])

        # ── GroupQueryAttention ───────────────────────────────────────────
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)
        attn_name = f"{b}/GroupQueryAttention"
        self.make_node(
            "GroupQueryAttention",
            inputs=[
                q_3d,
                k_3d,
                v_3d,
                past_k,
                past_v,
                f"{self.mask_attrs['seqlens_k']}/output_0",
                f"{self.mask_attrs['total_seq_len']}/output_0",
                "",  # cos_cache (not used; RoPE applied above)
                "",  # sin_cache
                "",  # position_ids
                "",  # attention_bias
                "",  # sinks
            ],
            outputs=[f"{attn_name}/output_0", present_k, present_v],
            name=attn_name,
            domain="com.microsoft",
            num_heads=h,
            kv_num_heads=h,
            scale=self.attention_attrs["scale"],
            local_window_size=self.window_size,
            softcap=self.attention_attrs["softcap"],
            do_rotary=0,
            rotary_interleaved=self.rope_attrs["interleaved"],
        )
        self.make_value(f"{attn_name}/output_0", self.io_dtype, shape=[bs, seq, h * hq])

        # ── Slice GQA output to v_head_dim ────────────────────────────────
        attn_out_4d = self._reshape_3d_to_4d(f"{b}/attn_out/Reshape_4d", f"{attn_name}/output_0", h, hq, [bs, seq, h, hq])
        attn_out_sliced = self._slice_last_dim(f"{b}/attn_out/Slice", attn_out_4d, 0, hv, [bs, seq, h, hv])
        attn_out_3d = self._reshape_4d_to_3d(f"{b}/attn_out/Reshape_3d", attn_out_sliced, h * hv, [bs, seq, h * hv])

        # ── Output projection o_proj ──────────────────────────────────────
        o_matmul_name = f"{b}/o_proj/MatMul"
        self.make_matmul(attention.o_proj, o_matmul_name, attn_out_3d)

        o_bias_exists = attention.o_proj.bias is not None
        if o_bias_exists:
            o_add_name = f"{b}/o_proj/Add"
            self.make_add_bias(attention.o_proj.bias, o_add_name, root_input=f"{o_matmul_name}/output_0")
        self.layernorm_attrs["skip_input"] = f"{o_matmul_name if not o_bias_exists else o_add_name}/output_0"

    # ------------------------------------------------------------------
    # MoE layer (routed experts + shared expert)
    # ------------------------------------------------------------------

    def make_moe_layer(self, layer_id, mlp, root_input):
        """MoE: fused routed experts + always-on shared expert."""
        moe_out = self._make_moe_fused(layer_id, mlp, root_input)

        # Shared expert: always applied
        shared_out = self._make_shared_expert(layer_id, mlp.shared_experts, root_input)

        # Add routed + shared expert outputs
        add_name = f"/model/layers.{layer_id}/moe/shared_add/Add"
        combined = self.make_add(add_name, [moe_out, shared_out], self.io_dtype, ["batch_size", "sequence_length", self.hidden_size])
        self.layernorm_attrs["skip_input"] = combined

    def _make_moe_fused(self, layer_id, mlp, root_input):
        """Build router + fused MoE kernel for the routed experts."""
        basename = f"/model/layers.{layer_id}/moe"

        # Router: hidden → logits [B*S, n_routed_experts]
        router_matmul_name = f"{basename}/router/MatMul"
        self.make_matmul(mlp.gate, router_matmul_name, root_input)
        # Reshape to [B*S, n_experts] for MoE kernel
        router_reshape_name = f"{basename}/router/Reshape"
        router_reshape_inputs = [f"{router_matmul_name}/output_0", f"/model/constants/INT64/{[-1, self.moe_attrs['num_experts']]}"]
        self.make_reshape(
            router_reshape_name,
            router_reshape_inputs,
            dtype=self.io_dtype,
            shape=["batch_size * sequence_length", self.moe_attrs["num_experts"]],
        )

        # Expert weights – already in the format expected by the ORT MoE kernel:
        #   gate_up_proj: [n_experts, 2*moe_inter, hidden]  (matches weight1 spec)
        #   down_proj:    [n_experts, hidden, moe_inter]     (matches weight2 spec)
        gate_up_weight = f"model.layers.{layer_id}.moe.experts.gate_up_proj.weight"
        down_weight = f"model.layers.{layer_id}.moe.experts.down_proj.weight"
        self.make_initializer(mlp.experts.gate_up_proj, gate_up_weight, to=self.io_dtype)
        self.make_initializer(mlp.experts.down_proj, down_weight, to=self.io_dtype)

        moe_name = f"{basename}/MoE"
        self.make_base_moe_op(
            moe_name, root_input=root_input, router_probs=f"{router_reshape_name}/output_0", weight1=gate_up_weight, weight2=down_weight
        )
        return f"{moe_name}/output_0"

    def _make_shared_expert(self, layer_id, shared_mlp, root_input):
        """Build the always-active shared expert MLP (SiLU-gated)."""
        b = f"/model/layers.{layer_id}/moe/shared_expert"
        inter = shared_mlp.gate_proj.weight.shape[0]  # shared intermediate size

        gate_name = self._make_mla_matmul(b, "gate_proj", shared_mlp.gate_proj, root_input, ["batch_size", "sequence_length", inter])
        up_name = self._make_mla_matmul(b, "up_proj", shared_mlp.up_proj, root_input, ["batch_size", "sequence_length", inter])

        # SiLU(gate) * up
        silu_name = f"{b}/silu/Sigmoid"
        self.make_sigmoid(silu_name, gate_name, self.io_dtype, ["batch_size", "sequence_length", inter])
        mul_name = f"{b}/silu/Mul"
        self.make_mul(mul_name, [f"{silu_name}/output_0", gate_name], self.io_dtype, ["batch_size", "sequence_length", inter])
        act_mul_name = f"{b}/act_mul/Mul"
        self.make_mul(act_mul_name, [f"{mul_name}/output_0", up_name], self.io_dtype, ["batch_size", "sequence_length", inter])

        # down_proj
        down_name = self._make_mla_matmul(
            b, "down_proj", shared_mlp.down_proj, f"{act_mul_name}/output_0", ["batch_size", "sequence_length", self.hidden_size]
        )
        return down_name

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------

    def _make_mla_matmul(self, basename, proj_name, linear, root_input, out_shape):
        """Build a single MatMul for an MLA projection and return output name."""
        matmul_name = f"{basename}/{proj_name}/MatMul"
        self.make_matmul(linear, matmul_name, root_input)
        return f"{matmul_name}/output_0"

    def _apply_rope(self, name, root_input, num_heads, shape, **kwargs):
        """Apply ORT RotaryEmbedding to *root_input*.

        Creates the cos/sin caches the first time it is called (they are cached
        lazily by the base class).  The cache size is
        [context_length, qk_rope_head_dim // 2], which is correct because
        partial_rotary_factor = qk_rope_head_dim / head_size and the base
        make_rotary_embedding_caches_from_scratch uses dim = partial_rotary_factor * head_size.
        """
        cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()
        position_ids = kwargs.get("position_ids", self.input_names.get("position_ids", "position_ids"))
        inputs = [root_input, position_ids, cos_cache_name, sin_cache_name]
        output = f"{name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            inputs=inputs,
            outputs=[output],
            name=name,
            domain="com.microsoft",
            interleaved=self.rope_attrs["interleaved"],
            num_heads=(num_heads if num_heads != 1 else 0),
            rotary_embedding_dim=0,  # full rotation on the (already split) input
        )
        self.make_value(output, self.io_dtype, shape=shape)
        return output

    def _reshape_3d_to_4d(self, name, root_input, num_heads, head_dim, shape):
        """[B, S, num_heads * head_dim] → [B, S, num_heads, head_dim]."""
        inputs = [root_input, f"/model/constants/INT64/[0, 0, {num_heads}, {head_dim}]"]
        return self.make_reshape(name, inputs, self.io_dtype, shape)

    def _reshape_4d_to_3d(self, name, root_input, flat_dim, shape):
        """[B, S, num_heads, head_dim] → [B, S, num_heads * head_dim]."""
        inputs = [root_input, f"/model/constants/INT64/[0, 0, {flat_dim}]"]
        return self.make_reshape(name, inputs, self.io_dtype, shape)

    def _slice_last_dim(self, name, root_input, start, end, shape):
        """Slice the last dimension: [..., start:end]."""
        return self.make_slice(name, root_input, self.io_dtype, shape, starts=[start], ends=[end], axes=[-1])

    def _concat_last_dim(self, name, a, b_tensor, shape):
        """Concat two tensors along the last axis."""
        return self.make_concat(name, [a, b_tensor], self.io_dtype, shape, axis=-1)

    def _expand_to_heads(self, name, k_pe_rotated, num_heads, rope_dim, k_nope_4d):
        """Expand k_pe_rotated [B, S, rope_dim] to [B, S, num_heads, rope_dim].

        Uses the Shape of k_nope_4d [B, S, num_heads, nope_dim] to get B and S
        dynamically, then builds a 4D expand target shape.
        """
        bs = "batch_size"
        seq = "sequence_length"

        # Unsqueeze: [B, S, rope_dim] → [B, S, 1, rope_dim]
        unsq_name = f"{name}/Unsqueeze"
        unsq_inputs = [k_pe_rotated, "/model/constants/INT64/[2]"]
        self.make_unsqueeze(unsq_name, unsq_inputs, self.io_dtype, [bs, seq, 1, rope_dim])

        # Tile by [1, 1, num_heads, 1] → [B, S, num_heads, rope_dim]
        tile_name = f"{name}/Tile"
        self.make_tile(
            tile_name,
            [f"{unsq_name}/output_0", f"/model/constants/INT64/[1, 1, {num_heads}, 1]"],
            self.io_dtype,
            [bs, seq, num_heads, rope_dim],
        )
        return f"{tile_name}/output_0"

    def _pad_v(self, name, v_4d, hv, hq, num_heads):
        """Zero-pad V from [..., hv] to [..., hq] along last axis.

        Returns output tensor name with shape [batch_size, sequence_length, num_heads, hq].
        """
        pad_amount = hq - hv
        bs = "batch_size"
        seq = "sequence_length"

        # ONNX Pad pads = [begin_dim0, begin_dim1, ..., end_dim0, end_dim1, ...]
        # We want to pad only the last dim at the end.
        # For a 4D tensor: pads = [0,0,0,0, 0,0,0,pad_amount]
        pads_name = f"/model/constants/INT64/[0, 0, 0, 0, 0, 0, 0, {pad_amount}]"
        pad_output = f"{name}/output_0"
        self.make_node("Pad", inputs=[v_4d, pads_name], outputs=[pad_output], name=name)
        self.make_value(pad_output, self.io_dtype, shape=[bs, seq, num_heads, hq])
        return pad_output
