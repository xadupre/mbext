# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os

import numpy as np

from .mistral import MistralModel


class GemmaModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embed_attrs["scale"] = np.round(np.sqrt(self.hidden_size), decimals=2)
        self.layernorm_attrs["add_offset"] = 1


class Gemma2Model(GemmaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = False
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = False
        self.attention_attrs["scale"] = config.query_pre_attn_scalar**-0.5

    def is_local(self, layer_id):
        return layer_id % 2 == 1

    def make_layernorm(self, layer_id, layernorm, skip, simple, location):
        if "final_norm" in location:
            # Set cast for final LayerNorm since it is a special case and not covered in `make_layer`
            self.layernorm_attrs["cast"]["root_input"] = False
        super().make_layernorm(layer_id, layernorm, skip, simple, location)

    def make_layer(self, layer_id, layer):
        # Gemma-2 decoder layer is typically defined as:
        # input_layernorm --> attention --> post_attention_layernorm --> pre_ffn_layernorm --> MLP --> post_ffn_layernorm

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_attention_layernorm
        # 2. Set skip_input to output of post_attention_layernorm
        # 3. Do not cast outputs from post_attention_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id, layer.post_attention_layernorm, skip=False, simple=self.layernorm_attrs["simple"], location="post_attention"
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        # Adjust LayerNorm attributes because of extra LayerNorms inserted
        # 1. Only cast root_input if the first layer of LayerNorms are being created
        original_cast_root_input = self.layernorm_attrs["cast"]["root_input"]
        self.layernorm_attrs["cast"]["root_input"] = self.layernorm_attrs["first_layernorm"]
        self.make_layernorm(
            layer_id, layer.pre_feedforward_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="pre_feedforward"
        )
        self.layernorm_attrs["cast"]["root_input"] = original_cast_root_input

        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        # Adjust LayerNorm attributes for extra LayerNorm to insert
        # 1. Temporarily set root_input for LayerNorm to skip_input for post_feedforward_layernorm
        # 2. Set skip_input to output of post_feedforward_layernorm
        # 3. Do not cast outputs from post_feedforward_layernorm
        original_root_input = self.layernorm_attrs["root_input"]
        original_cast_output_0 = self.layernorm_attrs["cast"]["output_0"]
        self.layernorm_attrs["root_input"] = self.layernorm_attrs["skip_input"]
        self.layernorm_attrs["cast"]["output_0"] = False
        self.make_layernorm(
            layer_id, layer.post_feedforward_layernorm, skip=False, simple=self.layernorm_attrs["simple"], location="post_feedforward"
        )
        self.layernorm_attrs["root_input"] = original_root_input
        self.layernorm_attrs["skip_input"] = self.layernorm_attrs["output_0"]
        self.layernorm_attrs["cast"]["output_0"] = original_cast_output_0

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        original_window_size = self.window_size
        self.window_size = original_window_size if self.is_local(layer_id) else -1  # default is -1 in GroupQueryAttention kernel
        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.window_size = original_window_size


class Gemma3Model(Gemma2Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Remember whether this was originally a conditional-generation config
        # (Gemma3ForConditionalGeneration) before builder.py overrides model_type
        # to "gemma3_text". load_weights needs to know which HF class to instantiate.
        self._original_architecture = config.architectures[0]

        if hasattr(config, "rope_local_base_freq"):
            # Older transformers: rope_local_base_freq and rope_theta are top-level fields
            self.rope_local_theta = config.rope_local_base_freq
        elif hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            # Newer transformers (v5+): rope info lives in a nested rope_parameters dict
            sliding_params = config.rope_parameters.get("sliding_attention", {})
            self.rope_local_theta = sliding_params.get("rope_theta", 10000.0)
            # Update the global theta (full-attention layers) which the base class
            # could not infer because config.rope_theta is absent in this format.
            full_params = config.rope_parameters.get("full_attention", {})
            self.rope_attrs["theta"] = full_params.get("rope_theta", self.rope_attrs["theta"])
        else:
            # Default local RoPE theta matching Gemma3's original rope_local_base_freq
            self.rope_local_theta = 10000.0
        self.make_rotary_embedding_multi_cache()

    def is_local(self, layer_id):
        return bool((layer_id + 1) % 6)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()

    def make_rotary_embedding_multi_cache(self):
        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        super().make_rotary_embedding_caches(cos_cache_name=self.cos_cache_global_name, sin_cache_name=self.sin_cache_global_name)

        # Create the new cos/sin caches for local attention layers with its own theta value
        self.rope_attrs["create_caches"] = True
        self.rope_attrs["theta"] = self.rope_local_theta

        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        super().make_rotary_embedding_caches(cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name)

    def load_weights(self, input_path):
        # Gemma3ForConditionalGeneration (VLM) does not accept the
        # ``num_hidden_layers`` keyword argument that the base class would
        # normally forward to ``AutoModelForCausalLM.from_pretrained``.
        # Load it directly here instead.
        if self._original_architecture == "Gemma3ForConditionalGeneration":
            if self.quant_type is not None or input_path.endswith(".gguf"):
                return super().load_weights(input_path)
            from transformers import Gemma3ForConditionalGeneration

            return Gemma3ForConditionalGeneration.from_pretrained(
                self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
            )
        return super().load_weights(input_path)

    def make_rotary_embedding_caches(self, **kwargs):
        cos_cache_name = kwargs.get("cos_cache_name", (self.cos_cache_global_name if self.window_size == -1 else self.cos_cache_local_name))
        sin_cache_name = kwargs.get("sin_cache_name", (self.sin_cache_global_name if self.window_size == -1 else self.sin_cache_local_name))
        return super().make_rotary_embedding_caches(cos_cache_name=cos_cache_name, sin_cache_name=sin_cache_name)


class Gemma4Model(Gemma3Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Store per-layer head-dim info BEFORE super().__init__ because Gemma3Model.__init__
        # calls make_rotary_embedding_multi_cache(), which Gemma4 overrides.
        self._layer_types = list(config.layer_types) if hasattr(config, "layer_types") and config.layer_types else None

        # Gemma4 full-attention layers may use a wider head_dim (global_head_dim).
        self._global_head_size = getattr(config, "global_head_dim", None) or config.head_dim
        self._global_num_kv_heads = getattr(config, "num_global_key_value_heads", None) or config.num_key_value_heads

        # Cache full-attention RoPE params so make_rotary_embedding_multi_cache can use them.
        self._full_rope_params = {}
        if hasattr(config, "rope_parameters") and isinstance(config.rope_parameters, dict):
            self._full_rope_params = config.rope_parameters.get("full_attention", {})

        # Gemma4 does not define query_pre_attn_scalar (it uses a fixed scale=1.0).
        # Set a placeholder so Gemma2Model.__init__ (which reads this field) doesn't fail.
        # The value 1.0 produces Gemma2Model's formula: 1.0**-0.5 == 1.0, which matches
        # Gemma4's intended scale. We explicitly re-set it to 1.0 after super().__init__
        # for clarity.
        if not hasattr(config, "query_pre_attn_scalar"):
            config.query_pre_attn_scalar = 1.0

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Override architecture tracking so load_weights picks the right HF class.
        self._original_architecture = config.architectures[0]

        # Gemma4 uses attention scale = 1.0 (no query_pre_attn_scalar).
        # The parent Gemma2Model set this via `config.query_pre_attn_scalar**-0.5`
        # using the placeholder value 1.0 added above, so this is a no-op for 1.0
        # but makes the intention explicit for any future config that sets
        # query_pre_attn_scalar to a non-1.0 value.
        self.attention_attrs["scale"] = 1.0

        # Gemma4RMSNorm stores the actual scale weight (initialised to ones) and applies it
        # as `normed_output * weight`, unlike Gemma3RMSNorm which stores a delta and applies
        # `normed_output * (1 + weight)`.  Clear the +1 offset set by GemmaModel.
        self.layernorm_attrs["add_offset"] = 0

        # Warn about features not yet supported in the ONNX export.
        if getattr(config, "hidden_size_per_layer_input", 0):
            print(
                "WARNING: Gemma4 Per-Layer Embeddings (PLE, hidden_size_per_layer_input="
                f"{config.hidden_size_per_layer_input}) are not exported to ONNX and will be zeroed out."
            )
        if self._global_head_size != self.head_size:
            print(
                f"WARNING: Gemma4 global_head_dim ({self._global_head_size}) differs from head_dim ({self.head_size}). "
                "Full-attention layers will use global_head_dim; genai_config.json head_size reflects the sliding-attention value."
            )

        # Build shared-KV metadata: the last `num_kv_shared_layers` layers reuse K/V
        # from the last non-shared layer of the same type (sliding or full attention).
        num_kv_shared = getattr(config, "num_kv_shared_layers", 0)
        self._first_kv_shared_layer_idx = config.num_hidden_layers - num_kv_shared
        self._shared_kv_donor_map: dict[int, int] = {}
        # Maps donor_id → (k_path_tensor_name, v_path_tensor_name); populated during
        # make_attention() as non-shared layers are built.
        self._donor_kv_paths: dict[int, tuple[str, str]] = {}
        if num_kv_shared > 0 and self._layer_types is not None:
            non_shared_types = self._layer_types[: self._first_kv_shared_layer_idx]
            for i in range(self._first_kv_shared_layer_idx, config.num_hidden_layers):
                layer_type = self._layer_types[i]
                # Find the last non-shared layer with the same type.
                for j in range(len(non_shared_types) - 1, -1, -1):
                    if non_shared_types[j] == layer_type:
                        self._shared_kv_donor_map[i] = j
                        break
                else:
                    raise ValueError(
                        f"Gemma4: shared-KV layer {i} (type '{layer_type}') has no donor: "
                        f"no non-shared layer of type '{layer_type}' found among "
                        f"layers 0..{self._first_kv_shared_layer_idx - 1}."
                    )

    def is_local(self, layer_id):
        if self._layer_types is not None and layer_id < len(self._layer_types):
            return self._layer_types[layer_id] == "sliding_attention"
        # Fallback: treat every layer as local (sliding)
        return True

    def is_shared_kv_layer(self, layer_id):
        """Return True if layer_id reuses K/V from a donor layer (num_kv_shared_layers > 0)."""
        return layer_id in self._shared_kv_donor_map

    def make_attention_init(self):
        super().make_attention_init()  # sets q_norm=True, k_norm=True via Gemma3Model
        self.attention_attrs["v_norm"] = True

    def make_rotary_embedding_multi_cache(self):
        import torch
        from onnx_ir.tensor_adapters import to_torch_dtype

        full_params = getattr(self, "_full_rope_params", {})
        global_partial_rotary_factor = full_params.get("partial_rotary_factor", 1.0)
        global_theta = full_params.get("rope_theta", self.rope_attrs["theta"])
        global_head_size = self._global_head_size

        # ----------------------------------------------------------------
        # Build the global (full-attention) cos/sin cache.
        #
        # Gemma4's "proportional" RoPE returns a FULL [max_len, head_dim//2]
        # cache.  Only the first `rope_angles` frequencies are non-zero; the
        # remaining `nope_angles` positions hold zeros, which evaluate to
        # cos=1, sin=0 → identity rotation for those dimensions.
        # ORT's RotaryEmbedding must be invoked with rotary_embedding_dim=0
        # (= full rotation) together with this full-sized cache, so that dims
        # are paired across the head-halves as expected by HF's rotate_half.
        # ----------------------------------------------------------------
        cache_length = self.rope_attrs["cache_length"]
        if global_partial_rotary_factor != 1.0:
            rope_angles = int(global_partial_rotary_factor * global_head_size // 2)
            nope_angles = global_head_size // 2 - rope_angles
            inv_freq_rot = 1.0 / (global_theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / global_head_size))
            inv_freq = torch.cat([inv_freq_rot, torch.zeros(nope_angles, dtype=torch.float32)])
        else:
            inv_freq = 1.0 / (global_theta ** (torch.arange(0, global_head_size, 2, dtype=torch.float32) / global_head_size))

        t = torch.arange(cache_length, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_global = emb.cos().to(to_torch_dtype(self.io_dtype))[:, : (global_head_size // 2)]
        sin_global = emb.sin().to(to_torch_dtype(self.io_dtype))[:, : (global_head_size // 2)]

        self.cos_cache_global_name, self.sin_cache_global_name = "cos_cache_global", "sin_cache_global"
        self.make_initializer(cos_global, self.cos_cache_global_name)
        self.make_initializer(sin_global, self.sin_cache_global_name)
        self.make_value(self.cos_cache_global_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])
        self.make_value(self.sin_cache_global_name, self.io_dtype, shape=["max_sequence_length", "head_dim / 2"])
        self.rope_attrs["create_caches"] = False

        # ----------------------------------------------------------------
        # Build the local (sliding-attention) cos/sin cache using the
        # standard base-class logic (theta = rope_local_theta, full rotation).
        # ----------------------------------------------------------------
        self.rope_attrs["create_caches"] = True
        self.rope_attrs["theta"] = self.rope_local_theta

        self.cos_cache_local_name, self.sin_cache_local_name = "cos_cache_local", "sin_cache_local"
        Gemma2Model.make_rotary_embedding_caches(self, cos_cache_name=self.cos_cache_local_name, sin_cache_name=self.sin_cache_local_name)

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # For full-attention (non-local) layers, temporarily use global head-dim and kv-heads.
        # Note: rotary_embedding_dim stays 0 (full rotation) because the global RoPE cache
        # already encodes the proportional structure via zero inv_freq padding.
        original_head_size = self.head_size
        original_num_kv_heads = self.num_kv_heads
        if not self.is_local(layer_id):
            self.head_size = self._global_head_size
            self.num_kv_heads = self._global_num_kv_heads

        if self.is_shared_kv_layer(layer_id):
            # Shared-KV layers have no k_proj/v_proj/k_norm/v_norm weights.
            # Set window_size for the GQA node, then delegate to the shared path.
            original_window_size = self.window_size
            self.window_size = original_window_size if self.is_local(layer_id) else -1
            self._make_shared_kv_attention(layer_id, attention, root_input, **kwargs)
            self.window_size = original_window_size
        else:
            # Gemma2Model.make_attention handles window_size toggle (local → full).
            Gemma2Model.make_attention(self, layer_id, attention, root_input, **kwargs)
            # Save the donor's pre-GQA K/V tensor names so shared layers can reuse them.
            # attention_attrs["k_path"] / ["v_path"] hold the post-norm, pre-GQA tensors.
            self._donor_kv_paths[layer_id] = (self.attention_attrs["k_path"], self.attention_attrs["v_path"])

        self.head_size = original_head_size
        self.num_kv_heads = original_num_kv_heads

    def _make_q_norm_only(self, layer_id, attention):
        """Apply Q RMSNorm (SimplifiedLayerNormalization) without touching K or V.

        Shared-KV layers have q_norm but no k_norm/v_norm weights: the donor's
        K/V are already normalized, so only Q needs normalization here.
        """
        import onnx_ir as ir

        layernorm_kwargs = {"epsilon": self.layernorm_attrs["epsilon"], "axis": -1, "stash_type": 1}
        old_io_dtype = self.io_dtype
        new_io_dtype = ir.DataType.FLOAT if self.layernorm_attrs["cast"]["use_fp32"] else self.io_dtype
        cast = old_io_dtype != new_io_dtype

        # Reshape Q from [B, S, D] to [B, S*N, H] before LayerNorm.
        q_reshape_1_name = f"/model/layers.{layer_id}/attn/q_norm/Reshape_1"
        q_reshape_1_inputs = [self.attention_attrs["q_path"], f"/model/constants/INT64/[0, -1, {self.head_size}]"]
        self.make_reshape(
            q_reshape_1_name,
            q_reshape_1_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length * num_attention_heads", self.head_size],
        )
        q_reshape_1_output = f"{q_reshape_1_name}/output_0"

        # Q LayerNorm.
        q_layernorm_name = f"/model/layers.{layer_id}/attn/q_norm/SimplifiedLayerNormalization"
        q_weight_name = f"model.layers.{layer_id}.attn.q_norm.layernorm.weight"
        q_layernorm_output = f"{q_layernorm_name}/output_0"
        self.make_initializer(attention.q_norm.weight + self.layernorm_attrs["add_offset"], q_weight_name, to=new_io_dtype)

        q_layernorm_inputs = [q_reshape_1_output, q_weight_name]
        q_layernorm_outputs = [q_layernorm_output]
        if cast:
            q_layernorm_inputs, q_layernorm_outputs = self.make_layernorm_casts(
                q_layernorm_name, q_layernorm_inputs, q_layernorm_outputs, old_io_dtype, new_io_dtype
            )

        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=q_layernorm_inputs,
            outputs=q_layernorm_outputs,
            name=q_layernorm_name,
            **layernorm_kwargs,
        )
        self.make_value(
            q_layernorm_outputs[0], dtype=new_io_dtype, shape=["batch_size", "sequence_length * num_attention_heads", self.head_size]
        )

        # Reshape Q back from [B, S*N, H] to [B, S, D].
        q_reshape_2_name = f"/model/layers.{layer_id}/attn/q_norm/Reshape_2"
        q_reshape_2_inputs = [q_layernorm_output, f"/model/constants/INT64/[0, -1, {self.num_attn_heads * self.head_size}]"]
        self.make_reshape(
            q_reshape_2_name,
            q_reshape_2_inputs,
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.num_attn_heads * self.head_size],
        )
        self.attention_attrs["q_path"] = f"{q_reshape_2_name}/output_0"

    def _make_shared_kv_attention(self, layer_id, attention, root_input, **kwargs):
        """Build the ONNX attention subgraph for a shared-KV layer.

        Shared-KV layers (``num_kv_shared_layers > 0``) have no k_proj/v_proj/
        k_norm/v_norm weights.  Instead they reuse the K/V tensors computed by
        the donor layer in the same forward pass.

        The donor's pre-GQA k/v paths (post-norm, same ``[B, S, kv_heads * H]``
        shape as Q) are wired as the current K/V inputs of this layer's GQA
        node.  The shared layer maintains its **own** KV-cache (past/present
        slots), which it updates at each step with the reused K/V — matching
        HF's ``shared_kv_states`` pattern.
        """
        donor_id = self._shared_kv_donor_map[layer_id]

        # --- Q projection ---
        q_matmul_basename = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        q_matmul_name = self.make_matmul(attention.q_proj, q_matmul_basename, root_input)
        self.attention_attrs["q_path"] = f"{q_matmul_name}/output_0"

        if attention.q_proj.bias is not None and attention.q_proj.bias.any():
            q_add_name = f"/model/layers.{layer_id}/attn/q_proj/Add"
            self.make_add_bias(attention.q_proj.bias, q_add_name, root_input=self.attention_attrs["q_path"])
            self.attention_attrs["q_path"] = f"{q_add_name}/output_0"

        # --- Q norm (shared layers always have q_norm, never k_norm/v_norm) ---
        self._make_q_norm_only(layer_id, attention)

        # --- Rotary embedding: passed as cos/sin caches to GQA (use_rope_in_attn=True) ---
        cos_cache_name, sin_cache_name = "", ""
        if self.attention_attrs["use_rope_in_attn"]:
            cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()
        else:
            # When rotary is a separate op (e.g. DML), apply it to Q only.
            q_rotary_name = f"/model/layers.{layer_id}/attn/q_rotary/RotaryEmbedding"
            self.make_rotary_embedding(
                q_rotary_name,
                root_input=self.attention_attrs["q_path"],
                position_ids=kwargs.get("position_ids", self.input_names["position_ids"]),
            )
            self.attention_attrs["q_path"] = f"{q_rotary_name}/output_0"

        # --- Donor's current K/V (post-norm, pre-GQA, shape [B, S, kv_heads * H]) ---
        donor_k_path, donor_v_path = self._donor_kv_paths[donor_id]

        # --- Shared layer's own KV-cache (past/present) ---
        past_k, past_v, present_k, present_v = self.make_key_value_cache_names(layer_id)

        # --- GroupQueryAttention node ---
        # k_path/v_path = donor's current K/V (same S as Q).
        # past_k/past_v = shared layer's own accumulated cache.
        # GQA concatenates past_k with donor's k → present_k for shared layer.
        # This mirrors HF's shared_kv_states semantics: both layers use identical
        # K/V (same donor weights + same donor hidden state), so the shared
        # layer's cache accumulates to the same content as the donor's cache.
        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_group_query_attention(
            attn_name,
            q_path=self.attention_attrs["q_path"],
            k_path=donor_k_path,
            v_path=donor_v_path,
            past_k=past_k,
            past_v=past_v,
            seqlens_k=f"{self.mask_attrs['seqlens_k']}/output_0",
            total_seq_len=f"{self.mask_attrs['total_seq_len']}/output_0",
            cos_cache=cos_cache_name,
            sin_cache=sin_cache_name,
            present_k=present_k,
            present_v=present_v,
        )

        # --- Output projection ---
        attn_output = f"{attn_name}/output_0"
        o_matmul_basename = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        o_matmul_name = self.make_matmul(attention.o_proj, o_matmul_basename, attn_output)

        o_bias_exists = attention.o_proj.bias is not None
        if o_bias_exists:
            o_add_name = f"/model/layers.{layer_id}/attn/o_proj/Add"
            self.make_add_bias(attention.o_proj.bias, o_add_name, root_input=f"{o_matmul_name}/output_0")

        self.layernorm_attrs["skip_input"] = f"{o_matmul_name if not o_bias_exists else o_add_name}/output_0"

    def make_key_value_cache_shape(self, layer_id, shape):
        """Return per-layer KV cache shape, honouring global_head_dim for full-attention layers."""
        if not self.is_local(layer_id) and self._global_head_size != self.head_size:
            base = super().make_key_value_cache_shape(layer_id, shape)
            return [base[0], self._global_num_kv_heads, base[2], self._global_head_size]
        return super().make_key_value_cache_shape(layer_id, shape)

    def load_weights(self, input_path):
        if self._original_architecture == "Gemma4ForConditionalGeneration":
            if self.quant_type is not None or input_path.endswith(".gguf"):
                return super().load_weights(input_path)
            from transformers import Gemma4ForConditionalGeneration

            return Gemma4ForConditionalGeneration.from_pretrained(
                self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
            )
        return super().load_weights(input_path)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """`gemma4` is not a recognised architecture in onnxruntime-genai; replace with `gemma2`."""
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["type"] = "gemma2"

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)
