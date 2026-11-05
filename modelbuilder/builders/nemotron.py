# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os

import torch
from onnx_light.onnx import TensorProto as ir

from .llama import LlamaModel


class NemotronHModel(LlamaModel):
    def load_weights(self, input_path):
        if input_path.endswith(".gguf") or "adapter_path" in self.extra_options:
            return super().load_weights(input_path)
        directory = input_path if os.path.isdir(input_path) else self.model_name_or_path
        if not os.path.isdir(directory):
            from huggingface_hub import constants, snapshot_download

            directory = snapshot_download(
                repo_id=directory,
                revision=getattr(self._checkpoint_config, "_commit_hash", None),
                cache_dir=self.cache_dir,
                token=self.hf_token,
                local_files_only=constants.HF_HUB_OFFLINE,
                allow_patterns=["*.json", "*.safetensors"],
            )
        if not any(name.endswith(".safetensors") for name in os.listdir(directory)):
            return super().load_weights(input_path)
        from ..helpers.nemotron import load_nemotron_checkpoint

        model = load_nemotron_checkpoint(directory, self._checkpoint_config, self.num_layers)
        self.quant_type = None
        return model

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # NemotronH uses `mlp_hidden_act` instead of `hidden_act`
        if not hasattr(config, "hidden_act"):
            config.hidden_act = getattr(config, "mlp_hidden_act", "relu2")
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        # NemotronH attention does not use rotary position embeddings (NoPE)
        self.attention_attrs["rope"] = False
        self.attention_attrs["use_rope_in_attn"] = False
        # NemotronH uses RMSNorm (simplified, no bias)
        self.layernorm_attrs["simple"] = True
        self.layernorm_attrs["epsilon"] = config.layer_norm_epsilon

        # Set up MoE attributes from NemotronH-specific config fields
        if hasattr(config, "n_routed_experts"):
            self.moe_attrs["num_experts"] = config.n_routed_experts
        if hasattr(config, "norm_topk_prob"):
            self.moe_attrs["normalize_routing_weights"] = 1 if config.norm_topk_prob else 0

        # Determine which layers have attention (and therefore need KV cache).
        # KV cache input/output names are re-indexed to only cover attention layers,
        # using the original layer index as the suffix so the names remain stable.
        layers_block_type = getattr(config, "layers_block_type", ["attention"] * config.num_hidden_layers)
        layers_block_type = [self._canonical_block_type(t) for t in layers_block_type]
        self._attn_layer_ids = [i for i, t in enumerate(layers_block_type) if t == "attention"]
        self._mamba_layer_ids = [i for i, t in enumerate(layers_block_type) if t == "mamba"]

        # Override KV cache inputs/outputs to only include attention layers.
        self.input_names["past_key_values.key"] = [f"past_key_values.{i}.key" for i in self._attn_layer_ids]
        self.input_names["past_key_values.value"] = [f"past_key_values.{i}.value" for i in self._attn_layer_ids]
        self.output_names["present.key"] = [f"present.{i}.key" for i in self._attn_layer_ids]
        self.output_names["present.value"] = [f"present.{i}.value" for i in self._attn_layer_ids]

        # Add conv_state and ssm_state for each mamba layer.
        if self._mamba_layer_ids:
            mamba_num_heads = getattr(config, "mamba_num_heads", 0)
            mamba_head_dim = getattr(config, "mamba_head_dim", 0)
            ssm_state_size = getattr(config, "ssm_state_size", 0)
            n_groups = getattr(config, "n_groups", 1)
            conv_kernel = getattr(config, "conv_kernel", 4)
            self.ssm_dtype = {"float32": ir.FLOAT, "float16": ir.FLOAT16, "bfloat16": ir.BFLOAT16}[
                getattr(config, "mamba_ssm_cache_dtype", "float32")
            ]
            mamba_intermediate = mamba_num_heads * mamba_head_dim
            mamba_conv_dim = mamba_intermediate + 2 * n_groups * ssm_state_size

            for layer_id in self._mamba_layer_ids:
                # Causal conv state: [batch_size, conv_dim, conv_kernel - 1]
                self.input_names[f"past_state.{layer_id}.conv"] = f"past_key_values.{layer_id}.conv_state"
                self.input_types[f"past_state.{layer_id}.conv"] = self.io_dtype
                self.input_shapes[f"past_state.{layer_id}.conv"] = ["batch_size", mamba_conv_dim, conv_kernel - 1]

                self.output_names[f"present_state.{layer_id}.conv"] = f"present.{layer_id}.conv_state"
                self.output_types[f"present_state.{layer_id}.conv"] = self.io_dtype
                self.output_shapes[f"present_state.{layer_id}.conv"] = ["batch_size", mamba_conv_dim, conv_kernel - 1]

                # SSM recurrent state: [batch_size, num_heads, head_dim, ssm_state_size]
                self.input_names[f"past_state.{layer_id}.ssm"] = f"past_key_values.{layer_id}.recurrent_state"
                self.input_types[f"past_state.{layer_id}.ssm"] = self.ssm_dtype
                self.input_shapes[f"past_state.{layer_id}.ssm"] = ["batch_size", mamba_num_heads, mamba_head_dim, ssm_state_size]

                self.output_names[f"present_state.{layer_id}.ssm"] = f"present.{layer_id}.recurrent_state"
                self.output_types[f"present_state.{layer_id}.ssm"] = self.ssm_dtype
                self.output_shapes[f"present_state.{layer_id}.ssm"] = ["batch_size", mamba_num_heads, mamba_head_dim, ssm_state_size]

    @staticmethod
    def _canonical_block_type(block_type):
        # transformers >= 5.14 renamed NemotronH block types: "attention" became
        # "full_attention" and "mamba" became "linear_attention". Normalize both
        # the old and new names to the canonical values used throughout this
        # builder so the same code supports every supported transformers version.
        return {"full_attention": "attention", "linear_attention": "mamba"}.get(block_type, block_type)

    def is_layer(self, module):
        return module.__class__.__name__ == "NemotronHBlock"

    def has_final_norm(self, module, orig_model):
        return hasattr(orig_model, "model") and hasattr(orig_model.model, "norm_f") and module == orig_model.model.norm_f

    def make_key_value_cache_names(self, layer_id):
        # Map the overall layer index to the KV-cache slot index (attention layers only).
        kv_idx = self._attn_layer_ids.index(layer_id)
        past_k = self.input_names["past_key_values.key"][kv_idx]
        past_v = self.input_names["past_key_values.value"][kv_idx]
        present_k = self.output_names["present.key"][kv_idx]
        present_v = self.output_names["present.value"][kv_idx]
        return past_k, past_v, present_k, present_v

    def make_layer(self, layer_id, layer):
        # Each NemotronH decoder block is defined as:
        # pre_norm --> mixer (attention / mamba / moe) --> residual add

        block_type = self._canonical_block_type(layer.block_type)
        if block_type == "attention":
            self.make_layernorm(layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input")
            self.make_attention(layer_id, layer.mixer, root_input=self.layernorm_attrs["output_0"])
        elif block_type == "moe":
            self.make_layernorm(layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input")
            self.make_nemotronh_moe(layer_id, layer.mixer, root_input=self.layernorm_attrs["output_0"])
        elif block_type == "mamba":
            self.make_layernorm(layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input")
            self.make_nemotronh_mamba(layer_id, layer.mixer, root_input=self.layernorm_attrs["output_0"])
        elif block_type == "mlp":
            self.make_layernorm(layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input")
            NemotronModel.make_mlp_proj(self, layer_id, layer.mixer, root_input=self.layernorm_attrs["output_0"])
        else:
            raise NotImplementedError(
                f"NemotronH block type '{layer.block_type}' is not supported for ONNX export. "
                "Only 'attention', 'mamba', 'mlp' and 'moe' layers are currently supported."
            )

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

    def make_nemotronh_mamba(self, layer_id, mamba, root_input):
        """Build ONNX nodes for the NemotronH Mamba2 block.

        Architecture (single-chunk SSD scan with recurrent state):
          hidden_states → in_proj → [gate, h_B_C, dt_raw]
          h_B_C + past_conv_state → CausalConvWithState → [conv_out, new_conv_state]
          conv_out → split → [hs, B_flat, C_flat]
          hs, B, C, dt, past_ssm_state → SSD scan → [y, new_ssm_state]
          y, gate → Zamba2RMSNormGated → scan_out
          scan_out → out_proj → output  (→ skip_input for next layer)
        """
        basename = f"/model/layers.{layer_id}/mamba"
        I = mamba.intermediate_size  # = num_heads * head_dim  # noqa: E741
        conv_dim = mamba.conv_dim  # = I + 2*G*N
        H = mamba.num_heads
        D = mamba.head_dim
        N = mamba.ssm_state_size
        G = mamba.n_groups
        K = mamba.conv_kernel_size
        GN = G * N  # n_groups * ssm_state_size
        rH = H // G  # repeat factor per group
        time_step_min = float(mamba.time_step_limit[0] if hasattr(mamba, "time_step_limit") else mamba.time_step_min)
        use_cast = self.io_dtype != ir.DataType.FLOAT
        ssm_dtype = ir.DataType.FLOAT  # SSM computations always in fp32

        past_conv = f"past_key_values.{layer_id}.conv_state"
        present_conv = f"present.{layer_id}.conv_state"
        past_ssm = f"past_key_values.{layer_id}.recurrent_state"
        present_ssm = f"present.{layer_id}.recurrent_state"

        # Helper: cast tensor to fp32 if io_dtype is not fp32.
        def cast_fp32(name, inp, shape):
            if use_cast:
                self.make_cast(name, inp, ssm_dtype, shape)
                return f"{name}/output_0"
            return inp

        # Helper: make a Exp node.
        def make_exp(name, inp, dtype, shape):
            self.make_node("Exp", inputs=[inp], outputs=[f"{name}/output_0"], name=name)
            self.make_value(f"{name}/output_0", dtype, shape=shape)
            return f"{name}/output_0"

        # Helper: make a MatMul node (not weight MatMul, just a plain op).
        def make_mm(name, a, b, dtype, shape):
            self.make_node("MatMul", inputs=[a, b], outputs=[f"{name}/output_0"], name=name)
            self.make_value(f"{name}/output_0", dtype, shape=shape)
            return f"{name}/output_0"

        # ================================================================
        # 1. Linear projection: in_proj  →  [gate, h_B_C, dt_raw]
        # ================================================================
        in_proj_name = self.make_matmul(mamba.in_proj, f"{basename}/in_proj/MatMul", root_input)
        # in_proj output: [B, S, I + conv_dim + H]
        gate_out = f"{basename}/in_proj/gate/output_0"
        h_B_C_out = f"{basename}/in_proj/h_B_C/output_0"
        dt_raw_out = f"{basename}/in_proj/dt_raw/output_0"
        self.make_node(
            "Split",
            inputs=[f"{in_proj_name}/output_0", f"/model/constants/INT64/[{I}, {conv_dim}, {H}]"],
            outputs=[gate_out, h_B_C_out, dt_raw_out],
            name=f"{basename}/in_proj/Split",
            axis=-1,
        )
        self.make_value(gate_out, self.io_dtype, ["batch_size", "sequence_length", I])
        self.make_value(h_B_C_out, self.io_dtype, ["batch_size", "sequence_length", conv_dim])
        self.make_value(dt_raw_out, self.io_dtype, ["batch_size", "sequence_length", H])

        # ================================================================
        # 2. Causal Conv1D with state  (channel-first: [B, conv_dim, S])
        # ================================================================
        h_B_C_T_name = f"{basename}/conv/Transpose_in"
        self.make_transpose(h_B_C_T_name, h_B_C_out, self.io_dtype, ["batch_size", conv_dim, "sequence_length"], [0, 2, 1])

        conv_w_name = f"model.layers.{layer_id}.mamba.conv1d.weight"
        self.make_initializer(mamba.conv1d.weight.detach(), conv_w_name, to=self.io_dtype)
        conv_b_name = f"model.layers.{layer_id}.mamba.conv1d.bias"
        conv_bias = mamba.conv1d.bias if mamba.conv1d.bias is not None else torch.zeros(conv_dim)
        self.make_initializer(conv_bias.detach(), conv_b_name, to=self.io_dtype)

        conv_op = f"{basename}/conv/CausalConvWithState"
        self.make_causal_conv_with_state(
            conv_op,
            root_input=f"{h_B_C_T_name}/output_0",
            weight=conv_w_name,
            bias=conv_b_name,
            past_conv_state=past_conv,
            present_conv_state=present_conv,
            output_shape=["batch_size", conv_dim, "sequence_length"],
            present_conv_shape=["batch_size", conv_dim, K - 1],
        )
        # Transpose back: [B, S, conv_dim]
        conv_out_T_name = f"{basename}/conv/Transpose_out"
        self.make_transpose(conv_out_T_name, f"{conv_op}/output_0", self.io_dtype, ["batch_size", "sequence_length", conv_dim], [0, 2, 1])
        conv_out = f"{conv_out_T_name}/output_0"

        # ================================================================
        # 3. Split conv output: [hs=I, B_flat=GN, C_flat=GN]
        # ================================================================
        hs_out = f"{basename}/split/hs/output_0"
        B_flat_out = f"{basename}/split/B_flat/output_0"
        C_flat_out = f"{basename}/split/C_flat/output_0"
        self.make_node(
            "Split",
            inputs=[conv_out, f"/model/constants/INT64/[{I}, {GN}, {GN}]"],
            outputs=[hs_out, B_flat_out, C_flat_out],
            name=f"{basename}/split/Split",
            axis=-1,
        )
        self.make_value(hs_out, self.io_dtype, ["batch_size", "sequence_length", I])
        self.make_value(B_flat_out, self.io_dtype, ["batch_size", "sequence_length", GN])
        self.make_value(C_flat_out, self.io_dtype, ["batch_size", "sequence_length", GN])

        # ================================================================
        # 4. Reshape B/C to [B, S, H, N] via repeat_interleave
        #    (each group is repeated rH = H//G times)
        # ================================================================
        def expand_groups(tag, flat_out):
            # [B, S, GN] → [B, S, G, N]
            rsh4 = f"{basename}/{tag}/reshape_GN"
            self.make_reshape(rsh4, [flat_out, [0, 0, G, N]], self.io_dtype, ["batch_size", "sequence_length", G, N])
            if rH == 1:
                rsh_out = f"{basename}/{tag}/reshape_HN"
                self.make_reshape(rsh_out, [f"{rsh4}/output_0", [0, 0, H, N]], self.io_dtype, ["batch_size", "sequence_length", H, N])
                return f"{rsh_out}/output_0"
            # [B, S, G, N] → [B, S, G, 1, N]
            unsq = f"{basename}/{tag}/unsqueeze"
            self.make_unsqueeze(
                unsq, [f"{rsh4}/output_0", "/model/constants/INT64/[3]"], self.io_dtype, ["batch_size", "sequence_length", G, 1, N]
            )
            # [B, S, G, 1, N] → [B, S, G, rH, N]  (broadcast Expand)
            exp5 = f"{basename}/{tag}/expand"
            self.make_expand(
                exp5,
                [f"{unsq}/output_0", f"/model/constants/INT64/[1, 1, {G}, {rH}, {N}]"],
                self.io_dtype,
                ["batch_size", "sequence_length", G, rH, N],
            )
            # [B, S, G, rH, N] → [B, S, H, N]
            rsh_out = f"{basename}/{tag}/reshape_HN"
            self.make_reshape(rsh_out, [f"{exp5}/output_0", [0, 0, H, N]], self.io_dtype, ["batch_size", "sequence_length", H, N])
            return f"{rsh_out}/output_0"

        B_rep = expand_groups("B", B_flat_out)  # [B, S, H, N]
        C_rep = expand_groups("C", C_flat_out)  # [B, S, H, N]

        # Reshape hs to [B, S, H, D]
        hs_4d_name = f"{basename}/hs_4d/Reshape"
        self.make_reshape(hs_4d_name, [hs_out, [0, 0, H, D]], self.io_dtype, ["batch_size", "sequence_length", H, D])
        hs_4d = f"{hs_4d_name}/output_0"

        # ================================================================
        # 5. Compute dt_clamped = clamp(softplus(dt_raw + dt_bias), min)
        # ================================================================
        dt_bias_name = f"model.layers.{layer_id}.mamba.dt_bias"
        self.make_initializer(mamba.dt_bias.detach(), dt_bias_name, to=ssm_dtype)

        dt_raw_fp32 = cast_fp32(f"{basename}/dt/Cast_in", dt_raw_out, ["batch_size", "sequence_length", H])
        dt_add_name = f"{basename}/dt/Add"
        self.make_add(dt_add_name, [dt_raw_fp32, dt_bias_name], ssm_dtype, ["batch_size", "sequence_length", H])
        dt_sp_name = f"{basename}/dt/Softplus"
        self.make_softplus(dt_sp_name, f"{dt_add_name}/output_0", ssm_dtype, ["batch_size", "sequence_length", H])
        dt_clip_name = f"{basename}/dt/Clip"
        self.make_clip(
            dt_clip_name,
            [f"{dt_sp_name}/output_0", f"/model/constants/FLOAT/{time_step_min}", ""],
            ssm_dtype,
            ["batch_size", "sequence_length", H],
        )
        dt = f"{dt_clip_name}/output_0"  # [B, S, H] fp32

        # ================================================================
        # 6. A_dt = (-exp(A_log)) * dt_clamped  →  [B, S, H]
        # ================================================================
        A_neg_name = f"model.layers.{layer_id}.mamba.A_neg"
        self.make_initializer((-torch.exp(mamba.A_log.float())).detach(), A_neg_name, to=ssm_dtype)

        A_dt_name = f"{basename}/A_dt/Mul"
        self.make_mul(A_dt_name, [dt, A_neg_name], ssm_dtype, ["batch_size", "sequence_length", H])
        A_dt_out = f"{A_dt_name}/output_0"  # [B, S, H]

        # ================================================================
        # 7. CumSum (inclusive and exclusive) of A_dt along sequence dim
        # ================================================================
        A_ci_name = f"{basename}/A_cumsum_incl/CumSum"
        self.make_node(
            "CumSum",
            inputs=[A_dt_out, "/model/constants/INT64/1"],
            outputs=[f"{A_ci_name}/output_0"],
            name=A_ci_name,
            exclusive=0,
            reverse=0,
        )
        self.make_value(f"{A_ci_name}/output_0", ssm_dtype, ["batch_size", "sequence_length", H])
        A_ci = f"{A_ci_name}/output_0"  # [B, S, H]  (inclusive)

        A_ce_name = f"{basename}/A_cumsum_excl/CumSum"
        self.make_node(
            "CumSum",
            inputs=[A_dt_out, "/model/constants/INT64/1"],
            outputs=[f"{A_ce_name}/output_0"],
            name=A_ce_name,
            exclusive=1,
            reverse=0,
        )
        self.make_value(f"{A_ce_name}/output_0", ssm_dtype, ["batch_size", "sequence_length", H])
        A_ce = f"{A_ce_name}/output_0"  # [B, S, H]  (exclusive)

        # ================================================================
        # 8. Prepare fp32 tensors and transpose to [B, H, S, *]
        # ================================================================
        B_fp32 = cast_fp32(f"{basename}/B_fp32/Cast", B_rep, ["batch_size", "sequence_length", H, N])
        C_fp32 = cast_fp32(f"{basename}/C_fp32/Cast", C_rep, ["batch_size", "sequence_length", H, N])
        hs_fp32 = cast_fp32(f"{basename}/hs_fp32/Cast", hs_4d, ["batch_size", "sequence_length", H, D])

        # x_bar = hs * dt[..., None]  →  [B, S, H, D]
        dt_unsq_name = f"{basename}/dt_unsq/Unsqueeze"
        self.make_unsqueeze(dt_unsq_name, [dt, "/model/constants/INT64/[-1]"], ssm_dtype, ["batch_size", "sequence_length", H, 1])
        x_bar_name = f"{basename}/x_bar/Mul"
        self.make_mul(x_bar_name, [hs_fp32, f"{dt_unsq_name}/output_0"], ssm_dtype, ["batch_size", "sequence_length", H, D])
        x_bar = f"{x_bar_name}/output_0"  # [B, S, H, D]

        def tp_BSHX(tag, inp, X, dtype):
            """Transpose [B,S,H,X] → [B,H,S,X]."""
            name = f"{basename}/{tag}/Transpose"
            self.make_transpose(name, inp, dtype, ["batch_size", H, "sequence_length", X], [0, 2, 1, 3])
            return f"{name}/output_0"

        def tp_BSH(tag, inp, dtype):
            """Transpose [B,S,H] → [B,H,S]."""
            name = f"{basename}/{tag}/Transpose"
            self.make_transpose(name, inp, dtype, ["batch_size", H, "sequence_length"], [0, 2, 1])
            return f"{name}/output_0"

        B_4d = tp_BSHX("B_T", B_fp32, N, ssm_dtype)  # [B, H, S, N]
        C_4d = tp_BSHX("C_T", C_fp32, N, ssm_dtype)  # [B, H, S, N]
        x_bar_4d = tp_BSHX("x_bar_T", x_bar, D, ssm_dtype)  # [B, H, S, D]
        hs_4d_T = tp_BSHX("hs_T", hs_fp32, D, ssm_dtype)  # [B, H, S, D]
        A_ci_T = tp_BSH("A_ci_T", A_ci, ssm_dtype)  # [B, H, S]
        A_ce_T = tp_BSH("A_ce_T", A_ce, ssm_dtype)  # [B, H, S]

        # ================================================================
        # 9. G = C @ B^T  →  [B, H, S, S]   (outer product over N)
        # ================================================================
        B_4d_Tn = f"{basename}/B_4d_Tn/Transpose"
        self.make_transpose(B_4d_Tn, B_4d, ssm_dtype, ["batch_size", H, N, "sequence_length"], [0, 1, 3, 2])
        G_out = make_mm(
            f"{basename}/G/MatMul", C_4d, f"{B_4d_Tn}/output_0", ssm_dtype, ["batch_size", H, "sequence_length", "sequence_length"]
        )

        # ================================================================
        # 10. L = causal_lower_tri(exp(A_ci_T[t] - A_ce_T[s]))
        # ================================================================
        # outer_sub[b,h,t,s] = A_ci_T[b,h,t] - A_ce_T[b,h,s]
        ci_unsq = f"{basename}/L/ci_unsq"
        self.make_unsqueeze(ci_unsq, [A_ci_T, "/model/constants/INT64/[-1]"], ssm_dtype, ["batch_size", H, "sequence_length", 1])
        ce_unsq = f"{basename}/L/ce_unsq"
        self.make_unsqueeze(ce_unsq, [A_ce_T, "/model/constants/INT64/[-2]"], ssm_dtype, ["batch_size", H, 1, "sequence_length"])
        outer_sub_name = f"{basename}/L/outer_sub"
        self.make_sub(
            outer_sub_name,
            [f"{ci_unsq}/output_0", f"{ce_unsq}/output_0"],
            ssm_dtype,
            ["batch_size", H, "sequence_length", "sequence_length"],
        )

        # Build causal mask [S, S] where mask[t,s] = -1e9 if t < s else 0
        S_shape_name = f"{basename}/causal/S_shape"
        self.make_shape(S_shape_name, A_dt_out, [3])
        S_scalar_name = f"{basename}/causal/S_scalar"
        self.make_gather(S_scalar_name, [f"{S_shape_name}/output_0", "/model/constants/INT64/1"], ir.DataType.INT64, [], axis=0)
        S_range_name = f"{basename}/causal/S_range"
        self.make_range(
            S_range_name,
            ["/model/constants/INT64/0", f"{S_scalar_name}/output_0", "/model/constants/INT64/1"],
            ir.DataType.INT64,
            ["sequence_length"],
        )
        row_name = f"{basename}/causal/row"
        self.make_unsqueeze(row_name, [f"{S_range_name}/output_0", "/model/constants/INT64/[1]"], ir.DataType.INT64, ["sequence_length", 1])
        col_name = f"{basename}/causal/col"
        self.make_unsqueeze(col_name, [f"{S_range_name}/output_0", "/model/constants/INT64/[0]"], ir.DataType.INT64, [1, "sequence_length"])
        lt_name = f"{basename}/causal/lt"
        self.make_less(lt_name, [f"{row_name}/output_0", f"{col_name}/output_0"])
        causal_mask_name = f"{basename}/causal/mask"
        self.make_where(
            causal_mask_name,
            [f"{lt_name}/output_0", "/model/constants/FLOAT/-1000000000.0", "/model/constants/FLOAT/0.0"],
            ssm_dtype,
            ["sequence_length", "sequence_length"],
        )

        L_pre_name = f"{basename}/L/add_mask"
        self.make_add(
            L_pre_name,
            [f"{outer_sub_name}/output_0", f"{causal_mask_name}/output_0"],
            ssm_dtype,
            ["batch_size", H, "sequence_length", "sequence_length"],
        )
        L_out = make_exp(f"{basename}/L/Exp", f"{L_pre_name}/output_0", ssm_dtype, ["batch_size", H, "sequence_length", "sequence_length"])

        # ================================================================
        # 11. M = G * L,   Y_diag = M @ x_bar_4d  →  [B, H, S, D]
        # ================================================================
        M_name = f"{basename}/M/Mul"
        self.make_mul(M_name, [G_out, L_out], ssm_dtype, ["batch_size", H, "sequence_length", "sequence_length"])
        Y_diag = make_mm(f"{basename}/Y_diag/MatMul", f"{M_name}/output_0", x_bar_4d, ssm_dtype, ["batch_size", H, "sequence_length", D])

        # ================================================================
        # 12. Y_init: initial SSM state contribution  →  [B, H, S, D]
        #     Y_init[b,h,s,d] = sum_n past_ssm[b,h,d,n] * C[b,h,s,n] * exp(A_ci_T[b,h,s])
        # ================================================================
        past_ssm_fp32 = past_ssm
        if self.ssm_dtype != ssm_dtype:
            self.make_cast(f"{basename}/Y_init/past_ssm_fp32", past_ssm, ssm_dtype, ["batch_size", H, D, N])
            past_ssm_fp32 = f"{basename}/Y_init/past_ssm_fp32/output_0"
        C_4d_Tn = f"{basename}/Y_init/C_T/Transpose"
        self.make_transpose(C_4d_Tn, C_4d, ssm_dtype, ["batch_size", H, N, "sequence_length"], [0, 1, 3, 2])
        # init_C = past_ssm @ C^T  →  [B, H, D, S]
        init_C = make_mm(
            f"{basename}/Y_init/MatMul", past_ssm_fp32, f"{C_4d_Tn}/output_0", ssm_dtype, ["batch_size", H, D, "sequence_length"]
        )
        exp_ci = make_exp(f"{basename}/Y_init/exp_ci", A_ci_T, ssm_dtype, ["batch_size", H, "sequence_length"])
        exp_ci_unsq = f"{basename}/Y_init/exp_ci_unsq"
        self.make_unsqueeze(exp_ci_unsq, [exp_ci, "/model/constants/INT64/[-2]"], ssm_dtype, ["batch_size", H, 1, "sequence_length"])
        Y_init_DS_name = f"{basename}/Y_init/Mul"
        self.make_mul(Y_init_DS_name, [init_C, f"{exp_ci_unsq}/output_0"], ssm_dtype, ["batch_size", H, D, "sequence_length"])
        Y_init_T_name = f"{basename}/Y_init/Transpose"
        self.make_transpose(Y_init_T_name, f"{Y_init_DS_name}/output_0", ssm_dtype, ["batch_size", H, "sequence_length", D], [0, 1, 3, 2])
        Y_init = f"{Y_init_T_name}/output_0"  # [B, H, S, D]

        # ================================================================
        # 13. D skip:  Y_D = D_param[h] * hs[b,h,s,d]  →  [B, H, S, D]
        # ================================================================
        D_param_name = f"model.layers.{layer_id}.mamba.D"
        self.make_initializer(mamba.D.detach(), D_param_name, to=ssm_dtype)
        D_4d_name = f"{basename}/D/unsqueeze"
        # [H] → [1, H, 1, 1]
        self.make_unsqueeze(D_4d_name, [D_param_name, "/model/constants/INT64/[0, 2, 3]"], ssm_dtype, [1, H, 1, 1])
        Y_D_name = f"{basename}/Y_D/Mul"
        self.make_mul(Y_D_name, [f"{D_4d_name}/output_0", hs_4d_T], ssm_dtype, ["batch_size", H, "sequence_length", D])
        Y_D = f"{Y_D_name}/output_0"  # [B, H, S, D]

        # ================================================================
        # 14. y = Y_diag + Y_init + Y_D  →  [B, S, I]
        # ================================================================
        y_sum1_name = f"{basename}/y_sum1/Add"
        self.make_add(y_sum1_name, [Y_diag, Y_init], ssm_dtype, ["batch_size", H, "sequence_length", D])
        y_sum2_name = f"{basename}/y_sum2/Add"
        self.make_add(y_sum2_name, [f"{y_sum1_name}/output_0", Y_D], ssm_dtype, ["batch_size", H, "sequence_length", D])
        # [B, H, S, D] → [B, S, H, D] → [B, S, I]
        y_BSHD_name = f"{basename}/y_BSHD/Transpose"
        self.make_transpose(y_BSHD_name, f"{y_sum2_name}/output_0", ssm_dtype, ["batch_size", "sequence_length", H, D], [0, 2, 1, 3])
        y_BSI_name = f"{basename}/y_BSI/Reshape"
        self.make_reshape(y_BSI_name, [f"{y_BSHD_name}/output_0", [0, 0, -1]], ssm_dtype, ["batch_size", "sequence_length", I])
        y_scan = f"{y_BSI_name}/output_0"  # [B, S, I]

        # ================================================================
        # 15. Zamba2RMSNormGated: scan_out = norm_w * group_rms_norm(silu(gate) * y)
        # ================================================================
        gate_fp32 = cast_fp32(f"{basename}/norm/gate_fp32", gate_out, ["batch_size", "sequence_length", I])
        # silu = x * sigmoid(x)
        sig_name = f"{basename}/norm/sigmoid"
        self.make_sigmoid(sig_name, gate_fp32, ssm_dtype, ["batch_size", "sequence_length", I])
        gate_silu_name = f"{basename}/norm/silu"
        self.make_mul(gate_silu_name, [gate_fp32, f"{sig_name}/output_0"], ssm_dtype, ["batch_size", "sequence_length", I])
        # y_gated = y * silu(gate)
        y_gated_name = f"{basename}/norm/y_gated"
        self.make_mul(y_gated_name, [y_scan, f"{gate_silu_name}/output_0"], ssm_dtype, ["batch_size", "sequence_length", I])

        # Group RMSNorm: group_size = I // G
        group_size = I // G
        rms_rsh_in_name = f"{basename}/norm/rms_reshape_in"
        self.make_reshape(
            rms_rsh_in_name,
            [f"{y_gated_name}/output_0", [0, 0, G, group_size]],
            ssm_dtype,
            ["batch_size", "sequence_length", G, group_size],
        )
        rms_pow_name = f"{basename}/norm/rms_pow"
        self.make_node(
            "Pow",
            inputs=[f"{rms_rsh_in_name}/output_0", "/model/constants/FLOAT/2"],
            outputs=[f"{rms_pow_name}/output_0"],
            name=rms_pow_name,
        )
        self.make_value(f"{rms_pow_name}/output_0", ssm_dtype, ["batch_size", "sequence_length", G, group_size])
        rms_var_name = f"{basename}/norm/rms_var"
        self.make_reduce_mean(
            rms_var_name,
            [f"{rms_pow_name}/output_0", "/model/constants/INT64/[-1]"],
            ssm_dtype,
            ["batch_size", "sequence_length", G, 1],
            keepdims=True,
        )
        eps = float(mamba.norm.variance_epsilon)
        rms_eps_name = f"{basename}/norm/rms_eps"
        self.make_add(
            rms_eps_name, [f"{rms_var_name}/output_0", f"/model/constants/FLOAT/{eps}"], ssm_dtype, ["batch_size", "sequence_length", G, 1]
        )
        rms_rsqrt_name = f"{basename}/norm/rms_rsqrt"
        self.make_rsqrt(rms_rsqrt_name, [f"{rms_eps_name}/output_0"], ssm_dtype, ["batch_size", "sequence_length", G, 1])
        rms_normed_name = f"{basename}/norm/rms_normed"
        self.make_mul(
            rms_normed_name,
            [f"{rms_rsh_in_name}/output_0", f"{rms_rsqrt_name}/output_0"],
            ssm_dtype,
            ["batch_size", "sequence_length", G, group_size],
        )
        rms_rsh_out_name = f"{basename}/norm/rms_reshape_out"
        self.make_reshape(rms_rsh_out_name, [f"{rms_normed_name}/output_0", [0, 0, -1]], ssm_dtype, ["batch_size", "sequence_length", I])

        norm_w_name = f"model.layers.{layer_id}.mamba.norm.weight"
        self.make_initializer(mamba.norm.weight.detach(), norm_w_name, to=ssm_dtype)
        scan_out_fp32_name = f"{basename}/norm/scan_out_fp32"
        self.make_mul(scan_out_fp32_name, [f"{rms_rsh_out_name}/output_0", norm_w_name], ssm_dtype, ["batch_size", "sequence_length", I])
        scan_out_fp32 = f"{scan_out_fp32_name}/output_0"

        # Cast back to io_dtype if needed
        if use_cast:
            scan_cast_name = f"{basename}/norm/scan_cast"
            self.make_cast(scan_cast_name, scan_out_fp32, self.io_dtype, ["batch_size", "sequence_length", I])
            scan_final = f"{scan_cast_name}/output_0"
        else:
            scan_final = scan_out_fp32

        # ================================================================
        # 16. Output projection
        # ================================================================
        out_proj_name = self.make_matmul(mamba.out_proj, f"{basename}/out_proj/MatMul", scan_final)
        mamba_output = f"{out_proj_name}/output_0"  # [B, S, hidden_size]

        # ================================================================
        # 17. Compute new SSM state  →  present_ssm [B, H, D, N]
        #
        # new_state = past_ssm * exp(A_cumsum_last)
        #           + sum_s [ exp(A_cumsum_last - A_ci_T[s]) * dt[s] * B[s] * hs[s] ]
        # ================================================================
        # A_cumsum_last[b,h]: last position of inclusive cumsum  →  [B, H, 1]
        A_last = self.make_slice(
            f"{basename}/ssm_state/A_last", A_ci_T, ssm_dtype, ["batch_size", H, 1], starts=[-1], ends=[2**62], axes=[-1]
        )

        # decay_s = exp(A_last - A_ci_T)  →  [B, H, S]
        decay_sub_name = f"{basename}/ssm_state/decay_sub"
        self.make_sub(decay_sub_name, [A_last, A_ci_T], ssm_dtype, ["batch_size", H, "sequence_length"])
        decay_s = make_exp(f"{basename}/ssm_state/decay_s", f"{decay_sub_name}/output_0", ssm_dtype, ["batch_size", H, "sequence_length"])

        # dt_T [B, H, S] = transpose(dt [B, S, H])
        dt_T = tp_BSH("dt_T", dt, ssm_dtype)

        # B_weighted[b,h,s,n] = B_4d * (decay_s * dt_T)[..., None]  →  [B, H, S, N]
        dt_decay_name = f"{basename}/ssm_state/dt_decay"
        self.make_mul(dt_decay_name, [dt_T, decay_s], ssm_dtype, ["batch_size", H, "sequence_length"])
        dt_decay_unsq_name = f"{basename}/ssm_state/dt_decay_unsq"
        self.make_unsqueeze(
            dt_decay_unsq_name,
            [f"{dt_decay_name}/output_0", "/model/constants/INT64/[-1]"],
            ssm_dtype,
            ["batch_size", H, "sequence_length", 1],
        )
        B_weighted_name = f"{basename}/ssm_state/B_weighted"
        self.make_mul(B_weighted_name, [B_4d, f"{dt_decay_unsq_name}/output_0"], ssm_dtype, ["batch_size", H, "sequence_length", N])

        # new_state_new = x_bar_4d^T @ B_weighted  →  [B, H, D, N]
        # x_bar_4d: [B,H,S,D] → [B,H,D,S]
        x_bar_T_name = f"{basename}/ssm_state/x_bar_T/Transpose"
        self.make_transpose(x_bar_T_name, x_bar_4d, ssm_dtype, ["batch_size", H, D, "sequence_length"], [0, 1, 3, 2])
        new_state_new = make_mm(
            f"{basename}/ssm_state/new_tokens/MatMul",
            f"{x_bar_T_name}/output_0",
            f"{B_weighted_name}/output_0",
            ssm_dtype,
            ["batch_size", H, D, N],
        )

        # new_state_init = past_ssm * exp(A_last)[..., None]  →  [B, H, D, N]
        exp_last = make_exp(f"{basename}/ssm_state/exp_last", A_last, ssm_dtype, ["batch_size", H, 1])
        exp_last_unsq_name = f"{basename}/ssm_state/exp_last_unsq"
        self.make_unsqueeze(exp_last_unsq_name, [exp_last, "/model/constants/INT64/[-1]"], ssm_dtype, ["batch_size", H, 1, 1])
        new_state_init_name = f"{basename}/ssm_state/init_contrib"
        self.make_mul(new_state_init_name, [past_ssm_fp32, f"{exp_last_unsq_name}/output_0"], ssm_dtype, ["batch_size", H, D, N])

        # new_ssm = new_state_new + new_state_init  →  present_ssm [B, H, D, N]
        new_ssm_name = f"{basename}/ssm_state/new_ssm"
        if self.ssm_dtype != ssm_dtype:
            # Compute in fp32, then cast to io_dtype for the output
            new_ssm_fp32_name = f"{basename}/ssm_state/new_ssm_fp32"
            self.make_node(
                "Add",
                inputs=[new_state_new, f"{new_state_init_name}/output_0"],
                outputs=[f"{new_ssm_fp32_name}/output_0"],
                name=new_ssm_fp32_name,
            )
            self.make_value(f"{new_ssm_fp32_name}/output_0", ssm_dtype, ["batch_size", H, D, N])
            self.make_cast(new_ssm_name, f"{new_ssm_fp32_name}/output_0", self.ssm_dtype, ["batch_size", H, D, N])
            self.make_node("Identity", inputs=[f"{new_ssm_name}/output_0"], outputs=[present_ssm], name=f"{basename}/ssm_state/identity")
            self.make_value(present_ssm, self.ssm_dtype, ["batch_size", H, D, N])
        else:
            self.make_node("Add", inputs=[new_state_new, f"{new_state_init_name}/output_0"], outputs=[present_ssm], name=new_ssm_name)
            self.make_value(present_ssm, ssm_dtype, ["batch_size", H, D, N])

        # ================================================================
        # Set skip_input for next SkipLayerNorm (residual add)
        # ================================================================
        self.layernorm_attrs["skip_input"] = mamba_output

    def make_nemotronh_moe(self, layer_id, moe, root_input):
        # Make nodes for the NemotronH MoE subgraph.
        #
        # Each NemotronH MoE layer consists of:
        #   - A top-k router with sigmoid activation (NemotronHTopkRouter)
        #   - Routed experts (non-gated: up_proj → relu2 → down_proj)
        #   - A shared expert (always active: up_proj → relu2 → down_proj)
        #
        #                      root_input
        #                     /           \
        #              Router path       Shared expert path
        #          gate MatMul              up_proj MatMul
        #              |                       |
        #           Sigmoid                  relu2
        #              |                       |
        #           TopK                   down_proj MatMul
        #              |                       |
        #              +----> experts <----+   |
        #                       |              |
        #                       +-----> Add <--+
        #                               |
        #                          moe_output  (→ skip_input for next layer)

        basename = f"/model/layers.{layer_id}/moe"

        # com.microsoft.MoE does not support relu2 activation on CPU, so fall back to a
        # decomposed implementation using standard ONNX ops when targeting CPU.
        if self.ep == "cpu" or hasattr(moe.gate, "e_score_correction_bias") or hasattr(moe.fc1_latent_proj, "weight"):
            moe_expert_output = self._make_nemotronh_moe_routed_decomposed(layer_id, moe, root_input)
        else:
            moe_expert_output = self._make_nemotronh_moe_routed_fused(layer_id, moe, root_input)

        # Shared expert: up_proj → relu2 (Relu + Pow(2)) → down_proj
        shared_basename = f"{basename}/shared_experts"
        shared_moe_interm_size = moe.shared_experts.up_proj.weight.shape[0]

        shared_up_name = self.make_matmul(moe.shared_experts.up_proj, f"{shared_basename}/up_proj/MatMul", root_input)

        # relu2(x) = relu(x)^2
        relu_name = f"{shared_basename}/act/Relu"
        self.make_node("Relu", inputs=[f"{shared_up_name}/output_0"], outputs=[f"{relu_name}/output_0"], name=relu_name)
        self.make_value(f"{relu_name}/output_0", self.io_dtype, shape=["batch_size", "sequence_length", shared_moe_interm_size])

        pow_name = f"{shared_basename}/act/Pow"
        self.make_node(
            "Pow", inputs=[f"{relu_name}/output_0", "/model/constants/INT32/[2]"], outputs=[f"{pow_name}/output_0"], name=pow_name
        )
        self.make_value(f"{pow_name}/output_0", self.io_dtype, shape=["batch_size", "sequence_length", shared_moe_interm_size])

        shared_down_name = self.make_matmul(moe.shared_experts.down_proj, f"{shared_basename}/down_proj/MatMul", f"{pow_name}/output_0")

        # Add routed-expert output and shared-expert output
        add_name = f"{basename}/Add"
        self.make_add(
            add_name,
            [moe_expert_output, f"{shared_down_name}/output_0"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )

        # Assign MoE output as skip input for the next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{add_name}/output_0"

    def _make_nemotronh_moe_routed_fused(self, layer_id, moe, root_input):
        """Routed experts via com.microsoft.MoE fused op (CUDA and other non-CPU EPs)."""
        basename = f"/model/layers.{layer_id}/moe"
        num_experts = self.moe_attrs["num_experts"]
        op_type = self.moe_attrs["op_type"]

        # Router: gate.weight MatMul → Sigmoid → Reshape(-1, n_experts)
        # Exclude the router from int4 quantization by default: quantising
        # the small routing matrix corrupts top-k expert selection.
        gate_matmul_name = self.make_matmul(moe.gate, f"{basename}/gate/MatMul", root_input)
        if not self.quant_attrs["int4"]["quantize_moe_router"]:
            if gate_matmul_name not in self.quant_attrs["int4"]["nodes_to_exclude"]:
                self.quant_attrs["int4"]["nodes_to_exclude"].append(gate_matmul_name)

        gate_sigmoid_name = f"{basename}/gate/Sigmoid"
        self.make_node(
            "Sigmoid", inputs=[f"{gate_matmul_name}/output_0"], outputs=[f"{gate_sigmoid_name}/output_0"], name=gate_sigmoid_name
        )
        self.make_value(f"{gate_sigmoid_name}/output_0", self.io_dtype, shape=["batch_size", "sequence_length", num_experts])

        gate_reshape_name = f"{basename}/gate/Reshape"
        self.make_reshape(
            gate_reshape_name,
            [f"{gate_sigmoid_name}/output_0", f"/model/constants/INT64/{[-1, num_experts]}"],
            dtype=self.io_dtype,
            shape=["batch_size * sequence_length", num_experts],
        )

        # Expert weights: store as initializers.
        #   experts.up_proj shape: (n_experts, moe_intermediate_size, hidden_size)
        #   experts.down_proj shape: (n_experts, hidden_size, moe_intermediate_size)
        #   Both are already in (out_features, in_features) format per expert.
        if op_type == "MoE":
            up_proj_weight = f"model.layers.{layer_id}.moe.experts.up_proj.weight"
            down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.weight"
            self.make_initializer(moe.experts.up_proj.detach(), up_proj_weight, to=self.io_dtype)
            self.make_initializer(moe.experts.down_proj.detach(), down_proj_weight, to=self.io_dtype)

            moe_op_name = f"{basename}/{op_type}"
            self.make_moe_op(
                moe_op_name,
                root_input=root_input,
                router_probs=f"{gate_reshape_name}/output_0",
                weight1=up_proj_weight,
                weight2=down_proj_weight,
            )
        else:
            # QMoE: quantize the expert weights per expert
            up_proj_qweight = f"model.layers.{layer_id}.moe.experts.up_proj.qweight"
            up_proj_scales = f"model.layers.{layer_id}.moe.experts.up_proj.scales"
            down_proj_qweight = f"model.layers.{layer_id}.moe.experts.down_proj.qweight"
            down_proj_scales = f"model.layers.{layer_id}.moe.experts.down_proj.scales"

            up_qweight_list, up_scales_list = [], []
            down_qweight_list, down_scales_list = [], []
            for i in range(num_experts):
                # experts.up_proj[i]: (moe_intermediate_size, hidden_size) already (out, in)
                qw1, s1 = self.make_qmoe_weights(moe.experts.up_proj[i])
                up_qweight_list.append(qw1)
                up_scales_list.append(s1)
                # experts.down_proj[i]: (hidden_size, moe_intermediate_size) already (out, in)
                qw2, s2 = self.make_qmoe_weights(moe.experts.down_proj[i])
                down_qweight_list.append(qw2)
                down_scales_list.append(s2)

            self.make_initializer(torch.stack(up_qweight_list, dim=0).to(torch.uint8), up_proj_qweight)
            self.make_initializer(torch.stack(up_scales_list, dim=0), up_proj_scales, to=self.io_dtype)
            self.make_initializer(torch.stack(down_qweight_list, dim=0).to(torch.uint8), down_proj_qweight)
            self.make_initializer(torch.stack(down_scales_list, dim=0), down_proj_scales, to=self.io_dtype)

            moe_op_name = f"{basename}/{op_type}"
            self.make_moe_op(
                moe_op_name,
                root_input=root_input,
                router_probs=f"{gate_reshape_name}/output_0",
                weight1=up_proj_qweight,
                scales1=up_proj_scales,
                weight2=down_proj_qweight,
                scales2=down_proj_scales,
            )

        return f"{moe_op_name}/output_0"

    def _make_nemotronh_moe_routed_decomposed(self, layer_id, moe, root_input):
        """Routed experts via standard ONNX ops (CPU-compatible; no com.microsoft.MoE).

        ORT's CPU com.microsoft.MoE kernel does not support the relu2 activation type
        used by NemotronH. Gather only the tokens assigned to each expert, then
        scatter their weighted outputs back. Never gather full expert matrices
        per token: that would exhaust memory on the 30B checkpoints.
        """
        basename = f"/model/layers.{layer_id}/moe"
        num_experts = self.moe_attrs["num_experts"]
        top_k = self.moe_attrs["top_k"]
        normalize = self.moe_attrs["normalize_routing_weights"]
        expert_size = moe.experts.up_proj.shape[-1]
        expert_input = root_input
        if hasattr(moe.fc1_latent_proj, "weight"):
            latent_name = self.make_matmul(moe.fc1_latent_proj, f"{basename}/fc1_latent_proj/MatMul", root_input)
            expert_input = f"{latent_name}/output_0"
        # Use fp32 intermediate for TopK and weighted-sum when io_dtype is not fp32.
        use_cast = self.io_dtype != ir.DataType.FLOAT
        # 1. Router: gate MatMul → Sigmoid → TopK
        # Exclude the router from int4 quantization by default: quantising
        # the small routing matrix corrupts top-k expert selection.
        router_input = root_input
        if use_cast:
            router_input = f"{basename}/gate/input_Cast/output_0"
            self.make_cast(f"{basename}/gate/input_Cast", root_input, ir.FLOAT, ["batch_size", "sequence_length", self.hidden_size])
        gate_weight = f"model.layers.{layer_id}.moe.gate.weight"
        self.make_initializer(moe.gate.weight.detach().float().T, gate_weight)
        gate_matmul_name = f"{basename}/gate/MatMul"
        self.make_node("MatMul", inputs=[router_input, gate_weight], outputs=[f"{gate_matmul_name}/output_0"], name=gate_matmul_name)
        self.make_value(f"{gate_matmul_name}/output_0", ir.FLOAT, ["batch_size", "sequence_length", num_experts])
        if not self.quant_attrs["int4"]["quantize_moe_router"]:
            if gate_matmul_name not in self.quant_attrs["int4"]["nodes_to_exclude"]:
                self.quant_attrs["int4"]["nodes_to_exclude"].append(gate_matmul_name)

        gate_sigmoid_name = f"{basename}/gate/Sigmoid"
        self.make_node(
            "Sigmoid", inputs=[f"{gate_matmul_name}/output_0"], outputs=[f"{gate_sigmoid_name}/output_0"], name=gate_sigmoid_name
        )
        self.make_value(f"{gate_sigmoid_name}/output_0", ir.FLOAT, shape=["batch_size", "sequence_length", num_experts])
        topk_input = f"{gate_sigmoid_name}/output_0"
        correction_bias = getattr(moe.gate, "e_score_correction_bias", None)
        if correction_bias is not None:
            bias_name = f"model.layers.{layer_id}.moe.gate.e_score_correction_bias"
            self.make_initializer(correction_bias.detach().float(), bias_name)
            self.make_add(f"{basename}/gate/choice", [topk_input, bias_name], ir.FLOAT, ["batch_size", "sequence_length", num_experts])
            topk_input = f"{basename}/gate/choice/output_0"

        num_group = getattr(moe.gate, "num_group", 1)
        topk_group = getattr(moe.gate, "topk_group", num_group)
        if topk_group < num_group:
            grouped = f"{basename}/gate/groups"
            self.make_reshape(
                grouped,
                [topk_input, [0, 0, num_group, num_experts // num_group]],
                ir.FLOAT,
                ["batch_size", "sequence_length", num_group, num_experts // num_group],
            )
            self.make_node(
                "TopK",
                [f"{grouped}/output_0", "/model/constants/INT64/[2]"],
                [f"{grouped}/top2", f"{grouped}/top2_indices"],
                name=f"{grouped}/TopK",
                axis=-1,
            )
            self.make_reduce_sum(
                f"{grouped}/sum",
                [f"{grouped}/top2", "/model/constants/INT64/[-1]"],
                ir.FLOAT,
                ["batch_size", "sequence_length", num_group],
                keepdims=False,
            )
            self.make_node(
                "TopK",
                [f"{grouped}/sum/output_0", f"/model/constants/INT64/[{topk_group}]"],
                [f"{grouped}/selected_scores", f"{grouped}/selected"],
                name=f"{grouped}/Select",
                axis=-1,
            )
            self.make_initializer(torch.arange(num_experts) // (num_experts // num_group), f"{grouped}/expert_groups")
            self.make_unsqueeze(
                f"{grouped}/indices",
                [f"{grouped}/selected", "/model/constants/INT64/[-1]"],
                ir.INT64,
                ["batch_size", "sequence_length", topk_group, 1],
            )
            self.make_node(
                "Equal", [f"{grouped}/indices/output_0", f"{grouped}/expert_groups"], [f"{grouped}/matches"], name=f"{grouped}/Equal"
            )
            self.make_cast(
                f"{grouped}/matches_Cast", f"{grouped}/matches", ir.INT64, ["batch_size", "sequence_length", topk_group, num_experts]
            )
            self.make_reduce_sum(
                f"{grouped}/mask",
                [f"{grouped}/matches_Cast/output_0", "/model/constants/INT64/[-2]"],
                ir.INT64,
                ["batch_size", "sequence_length", num_experts],
                keepdims=False,
            )
            self.make_cast(f"{grouped}/mask_Cast", f"{grouped}/mask/output_0", ir.BOOL, ["batch_size", "sequence_length", num_experts])
            self.make_initializer(torch.tensor(float("-inf")), f"{grouped}/negative_inf")
            self.make_node(
                "Where",
                [f"{grouped}/mask_Cast/output_0", topk_input, f"{grouped}/negative_inf"],
                [f"{grouped}/masked"],
                name=f"{grouped}/Where",
            )
            topk_input = f"{grouped}/masked"

        topk_name = f"{basename}/gate/TopK"
        topk_val_out = f"{topk_name}/output_0"
        topk_idx_out = f"{topk_name}/output_1"
        self.make_node(
            "TopK",
            inputs=[topk_input, f"/model/constants/INT64/[{top_k}]"],
            outputs=[topk_val_out, topk_idx_out],
            name=topk_name,
            axis=-1,
            largest=True,
            sorted=True,
        )
        self.make_value(topk_val_out, ir.DataType.FLOAT, shape=["batch_size", "sequence_length", top_k])
        self.make_value(topk_idx_out, ir.DataType.INT64, shape=["batch_size", "sequence_length", top_k])
        self.make_node(
            "GatherElements",
            inputs=[f"{gate_sigmoid_name}/output_0", topk_idx_out],
            outputs=[f"{topk_name}/weights"],
            name=f"{topk_name}/GatherElements",
            axis=-1,
        )
        topk_val_out = f"{topk_name}/weights"

        # 2. Optional: normalise routing weights (norm_topk_prob).
        if normalize:
            topk_sum_name = f"{basename}/gate/TopK_sum"
            self.make_reduce_sum(
                topk_sum_name,
                [topk_val_out, "/model/constants/INT64/[-1]"],
                dtype=ir.DataType.FLOAT,
                shape=["batch_size", "sequence_length", 1],
                keepdims=True,
            )
            self.make_add(
                f"{topk_sum_name}/epsilon",
                [f"{topk_sum_name}/output_0", "/model/constants/FLOAT/1e-20"],
                ir.FLOAT,
                ["batch_size", "sequence_length", 1],
            )
            topk_div_name = f"{basename}/gate/TopK_div"
            self.make_div(
                topk_div_name,
                [topk_val_out, f"{topk_sum_name}/epsilon/output_0"],
                dtype=ir.DataType.FLOAT,
                shape=["batch_size", "sequence_length", top_k],
            )
            topk_weights = f"{topk_div_name}/output_0"
        else:
            topk_weights = topk_val_out
        scaling_factor = getattr(moe.gate, "routed_scaling_factor", 1.0)
        if scaling_factor != 1.0:
            scale_name = f"{basename}/gate/scaling_factor"
            self.make_initializer(torch.tensor(float(scaling_factor)), scale_name)
            self.make_mul(f"{basename}/gate/scale", [topk_weights, scale_name], ir.FLOAT, ["batch_size", "sequence_length", top_k])
            topk_weights = f"{basename}/gate/scale/output_0"

        self.make_reshape(f"{basename}/tokens", [expert_input, [-1, expert_size]], self.io_dtype, ["tokens", expert_size])
        self.make_reshape(f"{basename}/indices", [topk_idx_out, [-1, top_k]], ir.INT64, ["tokens", top_k])
        self.make_reshape(f"{basename}/weights", [topk_weights, [-1, top_k]], ir.FLOAT, ["tokens", top_k])
        tokens = f"{basename}/tokens/output_0"
        sum_input = tokens
        if use_cast:
            self.make_cast(f"{basename}/sum_input", tokens, ir.FLOAT, ["tokens", expert_size])
            sum_input = f"{basename}/sum_input/output_0"
        self.make_mul(f"{basename}/zeros", [sum_input, "/model/constants/FLOAT/0"], ir.FLOAT, ["tokens", expert_size])
        accumulated = f"{basename}/zeros/output_0"
        for expert in range(num_experts):
            prefix = f"{basename}/experts.{expert}"
            self.make_node(
                "Equal", [f"{basename}/indices/output_0", f"/model/constants/INT64/{expert}"], [f"{prefix}/mask"], name=f"{prefix}/Equal"
            )
            self.make_node("NonZero", [f"{prefix}/mask"], [f"{prefix}/positions"], name=f"{prefix}/NonZero")
            self.make_transpose(f"{prefix}/pairs", f"{prefix}/positions", ir.INT64, ["selected_tokens", 2], [1, 0])
            self.make_gather(
                f"{prefix}/rows", [f"{prefix}/pairs/output_0", "/model/constants/INT64/[0]"], ir.INT64, ["selected_tokens", 1], axis=1
            )
            self.make_node("GatherND", [tokens, f"{prefix}/rows/output_0"], [f"{prefix}/tokens"], name=f"{prefix}/GatherND")
            self.make_node(
                "GatherND",
                [f"{basename}/weights/output_0", f"{prefix}/pairs/output_0"],
                [f"{prefix}/weights"],
                name=f"{prefix}/WeightGatherND",
            )
            self.make_unsqueeze(
                f"{prefix}/weight_column", [f"{prefix}/weights", "/model/constants/INT64/[-1]"], ir.FLOAT, ["selected_tokens", 1]
            )
            up_weight = f"model.layers.{layer_id}.moe.experts.{expert}.up_proj.weight"
            down_weight = f"model.layers.{layer_id}.moe.experts.{expert}.down_proj.weight"
            self.make_initializer(moe.experts.up_proj[expert].detach().T, up_weight, to=self.io_dtype)
            self.make_initializer(moe.experts.down_proj[expert].detach().T, down_weight, to=self.io_dtype)
            self.make_node("MatMul", [f"{prefix}/tokens", up_weight], [f"{prefix}/up"], name=f"{prefix}/up_proj/MatMul")
            self.make_node("Relu", [f"{prefix}/up"], [f"{prefix}/relu"], name=f"{prefix}/Relu")
            self.make_node("Mul", [f"{prefix}/relu", f"{prefix}/relu"], [f"{prefix}/relu2"], name=f"{prefix}/Relu2")
            self.make_node("MatMul", [f"{prefix}/relu2", down_weight], [f"{prefix}/down"], name=f"{prefix}/down_proj/MatMul")
            down_output = f"{prefix}/down"
            if use_cast:
                self.make_cast(f"{prefix}/down_Cast", down_output, ir.FLOAT, ["selected_tokens", expert_size])
                down_output = f"{prefix}/down_Cast/output_0"
            self.make_mul(
                f"{prefix}/weighted", [down_output, f"{prefix}/weight_column/output_0"], ir.FLOAT, ["selected_tokens", expert_size]
            )
            self.make_node("GatherND", [accumulated, f"{prefix}/rows/output_0"], [f"{prefix}/previous"], name=f"{prefix}/PreviousGatherND")
            self.make_add(
                f"{prefix}/updated", [f"{prefix}/previous", f"{prefix}/weighted/output_0"], ir.FLOAT, ["selected_tokens", expert_size]
            )
            self.make_node(
                "ScatterND",
                [accumulated, f"{prefix}/rows/output_0", f"{prefix}/updated/output_0"],
                [f"{prefix}/accumulated"],
                name=f"{prefix}/ScatterND",
            )
            accumulated = f"{prefix}/accumulated"
        self.make_shape(f"{basename}/output_shape", expert_input, shape=[3])
        self.make_reshape(
            f"{basename}/output",
            [accumulated, f"{basename}/output_shape/output_0"],
            ir.FLOAT,
            ["batch_size", "sequence_length", expert_size],
        )
        expert_output = f"{basename}/output/output_0"
        if use_cast:
            self.make_cast(f"{basename}/output_Cast", expert_output, self.io_dtype, ["batch_size", "sequence_length", expert_size])
            expert_output = f"{basename}/output_Cast/output_0"
        if hasattr(moe.fc2_latent_proj, "weight"):
            latent_name = self.make_matmul(moe.fc2_latent_proj, f"{basename}/fc2_latent_proj/MatMul", expert_output)
            expert_output = f"{latent_name}/output_0"
        return expert_output

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """`nemotronh` is not supported as an architecture, let's replace with `llama`."""
        super().make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        genai_config["model"]["type"] = "llama"

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)


class NemotronModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.layernorm_attrs["add_offset"] = 1
        if hasattr(config, "norm_eps"):
            self.layernorm_attrs["epsilon"] = config.norm_eps

    def make_mlp_proj(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #          root_input
        #              |
        #         UpProjMatMul
        #              |
        #           ActFunc
        #              |
        #         DownProjMatMul

        up_basename = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        up_name = self.make_matmul(mlp.up_proj, up_basename, root_input)

        act_fn_name = self.make_activation(layer_id, root_input=f"{up_name}/output_0")

        # Make output MatMul node
        down_basename = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        down_name = self.make_matmul(mlp.down_proj, down_basename, f"{act_fn_name}/output_0")

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_name}/output_0"
