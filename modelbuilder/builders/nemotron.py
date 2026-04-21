# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os

import onnx_ir as ir

from .llama import LlamaModel


class NemotronHModel(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # NemotronH uses `mlp_hidden_act` instead of `hidden_act`
        if not hasattr(config, "hidden_act"):
            config.hidden_act = getattr(config, "mlp_hidden_act", "relu2")

        # Record per-layer block types before super().__init__ is called
        self._layers_block_type = list(getattr(config, "layers_block_type", ["attention"] * config.num_hidden_layers))

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # NemotronH attention does not use rotary position embeddings (NoPE)
        self.attention_attrs["rope"] = False
        self.attention_attrs["use_rope_in_attn"] = False
        # NemotronH uses RMSNorm (simplified, no bias)
        self.layernorm_attrs["simple"] = True
        self.layernorm_attrs["epsilon"] = config.layer_norm_epsilon

        # Mamba-specific dimensions (used for all mamba layers, which share the same config)
        mamba_intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        mamba_conv_dim = mamba_intermediate_size + 2 * config.n_groups * config.ssm_state_size
        self.mamba_attrs = {
            "intermediate_size": mamba_intermediate_size,
            "conv_dim": mamba_conv_dim,
            "num_heads": config.mamba_num_heads,
            "head_dim": config.mamba_head_dim,
            "ssm_state_size": config.ssm_state_size,
            "conv_kernel": config.conv_kernel,
            "n_groups": config.n_groups,
            "time_step_min": float(getattr(config, "time_step_min", 0.001)),
            "layer_norm_epsilon": float(config.layer_norm_epsilon),
        }

        # Per-layer type helpers
        self._attention_layers = {i for i, t in enumerate(self._layers_block_type) if t == "attention"}
        self._mamba_layers = [i for i, t in enumerate(self._layers_block_type) if t == "mamba"]
        # Lookup from layer_id → index within mamba state lists
        self._mamba_state_idx = {layer_id: idx for idx, layer_id in enumerate(self._mamba_layers)}

        # Restrict KV-cache names to attention layers only (set None for non-attention)
        self.input_names["past_key_values.key"] = [
            f"past_key_values.{i}.key" if i in self._attention_layers else None for i in range(self.num_layers)
        ]
        self.input_names["past_key_values.value"] = [
            f"past_key_values.{i}.value" if i in self._attention_layers else None for i in range(self.num_layers)
        ]
        self.output_names["present.key"] = [f"present.{i}.key" if i in self._attention_layers else None for i in range(self.num_layers)]
        self.output_names["present.value"] = [f"present.{i}.value" if i in self._attention_layers else None for i in range(self.num_layers)]

        # Add mamba-state inputs/outputs for mamba layers
        if self._mamba_layers:
            conv_dim = self.mamba_attrs["conv_dim"]
            num_heads = self.mamba_attrs["num_heads"]
            head_dim = self.mamba_attrs["head_dim"]
            ssm_state_size = self.mamba_attrs["ssm_state_size"]
            conv_kernel = self.mamba_attrs["conv_kernel"]

            self.input_names["past_mamba.conv_state"] = [f"past_mamba.{i}.conv_state" for i in self._mamba_layers]
            self.input_names["past_mamba.recurrent_state"] = [f"past_mamba.{i}.recurrent_state" for i in self._mamba_layers]
            self.input_types["past_mamba.conv_state"] = ir.DataType.FLOAT
            self.input_types["past_mamba.recurrent_state"] = ir.DataType.FLOAT
            self.input_shapes["past_mamba.conv_state"] = ["batch_size", conv_dim, conv_kernel]
            self.input_shapes["past_mamba.recurrent_state"] = ["batch_size", num_heads, head_dim, ssm_state_size]

            self.output_names["present_mamba.conv_state"] = [f"present_mamba.{i}.conv_state" for i in self._mamba_layers]
            self.output_names["present_mamba.recurrent_state"] = [f"present_mamba.{i}.recurrent_state" for i in self._mamba_layers]
            self.output_types["present_mamba.conv_state"] = ir.DataType.FLOAT
            self.output_types["present_mamba.recurrent_state"] = ir.DataType.FLOAT
            self.output_shapes["present_mamba.conv_state"] = ["batch_size", conv_dim, conv_kernel]
            self.output_shapes["present_mamba.recurrent_state"] = ["batch_size", num_heads, head_dim, ssm_state_size]

    # ------------------------------------------------------------------
    # Overridden make_inputs_and_outputs to handle mixed layer model I/O
    # ------------------------------------------------------------------

    def make_inputs_and_outputs(self):
        inputs = self.model.graph.inputs
        for key in self.input_names:
            name = self.input_names[key]
            dtype = self.input_types[key]
            shape = self.input_shapes[key]

            if isinstance(name, list):
                if key.startswith("past_mamba."):
                    # Mamba-state inputs: each element has the same flat shape
                    for mamba_name in name:
                        inputs.append(self.make_value(mamba_name, dtype=dtype, shape=shape))
                else:
                    # KV-cache inputs: may have None entries for non-attention layers
                    for i, kv_name in enumerate(name):
                        if kv_name is None:
                            continue
                        kv_shape = self.make_key_value_cache_shape(i, shape)
                        inputs.append(self.make_value(kv_name, dtype=dtype, shape=kv_shape))
            else:
                inputs.append(self.make_value(name, dtype=dtype, shape=shape))

        outputs = self.model.graph.outputs
        for key in self.output_names:
            name = self.output_names[key]
            dtype = self.output_types[key]
            shape = self.output_shapes[key]

            if isinstance(name, list):
                if key.startswith("present_mamba."):
                    # Mamba-state outputs
                    for mamba_name in name:
                        outputs.append(self.make_value(mamba_name, dtype=dtype, shape=shape))
                else:
                    # KV-cache outputs: may have None entries
                    for i, kv_name in enumerate(name):
                        if kv_name is None:
                            continue
                        kv_shape = self.make_key_value_cache_shape(i, shape)
                        outputs.append(self.make_value(kv_name, dtype=dtype, shape=kv_shape))
            else:
                outputs.append(self.make_value(name, dtype=dtype, shape=shape))

    # ------------------------------------------------------------------
    # Layer dispatch
    # ------------------------------------------------------------------

    def is_layer(self, module):
        return module.__class__.__name__ == "NemotronHBlock"

    def has_final_norm(self, module, orig_model):
        return hasattr(orig_model, "model") and hasattr(orig_model.model, "norm_f") and module == orig_model.model.norm_f

    def make_layer(self, layer_id, layer):
        # Each NemotronH decoder block is defined as:
        # pre_norm --> mixer (attention / mamba / moe) --> residual add (via SkipLayerNorm)
        block_type = layer.block_type

        self.make_layernorm(layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input")
        root_input = self.layernorm_attrs["output_0"]

        if block_type == "attention":
            self.make_attention(layer_id, layer.mixer, root_input=root_input)
        elif block_type == "mamba":
            self.make_mamba(layer_id, layer.mixer, root_input=root_input)
        elif block_type == "moe":
            self.make_moe_nemotron_h(layer_id, layer.mixer, root_input=root_input)
        else:
            raise NotImplementedError(f"NemotronH block type '{block_type}' is not supported for ONNX export.")

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    # ------------------------------------------------------------------
    # Mamba2 single-step decode block
    # ------------------------------------------------------------------

    def _make_silu(self, basename, root_input, dtype, shape):
        """Helper: x * sigmoid(x) (SiLU activation)."""
        sig_name = f"{basename}/Sigmoid"
        self.make_sigmoid(sig_name, root_input, dtype, shape)
        mul_name = f"{basename}/Mul"
        self.make_mul(mul_name, [root_input, f"{sig_name}/output_0"], dtype, shape)
        return f"{mul_name}/output_0"

    def make_mamba(self, layer_id, mamba, root_input):
        """
        Make ONNX nodes for a Mamba2 mixer block (single-token stateful decode path).

        The ONNX model processes exactly one token at a time. State (conv_state,
        recurrent_state) is passed as explicit model inputs and returned as outputs,
        enabling incremental decode.

        Inputs  (model-level):
          root_input:                     (batch, 1, hidden_size)
          past_mamba.{i}.conv_state:      (batch, conv_dim, conv_kernel)   [float32]
          past_mamba.{i}.recurrent_state: (batch, num_heads, head_dim, ssm_state_size) [float32]

        Outputs (model-level):
          present_mamba.{i}.conv_state
          present_mamba.{i}.recurrent_state
          layernorm_attrs["skip_input"] set to the mixer output for the residual add.
        """
        basename = f"/model/layers.{layer_id}/mamba"

        # Retrieve dimensions
        int_size = self.mamba_attrs["intermediate_size"]
        conv_dim = self.mamba_attrs["conv_dim"]
        num_heads = self.mamba_attrs["num_heads"]
        head_dim = self.mamba_attrs["head_dim"]
        ssm_state_size = self.mamba_attrs["ssm_state_size"]
        conv_kernel = self.mamba_attrs["conv_kernel"]
        n_groups = self.mamba_attrs["n_groups"]
        time_step_min = self.mamba_attrs["time_step_min"]
        eps = self.mamba_attrs["layer_norm_epsilon"]
        group_state_size = n_groups * ssm_state_size

        mamba_idx = self._mamba_state_idx[layer_id]
        past_conv = self.input_names["past_mamba.conv_state"][mamba_idx]
        past_rec = self.input_names["past_mamba.recurrent_state"][mamba_idx]
        present_conv = self.output_names["present_mamba.conv_state"][mamba_idx]
        present_rec = self.output_names["present_mamba.recurrent_state"][mamba_idx]

        # ----------------------------------------------------------------
        # 1. in_proj  (batch, 1, hidden) → (batch, 1, proj_size)
        # ----------------------------------------------------------------
        proj_size = int_size + conv_dim + num_heads
        in_proj_node = self.make_matmul(mamba.in_proj, f"{basename}/in_proj/MatMul", root_input)
        in_proj_out = f"{in_proj_node}/output_0"

        # Cast to float32 for precision (SSM arithmetic is fp32 in HF reference)
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast(f"{basename}/in_proj/CastF32", in_proj_out, ir.DataType.FLOAT, ["batch_size", 1, proj_size])
            in_proj_out = f"{basename}/in_proj/CastF32/output_0"

        # ----------------------------------------------------------------
        # 2. Split in_proj into gate / hidden_B_C / raw_dt
        # ----------------------------------------------------------------
        gate_out = self.make_slice(
            f"{basename}/gate/Slice", in_proj_out, ir.DataType.FLOAT, ["batch_size", 1, int_size], starts=[0], ends=[int_size], axes=[2]
        )
        hidden_B_C_out = self.make_slice(
            f"{basename}/hidden_B_C/Slice",
            in_proj_out,
            ir.DataType.FLOAT,
            ["batch_size", 1, conv_dim],
            starts=[int_size],
            ends=[int_size + conv_dim],
            axes=[2],
        )
        raw_dt_out = self.make_slice(
            f"{basename}/raw_dt/Slice",
            in_proj_out,
            ir.DataType.FLOAT,
            ["batch_size", 1, num_heads],
            starts=[int_size + conv_dim],
            ends=[proj_size],
            axes=[2],
        )

        # ----------------------------------------------------------------
        # 3. Conv1d decode update
        #    Shift conv_state left by 1 (discard oldest), append new token
        # ----------------------------------------------------------------
        # Transpose hidden_B_C: (batch, 1, conv_dim) → (batch, conv_dim, 1)
        self.make_transpose(
            f"{basename}/hidden_B_C/Transpose", hidden_B_C_out, ir.DataType.FLOAT, ["batch_size", conv_dim, 1], perm=[0, 2, 1]
        )
        hidden_B_C_t = f"{basename}/hidden_B_C/Transpose/output_0"

        # Slice past_conv_state: remove oldest element → (batch, conv_dim, conv_kernel-1)
        conv_sliced = self.make_slice(
            f"{basename}/conv_state/Slice",
            past_conv,
            ir.DataType.FLOAT,
            ["batch_size", conv_dim, conv_kernel - 1],
            starts=[1],
            ends=[conv_kernel],
            axes=[2],
        )

        # Concat → new conv_state: (batch, conv_dim, conv_kernel)
        self.make_concat(
            f"{basename}/new_conv_state/Concat",
            [conv_sliced, hidden_B_C_t],
            ir.DataType.FLOAT,
            ["batch_size", conv_dim, conv_kernel],
            axis=2,
        )
        new_conv_state_out = f"{basename}/new_conv_state/Concat/output_0"

        # Store present conv_state
        self.make_node("Identity", inputs=[new_conv_state_out], outputs=[present_conv], name=f"{basename}/present_conv_state/Identity")
        self.make_value(present_conv, ir.DataType.FLOAT, shape=["batch_size", conv_dim, conv_kernel])

        # Save conv1d weight (conv_dim, conv_kernel) as initializer
        conv_weight = mamba.conv1d.weight.squeeze(1)  # (conv_dim, conv_kernel)
        conv_w_name = f"model.layers.{layer_id}.mamba.conv1d.weight"
        self.make_initializer(conv_weight, conv_w_name, to=ir.DataType.FLOAT)

        # new_conv_state * conv_weight → (batch, conv_dim, conv_kernel) elementwise
        self.make_mul(
            f"{basename}/conv/WeightMul", [new_conv_state_out, conv_w_name], ir.DataType.FLOAT, ["batch_size", conv_dim, conv_kernel]
        )
        # Sum over kernel dim: (batch, conv_dim)
        self.make_reduce_sum(
            f"{basename}/conv/ReduceSum",
            [f"{basename}/conv/WeightMul/output_0", "/model/constants/INT64/[-1]"],
            ir.DataType.FLOAT,
            ["batch_size", conv_dim],
            keepdims=False,
        )
        conv_linear = f"{basename}/conv/ReduceSum/output_0"

        # Add conv bias
        if mamba.conv1d.bias is not None:
            conv_b_name = f"model.layers.{layer_id}.mamba.conv1d.bias"
            self.make_initializer(mamba.conv1d.bias, conv_b_name, to=ir.DataType.FLOAT)
            self.make_add(f"{basename}/conv/AddBias", [conv_linear, conv_b_name], ir.DataType.FLOAT, ["batch_size", conv_dim])
            conv_linear = f"{basename}/conv/AddBias/output_0"

        # SiLU activation
        conv_out = self._make_silu(f"{basename}/conv/silu", conv_linear, ir.DataType.FLOAT, ["batch_size", conv_dim])

        # ----------------------------------------------------------------
        # 4. Split conv output into hidden_h / B_raw / C_raw
        # ----------------------------------------------------------------
        hidden_h_out = self.make_slice(
            f"{basename}/hidden_h/Slice", conv_out, ir.DataType.FLOAT, ["batch_size", int_size], starts=[0], ends=[int_size], axes=[1]
        )
        B_raw_out = self.make_slice(
            f"{basename}/B_raw/Slice",
            conv_out,
            ir.DataType.FLOAT,
            ["batch_size", group_state_size],
            starts=[int_size],
            ends=[int_size + group_state_size],
            axes=[1],
        )
        C_raw_out = self.make_slice(
            f"{basename}/C_raw/Slice",
            conv_out,
            ir.DataType.FLOAT,
            ["batch_size", group_state_size],
            starts=[int_size + group_state_size],
            ends=[conv_dim],
            axes=[1],
        )

        # ----------------------------------------------------------------
        # 5. dt processing
        #    raw_dt: (batch, 1, num_heads) → squeeze → (batch, num_heads)
        #    expand to (batch, num_heads, head_dim), add dt_bias, softplus, clamp
        # ----------------------------------------------------------------
        # Squeeze the seq dimension (index 1)
        self.make_squeeze(
            f"{basename}/dt/Squeeze", [raw_dt_out, "/model/constants/INT64/[1]"], ir.DataType.FLOAT, ["batch_size", num_heads]
        )
        dt_sq = f"{basename}/dt/Squeeze/output_0"

        # Unsqueeze → (batch, num_heads, 1), expand → (batch, num_heads, head_dim)
        self.make_unsqueeze(
            f"{basename}/dt/Unsqueeze", [dt_sq, "/model/constants/INT64/[-1]"], ir.DataType.FLOAT, ["batch_size", num_heads, 1]
        )
        self.make_expand(
            f"{basename}/dt/Expand",
            [f"{basename}/dt/Unsqueeze/output_0", f"/model/constants/INT64/[1, {num_heads}, {head_dim}]"],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim],
        )
        dt_expanded = f"{basename}/dt/Expand/output_0"

        # dt_bias initializer: (num_heads,) → expand to (num_heads, head_dim)
        dt_bias_name = f"model.layers.{layer_id}.mamba.dt_bias"
        self.make_initializer(mamba.dt_bias, dt_bias_name, to=ir.DataType.FLOAT)
        # Unsqueeze dt_bias: (num_heads, 1)
        dt_bias_unsq = f"{basename}/dt_bias/Unsqueeze"
        self.make_node(
            "Unsqueeze", inputs=[dt_bias_name, "/model/constants/INT64/[-1]"], outputs=[f"{dt_bias_unsq}/output_0"], name=dt_bias_unsq
        )
        self.make_value(f"{dt_bias_unsq}/output_0", ir.DataType.FLOAT, shape=[num_heads, 1])
        # Expand to (num_heads, head_dim)
        self.make_expand(
            f"{basename}/dt_bias/Expand",
            [f"{dt_bias_unsq}/output_0", f"/model/constants/INT64/[{num_heads}, {head_dim}]"],
            ir.DataType.FLOAT,
            [num_heads, head_dim],
        )
        dt_bias_exp = f"{basename}/dt_bias/Expand/output_0"

        # Add dt + dt_bias
        self.make_add(f"{basename}/dt/AddBias", [dt_expanded, dt_bias_exp], ir.DataType.FLOAT, ["batch_size", num_heads, head_dim])
        # Softplus
        self.make_softplus(
            f"{basename}/dt/Softplus", f"{basename}/dt/AddBias/output_0", ir.DataType.FLOAT, ["batch_size", num_heads, head_dim]
        )
        # Clamp (min = time_step_min)
        time_step_min_name = f"/model/constants/FLOAT/{time_step_min}"
        self.make_clip(
            f"{basename}/dt/Clamp",
            [f"{basename}/dt/Softplus/output_0", time_step_min_name, ""],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim],
        )
        dt_out = f"{basename}/dt/Clamp/output_0"

        # ----------------------------------------------------------------
        # 6. A = -exp(A_log)  and expand to (num_heads, head_dim, ssm_state_size)
        # ----------------------------------------------------------------
        A_log_name = f"model.layers.{layer_id}.mamba.A_log"
        self.make_initializer(mamba.A_log, A_log_name, to=ir.DataType.FLOAT)
        self.make_node("Exp", inputs=[A_log_name], outputs=[f"{basename}/A/Exp/output_0"], name=f"{basename}/A/Exp")
        self.make_value(f"{basename}/A/Exp/output_0", ir.DataType.FLOAT, shape=[num_heads])
        self.make_neg(f"{basename}/A/Neg", f"{basename}/A/Exp/output_0", ir.DataType.FLOAT, [num_heads])
        A_out = f"{basename}/A/Neg/output_0"

        # Reshape A: (num_heads,) → (num_heads, 1, 1)
        self.make_reshape(f"{basename}/A/Reshape", [A_out, [num_heads, 1, 1]], ir.DataType.FLOAT, [num_heads, 1, 1])
        # Expand A: (num_heads, 1, 1) → (num_heads, head_dim, ssm_state_size)
        self.make_expand(
            f"{basename}/A/Expand",
            [f"{basename}/A/Reshape/output_0", f"/model/constants/INT64/[{num_heads}, {head_dim}, {ssm_state_size}]"],
            ir.DataType.FLOAT,
            [num_heads, head_dim, ssm_state_size],
        )
        A_expand = f"{basename}/A/Expand/output_0"

        # dA = exp(dt[..., None] * A_expand): (batch, num_heads, head_dim, ssm_state_size)
        self.make_unsqueeze(
            f"{basename}/dA/dt_unsq", [dt_out, "/model/constants/INT64/[-1]"], ir.DataType.FLOAT, ["batch_size", num_heads, head_dim, 1]
        )
        self.make_mul(
            f"{basename}/dA/DtA",
            [f"{basename}/dA/dt_unsq/output_0", A_expand],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim, ssm_state_size],
        )
        self.make_node("Exp", inputs=[f"{basename}/dA/DtA/output_0"], outputs=[f"{basename}/dA/Exp/output_0"], name=f"{basename}/dA/Exp")
        self.make_value(f"{basename}/dA/Exp/output_0", ir.DataType.FLOAT, shape=["batch_size", num_heads, head_dim, ssm_state_size])
        dA = f"{basename}/dA/Exp/output_0"

        # ----------------------------------------------------------------
        # 7. B processing: (batch, group_state_size) → (batch, num_heads, ssm_state_size)
        # ----------------------------------------------------------------
        B_out = self._expand_state_groups(basename, "B", B_raw_out, n_groups, num_heads, ssm_state_size)

        # dB = dt[..., None] * B[..., None, :]: (batch, num_heads, head_dim, ssm_state_size)
        self.make_unsqueeze(
            f"{basename}/dB/B_unsq", [B_out, "/model/constants/INT64/[-2]"], ir.DataType.FLOAT, ["batch_size", num_heads, 1, ssm_state_size]
        )
        self.make_mul(
            f"{basename}/dB/Mul",
            [
                f"{basename}/dA/dt_unsq/output_0",  # (batch, num_heads, head_dim, 1)
                f"{basename}/dB/B_unsq/output_0",
            ],  # (batch, num_heads, 1, ssm_state_size)
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim, ssm_state_size],
        )
        dB = f"{basename}/dB/Mul/output_0"

        # ----------------------------------------------------------------
        # 8. x = reshape(hidden_h): (batch, int_size) → (batch, num_heads, head_dim)
        #    dBx = dB * x[..., None]: (batch, num_heads, head_dim, ssm_state_size)
        # ----------------------------------------------------------------
        self.make_reshape(
            f"{basename}/x/Reshape",
            [hidden_h_out, ["batch_size", num_heads, head_dim]],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim],
        )
        x_out = f"{basename}/x/Reshape/output_0"

        self.make_unsqueeze(
            f"{basename}/dBx/x_unsq", [x_out, "/model/constants/INT64/[-1]"], ir.DataType.FLOAT, ["batch_size", num_heads, head_dim, 1]
        )
        self.make_mul(
            f"{basename}/dBx/Mul",
            [dB, f"{basename}/dBx/x_unsq/output_0"],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim, ssm_state_size],
        )
        dBx = f"{basename}/dBx/Mul/output_0"

        # ----------------------------------------------------------------
        # 9. Recurrent state update:
        #    new_state = dA * past_recurrent_state + dBx
        # ----------------------------------------------------------------
        self.make_mul(f"{basename}/state/dA_state", [dA, past_rec], ir.DataType.FLOAT, ["batch_size", num_heads, head_dim, ssm_state_size])
        self.make_add(
            f"{basename}/state/Update",
            [f"{basename}/state/dA_state/output_0", dBx],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim, ssm_state_size],
        )
        new_rec_out = f"{basename}/state/Update/output_0"

        # Store present recurrent_state
        self.make_node("Identity", inputs=[new_rec_out], outputs=[present_rec], name=f"{basename}/present_rec_state/Identity")
        self.make_value(present_rec, ir.DataType.FLOAT, shape=["batch_size", num_heads, head_dim, ssm_state_size])

        # ----------------------------------------------------------------
        # 10. C processing (same pattern as B)
        # ----------------------------------------------------------------
        C_out = self._expand_state_groups(basename, "C", C_raw_out, n_groups, num_heads, ssm_state_size)

        # ----------------------------------------------------------------
        # 11. Output: y = C @ new_state + D * x
        #     C: (batch, num_heads, ssm_state_size)
        #     new_state: (batch, num_heads, head_dim, ssm_state_size)
        #     → reshape to batched-matmul form
        # ----------------------------------------------------------------
        # new_state: (batch * num_heads, head_dim, ssm_state_size)
        self.make_reshape(
            f"{basename}/y/state_reshape", [new_rec_out, [-1, head_dim, ssm_state_size]], ir.DataType.FLOAT, [-1, head_dim, ssm_state_size]
        )
        # C: (batch * num_heads, ssm_state_size, 1)
        self.make_reshape(f"{basename}/y/C_reshape", [C_out, [-1, ssm_state_size, 1]], ir.DataType.FLOAT, [-1, ssm_state_size, 1])
        # Batched MatMul: (batch*num_heads, head_dim, ssm_state_size) @ (batch*num_heads, ssm_state_size, 1)
        #               = (batch*num_heads, head_dim, 1)
        self.make_node(
            "MatMul",
            inputs=[f"{basename}/y/state_reshape/output_0", f"{basename}/y/C_reshape/output_0"],
            outputs=[f"{basename}/y/MatMul/output_0"],
            name=f"{basename}/y/MatMul",
        )
        self.make_value(f"{basename}/y/MatMul/output_0", ir.DataType.FLOAT, shape=[-1, head_dim, 1])

        # Reshape result: (batch, num_heads, head_dim)
        self.make_reshape(
            f"{basename}/y/y_reshape",
            [f"{basename}/y/MatMul/output_0", ["batch_size", num_heads, head_dim]],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, head_dim],
        )
        y_no_D = f"{basename}/y/y_reshape/output_0"

        # D residual: D (num_heads,) → (num_heads, head_dim) → add D * x
        D_name = f"model.layers.{layer_id}.mamba.D"
        self.make_initializer(mamba.D, D_name, to=ir.DataType.FLOAT)
        self.make_node(
            "Unsqueeze", inputs=[D_name, "/model/constants/INT64/[-1]"], outputs=[f"{basename}/D/unsq/output_0"], name=f"{basename}/D/unsq"
        )
        self.make_value(f"{basename}/D/unsq/output_0", ir.DataType.FLOAT, shape=[num_heads, 1])
        self.make_expand(
            f"{basename}/D/Expand",
            [f"{basename}/D/unsq/output_0", f"/model/constants/INT64/[{num_heads}, {head_dim}]"],
            ir.DataType.FLOAT,
            [num_heads, head_dim],
        )
        D_exp = f"{basename}/D/Expand/output_0"

        self.make_mul(f"{basename}/D/DxMul", [D_exp, x_out], ir.DataType.FLOAT, ["batch_size", num_heads, head_dim])
        self.make_add(
            f"{basename}/y/PlusDx", [y_no_D, f"{basename}/D/DxMul/output_0"], ir.DataType.FLOAT, ["batch_size", num_heads, head_dim]
        )
        y_with_D = f"{basename}/y/PlusDx/output_0"

        # Reshape y: (batch, num_heads, head_dim) → (batch, 1, intermediate_size)
        self.make_reshape(
            f"{basename}/y/final_reshape", [y_with_D, ["batch_size", 1, int_size]], ir.DataType.FLOAT, ["batch_size", 1, int_size]
        )
        y_out = f"{basename}/y/final_reshape/output_0"

        # ----------------------------------------------------------------
        # 12. Gate normalization (Zamba2RMSNormGated)
        #     out = weight * RMSNorm_per_group( y * silu(gate) )
        # ----------------------------------------------------------------
        group_size = int_size // n_groups

        # gate → silu
        gate_silu = self._make_silu(f"{basename}/gate_norm/silu", gate_out, ir.DataType.FLOAT, ["batch_size", 1, int_size])
        # y * silu(gate)
        self.make_mul(f"{basename}/gate_norm/GatedMul", [y_out, gate_silu], ir.DataType.FLOAT, ["batch_size", 1, int_size])
        gated = f"{basename}/gate_norm/GatedMul/output_0"

        # Reshape for grouped RMS: (batch, 1, n_groups, group_size)
        self.make_reshape(
            f"{basename}/gate_norm/GroupReshape",
            [gated, ["batch_size", 1, n_groups, group_size]],
            ir.DataType.FLOAT,
            ["batch_size", 1, n_groups, group_size],
        )
        # Variance: mean(x^2, axis=-1, keepdim=True) → (batch, 1, n_groups, 1)
        self.make_node(
            "Pow",
            inputs=[f"{basename}/gate_norm/GroupReshape/output_0", "/model/constants/INT32/[2]"],
            outputs=[f"{basename}/gate_norm/Pow/output_0"],
            name=f"{basename}/gate_norm/Pow",
        )
        self.make_value(f"{basename}/gate_norm/Pow/output_0", ir.DataType.FLOAT, shape=["batch_size", 1, n_groups, group_size])
        self.make_reduce_mean(
            f"{basename}/gate_norm/Variance",
            [f"{basename}/gate_norm/Pow/output_0", "/model/constants/INT64/[-1]"],
            ir.DataType.FLOAT,
            ["batch_size", 1, n_groups, 1],
            keepdims=True,
        )
        # variance + eps
        eps_name = f"/model/constants/FLOAT/{eps}"
        self.make_add(
            f"{basename}/gate_norm/VarPlusEps",
            [f"{basename}/gate_norm/Variance/output_0", eps_name],
            ir.DataType.FLOAT,
            ["batch_size", 1, n_groups, 1],
        )
        # rsqrt(variance + eps)
        self.make_rsqrt(
            f"{basename}/gate_norm/Rsqrt", [f"{basename}/gate_norm/VarPlusEps/output_0"], ir.DataType.FLOAT, ["batch_size", 1, n_groups, 1]
        )
        rsqrt_out = f"{basename}/gate_norm/Rsqrt/output_0"

        # Normalize
        self.make_mul(
            f"{basename}/gate_norm/Normalize",
            [f"{basename}/gate_norm/GroupReshape/output_0", rsqrt_out],
            ir.DataType.FLOAT,
            ["batch_size", 1, n_groups, group_size],
        )
        # Reshape back: (batch, 1, intermediate_size)
        self.make_reshape(
            f"{basename}/gate_norm/UngroupReshape",
            [f"{basename}/gate_norm/Normalize/output_0", ["batch_size", 1, int_size]],
            ir.DataType.FLOAT,
            ["batch_size", 1, int_size],
        )

        # Multiply by learnable weight (norm.weight)
        norm_w_name = f"model.layers.{layer_id}.mamba.norm.weight"
        self.make_initializer(mamba.norm.weight, norm_w_name, to=ir.DataType.FLOAT)
        self.make_mul(
            f"{basename}/gate_norm/WeightMul",
            [f"{basename}/gate_norm/UngroupReshape/output_0", norm_w_name],
            ir.DataType.FLOAT,
            ["batch_size", 1, int_size],
        )
        norm_out = f"{basename}/gate_norm/WeightMul/output_0"

        # ----------------------------------------------------------------
        # 13. out_proj  (batch, 1, int_size) → (batch, 1, hidden_size)
        # ----------------------------------------------------------------
        # Cast back to io_dtype before out_proj if needed
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast(f"{basename}/out_proj/CastIO", norm_out, self.io_dtype, ["batch_size", 1, int_size])
            norm_out = f"{basename}/out_proj/CastIO/output_0"

        out_proj_node = self.make_matmul(mamba.out_proj, f"{basename}/out_proj/MatMul", norm_out)
        out = f"{out_proj_node}/output_0"

        self.layernorm_attrs["skip_input"] = out

    def _expand_state_groups(self, basename, label, raw_out, n_groups, num_heads, ssm_state_size):
        """
        Reshape (batch, n_groups * ssm_state_size) → (batch, num_heads, ssm_state_size)
        by repeating group vectors to match num_heads.
        """
        # Reshape: (batch, n_groups, ssm_state_size)
        self.make_reshape(
            f"{basename}/{label}/GroupReshape",
            [raw_out, ["batch_size", n_groups, ssm_state_size]],
            ir.DataType.FLOAT,
            ["batch_size", n_groups, ssm_state_size],
        )
        # Unsqueeze: (batch, n_groups, 1, ssm_state_size)
        self.make_unsqueeze(
            f"{basename}/{label}/Unsqueeze",
            [f"{basename}/{label}/GroupReshape/output_0", "/model/constants/INT64/[2]"],
            ir.DataType.FLOAT,
            ["batch_size", n_groups, 1, ssm_state_size],
        )
        reps = num_heads // n_groups
        # Expand: (batch, n_groups, num_heads//n_groups, ssm_state_size)
        self.make_expand(
            f"{basename}/{label}/Expand",
            [f"{basename}/{label}/Unsqueeze/output_0", f"/model/constants/INT64/[1, {n_groups}, {reps}, {ssm_state_size}]"],
            ir.DataType.FLOAT,
            ["batch_size", n_groups, reps, ssm_state_size],
        )
        # Reshape: (batch, num_heads, ssm_state_size)
        self.make_reshape(
            f"{basename}/{label}/FinalReshape",
            [f"{basename}/{label}/Expand/output_0", ["batch_size", num_heads, ssm_state_size]],
            ir.DataType.FLOAT,
            ["batch_size", num_heads, ssm_state_size],
        )
        return f"{basename}/{label}/FinalReshape/output_0"

    # ------------------------------------------------------------------
    # NemotronH MoE block
    # ------------------------------------------------------------------

    def make_moe_nemotron_h(self, layer_id, moe, root_input):
        """
        Make ONNX nodes for the NemotronH MoE mixer (no state).

        Implements:
          1. Shared expert: up_proj → relu² → down_proj
          2. Routed experts:
             a. Router: sigmoid(gate.weight @ x) + correction_bias → topk
             b. Expert weights gather + MatMul + relu² + MatMul
             c. Weight by routing scores, sum over top-k
          3. output = shared_out + routed_out
        """
        basename = f"/model/layers.{layer_id}/moe"

        n_experts = moe.n_routed_experts
        top_k = moe.top_k
        moe_int = moe.experts.intermediate_dim  # intermediate size per expert
        norm_topk = getattr(moe, "norm_topk_prob", True)
        scaling = float(getattr(moe, "routed_scaling_factor", 1.0))
        hidden = self.hidden_size

        # ----------------------------------------------------------------
        # Shared expert
        # ----------------------------------------------------------------
        shared_basename = f"{basename}/shared_expert"
        shared_up_node = self.make_matmul(moe.shared_experts.up_proj, f"{shared_basename}/up_proj/MatMul", root_input)
        shared_up = f"{shared_up_node}/output_0"

        # Activation: relu² = (relu(x))²  [NemotronH uses relu2]
        shared_relu_name = f"{shared_basename}/act_fn/Relu"
        self.make_node("Relu", inputs=[shared_up], outputs=[f"{shared_relu_name}/output_0"], name=shared_relu_name)
        self.make_value(
            f"{shared_relu_name}/output_0", self.io_dtype, shape=["batch_size", "sequence_length", moe.shared_experts.intermediate_size]
        )
        shared_sq_name = f"{shared_basename}/act_fn/Pow"
        self.make_node(
            "Pow",
            inputs=[f"{shared_relu_name}/output_0", "/model/constants/INT32/[2]"],
            outputs=[f"{shared_sq_name}/output_0"],
            name=shared_sq_name,
        )
        self.make_value(
            f"{shared_sq_name}/output_0", self.io_dtype, shape=["batch_size", "sequence_length", moe.shared_experts.intermediate_size]
        )

        shared_down_node = self.make_matmul(
            moe.shared_experts.down_proj, f"{shared_basename}/down_proj/MatMul", f"{shared_sq_name}/output_0"
        )
        shared_out = f"{shared_down_node}/output_0"

        # ----------------------------------------------------------------
        # Routing
        # ----------------------------------------------------------------
        # Flatten to (batch*seq, hidden) for routing
        self.make_reshape(f"{basename}/router/Flatten", [root_input, [-1, hidden]], self.io_dtype, [-1, hidden])
        flat_input = f"{basename}/router/Flatten/output_0"

        # Router linear: (batch*seq, hidden) @ gate.weight.T → (batch*seq, n_experts)
        router_w_name = f"model.layers.{layer_id}.moe.gate.weight"
        self.make_initializer(moe.gate.weight, router_w_name, to=self.io_dtype)
        self.make_transpose(f"{basename}/router/WeightTranspose", router_w_name, self.io_dtype, [hidden, n_experts], perm=[1, 0])
        self.make_node(
            "MatMul",
            inputs=[flat_input, f"{basename}/router/WeightTranspose/output_0"],
            outputs=[f"{basename}/router/MatMul/output_0"],
            name=f"{basename}/router/MatMul",
        )
        self.make_value(f"{basename}/router/MatMul/output_0", self.io_dtype, shape=[-1, n_experts])

        # Cast to float32 for routing arithmetic
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast(f"{basename}/router/CastF32", f"{basename}/router/MatMul/output_0", ir.DataType.FLOAT, [-1, n_experts])
            router_logits = f"{basename}/router/CastF32/output_0"
        else:
            router_logits = f"{basename}/router/MatMul/output_0"

        # Sigmoid
        self.make_sigmoid(f"{basename}/router/Sigmoid", router_logits, ir.DataType.FLOAT, [-1, n_experts])
        sigmoid_probs = f"{basename}/router/Sigmoid/output_0"

        # Add correction bias
        corr_name = f"model.layers.{layer_id}.moe.gate.e_score_correction_bias"
        self.make_initializer(moe.gate.e_score_correction_bias, corr_name, to=ir.DataType.FLOAT)
        self.make_add(f"{basename}/router/AddCorr", [sigmoid_probs, corr_name], ir.DataType.FLOAT, [-1, n_experts])
        corrected = f"{basename}/router/AddCorr/output_0"

        # TopK: select top_k experts based on corrected scores
        topk_node = f"{basename}/router/TopK"
        topk_vals_out = f"{topk_node}/output_0"
        topk_idx_out = f"{topk_node}/output_1"
        self.make_node(
            "TopK",
            inputs=[corrected, f"/model/constants/INT64/[{top_k}]"],
            outputs=[topk_vals_out, topk_idx_out],
            name=topk_node,
            axis=-1,
            largest=True,
            sorted=False,
        )
        self.make_value(topk_vals_out, ir.DataType.FLOAT, shape=[-1, top_k])
        self.make_value(topk_idx_out, ir.DataType.INT64, shape=[-1, top_k])

        # Gather routing weights from the UNCORRECTED sigmoid probs
        self.make_gather(f"{basename}/router/GatherWeights", [sigmoid_probs, topk_idx_out], ir.DataType.FLOAT, [-1, top_k], axis=1)
        topk_weights = f"{basename}/router/GatherWeights/output_0"

        # Optional normalization: divide by sum
        if norm_topk:
            self.make_reduce_sum(
                f"{basename}/router/WeightSum", [topk_weights, "/model/constants/INT64/[-1]"], ir.DataType.FLOAT, [-1, 1], keepdims=True
            )
            # Add small epsilon to avoid division by zero
            self.make_add(
                f"{basename}/router/WeightSumEps",
                [f"{basename}/router/WeightSum/output_0", "/model/constants/FLOAT/1e-20"],
                ir.DataType.FLOAT,
                [-1, 1],
            )
            self.make_div(
                f"{basename}/router/Normalize", [topk_weights, f"{basename}/router/WeightSumEps/output_0"], ir.DataType.FLOAT, [-1, top_k]
            )
            topk_weights = f"{basename}/router/Normalize/output_0"

        # Scale
        if scaling != 1.0:
            self.make_mul(f"{basename}/router/Scale", [topk_weights, f"/model/constants/FLOAT/{scaling}"], ir.DataType.FLOAT, [-1, top_k])
            topk_weights = f"{basename}/router/Scale/output_0"

        # ----------------------------------------------------------------
        # Expert dispatch (decomposed, per-expert gather + matmul)
        # ----------------------------------------------------------------
        # Save expert weights as initializers: (n_experts, moe_int, hidden), (n_experts, hidden, moe_int)
        up_w_name = f"model.layers.{layer_id}.moe.experts.up_proj"
        down_w_name = f"model.layers.{layer_id}.moe.experts.down_proj"
        self.make_initializer(moe.experts.up_proj, up_w_name, to=self.io_dtype)
        self.make_initializer(moe.experts.down_proj, down_w_name, to=self.io_dtype)

        # Gather: up_proj[topk_indices] → (batch*seq, top_k, moe_int, hidden)
        self.make_gather(f"{basename}/experts/GatherUp", [up_w_name, topk_idx_out], self.io_dtype, [-1, top_k, moe_int, hidden], axis=0)
        # Gather: down_proj[topk_indices] → (batch*seq, top_k, hidden, moe_int)
        self.make_gather(f"{basename}/experts/GatherDown", [down_w_name, topk_idx_out], self.io_dtype, [-1, top_k, hidden, moe_int], axis=0)

        # Input expansion: (batch*seq, hidden) → (batch*seq, 1, 1, hidden)
        self.make_unsqueeze(f"{basename}/experts/InputUnsq1", [flat_input, "/model/constants/INT64/[1]"], self.io_dtype, [-1, 1, hidden])
        self.make_unsqueeze(
            f"{basename}/experts/InputUnsq2",
            [f"{basename}/experts/InputUnsq1/output_0", "/model/constants/INT64/[-1]"],
            self.io_dtype,
            [-1, 1, hidden, 1],
        )
        x_for_experts = f"{basename}/experts/InputUnsq2/output_0"

        # MatMul up: (batch*seq, top_k, moe_int, hidden) @ (batch*seq, 1, hidden, 1)
        #           = (batch*seq, top_k, moe_int, 1)
        self.make_node(
            "MatMul",
            inputs=[f"{basename}/experts/GatherUp/output_0", x_for_experts],
            outputs=[f"{basename}/experts/MatMulUp/output_0"],
            name=f"{basename}/experts/MatMulUp",
        )
        self.make_value(f"{basename}/experts/MatMulUp/output_0", self.io_dtype, shape=[-1, top_k, moe_int, 1])

        # Squeeze last dim: (batch*seq, top_k, moe_int)
        self.make_squeeze(
            f"{basename}/experts/Squeeze",
            [f"{basename}/experts/MatMulUp/output_0", "/model/constants/INT64/[-1]"],
            self.io_dtype,
            [-1, top_k, moe_int],
        )

        # Activation: relu²
        relu_e_name = f"{basename}/experts/act_fn/Relu"
        self.make_node("Relu", inputs=[f"{basename}/experts/Squeeze/output_0"], outputs=[f"{relu_e_name}/output_0"], name=relu_e_name)
        self.make_value(f"{relu_e_name}/output_0", self.io_dtype, shape=[-1, top_k, moe_int])
        sq_e_name = f"{basename}/experts/act_fn/Pow"
        self.make_node(
            "Pow", inputs=[f"{relu_e_name}/output_0", "/model/constants/INT32/[2]"], outputs=[f"{sq_e_name}/output_0"], name=sq_e_name
        )
        self.make_value(f"{sq_e_name}/output_0", self.io_dtype, shape=[-1, top_k, moe_int])

        # Unsqueeze for batched down-proj matmul: (batch*seq, top_k, 1, moe_int)
        self.make_unsqueeze(
            f"{basename}/experts/ActUnsq", [f"{sq_e_name}/output_0", "/model/constants/INT64/[-2]"], self.io_dtype, [-1, top_k, 1, moe_int]
        )

        # MatMul down: (batch*seq, top_k, 1, moe_int) @ (batch*seq, top_k, moe_int, hidden)
        #            = (batch*seq, top_k, 1, hidden)
        self.make_node(
            "MatMul",
            inputs=[f"{basename}/experts/ActUnsq/output_0", f"{basename}/experts/GatherDown/output_0"],
            outputs=[f"{basename}/experts/MatMulDown/output_0"],
            name=f"{basename}/experts/MatMulDown",
        )
        self.make_value(f"{basename}/experts/MatMulDown/output_0", self.io_dtype, shape=[-1, top_k, 1, hidden])

        # Squeeze: (batch*seq, top_k, hidden)
        self.make_squeeze(
            f"{basename}/experts/SqzDown",
            [f"{basename}/experts/MatMulDown/output_0", "/model/constants/INT64/[-2]"],
            self.io_dtype,
            [-1, top_k, hidden],
        )
        expert_outputs = f"{basename}/experts/SqzDown/output_0"

        # Weight the expert outputs: topk_weights (batch*seq, top_k) → unsqueeze → (batch*seq, top_k, 1)
        if self.io_dtype != ir.DataType.FLOAT:
            # Cast weights back to io_dtype for multiplication
            self.make_cast(f"{basename}/router/WeightsCastIO", topk_weights, self.io_dtype, [-1, top_k])
            topk_weights = f"{basename}/router/WeightsCastIO/output_0"

        self.make_unsqueeze(f"{basename}/router/WeightsUnsq", [topk_weights, "/model/constants/INT64/[-1]"], self.io_dtype, [-1, top_k, 1])
        self.make_mul(
            f"{basename}/experts/WeightedMul",
            [expert_outputs, f"{basename}/router/WeightsUnsq/output_0"],
            self.io_dtype,
            [-1, top_k, hidden],
        )

        # Sum over top_k: (batch*seq, hidden)
        self.make_reduce_sum(
            f"{basename}/experts/Sum",
            [f"{basename}/experts/WeightedMul/output_0", "/model/constants/INT64/[1]"],
            self.io_dtype,
            [-1, hidden],
            keepdims=False,
        )

        # Reshape back to (batch, seq, hidden)
        self.make_reshape(
            f"{basename}/experts/Unflatten",
            [f"{basename}/experts/Sum/output_0", ["batch_size", "sequence_length", hidden]],
            self.io_dtype,
            ["batch_size", "sequence_length", hidden],
        )
        routed_out = f"{basename}/experts/Unflatten/output_0"

        # ----------------------------------------------------------------
        # Final output: routed + shared
        # ----------------------------------------------------------------
        self.make_add(f"{basename}/Output", [routed_out, shared_out], self.io_dtype, ["batch_size", "sequence_length", hidden])

        self.layernorm_attrs["skip_input"] = f"{basename}/Output/output_0"

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
