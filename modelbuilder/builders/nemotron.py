# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os

import torch

from .llama import LlamaModel


class NemotronHModel(LlamaModel):
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
        self._attn_layer_ids = [i for i, t in enumerate(layers_block_type) if t == "attention"]

        # Override KV cache inputs/outputs to only include attention layers.
        self.input_names["past_key_values.key"] = [f"past_key_values.{i}.key" for i in self._attn_layer_ids]
        self.input_names["past_key_values.value"] = [f"past_key_values.{i}.value" for i in self._attn_layer_ids]
        self.output_names["present.key"] = [f"present.{i}.key" for i in self._attn_layer_ids]
        self.output_names["present.value"] = [f"present.{i}.value" for i in self._attn_layer_ids]

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

        if layer.block_type == "attention":
            self.make_layernorm(layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input")
            self.make_attention(layer_id, layer.mixer, root_input=self.layernorm_attrs["output_0"])
        elif layer.block_type == "moe":
            self.make_layernorm(layer_id, layer.norm, skip=not self.layernorm_attrs["first_layernorm"], simple=True, location="input")
            self.make_nemotronh_moe(layer_id, layer.mixer, root_input=self.layernorm_attrs["output_0"])
        else:
            raise NotImplementedError(
                f"NemotronH block type '{layer.block_type}' is not supported for ONNX export. "
                "Only 'attention' and 'moe' layers are currently supported."
            )

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

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
        #           Reshape                down_proj MatMul
        #              |                       |
        #              +----> MoE op <----+    |
        #                       |              |
        #                       +-----> Add <--+
        #                               |
        #                          moe_output  (→ skip_input for next layer)

        basename = f"/model/layers.{layer_id}/moe"
        num_experts = self.moe_attrs["num_experts"]
        op_type = self.moe_attrs["op_type"]

        # 1. Router: gate.weight MatMul → Sigmoid → Reshape(-1, n_experts)
        gate_matmul_name = self.make_matmul(moe.gate, f"{basename}/gate/MatMul", root_input)

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

        # 2. Expert weights: store as initializers.
        #    experts.up_proj shape: (n_experts, moe_intermediate_size, hidden_size)
        #    experts.down_proj shape: (n_experts, hidden_size, moe_intermediate_size)
        #    Both are already in (out_features, in_features) format per expert, as expected by the MoE op.

        if op_type == "MoE":
            up_proj_weight = f"model.layers.{layer_id}.moe.experts.up_proj.weight"
            down_proj_weight = f"model.layers.{layer_id}.moe.experts.down_proj.weight"
            self.make_initializer(moe.experts.up_proj.detach(), up_proj_weight, to=self.io_dtype)
            self.make_initializer(moe.experts.down_proj.detach(), down_proj_weight, to=self.io_dtype)

            # 3. MoE op
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

        moe_expert_output = f"{moe_op_name}/output_0"

        # 4. Shared expert: up_proj → relu2 (Relu + Pow(2)) → down_proj
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

        # 5. Add routed-expert output and shared-expert output
        add_name = f"{basename}/Add"
        self.make_add(
            add_name,
            [moe_expert_output, f"{shared_down_name}/output_0"],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )

        # Assign MoE output as skip input for the next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{add_name}/output_0"

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
