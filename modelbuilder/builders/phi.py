# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import json
import os

import numpy as np
import onnx_ir as ir
import torch

from .base import Model
from .base_embedding import EmbeddingModel
from .base_vision import VisionEncoderModel
from .mistral import MistralModel


class PhiModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.mlp_attrs["use_proj"], self.mlp_attrs["use_fc"] = False, True

    def make_layer(self, layer_id, layer):
        # Each Phi decoder layer is defined as:
        # input_layernorm --> attention --> MLP --> residual_add
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, layer.self_attn, root_input=self.layernorm_attrs["output_0"])

        old_skip_input = self.layernorm_attrs["skip_input"]
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        residual_add_name = f"/model/layers.{layer_id}/residual_add/Add"
        residual_add_inputs = [old_skip_input, self.layernorm_attrs["skip_input"]]
        self.make_add(
            residual_add_name, residual_add_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size]
        )

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True

        # Assign output 0 of residual Add as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{residual_add_name}/output_0"


class Phi3MiniModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Phi3MiniLongRoPEModel(Phi3MiniModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_rotary_embedding_multi_cache()

        # Set position_ids_name based on whether position_ids is available as an input
        if "position_ids" in self.input_names:
            position_ids_result = self.make_position_ids_reformatting()
            self.position_ids_name = (
                f"{position_ids_result}/output_0"
                if position_ids_result != self.input_names["position_ids"]
                else self.input_names["position_ids"]
            )
        else:
            # When position_ids is not an input (use_rope_in_attn is True),
            # position_ids won't be used since rotary embeddings are handled in GQA
            self.position_ids_name = None

    def make_position_ids_reformatting(self):
        if self.ep not in self.eps_without_if_support:
            position_ids_input_to_rotemb = super().make_position_ids_reformatting()
            return position_ids_input_to_rotemb

        # Make Long RoPE scaling subgraph to adjust position_ids for sequences beyond original context length
        # For WebGPU: use int32 for all ops due to limited int64 support, then cast back to int64
        # For other EPs: use native int64 throughout
        #
        # WebGPU graph:                              Other EPs graph:
        #   position_ids                               position_ids
        #        |                                          |
        #   Cast (int32)                               ReduceMax (int64)
        #        |                                          |
        #   ReduceMax (int32)                          GreaterOrEqual
        #        |                                          |
        #   GreaterOrEqual                             Cast (int64)
        #        |                                          |
        #   Cast (int32)                               Mul (int64)
        #        |                                          |
        #   Mul (int32)                                Add (int64)
        #        |
        #   Add (int32)
        #        |
        #   Cast (int64)

        basename = "/model/pos_ids_reformat"
        proto_dtype = self.input_types["position_ids"]

        # For WebGPU, use int32 for computation due to limited int64 ops support
        is_webgpu = self.extra_options.get("enable_webgpu_graph", False)
        compute_dtype = ir.DataType.INT32 if is_webgpu else proto_dtype
        compute_str_dtype = self.to_str_dtype(compute_dtype)

        # Cast position_ids to int32 for WebGPU
        input_tensor = self.input_names["position_ids"]
        if is_webgpu:
            cast_input_name = f"{basename}/Cast_input"
            self.make_cast(cast_input_name, input_tensor, dtype=ir.DataType.INT32, shape=["batch_size", "sequence_length"])
            input_tensor = f"{cast_input_name}/output_0"

        reduce_max_name = f"{basename}/ReduceMax"
        reduce_max_inputs = [input_tensor]
        self.make_reduce_max(reduce_max_name, reduce_max_inputs, dtype=compute_dtype, shape=[1])
        greater_or_equal_name = f"{basename}/GreaterOrEqual"
        greater_or_equal_inputs = [f"{reduce_max_name}/output_0", f"/model/constants/{compute_str_dtype}/{self.original_context_length}"]
        self.make_greater_or_equal(greater_or_equal_name, greater_or_equal_inputs, shape=[])
        cast_name = f"{basename}/Cast"
        self.make_cast(cast_name, f"{greater_or_equal_name}/output_0", dtype=compute_dtype, shape=None)
        mul_name = f"{basename}/Mul"
        mul_inputs = [f"{cast_name}/output_0", f"/model/constants/{compute_str_dtype}/{self.original_context_length}"]
        self.make_mul(mul_name, mul_inputs, dtype=compute_dtype, shape=None)
        add_1_name = f"{basename}/Add_1"
        add_1_inputs = [f"{mul_name}/output_0", input_tensor]
        self.make_add(add_1_name, add_1_inputs, dtype=compute_dtype, shape=["batch_size", "sequence_length"])

        # Cast back to int64 for WebGPU to maintain compatibility
        result_name = add_1_name
        if is_webgpu:
            cast_output_name = f"{basename}/Cast_output"
            self.make_cast(cast_output_name, f"{add_1_name}/output_0", dtype=ir.DataType.INT64, shape=["batch_size", "sequence_length"])
            result_name = cast_output_name

        return result_name

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        if self.position_ids_name is not None:
            super().make_attention(layer_id, attention, root_input, position_ids=self.position_ids_name, **kwargs)
        else:
            super().make_attention(layer_id, attention, root_input, **kwargs)


class Phi3SmallModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layernorm_attrs["simple"] = False
        self.embed_attrs["scale"] = config.mup_embedding_multiplier
        self.rope_attrs["t_dtype"] = torch.float32
        self.lm_head_attrs["scale"] = 1 / config.mup_width_multiplier

        self.calculate_block_mask()
        self.dense_attention_every_n_layers = config.dense_attention_every_n_layers
        if config.mup_use_scaling:
            self.attention_attrs["scale"] = config.mup_attn_multiplier / self.head_size

        self.clamp_limit = config.gegelu_limit

    def calculate_cdiv(self, a, b):
        return -(a // -b)

    def calculate_block_mask(self):
        # Initialize parameters for calculating block dense mask
        n_heads = self.num_attn_heads
        q_len = self.context_length
        N_CTX = self.context_length
        BLOCK = self.attention_attrs["block_sparse"]["sparse_block_size"]
        local_blocks = self.attention_attrs["block_sparse"]["local_blocks"]
        vert_stride = self.attention_attrs["block_sparse"]["vert_stride"]
        homo_head = self.attention_attrs["block_sparse"]["homo_head"]

        N_BLOCK = self.calculate_cdiv(N_CTX, BLOCK)
        if homo_head:
            q_pos = torch.arange(N_BLOCK)[:, None]
            k_pos = torch.arange(N_BLOCK)[None]
            mask_vert_strided = (torch.arange(N_BLOCK) + 1) % vert_stride == 0
            block_mask_dense = (q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)
            N_BLOCK_Q = self.calculate_cdiv(q_len, BLOCK)
            block_mask_dense_output = block_mask_dense[-N_BLOCK_Q:].to_sparse_csr()

            crows = block_mask_dense_output.crow_indices()
            cols = block_mask_dense_output.col_indices()

            crows = crows[None].expand(n_heads, crows.shape[0])
            cols = cols[None].expand(n_heads, cols.shape[0])
        else:
            q_pos = torch.arange(N_BLOCK)[None, :, None]
            k_pos = torch.arange(N_BLOCK)[None, None]
            head_sliding_step = max(1, int(vert_stride / n_heads))  # if vert_stride <= n_heads, rotating the heads
            mask_vert_strided = [(torch.arange(N_BLOCK) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(n_heads)]
            mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
            block_mask_dense = (q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)
            N_BLOCK_Q = self.calculate_cdiv(q_len, BLOCK)
            block_mask_dense_output = block_mask_dense[:, -N_BLOCK_Q:]

            # Dense to crow_col
            pad = -1
            dim = block_mask_dense_output.dim()
            assert dim in (2, 3)
            if dim == 2:
                block_mask_dense_output = block_mask_dense_output[None]
            block_mask_dense_output = [xi.to_sparse_csr() for xi in block_mask_dense_output]
            crows = torch.vstack([xi.crow_indices() for xi in block_mask_dense_output])
            cols = [xi.col_indices() for xi in block_mask_dense_output]
            max_cols = max(len(xi) for xi in cols)
            cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
            cols = torch.vstack(cols)
            if dim == 2:
                crows = crows[0]
                cols = cols[0]

        # Create tensors for row indices and col indices
        crows_name = "block_row_indices"
        self.make_initializer(crows, crows_name, to=ir.DataType.INT32)
        self.mask_attrs["block_row_indices"] = crows_name

        cols_name = "block_col_indices"
        self.make_initializer(cols, cols_name, to=ir.DataType.INT32)
        self.mask_attrs["block_col_indices"] = cols_name

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        dense_attention_op = self.attention_attrs["op_type"]
        sparse_attention_op = "SparseAttention"

        # Use dense attention every n layers and use sparse attention otherwise
        if (self.layer_id + 1) % self.dense_attention_every_n_layers != 0:
            # Use sparse attention
            self.attention_attrs["op_type"] = sparse_attention_op

        q_size = self.num_attn_heads * self.head_size
        kv_size = self.num_kv_heads * self.head_size

        qkv_weight = attention.query_key_value.weight.T.view(
            self.hidden_size, self.num_kv_heads, (self.num_attn_heads // self.num_kv_heads) + 2, self.head_size
        )
        qkv_bias = attention.query_key_value.bias.view(self.num_kv_heads, (self.num_attn_heads // self.num_kv_heads) + 2, self.head_size)

        attention.q_proj = torch.nn.Linear(in_features=q_size, out_features=q_size)
        attention.q_proj.weight = torch.nn.Parameter(qkv_weight[:, :, :-2].reshape(q_size, q_size).T, requires_grad=False)
        attention.q_proj.bias = (
            None if attention.query_key_value.bias is None else torch.nn.Parameter(qkv_bias[:, :-2].flatten(), requires_grad=False)
        )

        attention.k_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.k_proj.weight = torch.nn.Parameter(qkv_weight[:, :, [-2]].reshape(q_size, kv_size).T, requires_grad=False)
        attention.k_proj.bias = (
            None if attention.query_key_value.bias is None else torch.nn.Parameter(qkv_bias[:, [-2]].flatten(), requires_grad=False)
        )

        attention.v_proj = torch.nn.Linear(in_features=q_size, out_features=kv_size)
        attention.v_proj.weight = torch.nn.Parameter(qkv_weight[:, :, [-1]].reshape(q_size, kv_size).T, requires_grad=False)
        attention.v_proj.bias = (
            None if attention.query_key_value.bias is None else torch.nn.Parameter(qkv_bias[:, [-1]].flatten(), requires_grad=False)
        )

        del qkv_weight
        del qkv_bias
        del attention.query_key_value

        super().make_attention(layer_id, attention, root_input, **kwargs)
        self.attention_attrs["op_type"] = dense_attention_op

    def make_mlp_proj(self, layer_id, mlp, root_input):
        # Make nodes for the MLP subgraph
        #
        #           root_input
        #               |
        #          UpProjMatMul
        #               |
        #           UpProjAdd
        #          /          \
        #         /            \
        #        /              \
        #      Slice             Slice
        #    (even idx)        (odd idx)
        #    /   |   \         /   |   \
        #  Cast  |    |      Cast  |    |
        #   |    |    |       |    |    |
        # IsInf  |   Clip   IsInf  |   Clip
        #   |    |    |       |    |    |
        #    \   |   /         \   |   /
        #     \  |  /           \  |  /
        #      Where             Where
        #        |                 |
        #    QuickGelu            Add
        #        |                 |
        #        +--------+--------+
        #                 |
        #                Mul
        #                 |
        #           DownProjMatMul
        #                 |
        #            DownProjAdd

        # Make input MatMul and Add nodes
        up_matmul_name = f"/model/layers.{layer_id}/mlp/up_proj/MatMul"
        self.make_matmul(mlp.up_proj, up_matmul_name, root_input)
        up_add_name = f"/model/layers.{layer_id}/mlp/up_proj/Add"
        self.make_add_bias(mlp.up_proj.bias, up_add_name, f"{up_matmul_name}/output_0")

        # Left path
        slice_1_name = f"/model/layers.{layer_id}/mlp/gelu/Slice"
        slice_1_inputs = [
            f"{up_add_name}/output_0",
            "/model/constants/INT64/[0]",
            f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]",
            "/model/constants/INT64/[-1]",
            "/model/constants/INT64/[2]",
        ]
        self.make_slice(slice_1_name, slice_1_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        cast_1_name = f"/model/layers.{layer_id}/mlp/gelu/Cast"
        self.make_cast(
            cast_1_name,
            f"{slice_1_name}/output_0",
            dtype=ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        isinf_1_name = f"/model/layers.{layer_id}/mlp/gelu/IsInf"
        self.make_isinf(isinf_1_name, f"{cast_1_name}/output_0", shape=["batch_size", "sequence_length", self.intermediate_size])
        clip_1_name = f"/model/layers.{layer_id}/mlp/gelu/Clip"
        clip_1_inputs = [f"{slice_1_name}/output_0", "", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.clamp_limit}"]
        self.make_clip(clip_1_name, clip_1_inputs, self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        where_1_name = f"/model/layers.{layer_id}/mlp/gelu/Where"
        where_1_inputs = [f"{isinf_1_name}/output_0", f"{slice_1_name}/output_0", f"{clip_1_name}/output_0"]
        self.make_where(where_1_name, where_1_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        # Make activation
        act_fn_name = self.make_activation(layer_id, root_input=f"{where_1_name}/output_0")

        # Right path
        slice_2_name = f"/model/layers.{layer_id}/mlp/linear/Slice"
        slice_2_inputs = [
            f"{up_add_name}/output_0",
            "/model/constants/INT64/[1]",
            f"/model/constants/INT64/[{torch.iinfo(torch.int64).max}]",
            "/model/constants/INT64/[-1]",
            "/model/constants/INT64/[2]",
        ]
        self.make_slice(slice_2_name, slice_2_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        cast_2_name = f"/model/layers.{layer_id}/mlp/linear/Cast"
        self.make_cast(
            cast_2_name,
            f"{slice_2_name}/output_0",
            dtype=ir.DataType.FLOAT,
            shape=["batch_size", "sequence_length", self.intermediate_size],
        )
        isinf_2_name = f"/model/layers.{layer_id}/mlp/linear/IsInf"
        self.make_isinf(isinf_2_name, f"{cast_2_name}/output_0", shape=["batch_size", "sequence_length", self.intermediate_size])
        clip_2_name = f"/model/layers.{layer_id}/mlp/linear/Clip"
        clip_2_inputs = [
            f"{slice_2_name}/output_0",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/-{self.clamp_limit}",
            f"/model/constants/{self.to_str_dtype(self.io_dtype)}/{self.clamp_limit}",
        ]
        self.make_clip(clip_2_name, clip_2_inputs, self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        where_2_name = f"/model/layers.{layer_id}/mlp/linear/Where"
        where_2_inputs = [f"{isinf_2_name}/output_0", f"{slice_2_name}/output_0", f"{clip_2_name}/output_0"]
        self.make_where(where_2_name, where_2_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])
        add_name = f"/model/layers.{layer_id}/mlp/linear/Add"
        add_inputs = [f"{where_2_name}/output_0", f"/model/constants/{self.to_str_dtype(self.io_dtype)}/1"]
        self.make_add(add_name, add_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        # Make Mul node after activation
        mul_name = f"/model/layers.{layer_id}/mlp/Mul"
        mul_inputs = [f"{act_fn_name}/output_0", f"{add_name}/output_0"]
        self.make_mul(mul_name, mul_inputs, dtype=self.io_dtype, shape=["batch_size", "sequence_length", self.intermediate_size])

        # Make output MatMul and Add nodes
        down_matmul_name = f"/model/layers.{layer_id}/mlp/down_proj/MatMul"
        self.make_matmul(mlp.down_proj, down_matmul_name, f"{mul_name}/output_0")
        down_add_name = f"/model/layers.{layer_id}/mlp/down_proj/Add"
        self.make_add_bias(mlp.down_proj.bias, down_add_name, f"{down_matmul_name}/output_0")

        # Assign output 0 of previous MatMul as skip input to next SkipLayerNorm
        self.layernorm_attrs["skip_input"] = f"{down_add_name}/output_0"


class Phi3SmallLongRoPEModel(Phi3SmallModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.make_rotary_embedding_multi_cache()


class Phi3VModel(Phi3MiniLongRoPEModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Phi3MoELongRoPEModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        assert io_dtype == ir.DataType.FLOAT16, "This model only supports float16 io type."
        self.layernorm_attrs["simple"] = False
        self.moe_attrs["use_sparse_mixer"] = True
        self.make_rotary_embedding_multi_cache()

    def make_layer(self, layer_id, layer):
        # Each LLM decoder layer is typically defined as:
        # input_layernorm --> attention --> output_layernorm --> MoE
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
        self.make_block_sparse_moe(layer_id, layer.block_sparse_moe, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            # Norm after last decoder layer of model (last layer --> norm)
            self.layernorm_attrs["last_layernorm"] = True


class Phi4MMModel(Phi3VModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.matmul_attrs["use_lora"] = True
        self.attention_attrs["use_packed_matmul"] = False

    def make_layer(self, layer_id, layer):
        layer.self_attn.qkv_proj.lora_A.default = layer.self_attn.qkv_proj.lora_A.vision
        layer.self_attn.qkv_proj.lora_B.default = layer.self_attn.qkv_proj.lora_B.vision
        layer.self_attn.qkv_proj.scaling["default"] = layer.self_attn.qkv_proj.scaling["vision"]
        layer.self_attn.o_proj.lora_A.default = layer.self_attn.o_proj.lora_A.vision
        layer.self_attn.o_proj.lora_B.default = layer.self_attn.o_proj.lora_B.vision
        layer.self_attn.o_proj.scaling["default"] = layer.self_attn.o_proj.scaling["vision"]

        layer.mlp.gate_up_proj.lora_A.default = layer.mlp.gate_up_proj.lora_A.vision
        layer.mlp.gate_up_proj.lora_B.default = layer.mlp.gate_up_proj.lora_B.vision
        layer.mlp.gate_up_proj.scaling["default"] = layer.mlp.gate_up_proj.scaling["vision"]
        layer.mlp.down_proj.lora_A.default = layer.mlp.down_proj.lora_A.vision
        layer.mlp.down_proj.lora_B.default = layer.mlp.down_proj.lora_B.vision
        layer.mlp.down_proj.scaling["default"] = layer.mlp.down_proj.scaling["vision"]

        super().make_layer(layer_id, layer)


class Phi4MultimodalVisionEncoderModel(VisionEncoderModel):
    """ONNX graph builder for the Phi-4-multimodal-instruct vision encoder.

    Exports the vision tower (patch embedding + transformer blocks + AvgPool2d
    compression + crop merging + projection) to ``vision_encoder.onnx``.

    For a fixed 448×448 image the processor creates exactly two crops:
    one sub-crop and one global crop, both of size 448×448.

    Inputs
    ------
    pixel_values : float [2, 3, 448, 448]
        Pre-processed pixel values for all crops (sub-crop first, then global).

    Outputs
    -------
    image_features : io_dtype [num_image_tokens, text_hidden_size]
        Merged and projected image features for injection into text embeddings.
        For a 448×448 image: num_image_tokens = 545.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        vc = config.vision_config

        # Patch a copy of config with vision encoder attributes so that
        # Model.__init__ can initialise the shared graph/values state.
        vis_config = copy.deepcopy(config)
        vis_config.hidden_size = vc.hidden_size
        vis_config.intermediate_size = vc.intermediate_size
        vis_config.num_attention_heads = vc.num_attention_heads
        vis_config.num_key_value_heads = vc.num_attention_heads
        vis_config.num_hidden_layers = vc.num_hidden_layers
        vis_config.head_dim = vc.hidden_size // vc.num_attention_heads
        vis_config.hidden_act = vc.hidden_act
        vis_config.vocab_size = 1
        vis_config.max_position_embeddings = (vc.image_size // vc.patch_size) ** 2
        vis_config.rms_norm_eps = vc.layer_norm_eps
        vis_config.rope_scaling = None

        extra_options = {**extra_options, "filename": self.FILENAME, "exclude_lm_head": True, "exclude_embeds": True}

        super().__init__(vis_config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.graph.name = "phi4mm_vision_encoder"

        # Store the original full config for text_hidden_size and image_token_id.
        self.config = config
        self.vision_config = vc

        # Vision-specific dimensions.
        self.crop_size = vc.image_size  # 448
        self.patch_size = vc.patch_size  # 14
        self.num_channels = vc.num_channels  # 3
        self.n_patches_per_side = vc.image_size // vc.patch_size  # 32
        self.n_patches = self.n_patches_per_side**2  # 1024
        self.vis_hidden_size = vc.hidden_size  # 1152
        self.vis_intermediate_size = vc.intermediate_size  # 4304
        self.vis_num_heads = vc.num_attention_heads  # 16
        self.vis_head_dim = vc.hidden_size // vc.num_attention_heads  # 72
        self.vis_attn_scale = float(self.vis_head_dim**-0.5)
        self.vis_rms_norm_eps = vc.layer_norm_eps  # 1e-6

        # feature_layer = -2 means we extract the hidden state after
        # layer (num_hidden_layers - 2) = layer 25 (0-indexed).
        self.vis_feature_layer_idx = vc.num_hidden_layers + vc.feature_layer  # 25

        # After AvgPool2d(kernel=2, stride=2): n_patches_per_side → n//2.
        self.n_compressed_per_side = self.n_patches_per_side // 2  # 16
        self.n_compressed_patches = self.n_compressed_per_side**2  # 256

        # Text hidden size for the projector output.
        self.text_hidden_size = config.hidden_size  # 3072

        # Fixed: one sub-crop + one global crop (total 2 crops for 448×448).
        self.n_crops = 2

        # Total image tokens per image (for a 448×448 image):
        #   sub:    n_cs * (n_cs + 1) = 16 * 17 = 272
        #   global extensor: 1
        #   global: n_cs * (n_cs + 1) = 16 * 17 = 272
        #   total:  545
        n_cs = self.n_compressed_per_side
        self.n_sub_tokens = n_cs * (n_cs + 1)
        self.n_global_tokens = n_cs * (n_cs + 1)
        self.n_image_tokens = self.n_sub_tokens + 1 + self.n_global_tokens

    # ------------------------------------------------------------------
    # Vision attention (no RoPE, standard SDPA)
    # ------------------------------------------------------------------

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Build one Phi4MultimodalVisionAttention layer.

        root_input: [n_crops, n_patches, vis_hidden_size]
        Stores the attention output in layernorm_attrs["skip_input"].
        """
        b = f"/vision/layers.{layer_id}/attn"
        nc = self.n_crops
        n_p = self.n_patches
        d = self.vis_hidden_size
        nh = self.vis_num_heads
        hd = self.vis_head_dim

        # Q / K / V projections (with bias).
        q = f"{self.make_matmul(attention.q_proj, f'{b}/q_proj/MatMul', root_input)}/output_0"
        if attention.q_proj.bias is not None:
            self.make_add_bias(attention.q_proj.bias, f"{b}/q_proj/Add", root_input=q)
            q = f"{b}/q_proj/Add/output_0"

        k = f"{self.make_matmul(attention.k_proj, f'{b}/k_proj/MatMul', root_input)}/output_0"
        if attention.k_proj.bias is not None:
            self.make_add_bias(attention.k_proj.bias, f"{b}/k_proj/Add", root_input=k)
            k = f"{b}/k_proj/Add/output_0"

        v = f"{self.make_matmul(attention.v_proj, f'{b}/v_proj/MatMul', root_input)}/output_0"
        if attention.v_proj.bias is not None:
            self.make_add_bias(attention.v_proj.bias, f"{b}/v_proj/Add", root_input=v)
            v = f"{b}/v_proj/Add/output_0"

        # Reshape to [nc, n_patches, n_heads, head_dim] then transpose to [nc, n_heads, n_patches, head_dim].
        qkv_4d = [nc, n_p, nh, hd]
        q_4d = self.make_reshape(f"{b}/q_reshape", [q, [nc, n_p, nh, hd]], self.io_dtype, qkv_4d)
        k_4d = self.make_reshape(f"{b}/k_reshape", [k, [nc, n_p, nh, hd]], self.io_dtype, qkv_4d)
        v_4d = self.make_reshape(f"{b}/v_reshape", [v, [nc, n_p, nh, hd]], self.io_dtype, qkv_4d)

        qkv_t = [nc, nh, n_p, hd]
        q_t = self.make_transpose(f"{b}/q_t", q_4d, self.io_dtype, qkv_t, perm=[0, 2, 1, 3])
        v_t = self.make_transpose(f"{b}/v_t", v_4d, self.io_dtype, qkv_t, perm=[0, 2, 1, 3])
        # k_T: [nc, nh, hd, n_p] — pre-transposed for matmul
        k_T = self.make_transpose(f"{b}/k_T", k_4d, self.io_dtype, [nc, nh, hd, n_p], perm=[0, 2, 3, 1])

        # Scaled dot-product attention with causal mask.
        # The HF vision attention uses SDPA with is_causal=True; replicate that here.
        attn_w_out = f"{b}/attn_w/MatMul/output_0"
        self.make_node("MatMul", inputs=[q_t, k_T], outputs=[attn_w_out], name=f"{b}/attn_w/MatMul")
        self.make_value(attn_w_out, self.io_dtype, shape=[nc, nh, n_p, n_p])

        np_dtype = {ir.DataType.FLOAT: np.float32, ir.DataType.FLOAT16: np.float16}.get(self.io_dtype, np.float32)
        scale_name = f"{b}/scale"
        self.make_initializer(np.array(self.vis_attn_scale, dtype=np_dtype), scale_name)
        attn_ws = self.make_mul(f"{b}/attn_scale", [attn_w_out, scale_name], self.io_dtype, [nc, nh, n_p, n_p])

        # Causal mask: upper-triangular entries set to -inf so softmax ignores them.
        # Shape [1, 1, n_p, n_p] broadcasts over (nc, nh).
        causal_mask_name = f"{b}/causal_mask"
        causal_np = np.full((1, 1, n_p, n_p), fill_value=0.0, dtype=np_dtype)
        causal_np[:, :, np.triu_indices(n_p, k=1)[0], np.triu_indices(n_p, k=1)[1]] = np.finfo(np_dtype).min / 2
        self.make_initializer(causal_np, causal_mask_name)
        attn_ws_masked = self.make_add(f"{b}/causal_add", [attn_ws, causal_mask_name], self.io_dtype, [nc, nh, n_p, n_p])
        attn_probs = self.make_softmax(f"{b}/softmax", attn_ws_masked, self.io_dtype, [nc, nh, n_p, n_p])

        attn_out_t = f"{b}/attn_out/MatMul/output_0"
        self.make_node("MatMul", inputs=[attn_probs, v_t], outputs=[attn_out_t], name=f"{b}/attn_out/MatMul")
        self.make_value(attn_out_t, self.io_dtype, shape=qkv_t)

        # Transpose + Reshape back to [nc, n_patches, hidden].
        attn_out = self.make_transpose(f"{b}/attn_out_t", attn_out_t, self.io_dtype, [nc, n_p, nh, hd], perm=[0, 2, 1, 3])
        attn_out_3d = self.make_reshape(f"{b}/attn_out_reshape", [attn_out, [nc, n_p, d]], self.io_dtype, [nc, n_p, d])

        # O projection (with bias).
        o = f"{self.make_matmul(attention.out_proj, f'{b}/out_proj/MatMul', attn_out_3d)}/output_0"
        if attention.out_proj.bias is not None:
            self.make_add_bias(attention.out_proj.bias, f"{b}/out_proj/Add", root_input=o)
            o = f"{b}/out_proj/Add/output_0"

        self.layernorm_attrs["skip_input"] = o

    # ------------------------------------------------------------------
    # Single transformer layer
    # ------------------------------------------------------------------

    def make_layer(self, layer_id, layer):
        """Build one Phi4MultimodalVisionEncoderLayer.

        Pipeline: LN1 → attention → residual → LN2 → MLP → residual.
        """
        root_input = self.layernorm_attrs["root_input"]
        b = f"/vision/layers.{layer_id}"
        nc = self.n_crops
        n_p = self.n_patches
        d = self.vis_hidden_size
        ff = self.vis_intermediate_size

        # Layer norm 1.
        norm1 = self.make_layer_norm(
            f"{b}/layer_norm1/LayerNorm",
            root_input,
            layer.layer_norm1.weight,
            layer.layer_norm1.bias,
            shape=[nc, n_p, d],
            weight_name=f"vision.layers.{layer_id}.layer_norm1.weight",
            bias_name=f"vision.layers.{layer_id}.layer_norm1.bias",
        )

        # Attention.
        self.make_attention(layer_id, layer.self_attn, norm1)
        attn_out = self.layernorm_attrs["skip_input"]

        # Residual 1.
        res1 = self.make_add(f"{b}/residual1/Add", [root_input, attn_out], self.io_dtype, [nc, n_p, d])

        # Layer norm 2.
        norm2 = self.make_layer_norm(
            f"{b}/layer_norm2/LayerNorm",
            res1,
            layer.layer_norm2.weight,
            layer.layer_norm2.bias,
            shape=[nc, n_p, d],
            weight_name=f"vision.layers.{layer_id}.layer_norm2.weight",
            bias_name=f"vision.layers.{layer_id}.layer_norm2.bias",
        )

        # MLP: fc1 → GELU → fc2 (with optional bias on each).
        fc1_out = f"{self.make_matmul(layer.mlp.fc1, f'{b}/mlp/fc1/MatMul', norm2)}/output_0"
        if layer.mlp.fc1.bias is not None:
            self.make_add_bias(layer.mlp.fc1.bias, f"{b}/mlp/fc1/Add", root_input=fc1_out)
            fc1_out = f"{b}/mlp/fc1/Add/output_0"

        gelu_out = f"{b}/mlp/gelu/output_0"
        self.make_node("Gelu", inputs=[fc1_out], outputs=[gelu_out], name=f"{b}/mlp/gelu/Gelu", domain="com.microsoft")
        self.make_value(gelu_out, self.io_dtype, shape=[nc, n_p, ff])

        fc2_out = f"{self.make_matmul(layer.mlp.fc2, f'{b}/mlp/fc2/MatMul', gelu_out)}/output_0"
        if layer.mlp.fc2.bias is not None:
            self.make_add_bias(layer.mlp.fc2.bias, f"{b}/mlp/fc2/Add", root_input=fc2_out)
            fc2_out = f"{b}/mlp/fc2/Add/output_0"

        # Residual 2.
        res2 = self.make_add(f"{b}/residual2/Add", [res1, fc2_out], self.io_dtype, [nc, n_p, d])

        self.layernorm_attrs["root_input"] = res2

    # ------------------------------------------------------------------
    # Patch embedding (Conv2d + position embeddings)
    # ------------------------------------------------------------------

    def _build_patch_embedding(self, vision_embed):
        """Build Conv2d patch embedding + fixed position embeddings.

        Returns value name of shape [n_crops, n_patches, vis_hidden_size].
        """
        nc = self.n_crops
        n_p = self.n_patches
        d = self.vis_hidden_size

        # Graph input.
        pixel_values_in = self.make_value("pixel_values", self.io_dtype, shape=[nc, self.num_channels, self.crop_size, self.crop_size])
        self.graph.inputs.append(pixel_values_in)

        patch_embed = self.make_patch_embedding(
            "pixel_values", vision_embed.patch_embedding.weight, "vision.embeddings.patch_embedding.weight", [nc]
        )

        # Fixed position embeddings (no interpolation for fixed crop_size).
        # shape: [1, n_patches, d] broadcast-added to [nc, n_patches, d].
        pos_w = vision_embed.position_embedding.weight.detach().unsqueeze(0)  # [1, n_patches, d]
        pos_name = "vision.embeddings.position_embedding.weight"
        self.make_initializer(pos_w, pos_name, to=self.io_dtype)

        embed_out = self.make_add("/vision/pos_embed/Add", [patch_embed, pos_name], self.io_dtype, [nc, n_p, d])
        return embed_out

    # ------------------------------------------------------------------
    # AvgPool2d(kernel=2, stride=2) compression
    # ------------------------------------------------------------------

    def _build_avg_pool(self, x):
        """Apply AvgPool2d(2, 2) to compress patch features.

        Input shape:  [n_crops, n_patches, d]  =  [2, 1024, 1152]
        Output shape: [n_crops, n_compressed_patches, d]  =  [2, 256, 1152]
        """
        nc = self.n_crops
        n_h = n_w = self.n_patches_per_side  # 32
        d = self.vis_hidden_size  # 1152
        nc_s = self.n_compressed_per_side  # 16
        n_cp = self.n_compressed_patches  # 256

        # [nc, n_patches, d] → [nc, n_h, n_w, d] → [nc, d, n_h, n_w]
        r1 = self.make_reshape("/vision/pool/Reshape1", [x, [nc, n_h, n_w, d]], self.io_dtype, [nc, n_h, n_w, d])
        r2 = self.make_transpose("/vision/pool/Transpose1", r1, self.io_dtype, [nc, d, n_h, n_w], perm=[0, 3, 1, 2])

        # AveragePool: [nc, d, n_h, n_w] → [nc, d, nc_s, nc_s]
        pool_out = "/vision/pool/AveragePool/output_0"
        self.make_node("AveragePool", inputs=[r2], outputs=[pool_out], name="/vision/pool/AveragePool", kernel_shape=[2, 2], strides=[2, 2])
        self.make_value(pool_out, self.io_dtype, shape=[nc, d, nc_s, nc_s])

        # [nc, d, nc_s, nc_s] → [nc, nc_s, nc_s, d] → [nc, n_compressed_patches, d]
        r3 = self.make_transpose("/vision/pool/Transpose2", pool_out, self.io_dtype, [nc, nc_s, nc_s, d], perm=[0, 2, 3, 1])
        r4 = self.make_reshape("/vision/pool/Reshape2", [r3, [nc, n_cp, d]], self.io_dtype, [nc, n_cp, d])
        return r4

    # ------------------------------------------------------------------
    # Crop merging: sub_img + global_extensor + global_img
    # ------------------------------------------------------------------

    def _build_crop_merging(self, x, image_embed):
        """Merge sub-crop and global-crop with feature extensors.

        The pixel_values layout mirrors the HF model: index 0 = global image,
        index 1 = sub-crop (for a single 448×448 image with area_ratio=1).

        The HF model then combines them as:
          cat([sub_img, global_img_feature_extensor, global_img], dim=1)

        Input:  x [n_crops=2, n_compressed_patches=256, d=1152]
        Output: merged [n_image_tokens=545, d=1152]
        """
        nc = self.n_crops  # noqa: F841 - kept for readability alongside n_cp/n_cs/d
        n_cp = self.n_compressed_patches  # 256
        n_cs = self.n_compressed_per_side  # 16
        d = self.vis_hidden_size  # 1152

        # ---- Sub-crop (index 1: processor puts global first) ----
        sub_slice = self.make_slice("/vision/merge/sub_slice", x, self.io_dtype, [1, n_cp, d], starts=[1], ends=[2], axes=[0])
        sub_r = self.make_reshape("/vision/merge/sub_reshape", [sub_slice, [1, n_cs, n_cs, d]], self.io_dtype, [1, n_cs, n_cs, d])

        # Append feature-extensor column (repeated sub_img_feature_extensor).
        # sub_img_feature_extensor has shape [1, 1, 1, d]; expand to [1, n_cs, 1, d].
        sub_ext = image_embed.sub_img_feature_extensor.expand(1, n_cs, 1, d).reshape(1, n_cs, 1, d).detach()
        sub_ext_name = "vision.sub_img_feature_extensor"
        self.make_initializer(sub_ext, sub_ext_name, to=self.io_dtype)
        sub_cat = self.make_concat("/vision/merge/sub_cat", [sub_r, sub_ext_name], self.io_dtype, [1, n_cs, n_cs + 1, d], axis=2)
        sub_flat = self.make_reshape(
            "/vision/merge/sub_flat", [sub_cat, [1, self.n_sub_tokens, d]], self.io_dtype, [1, self.n_sub_tokens, d]
        )

        # ---- Global-image extensor (single token separator) ----
        global_ext_name = "vision.global_img_feature_extensor"
        self.make_initializer(image_embed.global_img_feature_extensor.detach(), global_ext_name, to=self.io_dtype)

        # ---- Global crop (index 0: first in HF layout) ----
        global_slice = self.make_slice("/vision/merge/global_slice", x, self.io_dtype, [1, n_cp, d], starts=[0], ends=[1], axes=[0])
        global_r = self.make_reshape("/vision/merge/global_reshape", [global_slice, [1, n_cs, n_cs, d]], self.io_dtype, [1, n_cs, n_cs, d])

        # The HF code uses sub_img_feature_extensor for the global image column too.
        global_col_name = "vision.global_col_feature_extensor"
        self.make_initializer(sub_ext, global_col_name, to=self.io_dtype)
        global_cat = self.make_concat(
            "/vision/merge/global_cat", [global_r, global_col_name], self.io_dtype, [1, n_cs, n_cs + 1, d], axis=2
        )
        global_flat = self.make_reshape(
            "/vision/merge/global_flat", [global_cat, [1, self.n_global_tokens, d]], self.io_dtype, [1, self.n_global_tokens, d]
        )

        # Concatenate: [sub | global_extensor | global] along seq dim.
        merged = self.make_concat(
            "/vision/merge/concat", [sub_flat, global_ext_name, global_flat], self.io_dtype, [1, self.n_image_tokens, d], axis=1
        )

        # Squeeze the batch=1 dimension: [1, n_tokens, d] → [n_tokens, d].
        squeezed = self.make_reshape("/vision/merge/squeeze", [merged, [self.n_image_tokens, d]], self.io_dtype, [self.n_image_tokens, d])
        return squeezed

    # ------------------------------------------------------------------
    # Projection: Linear(d, t_hid) + GELU + Linear(t_hid, t_hid)
    # ------------------------------------------------------------------

    def _build_projector(self, image_embed, x):
        """Project compressed features into the text hidden space.

        Input:  x [n_image_tokens, image_dim_out]   e.g. [545, 1152]
        Output:   [n_image_tokens, text_hidden_size] e.g. [545, 3072]
        """
        return self.make_gelu_mlp(
            image_embed.img_projection_up, image_embed.img_projection_down, x, [self.n_image_tokens, self.text_hidden_size], "/vision/proj"
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def _load_hf_model(self, input_path):
        from transformers import Phi4MultimodalForCausalLM

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Phi4MultimodalForCausalLM.from_pretrained(src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)

    def make_model(self, input_path):
        """Load HF weights and build the vision encoder ONNX graph."""
        hf_model = self._load_hf_model(input_path)
        hf_model.eval()

        image_embed = hf_model.model.embed_tokens_extend.image_embed
        vis_model = image_embed.img_processor  # Phi4MultimodalVisionModel

        # Step 1: Patch embedding + position embedding.
        x = self._build_patch_embedding(vis_model.embeddings)

        # Step 2: Transformer layers 0 … vis_feature_layer_idx (inclusive).
        self.layernorm_attrs["root_input"] = x
        for layer_id in range(self.vis_feature_layer_idx + 1):
            self.make_layer(layer_id, vis_model.encoder.layers[layer_id])
        x = self.layernorm_attrs["root_input"]

        # Step 3: AvgPool2d(2, 2) compression.
        x = self._build_avg_pool(x)

        # Step 4: Merge crops (sub + global_extensor + global).
        x = self._build_crop_merging(x, image_embed)

        # Step 5: Project to text hidden size.
        image_features = self._build_projector(image_embed, x)

        # Graph output.
        self.make_node("Identity", inputs=[image_features], outputs=["image_features"], name="/vision/output/Identity")
        out_val = self.make_value("image_features", self.io_dtype, shape=[self.n_image_tokens, self.text_hidden_size])
        self.graph.outputs.append(out_val)

        self.graph.sort()


class Phi4MultimodalEmbeddingModel(EmbeddingModel):
    """ONNX embedding model for the Phi-4-multimodal-instruct phi3v pipeline.

    Scatters vision-encoder ``image_features`` into the text-token embeddings
    at positions marked with ``image_token_id``.
    """

    def _load_hf_model(self, input_path):
        from transformers import Phi4MultimodalForCausalLM

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Phi4MultimodalForCausalLM.from_pretrained(src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)

    # EmbeddingModel.make_model calls self.load_hf_model (note: no leading underscore).
    def load_hf_model(self, input_path):
        return self._load_hf_model(input_path)

    def get_embed_weight(self, hf_model):
        return hf_model.model.embed_tokens.weight.detach().float().numpy()


class Phi4MultimodalTextModel(MistralModel):
    """Text decoder for ``Phi4MultimodalForCausalLM``.

    Structurally identical to Phi3MiniModel (MistralModel) but loaded from
    the full multimodal checkpoint.  The model is built with
    ``exclude_embeds=True`` so it accepts ``inputs_embeds`` directly.
    """

    def load_weights(self, input_path):
        from transformers import Phi4MultimodalForCausalLM

        model = Phi4MultimodalForCausalLM.from_pretrained(
            input_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
        )
        return model


class Phi4MultimodalConditionalGenerationModel(Model):
    """Orchestrates exporting the Phi-4-multimodal-instruct full pipeline.

    Exports three ONNX artefacts:

    * ``vision_encoder.onnx`` — Phi4MM vision tower + AvgPool2d + projector.
    * ``embedding.onnx``      — token embeddings + ScatterND image scatter.
    * ``model.onnx``          — text decoder (``inputs_embeds`` → logits).
    * ``genai_config.json``   — ``phi3v`` VLM config for onnxruntime-genai.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # --- Vision encoder ---
        self.vision_encoder = Phi4MultimodalVisionEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # --- Embedding model ---
        embed_extra_options = dict(extra_options)
        embed_extra_options["image_token_id"] = config.vision_config.image_token_id
        self.embedding_model = Phi4MultimodalEmbeddingModel(config, io_dtype, ir.DataType.FLOAT, ep, cache_dir, embed_extra_options)

        # --- Text decoder (with exclude_embeds=True) ---
        text_extra_options = dict(extra_options)
        text_extra_options["exclude_embeds"] = True
        self.text_model = Phi4MultimodalTextModel(config, io_dtype, onnx_dtype, ep, cache_dir, text_extra_options)

    # ------------------------------------------------------------------
    # builder.py interface
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        print("Building vision encoder for Phi4MultimodalForCausalLM...")
        self.vision_encoder.make_model(input_path)
        print("Building embedding model for Phi4MultimodalForCausalLM...")
        self.embedding_model.make_model(input_path)
        print("Building text decoder for Phi4MultimodalForCausalLM...")
        self.text_model.make_model(input_path)

    def save_model(self, out_dir):
        self.vision_encoder.save_model(out_dir)
        self.embedding_model.save_model(out_dir)
        self.text_model.save_model(out_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        # Let the text model write genai_config.json first, then extend it
        # with vision + embedding sections for the phi3v pipeline.
        self.text_model.make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        # onnxruntime-genai uses "phi3v" as the model type for the
        # Vision + Embedding + Decoder multimodal pipeline.
        genai_config["model"]["type"] = "phi3v"

        genai_config["model"]["vision"] = {
            "filename": self.vision_encoder.FILENAME,
            "num_img_tokens": self.vision_encoder.n_image_tokens,
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "image_features"},
        }

        genai_config["model"]["embedding"] = {
            "filename": self.embedding_model.FILENAME,
            "inputs": {"input_ids": "input_ids", "image_features": "image_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.text_model.save_processing(model_name_or_path, extra_kwargs, out_dir)
