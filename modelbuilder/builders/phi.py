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
from .base_audio import AudioEncoderModel
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
        # Q, K, V: [nc, n_patches, hidden]
        q = self.make_vis_proj(attention.q_proj, f"{b}/q_proj/MatMul", root_input)
        k = self.make_vis_proj(attention.k_proj, f"{b}/k_proj/MatMul", root_input)
        v = self.make_vis_proj(attention.v_proj, f"{b}/v_proj/MatMul", root_input)

        # Causal MultiHeadAttention (unidirectional=1 replaces the explicit upper-triangular mask).
        attn_out = self.make_vis_mha(b, q, k, v, nh, hd, d, [nc], n_p, unidirectional=1)

        # O projection (with bias).
        o = self.make_vis_proj(attention.out_proj, f"{b}/out_proj/MatMul", attn_out)

        self.layernorm_attrs["skip_input"] = o

    # ------------------------------------------------------------------
    # Single transformer layer
    # ------------------------------------------------------------------

    def make_layer(self, layer_id, layer):
        """Build one Phi4MultimodalVisionEncoderLayer.

        Pipeline: LN1 → attention → residual → LN2 → MLP → residual.
        """
        b = f"/vision/layers.{layer_id}"
        nc = self.n_crops
        n_p = self.n_patches
        d = self.vis_hidden_size
        ff = self.vis_intermediate_size

        self._make_standard_vision_layer(
            layer_id,
            layer.layer_norm1.weight,
            layer.self_attn,
            layer.layer_norm2.weight,
            lambda norm2_out: self.make_gelu_mlp(layer.mlp.fc1, layer.mlp.fc2, norm2_out, [nc, n_p, ff], f"{b}/mlp"),
            [nc, n_p, d],
            norm1_name=f"{b}/layer_norm1/LayerNorm",
            norm2_name=f"{b}/layer_norm2/LayerNorm",
            res1_name=f"{b}/residual1/Add",
            res2_name=f"{b}/residual2/Add",
            norm1_bias=layer.layer_norm1.bias,
            norm2_bias=layer.layer_norm2.bias,
            norm1_weight_name=f"vision.layers.{layer_id}.layer_norm1.weight",
            norm2_weight_name=f"vision.layers.{layer_id}.layer_norm2.weight",
            norm1_bias_name=f"vision.layers.{layer_id}.layer_norm1.bias",
            norm2_bias_name=f"vision.layers.{layer_id}.layer_norm2.bias",
        )

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

    def make_projector(self, image_embed, x):
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
        image_features = self.make_projector(image_embed, x)

        # Graph output.
        self.make_node("Identity", inputs=[image_features], outputs=["image_features"], name="/vision/output/Identity")
        out_val = self.make_value("image_features", self.io_dtype, shape=[self.n_image_tokens, self.text_hidden_size])
        self.graph.outputs.append(out_val)

        self.graph.sort()


class Phi4MultimodalAudioEncoderModel(AudioEncoderModel):
    """ONNX graph builder for the Phi-4-multimodal audio encoder.

    Exports the Conformer-based audio tower to ``audio_encoder.onnx``.

    Inputs
    ------
    audio_features : float32 [1, n_frames, input_size]
        Mel-spectrogram features for a single audio item (variable length).

    Outputs
    -------
    audio_features : io_dtype [n_audio_tokens, text_hidden_size]
        Projected audio features suitable for injection into text embeddings
        at ``audio_token_id`` placeholder positions.
    """

    FILENAME = "audio_encoder.onnx"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        ac = config.audio_config

        audio_config = copy.deepcopy(config)
        audio_config.hidden_size = ac.hidden_size
        audio_config.intermediate_size = ac.intermediate_size
        audio_config.num_attention_heads = ac.num_attention_heads
        audio_config.num_key_value_heads = ac.num_attention_heads
        audio_config.num_hidden_layers = ac.num_blocks
        audio_config.head_dim = ac.hidden_size // ac.num_attention_heads
        audio_config.hidden_act = "silu"
        audio_config.vocab_size = 1
        audio_config.max_position_embeddings = 1
        audio_config.rms_norm_eps = 1e-5
        audio_config.rope_scaling = None

        audio_extra_options = {**extra_options, "filename": self.FILENAME, "exclude_lm_head": True, "exclude_embeds": True}
        super().__init__(audio_config, io_dtype, onnx_dtype, ep, cache_dir, audio_extra_options)

        self.graph.name = "phi4mm_audio_encoder"
        self.config = config
        self.audio_config = ac

        self.input_size = ac.input_size  # 80
        self.audio_hidden_size = ac.hidden_size  # 1024
        self.num_heads = ac.num_attention_heads  # 16
        self.head_dim = ac.hidden_size // ac.num_attention_heads  # 64
        self.audio_num_layers = self.num_layers  # may be reduced by num_hidden_layers option
        self.audio_intermediate_size = ac.intermediate_size  # 1536
        self.nemo_conv_channels = ac.nemo_conv_channels  # 1024
        self.nemo_final_size = ac.nemo_final_size  # 10
        self.ext_pw_out_channel = ac.ext_pw_out_channel  # 1024
        self.depthwise_sep_out = ac.depthwise_separable_out_channel  # 1024
        self.kernel_size = ac.kernel_size  # 3
        self.bias_max_distance = ac.bias_max_distance  # 1000
        self.num_buckets = 2 * ac.bias_max_distance  # 2000 (bias_symmetric=False)
        self.text_hidden_size = config.hidden_size

    # ------------------------------------------------------------------
    # Encoder embedding (mean-variance normalization)
    # ------------------------------------------------------------------

    def _make_encoder_embedding(self, encoder_embedding, audio_in):
        """(x - global_mean) * global_invstd.  Input/output: [1, T, input_size]."""
        mean_name = "audio.encoder_embedding.global_mean"
        invstd_name = "audio.encoder_embedding.global_invstd"
        self.make_initializer(encoder_embedding.global_mean.detach(), mean_name, to=self.io_dtype)
        self.make_initializer(encoder_embedding.global_invstd.detach(), invstd_name, to=self.io_dtype)

        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast("/audio/encoder_embed/Cast", audio_in, self.io_dtype, [1, None, self.input_size])
            audio_in = "/audio/encoder_embed/Cast/output_0"

        sub_out = "/audio/encoder_embed/Sub/output_0"
        self.make_node("Sub", inputs=[audio_in, mean_name], outputs=[sub_out], name="/audio/encoder_embed/Sub")
        self.make_value(sub_out, self.io_dtype, shape=[1, None, self.input_size])

        mul_out = "/audio/encoder_embed/Mul/output_0"
        self.make_node("Mul", inputs=[sub_out, invstd_name], outputs=[mul_out], name="/audio/encoder_embed/Mul")
        self.make_value(mul_out, self.io_dtype, shape=[1, None, self.input_size])
        return mul_out

    # ------------------------------------------------------------------
    # NemoConvSubsampling: 3-stage strided Conv2d → linear
    # ------------------------------------------------------------------

    def _make_nemo_subsampling(self, embed, audio_in):
        """NemoConvSubsampling: [1, T, 80] → [1, T/8, hidden_size]."""
        nc = self.nemo_conv_channels  # 1024
        nf = self.nemo_final_size  # 10
        d = self.audio_hidden_size  # 1024

        # Unsqueeze: [1, T, 80] → [1, 1, T, 80]
        ax1 = self._make_constant_i64_1d("/audio/nemo/unsq_ax", [1])
        self.make_unsqueeze("/audio/nemo/unsqueeze", [audio_in, ax1], self.io_dtype, [1, 1, None, self.input_size])
        x = "/audio/nemo/unsqueeze/output_0"

        def _make_conv2d(tag, conv, inp, out_shape):
            w_name = f"audio.embed.conv.{tag}.weight"
            self.make_initializer(conv.weight.detach(), w_name, to=self.io_dtype)
            inputs = [inp, w_name]
            if conv.bias is not None:
                b_name = f"audio.embed.conv.{tag}.bias"
                self.make_initializer(conv.bias.detach(), b_name, to=self.io_dtype)
                inputs.append(b_name)
            stride = conv.stride if isinstance(conv.stride, int) else conv.stride[0]
            pad = conv.padding if isinstance(conv.padding, int) else conv.padding[0]
            k = conv.kernel_size if isinstance(conv.kernel_size, int) else conv.kernel_size[0]
            groups = conv.groups
            self.make_conv(
                f"/audio/nemo/conv{tag}",
                inputs,
                self.io_dtype,
                out_shape,
                dilations=[1, 1],
                group=groups,
                kernel_shape=[k, k],
                pads=[pad, pad, pad, pad],
                strides=[stride, stride],
            )
            return f"/audio/nemo/conv{tag}/output_0"

        # Stage 1: Conv2d(1, nc, 3, stride=2, pad=1) + ReLU
        x = _make_conv2d("0", embed.conv[0], x, [1, nc, None, 40])
        relu0_out = "/audio/nemo/relu0/output_0"
        self.make_node("Relu", inputs=[x], outputs=[relu0_out], name="/audio/nemo/relu0")
        self.make_value(relu0_out, self.io_dtype, shape=[1, nc, None, 40])
        x = relu0_out

        # Stage 2: DW + PW + ReLU
        x = _make_conv2d("2", embed.conv[2], x, [1, nc, None, 20])
        x = _make_conv2d("3", embed.conv[3], x, [1, nc, None, 20])
        relu2_out = "/audio/nemo/relu2/output_0"
        self.make_node("Relu", inputs=[x], outputs=[relu2_out], name="/audio/nemo/relu2")
        self.make_value(relu2_out, self.io_dtype, shape=[1, nc, None, 20])
        x = relu2_out

        # Stage 3: DW + PW + ReLU
        x = _make_conv2d("5", embed.conv[5], x, [1, nc, None, nf])
        x = _make_conv2d("6", embed.conv[6], x, [1, nc, None, nf])
        relu5_out = "/audio/nemo/relu5/output_0"
        self.make_node("Relu", inputs=[x], outputs=[relu5_out], name="/audio/nemo/relu5")
        self.make_value(relu5_out, self.io_dtype, shape=[1, nc, None, nf])
        x = relu5_out

        # Transpose: [1, nc, T/8, nf] → [1, T/8, nc, nf]
        x = self.make_transpose("/audio/nemo/transpose", x, self.io_dtype, [1, None, nc, nf], perm=[0, 2, 1, 3])

        # Reshape: [1, T/8, nc, nf] → [1, T/8, nc*nf] (0 = copy T/8 dim)
        x = self.make_reshape("/audio/nemo/reshape", [x, [1, 0, nc * nf]], self.io_dtype, [1, None, nc * nf])

        # Linear out: [1, T/8, nc*nf] → [1, T/8, d]
        x = self._make_audio_linear("/audio/nemo/out", embed.out, x, [1, None, d])
        return x

    # ------------------------------------------------------------------
    # Relative attention bias (dynamic, computed once per forward pass)
    # ------------------------------------------------------------------

    def _make_relative_attention_bias(self, encoder, hidden):
        """Build [1, num_heads, n, n] relative attention bias tensor.

        All computations are in int64; the final Gather result is io_dtype.
        """
        nh = self.num_heads

        # Store bias_values embedding: [num_buckets, num_heads]
        bias_val_name = "audio.relative_attention_bias.weight"
        self.make_initializer(encoder.relative_attention_bias_layer.bias_values.weight.detach(), bias_val_name, to=self.io_dtype)

        # Get sequence length n from hidden [1, n, d]
        self.make_shape("/audio/rel_bias/hidden_shape", hidden, [3])
        n_idx = self._make_constant_i64("/audio/rel_bias/n_idx", 1)
        self.make_gather("/audio/rel_bias/n_scalar", ["/audio/rel_bias/hidden_shape/output_0", n_idx], ir.DataType.INT64, [], axis=0)

        zero = self._make_constant_i64("/audio/rel_bias/zero", 0)
        one = self._make_constant_i64("/audio/rel_bias/one", 1)
        self.make_range("/audio/rel_bias/range_n", [zero, "/audio/rel_bias/n_scalar/output_0", one], ir.DataType.INT64, [None])

        # Reshape to [n, 1] (context) and [1, n] (memory)
        ctx = self.make_reshape("/audio/rel_bias/ctx", ["/audio/rel_bias/range_n/output_0", [-1, 1]], ir.DataType.INT64, [None, 1])
        mem = self.make_reshape("/audio/rel_bias/mem", ["/audio/rel_bias/range_n/output_0", [1, -1]], ir.DataType.INT64, [1, None])

        # rel_pos = memory - context: [n, n]
        rel_pos = "/audio/rel_bias/rel_pos/output_0"
        self.make_node("Sub", inputs=[mem, ctx], outputs=[rel_pos], name="/audio/rel_bias/rel_pos/Sub")
        self.make_value(rel_pos, ir.DataType.INT64, shape=[None, None])

        # Clip to [-bias_max_distance, bias_max_distance - 1]
        clip_min = self._make_constant_i64("/audio/rel_bias/clip_min", -self.bias_max_distance)
        clip_max = self._make_constant_i64("/audio/rel_bias/clip_max", self.bias_max_distance - 1)
        rel_clip = "/audio/rel_bias/rel_clip/output_0"
        self.make_node("Clip", inputs=[rel_pos, clip_min, clip_max], outputs=[rel_clip], name="/audio/rel_bias/rel_clip/Clip")
        self.make_value(rel_clip, ir.DataType.INT64, shape=[None, None])

        # bias_idx = rel_clip + num_buckets // 2: indices into [0, num_buckets)
        half_bkt = self._make_constant_i64("/audio/rel_bias/half_bkt", self.num_buckets // 2)
        bias_idx = self.make_add("/audio/rel_bias/bias_idx", [rel_clip, half_bkt], ir.DataType.INT64, [None, None])

        # Gather: [n, n] → [n, n, num_heads]
        att_bias_raw = "/audio/rel_bias/att_bias_raw/output_0"
        self.make_node(
            "Gather", inputs=[bias_val_name, bias_idx], outputs=[att_bias_raw], name="/audio/rel_bias/att_bias_raw/Gather", axis=0
        )
        self.make_value(att_bias_raw, self.io_dtype, shape=[None, None, nh])

        # Transpose: [n, n, nh] → [nh, n, n]
        att_bias_t = self.make_transpose("/audio/rel_bias/att_bias_t", att_bias_raw, self.io_dtype, [nh, None, None], perm=[2, 0, 1])

        # Unsqueeze batch: [nh, n, n] → [1, nh, n, n]
        bax = self._make_constant_i64_1d("/audio/rel_bias/bax", [0])
        self.make_unsqueeze("/audio/rel_bias/att_bias", [att_bias_t, bax], self.io_dtype, [1, nh, None, None])
        return "/audio/rel_bias/att_bias/output_0"

    # ------------------------------------------------------------------
    # AudioMLP (feed_forward_in / feed_forward_out)
    # ------------------------------------------------------------------

    def _make_audio_mlp(self, name, mlp, root_input):
        """Conformer FFN: LayerNorm → gate_up_proj → split → swish-gate → down_proj.

        Input/output shape: [1, n, audio_hidden_size].
        """
        d = self.audio_hidden_size
        mid = self.audio_intermediate_size

        # LayerNorm
        ln_out = self.make_audio_layer_norm(f"{name}/layer_norm", root_input, mlp.layer_norm.weight, mlp.layer_norm.bias, [1, None, d])

        # gate_up_proj: [1, n, d] → [1, n, 2*mid]
        combined = self._make_audio_linear(f"{name}/gate_up_proj", mlp.gate_up_proj, ln_out, [1, None, 2 * mid])

        # Split into gate and up: each [1, n, mid]
        split_g = f"{name}/Split/output_0"
        split_u = f"{name}/Split/output_1"
        split_sizes = ir.Tensor(np.array([mid, mid], dtype=np.int64), name=f"{name}/split_sizes")
        self.make_node("Constant", inputs=[], outputs=[f"{name}/split_sizes"], name=f"{name}/split_sizes/Constant", value=split_sizes)
        self.make_value(f"{name}/split_sizes", ir.DataType.INT64, shape=[2])
        self.make_node("Split", inputs=[combined, f"{name}/split_sizes"], outputs=[split_g, split_u], name=f"{name}/Split", axis=-1)
        self.make_value(split_g, self.io_dtype, shape=[1, None, mid])
        self.make_value(split_u, self.io_dtype, shape=[1, None, mid])

        # gate_act = swish(gate): [1, n, mid]
        gate_act = self._make_swish(f"{name}/swish", split_g, [1, None, mid])

        # up * gate_act: [1, n, mid]
        gated = self.make_mul(f"{name}/gated_mul", [split_u, gate_act], self.io_dtype, [1, None, mid])

        # down_proj: [1, n, mid] → [1, n, d]
        return self._make_audio_linear(f"{name}/down_proj", mlp.down_proj, gated, [1, None, d])

    # ------------------------------------------------------------------
    # GLU pointwise conv (part of ConvModule)
    # ------------------------------------------------------------------

    def _make_glu(self, name, glu, root_input):
        """GluPointWiseConv.

        Input:  [1, n, ext_pw_out_channel]
        Output: [1, n, ext_pw_out_channel]
        """
        ep = self.ext_pw_out_channel
        prefix = name[1:].replace("/", ".")

        # Transpose: [1, n, ep] → [1, ep, n]
        h = self.make_transpose(f"{name}/perm_in", root_input, self.io_dtype, [1, ep, None], perm=[0, 2, 1])

        # ext_pw_conv_1d: Conv1d(ep, ep*2, 1) → [1, ep*2, n]
        w_name = f"{prefix}.ext_pw_conv_1d.weight"
        self.make_initializer(glu.ext_pw_conv_1d.weight.detach(), w_name, to=self.io_dtype)
        inputs = [h, w_name]
        if glu.ext_pw_conv_1d.bias is not None:
            b_name = f"{prefix}.ext_pw_conv_1d.bias"
            self.make_initializer(glu.ext_pw_conv_1d.bias.detach(), b_name, to=self.io_dtype)
            inputs.append(b_name)
        self.make_conv(
            f"{name}/ext_pw_conv_1d",
            inputs,
            self.io_dtype,
            [1, ep * 2, None],
            dilations=[1],
            group=1,
            kernel_shape=[1],
            pads=[0, 0],
            strides=[1],
        )
        conv_out = f"{name}/ext_pw_conv_1d/output_0"

        # Split: [1, ep*2, n] → left [1, ep, n], right [1, ep, n]
        split_l = f"{name}/Split/output_0"
        split_r = f"{name}/Split/output_1"
        glu_split_sizes = ir.Tensor(np.array([ep, ep], dtype=np.int64), name=f"{name}/glu_split_sizes")
        self.make_node(
            "Constant", inputs=[], outputs=[f"{name}/glu_split_sizes"], name=f"{name}/glu_split_sizes/Constant", value=glu_split_sizes
        )
        self.make_value(f"{name}/glu_split_sizes", ir.DataType.INT64, shape=[2])
        self.make_node("Split", inputs=[conv_out, f"{name}/glu_split_sizes"], outputs=[split_l, split_r], name=f"{name}/Split", axis=1)
        self.make_value(split_l, self.io_dtype, shape=[1, ep, None])
        self.make_value(split_r, self.io_dtype, shape=[1, ep, None])

        # b1, b2: [1, ep, 1]
        b1_name = f"{prefix}.b1"
        b2_name = f"{prefix}.b2"
        self.make_initializer(glu.b1.detach(), b1_name, to=self.io_dtype)
        self.make_initializer(glu.b2.detach(), b2_name, to=self.io_dtype)

        # (left + b1)
        lb = self.make_add(f"{name}/left_bias", [split_l, b1_name], self.io_dtype, [1, ep, None])
        # swish(right + b2)
        rb = self.make_add(f"{name}/right_bias", [split_r, b2_name], self.io_dtype, [1, ep, None])
        rb_act = self._make_swish(f"{name}/right_swish", rb, [1, ep, None])

        # (left + b1) * swish(right + b2)
        out_c = self.make_mul(f"{name}/glu_mul", [lb, rb_act], self.io_dtype, [1, ep, None])

        # Transpose back: [1, ep, n] → [1, n, ep]
        return self.make_transpose(f"{name}/perm_out", out_c, self.io_dtype, [1, None, ep], perm=[0, 2, 1])

    # ------------------------------------------------------------------
    # DepthWiseSeparableConv1d (inside ConvModule)
    # ------------------------------------------------------------------

    def _make_dw_sep_conv1d(self, name, dw_sep, root_input):
        """DW + PW Conv1d with causal left-only padding trimming.

        Input/output: [1, n, ext_pw_out_channel] (after transposing internally).
        Returns tensor of shape [1, n, depthwise_sep_out].
        """
        ep = self.ext_pw_out_channel
        ds = self.depthwise_sep_out
        k = self.kernel_size  # 3
        pad = k - 1  # 2
        prefix = name[1:].replace("/", ".")

        # Transpose input: [1, n, ep] → [1, ep, n]
        h = self.make_transpose(f"{name}/perm_in", root_input, self.io_dtype, [1, ep, None], perm=[0, 2, 1])

        # DW Conv1d: [1, ep, n] → [1, ep, n + 2*pad - (k-1)] = [1, ep, n + pad]
        dw_w = f"{prefix}.dw_conv.weight"
        self.make_initializer(dw_sep.dw_conv.weight.detach(), dw_w, to=self.io_dtype)
        dw_inputs = [h, dw_w]
        if dw_sep.dw_conv.bias is not None:
            dw_b = f"{prefix}.dw_conv.bias"
            self.make_initializer(dw_sep.dw_conv.bias.detach(), dw_b, to=self.io_dtype)
            dw_inputs.append(dw_b)
        self.make_conv(
            f"{name}/dw_conv",
            dw_inputs,
            self.io_dtype,
            [1, ep, None],
            dilations=[1],
            group=ep,
            kernel_shape=[k],
            pads=[pad, pad],
            strides=[1],
        )
        dw_out = f"{name}/dw_conv/output_0"

        # Causal trim: the DW conv uses symmetric padding [pad, pad], producing
        # n + (k-1) output frames.  Remove the right-padded (k-1) tail to obtain
        # exactly n frames, emulating left-only (causal) padding.
        dw_trimmed = self.make_slice(f"{name}/dw_trim", dw_out, self.io_dtype, [1, ep, None], starts=[0], ends=[-(k - 1)], axes=[2])

        # PW Conv1d: [1, ep, n] → [1, ds, n]
        pw_w = f"{prefix}.pw_conv.weight"
        self.make_initializer(dw_sep.pw_conv.weight.detach(), pw_w, to=self.io_dtype)
        pw_inputs = [dw_trimmed, pw_w]
        if dw_sep.pw_conv.bias is not None:
            pw_b = f"{prefix}.pw_conv.bias"
            self.make_initializer(dw_sep.pw_conv.bias.detach(), pw_b, to=self.io_dtype)
            pw_inputs.append(pw_b)
        self.make_conv(
            f"{name}/pw_conv", pw_inputs, self.io_dtype, [1, ds, None], dilations=[1], group=1, kernel_shape=[1], pads=[0, 0], strides=[1]
        )
        pw_out = f"{name}/pw_conv/output_0"

        # Transpose back: [1, ds, n] → [1, n, ds]
        return self.make_transpose(f"{name}/perm_out", pw_out, self.io_dtype, [1, None, ds], perm=[0, 2, 1])

    # ------------------------------------------------------------------
    # AudioConvModule (conv sub-block of each Conformer layer)
    # ------------------------------------------------------------------

    def _make_audio_conv_module(self, name, conv_mod, root_input):
        """Conformer conv module.

        Input/output: [1, n, audio_hidden_size].
        """
        ep = self.ext_pw_out_channel
        ds = self.depthwise_sep_out
        prefix = name[1:].replace("/", ".")

        # LayerNorm
        ln_out = self.make_audio_layer_norm(
            f"{name}/layer_norm", root_input, conv_mod.layer_norm.weight, conv_mod.layer_norm.bias, [1, None, ep]
        )

        # GLU
        glu_out = self._make_glu(f"{name}/glu", conv_mod.glu, ln_out)

        # DW sep conv1d
        dw_out = self._make_dw_sep_conv1d(f"{name}/dw_sep", conv_mod.dw_sep_conv_1d, glu_out)

        # swish
        sw_out = self._make_swish(f"{name}/swish_after_dw", dw_out, [1, None, ds])

        # ext_pw_conv_1d: [1, n, ds] → via Conv1d(ds, ep, 1)
        # transpose to [1, ds, n]
        h = self.make_transpose(f"{name}/ext_perm_in", sw_out, self.io_dtype, [1, ds, None], perm=[0, 2, 1])
        ext_w = f"{prefix}.ext_pw_conv_1d.weight"
        self.make_initializer(conv_mod.ext_pw_conv_1d.weight.detach(), ext_w, to=self.io_dtype)
        ext_inputs = [h, ext_w]
        if conv_mod.ext_pw_conv_1d.bias is not None:
            ext_b = f"{prefix}.ext_pw_conv_1d.bias"
            self.make_initializer(conv_mod.ext_pw_conv_1d.bias.detach(), ext_b, to=self.io_dtype)
            ext_inputs.append(ext_b)
        self.make_conv(
            f"{name}/ext_pw_conv_1d",
            ext_inputs,
            self.io_dtype,
            [1, ep, None],
            dilations=[1],
            group=1,
            kernel_shape=[1],
            pads=[0, 0],
            strides=[1],
        )
        ext_out = f"{name}/ext_pw_conv_1d/output_0"

        # Transpose back: [1, ep, n] → [1, n, ep]
        return self.make_transpose(f"{name}/ext_perm_out", ext_out, self.io_dtype, [1, None, ep], perm=[0, 2, 1])

    # ------------------------------------------------------------------
    # Conformer self-attention with relative bias
    # ------------------------------------------------------------------

    def _make_audio_attention(self, name, attn, att_bias, root_input):
        """Conformer self-attention.

        Input/output: [1, n, audio_hidden_size].
        att_bias: [1, num_heads, n, n] relative attention bias.
        """
        d = self.audio_hidden_size
        nh = self.num_heads
        hd = self.head_dim
        scale = float(hd) ** -0.5

        # Q, K, V projections: [1, n, d] → [1, n, d]
        q = self._make_audio_linear(f"{name}/q_proj", attn.q_proj, root_input, [1, None, d])
        k = self._make_audio_linear(f"{name}/k_proj", attn.k_proj, root_input, [1, None, d])
        v = self._make_audio_linear(f"{name}/v_proj", attn.v_proj, root_input, [1, None, d])

        # Reshape: [1, n, d] → [1, n, nh, hd]
        q4 = self.make_reshape(f"{name}/q_reshape", [q, [1, 0, nh, hd]], self.io_dtype, [1, None, nh, hd])
        k4 = self.make_reshape(f"{name}/k_reshape", [k, [1, 0, nh, hd]], self.io_dtype, [1, None, nh, hd])
        v4 = self.make_reshape(f"{name}/v_reshape", [v, [1, 0, nh, hd]], self.io_dtype, [1, None, nh, hd])

        # Transpose: [1, n, nh, hd] → [1, nh, n, hd]
        qt = self.make_transpose(f"{name}/q_t", q4, self.io_dtype, [1, nh, None, hd], perm=[0, 2, 1, 3])
        kt = self.make_transpose(f"{name}/k_t", k4, self.io_dtype, [1, nh, None, hd], perm=[0, 2, 1, 3])
        vt = self.make_transpose(f"{name}/v_t", v4, self.io_dtype, [1, nh, None, hd], perm=[0, 2, 1, 3])

        # K transposed for matmul: [1, nh, hd, n]
        ktT = self.make_transpose(f"{name}/kt_T", kt, self.io_dtype, [1, nh, hd, None], perm=[0, 1, 3, 2])

        # Attention logits: Q @ K^T → [1, nh, n, n]
        attn_logits = f"{name}/attn_logits/output_0"
        self.make_node("MatMul", inputs=[qt, ktT], outputs=[attn_logits], name=f"{name}/attn_logits/MatMul")
        self.make_value(attn_logits, self.io_dtype, shape=[1, nh, None, None])

        # Scale
        np_dtype = np.float32 if self.io_dtype in (ir.DataType.FLOAT, ir.DataType.BFLOAT16) else np.float16
        scale_t = ir.Tensor(np.array(scale, dtype=np_dtype), name=f"{name}/scale")
        self.make_node("Constant", inputs=[], outputs=[f"{name}/scale"], name=f"{name}/scale/Constant", value=scale_t)
        self.make_value(f"{name}/scale", self.io_dtype, shape=[])
        attn_scaled = self.make_mul(f"{name}/attn_scale", [attn_logits, f"{name}/scale"], self.io_dtype, [1, nh, None, None])

        # Add relative attention bias
        attn_biased = self.make_add(f"{name}/attn_bias_add", [attn_scaled, att_bias], self.io_dtype, [1, nh, None, None])

        # Softmax
        attn_probs = self.make_softmax(f"{name}/softmax", attn_biased, self.io_dtype, [1, nh, None, None])

        # Attn @ V: [1, nh, n, n] @ [1, nh, n, hd] → [1, nh, n, hd]
        attn_out = f"{name}/attn_out/output_0"
        self.make_node("MatMul", inputs=[attn_probs, vt], outputs=[attn_out], name=f"{name}/attn_out/MatMul")
        self.make_value(attn_out, self.io_dtype, shape=[1, nh, None, hd])

        # Transpose back: [1, nh, n, hd] → [1, n, nh, hd]
        attn_back = self.make_transpose(f"{name}/attn_back", attn_out, self.io_dtype, [1, None, nh, hd], perm=[0, 2, 1, 3])

        # Flatten: [1, n, nh, hd] → [1, n, d]
        attn_flat = self.make_reshape(f"{name}/attn_flat", [attn_back, [1, 0, d]], self.io_dtype, [1, None, d])

        # o_proj
        return self._make_audio_linear(f"{name}/o_proj", attn.o_proj, attn_flat, [1, None, d])

    # ------------------------------------------------------------------
    # One Conformer encoder layer
    # ------------------------------------------------------------------

    def _make_conformer_layer(self, i, layer, hidden, att_bias, half_scale):
        """Build one Phi4MultimodalConformerEncoderLayer.

        Pipeline:
          residual = x + 0.5 * feed_forward_in(x)
          h = layer_norm_att(residual)
          h = residual + self_attn(h, att_bias)
          h = h + conv(h)
          h = h + 0.5 * feed_forward_out(h)
          out = layer_norm(h)
        """
        d = self.audio_hidden_size
        b = f"/audio/encoders.{i}"

        # 0.5 * feed_forward_in
        ffin_out = self._make_audio_mlp(f"{b}/feed_forward_in", layer.feed_forward_in, hidden)
        ffin_scaled = self.make_mul(f"{b}/ffin_scale", [ffin_out, half_scale], self.io_dtype, [1, None, d])
        residual = self.make_add(f"{b}/res_ffin", [hidden, ffin_scaled], self.io_dtype, [1, None, d])

        # Attention
        attn_ln = self.make_audio_layer_norm(
            f"{b}/layer_norm_att", residual, layer.layer_norm_att.weight, layer.layer_norm_att.bias, [1, None, d]
        )
        attn_out = self._make_audio_attention(f"{b}/self_attn", layer.self_attn, att_bias, attn_ln)
        h = self.make_add(f"{b}/res_attn", [residual, attn_out], self.io_dtype, [1, None, d])

        # Conv module
        conv_out = self._make_audio_conv_module(f"{b}/conv", layer.conv, h)
        h = self.make_add(f"{b}/res_conv", [h, conv_out], self.io_dtype, [1, None, d])

        # 0.5 * feed_forward_out
        ffout_out = self._make_audio_mlp(f"{b}/feed_forward_out", layer.feed_forward_out, h)
        ffout_scaled = self.make_mul(f"{b}/ffout_scale", [ffout_out, half_scale], self.io_dtype, [1, None, d])
        h = self.make_add(f"{b}/res_ffout", [h, ffout_scaled], self.io_dtype, [1, None, d])

        # Final LayerNorm
        return self.make_audio_layer_norm(f"{b}/layer_norm", h, layer.layer_norm.weight, layer.layer_norm.bias, [1, None, d])

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_hf_model(self, input_path):
        from transformers import Phi4MultimodalForCausalLM

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Phi4MultimodalForCausalLM.from_pretrained(src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        """Load HF weights and build the audio encoder ONNX graph."""
        hf_model = self.load_hf_model(input_path)
        hf_model.eval()

        audio_embed = hf_model.model.embed_tokens_extend.audio_embed
        encoder = audio_embed.encoder

        th = self.text_hidden_size

        # Graph input: float32 [1, T, input_size]
        in_val = self.make_value("audio_features", ir.DataType.FLOAT, shape=[1, None, self.input_size])
        self.graph.inputs.append(in_val)

        # 1. Encoder embedding: mean-variance normalization
        hidden = self._make_encoder_embedding(encoder.encoder_embedding, "audio_features")

        # 2. NemoConvSubsampling: [1, T, 80] → [1, T/8, d]
        hidden = self._make_nemo_subsampling(encoder.embed, hidden)

        # 3. Relative attention bias: [1, num_heads, n, n]  (dynamic, computed once)
        att_bias = self._make_relative_attention_bias(encoder, hidden)

        # Shared 0.5 scale initializer (reused by all Conformer layers)
        np_dtype = np.float32 if self.io_dtype in (ir.DataType.FLOAT, ir.DataType.BFLOAT16) else np.float16
        half_t = ir.Tensor(np.array(0.5, dtype=np_dtype), name="audio.half_scale")
        self.make_node("Constant", inputs=[], outputs=["audio.half_scale"], name="audio.half_scale/Constant", value=half_t)
        self.make_value("audio.half_scale", self.io_dtype, shape=[])

        # 4. Conformer encoder layers
        for i in range(self.audio_num_layers):
            hidden = self._make_conformer_layer(i, encoder.encoders[i], hidden, att_bias, "audio.half_scale")

        # 5. Projection: up_proj_for_speech + GELU + down_proj_for_speech
        # up_proj: [1, n, d] → [1, n, th]
        up = self._make_audio_linear("/audio/proj/up", audio_embed.up_proj_for_speech, hidden, [1, None, th])
        # GELU: com.microsoft Gelu uses the tanh approximation of GELU (same as
        # torch.nn.functional.gelu(..., approximate='tanh')).
        gelu_out = "/audio/proj/gelu/output_0"
        self.make_node("Gelu", inputs=[up], outputs=[gelu_out], name="/audio/proj/gelu/Gelu", domain="com.microsoft")
        self.make_value(gelu_out, self.io_dtype, shape=[1, None, th])
        # down_proj: [1, n, th] → [1, n, th]
        down = self._make_audio_linear("/audio/proj/down", audio_embed.down_proj_for_speech, gelu_out, [1, None, th])

        # 6. Squeeze batch: [1, n, th] → [n, th]
        bax = self._make_constant_i64_1d("/audio/output/bax", [0])
        self.make_squeeze("/audio/output/squeeze", [down, bax], self.io_dtype, [None, th])
        squeezed = "/audio/output/squeeze/output_0"

        # Graph output — named differently from the graph input ("audio_features")
        self.make_node("Identity", inputs=[squeezed], outputs=["audio_features_out"], name="/audio/output/Identity")
        out_val = self.make_value("audio_features_out", self.io_dtype, shape=[None, th])
        self.graph.outputs.append(out_val)

        self.graph.sort()


class Phi4MultimodalEmbeddingModel(EmbeddingModel):
    """ONNX embedding model for the Phi-4-multimodal-instruct phi3v pipeline.

    Extends the base ScatterND embedding graph to handle both image and audio
    token placeholders.  Image features are scattered at ``image_token_id``
    positions; audio features are scattered at ``audio_token_id`` positions.

    Graph (2-D ``input_ids [1, T]`` from ORT-GenAI's ``EmbeddingState``)::

        text_embeds   = Gather(embed_tokens_weight, input_ids)   # [1, T, H]
        text_2d       = Squeeze(text_embeds, [0])                # [T, H]
        flat_ids      = Squeeze(input_ids, [0])                  # [T]
        is_img        = Equal(flat_ids, image_token_id_const)    # [T] bool
        img_pos       = NonZero(is_img)                          # [1, N_img]
        img_pos_idx   = Transpose(img_pos, [1, 0])               # [N_img, 1]
        scattered_img = ScatterND(text_2d, img_pos_idx,
                                  image_features)                # [T, H]
        is_audio      = Equal(flat_ids, audio_token_id_const)    # [T] bool
        aud_pos       = NonZero(is_audio)                        # [1, N_aud]
        aud_pos_idx   = Transpose(aud_pos, [1, 0])               # [N_aud, 1]
        scattered_2d  = ScatterND(scattered_img, aud_pos_idx,
                                  audio_features)                # [T, H]
        inputs_embeds = Unsqueeze(scattered_2d, [0])             # [1, T, H]
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        audio_config = getattr(config, "audio_config", None)
        default_audio_token_id = audio_config.audio_token_id if audio_config is not None else 200011
        self.audio_token_id = extra_options.get("audio_token_id", default_audio_token_id)

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

    def make_model(self, input_path):
        """Load HF weights and build the embedding + scatter ONNX graph."""
        hf_model = self.load_hf_model(input_path)
        hf_model.eval()
        embed_weight = self.get_embed_weight(hf_model)

        # Initializers
        self.make_initializer(embed_weight, name="embed_tokens_weight")
        self.make_initializer(np.array(self.image_token_id, dtype=np.int64), name="image_token_id_const")
        self.make_initializer(np.array(self.audio_token_id, dtype=np.int64), name="audio_token_id_const")

        _squeeze_axes = ir.Tensor(np.array([0], dtype=np.int64), name="squeeze_batch_axes")
        self.make_node(
            "Constant", inputs=[], outputs=["squeeze_batch_axes"], name="/embed/squeeze_batch_axes/Constant", value=_squeeze_axes
        )
        self.make_value("squeeze_batch_axes", ir.DataType.INT64, shape=[1])

        # Graph inputs
        self.graph.inputs.append(self.make_value("input_ids", ir.DataType.INT64, shape=[None, None]))
        self.graph.inputs.append(self.make_value("image_features", self.io_dtype, shape=[None, self.hidden_size]))
        self.graph.inputs.append(self.make_value("audio_features", self.io_dtype, shape=[None, self.hidden_size]))

        # 1. Embed all tokens: [1, T] → [1, T, H] (float32)
        self.make_node("Gather", inputs=["embed_tokens_weight", "input_ids"], outputs=["text_embeds"], name="/embed/Gather", axis=0)
        # 2. Squeeze batch dim: [1, T, H] → [T, H]
        self.make_node("Squeeze", inputs=["text_embeds", "squeeze_batch_axes"], outputs=["text_2d_fp32"], name="/embed/Squeeze_3d")
        # 3. Cast to io_dtype
        self.make_cast("/embed/Cast_text_2d", "text_2d_fp32", self.io_dtype, [None, self.hidden_size])
        # 4. Flatten input_ids: [1, T] → [T]
        self.make_node("Squeeze", inputs=["input_ids", "squeeze_batch_axes"], outputs=["flat_ids"], name="/embed/Squeeze_ids")

        # 5. Scatter image features at image_token_id positions
        self.make_node("Equal", inputs=["flat_ids", "image_token_id_const"], outputs=["is_image"], name="/embed/Equal_img")
        self.make_node("NonZero", inputs=["is_image"], outputs=["img_pos"], name="/embed/NonZero_img")
        self.make_node("Transpose", inputs=["img_pos"], outputs=["img_pos_idx"], name="/embed/Transpose_img", perm=[1, 0])
        self.make_node(
            "ScatterND",
            inputs=["/embed/Cast_text_2d/output_0", "img_pos_idx", "image_features"],
            outputs=["scattered_img"],
            name="/embed/ScatterND_img",
        )

        # 6. Scatter audio features at audio_token_id positions
        self.make_node("Equal", inputs=["flat_ids", "audio_token_id_const"], outputs=["is_audio"], name="/embed/Equal_aud")
        self.make_node("NonZero", inputs=["is_audio"], outputs=["aud_pos"], name="/embed/NonZero_aud")
        self.make_node("Transpose", inputs=["aud_pos"], outputs=["aud_pos_idx"], name="/embed/Transpose_aud", perm=[1, 0])
        self.make_node(
            "ScatterND", inputs=["scattered_img", "aud_pos_idx", "audio_features"], outputs=["scattered_2d"], name="/embed/ScatterND_aud"
        )

        # 7. Re-add batch dimension: [T, H] → [1, T, H]
        self.make_node("Unsqueeze", inputs=["scattered_2d", "squeeze_batch_axes"], outputs=["inputs_embeds"], name="/embed/Unsqueeze")

        # Graph output
        self.graph.outputs.append(self.make_value("inputs_embeds", self.io_dtype, shape=[1, None, self.hidden_size]))

        self.graph.sort()


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

    Exports four ONNX artefacts:

    * ``vision_encoder.onnx`` — Phi4MM vision tower + AvgPool2d + projector.
    * ``audio_encoder.onnx``  — Phi4MM Conformer audio tower + projector.
    * ``embedding.onnx``      — token embeddings + ScatterND image/audio scatter.
    * ``model.onnx``          — text decoder (``inputs_embeds`` → logits).
    * ``genai_config.json``   — ``phi3v`` VLM config for onnxruntime-genai.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # --- Vision encoder ---
        self.vision_encoder = Phi4MultimodalVisionEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # --- Audio encoder ---
        self.audio_encoder = Phi4MultimodalAudioEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # --- Embedding model ---
        embed_extra_options = dict(extra_options)
        embed_extra_options["image_token_id"] = config.vision_config.image_token_id
        embed_extra_options["audio_token_id"] = config.audio_config.audio_token_id
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
        print("Building audio encoder for Phi4MultimodalForCausalLM...")
        self.audio_encoder.make_model(input_path)
        print("Building embedding model for Phi4MultimodalForCausalLM...")
        self.embedding_model.make_model(input_path)
        print("Building text decoder for Phi4MultimodalForCausalLM...")
        self.text_model.make_model(input_path)

    def save_model(self, out_dir):
        self.vision_encoder.save_model(out_dir)
        self.audio_encoder.save_model(out_dir)
        self.embedding_model.save_model(out_dir)
        self.text_model.save_model(out_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        # Let the text model write genai_config.json first, then extend it
        # with vision + audio + embedding sections for the phi3v pipeline.
        self.text_model.make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        # onnxruntime-genai uses "phi3v" as the model type for the
        # Vision + Audio + Embedding + Decoder multimodal pipeline.
        genai_config["model"]["type"] = "phi3v"

        genai_config["model"]["vision"] = {
            "filename": self.vision_encoder.FILENAME,
            "num_img_tokens": self.vision_encoder.n_image_tokens,
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "image_features"},
        }

        genai_config["model"]["speech"] = {
            "filename": self.audio_encoder.FILENAME,
            "inputs": {"audio_embeds": "audio_features"},
            "outputs": {"audio_features": "audio_features_out"},
        }

        genai_config["model"]["embedding"] = {
            "filename": self.embedding_model.FILENAME,
            "inputs": {"input_ids": "input_ids", "image_features": "image_features", "audio_features": "audio_features"},
            "outputs": {"inputs_embeds": "inputs_embeds"},
        }

        with open(config_path, "w") as f:
            json.dump(genai_config, f, indent=4)

    def save_processing(self, model_name_or_path, extra_kwargs, out_dir):
        self.text_model.save_processing(model_name_or_path, extra_kwargs, out_dir)
