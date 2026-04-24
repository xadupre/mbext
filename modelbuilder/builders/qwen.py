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
from onnxruntime.quantization.matmul_nbits_quantizer import RTNWeightOnlyQuantConfig
from transformers import AutoConfig

from .base import Model
from .base_embedding import EmbeddingModel
from .base_vision import VisionEncoderModel

# Default token IDs for Qwen2.5-Omni; overridden by values from the HF config when available.
_QWEN25OMNI_DEFAULT_IMAGE_TOKEN_ID = 151655
_QWEN25OMNI_DEFAULT_AUDIO_TOKEN_ID = 151646


class QwenModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class Qwen3Model(QwenModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

    def make_attention_init(self):
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        super().make_attention_init()


class Qwen25VLTextModel(Model):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # The HF model (Qwen2RMSNorm) *always* computes LayerNorm in float32.
        # By inheriting from `base.Model`, all `layernorm_attrs["cast"]` flags
        # are `False`. This causes parity loss and type mismatch error.
        #
        # SOLUTION: Manually set all `cast` flags to `True`. This forces the
        # builder to cast bf16 inputs -> fp32, compute LN, and cast fp32
        # outputs -> bf16, matching the HF model and fixing both errors.
        #
        print("Forcing LayerNorm computation to float32 (and enabling all casts) for Qwen2.5-VL parity.")
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True

        # Qwen2's RoPE *always* computes in float32.
        # We must replicate this behavior.
        print("Forcing RoPE computation to float32 for Qwen2.5-VL parity.")
        self.rope_attrs["cast_to_fp32"] = True

        # Check rope type since huggingface model supports yarn but that is not recommended as mentioned in model card. Example:
        #    "rope_scaling": {"type": "mrope", "mrope_section": [16, 24,24]}
        if config.rope_scaling and "type" in config.rope_scaling:
            assert config.rope_scaling["type"] in ["mrope", "default"]

        # Qwen 2.5 VL applies RoPE manually before attention, not fused in the op
        self.attention_attrs["use_rope_in_attn"] = False

        # We need separate Q, K, V tensors to apply MRoPE manually.
        # Packed MatMul provides a single output which would require splitting.
        self.attention_attrs["use_packed_matmul"] = False

        self.input_names["position_ids"] = "position_ids"

        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")

        # The HF logic is `mrope_section * 2`, not `[s * 2 for s in mrope_section]`.
        # This results in [16, 24, 24, 16, 24, 24]
        self.mrope_splits = self.mrope_sections * 2

        if sum(self.mrope_splits) != self.head_size:
            # The sum (128) should now correctly match self.head_size (128)
            raise ValueError(f"MRoPE splits {self.mrope_splits} sum ({sum(self.mrope_splits)}) does not match head size ({self.head_size})")

        # Force GroupQueryAttention since make_attention() below only implements GQA.
        self.attention_attrs["op_type"] = "GroupQueryAttention"

        if not self.is_gqa_supported():
            print(f"Warning: {self.ep} does not support GQA for {self.io_dtype}, so GQA might fallback to CPU!")

        # Create and save the inv_freq tensor
        self.make_inv_freq_tensor()

    def make_inv_freq_tensor(self):
        """
        Calculates and saves the `inv_freq` tensor as an initializer.
        This is copied from base.py:make_rotary_embedding_caches_from_scratch
        """
        dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)
        inv_freq = 1.0 / (
            self.rope_attrs["rescale_factors"] * (self.rope_attrs["theta"] ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        )

        # The HF model expects H/2, not R/2
        if dim != self.head_size:
            print(f"Warning: partial_rotary_factor ({self.rope_attrs['partial_rotary_factor']}) is not 1. This might be unsupported.")
            inv_freq = inv_freq[: (self.head_size // 2)]

        self.make_initializer(inv_freq, "model.inv_freq", to=ir.DataType.FLOAT)
        print("Created and saved 'model.inv_freq' initializer.")

    def make_inputs_and_outputs(self):
        # Qwen2.5-VL uses 3D position_ids
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]

        # Call the base Model's make_inputs_and_outputs (skipping MistralModel's)
        super().make_inputs_and_outputs()

    def make_dynamic_rope_caches(self, layer_id, basename):
        # Make nodes for the Dynamic RoPE Cache subgraph
        #
        # Re-implements Qwen2_5_VLRotaryEmbedding.forward using ONNX ops.
        # Takes 3D position_ids and inv_freq and dynamically creates
        # the cos/sin caches.
        #
        #         inv_freq (H/2)                                     position_ids (3, B, S)
        #             |                                                      |
        #         Unsqueeze                                              Unsqueeze
        #             |                                                      |
        #           Expand                                                  Cast
        #      (3, B, H/2, 1)                                           (3, B, 1, S)
        #             |                                                      |
        #             +--------------------------+---------------------------+
        #                                        |
        #                                      MatMul
        #                                   (3, B, H/2, S)
        #                                        |
        #                                    Transpose
        #                                   (3, B, S, H/2)
        #                                        |
        #                                     Concat
        #                                  (3, B, S, H)
        #                                        |
        #                          +-------------+-------------+
        #                          |                           |
        #                         Cos                         Sin
        #                          |                           |
        #                         Mul                         Mul
        #                   (apply scaling)             (apply scaling)
        #
        pos_ids_name = self.input_names["position_ids"]
        inv_freq_name = "model.inv_freq"
        head_dim_half = self.head_size // 2

        # Get Batch Size from position_ids.shape[1]
        shape_pos_ids_name = f"{basename}/pos_ids/Shape"
        shape_pos_ids_output = f"{shape_pos_ids_name}/output_0"
        self.make_shape(shape_pos_ids_name, pos_ids_name, [3])

        gather_batch_size_name = f"{basename}/pos_ids/Gather"
        gather_batch_size_output = f"{gather_batch_size_name}/output_0"
        self.make_gather(gather_batch_size_name, [shape_pos_ids_output, "/model/constants/INT64/[1]"], ir.DataType.INT64, [1], axis=0)

        # Expand inv_freq: [H/2] -> [1, 1, H/2, 1]
        unsqueeze_1_name = f"{basename}/inv_freq/Unsqueeze"
        unsqueeze_1_output = f"{unsqueeze_1_name}/output_0"
        self.make_unsqueeze(
            unsqueeze_1_name, [inv_freq_name, "/model/constants/INT64/[0, 1, 3]"], ir.DataType.FLOAT, [1, 1, head_dim_half, 1]
        )

        # Create target shape for Expand: [3, B, H/2, 1]
        concat_expand_shape_name = f"{basename}/expand_shape/Concat"
        concat_expand_shape_output = f"{concat_expand_shape_name}/output_0"
        self.make_concat(
            concat_expand_shape_name,
            ["/model/constants/INT64/[3]", gather_batch_size_output, f"/model/constants/INT64/[{head_dim_half}, 1]"],
            ir.DataType.INT64,
            [4],
            axis=0,
        )

        expand_name = f"{basename}/inv_freq/Expand"
        expand_output = f"{expand_name}/output_0"
        self.make_expand(
            expand_name, [unsqueeze_1_output, concat_expand_shape_output], ir.DataType.FLOAT, [3, "batch_size", head_dim_half, 1]
        )

        # Expand position_ids: [3, B, S] -> [3, B, 1, S]
        unsqueeze_2_name = f"{basename}/pos_ids/Unsqueeze"
        unsqueeze_2_output = f"{unsqueeze_2_name}/output_0"
        self.make_unsqueeze(
            unsqueeze_2_name, [pos_ids_name, "/model/constants/INT64/[2]"], ir.DataType.INT64, [3, "batch_size", 1, "sequence_length"]
        )

        # Cast position_ids to float
        cast_name = f"{basename}/pos_ids/Cast"
        cast_output = f"{cast_name}/output_0"
        self.make_cast(cast_name, unsqueeze_2_output, ir.DataType.FLOAT, [3, "batch_size", 1, "sequence_length"])

        # MatMul: [3, B, H/2, 1] @ [3, B, 1, S] -> [3, B, H/2, S]
        matmul_name = f"{basename}/freqs/MatMul"
        matmul_output = f"{matmul_name}/output_0"
        self.make_node("MatMul", [expand_output, cast_output], [matmul_output], name=matmul_name)
        self.make_value(matmul_output, ir.DataType.FLOAT, [3, "batch_size", head_dim_half, "sequence_length"])

        # Transpose: [3, B, H/2, S] -> [3, B, S, H/2]
        transpose_name = f"{basename}/freqs/Transpose"
        transpose_output = f"{transpose_name}/output_0"
        self.make_transpose(
            transpose_name, matmul_output, ir.DataType.FLOAT, [3, "batch_size", "sequence_length", head_dim_half], perm=[0, 1, 3, 2]
        )

        # Concat (freqs, freqs): [3, B, S, H/2] -> [3, B, S, H]
        concat_name = f"{basename}/Concat"
        concat_output = f"{concat_name}/output_0"
        self.make_concat(
            concat_name,
            [transpose_output, transpose_output],
            ir.DataType.FLOAT,
            [3, "batch_size", "sequence_length", self.head_size],
            axis=-1,
        )

        # Cos(emb) and Sin(emb)
        cos_name = f"{basename}/Cos"
        cos_cache_shape = [3, "batch_size", "sequence_length", self.head_size]
        self.make_cos(cos_name, concat_output, ir.DataType.FLOAT, cos_cache_shape)
        cos_output = f"{cos_name}/output_0"

        sin_name = f"{basename}/Sin"
        self.make_sin(sin_name, concat_output, ir.DataType.FLOAT, cos_cache_shape)
        sin_output = f"{sin_name}/output_0"

        return cos_output, sin_output

    def make_mrope_flattened_caches(self, layer_id, dyn_cos, dyn_sin):
        # Converts the 3D MRoPE caches [3, B, S, H] into flattened, interleaved caches [B*S, H/2]
        # suitable for the RotaryEmbedding operator.
        # The logic is:
        #   1. Slice dynamic caches to H/2.
        #   2. Split into 3 chunks based on mrope_sections (e.g. 16, 24, 24).
        #   3. Gather Temporal(0), Height(1), Width(2) specific slices for each chunk.
        #   4. Concat back to H/2.
        #   5. Flatten to [B*S, H/2].
        # The subgraph looks like:
        #      dyn_cos (3, B, S, H)
        #             |
        #           Slice
        #      (3, B, S, H/2)
        #             |
        #           Split
        #   (3, B, S, sections[i])
        #       /     |     \
        #  Gather  Gather  Gather
        #   idx=0   idx=1   idx=2
        #    /        |       \
        # Squeeze  Squeeze  Squeeze
        #    \        |       /
        #     \       |      /
        #      \      |     /
        #          Concat
        #       (B, S, H/2)
        #             |
        #          Reshape
        #        (B*S, H/2)

        basename = f"/model/layers.{layer_id}/attn/mrope_flattened_cache"

        def process_cache(input_name, name_suffix):
            # 1. Slice to H/2: [3, B, S, H] -> [3, B, S, H/2]
            slice_name = f"{basename}/{name_suffix}/half/Slice"
            slice_output = f"{slice_name}/output_0"
            self.make_slice(
                slice_name,
                [
                    input_name,
                    "/model/constants/INT64/[0]",
                    f"/model/constants/INT64/[{self.head_size // 2}]",
                    "/model/constants/INT64/[-1]",
                ],
                ir.DataType.FLOAT,
                [3, "batch_size", "sequence_length", self.head_size // 2],
            )

            # Create a Constant node for mrope_sections: [16, 24, 24]
            sections_name = f"{basename}/mrope_sections/Constant"
            sections_output = f"{basename}/mrope_sections"
            self.make_node(
                "Constant",
                [],
                [sections_output],
                name=sections_name,
                value=ir.tensor(torch.tensor(self.mrope_sections, dtype=torch.int64), name=sections_output),
            )
            self.make_value(sections_output, ir.DataType.INT64, [3])

            # 2. Split: [3, B, S, H/2] -> 3 * [3, B, S, section_dim]
            split_name = f"{basename}/{name_suffix}/Split"
            split_outputs = [f"{split_name}/output_{i}" for i in range(3)]
            self.make_node("Split", [slice_output, sections_output], split_outputs, name=split_name, axis=-1)

            # 3. Gather + Squeeze: Reorder T, H, W
            gathered_chunks = []
            for i in range(3):
                # Chunk 0->T(0), Chunk 1->H(1), Chunk 2->W(2)
                gather_name = f"{basename}/{name_suffix}/chunk_{i}/Gather"
                gather_output = f"{gather_name}/output_0"
                self.make_node("Gather", [split_outputs[i], f"/model/constants/INT64/[{i}]"], [gather_output], name=gather_name, axis=0)
                # Gather output is [1, B, S, dim]

                squeeze_name = f"{basename}/{name_suffix}/chunk_{i}/Squeeze"
                squeeze_output = f"{squeeze_name}/output_0"
                self.make_squeeze(
                    squeeze_name,
                    [gather_output, "/model/constants/INT64/[0]"],
                    ir.DataType.FLOAT,
                    ["batch_size", "sequence_length", self.mrope_sections[i]],
                )
                gathered_chunks.append(squeeze_output)

            # 4. Concat: -> [B, S, H/2]
            concat_name = f"{basename}/{name_suffix}/Concat"
            concat_output = f"{concat_name}/output_0"
            self.make_concat(
                concat_name, gathered_chunks, ir.DataType.FLOAT, ["batch_size", "sequence_length", self.head_size // 2], axis=-1
            )

            # 5. Flatten: -> [B*S, H/2]
            reshape_name = f"{basename}/{name_suffix}_flat/Reshape"
            reshape_output = f"{reshape_name}/output_0"
            self.make_reshape(
                reshape_name,
                [concat_output, f"/model/constants/INT64/[-1, {self.head_size // 2}]"],
                ir.DataType.FLOAT,
                ["total_token_count", self.head_size // 2],
            )
            return reshape_output

        flat_cos = process_cache(dyn_cos, "cos")
        flat_sin = process_cache(dyn_sin, "sin")

        return flat_cos, flat_sin

    def apply_mrope_rotation(self, layer_id, q_or_k_path, q_or_k_shape, dyn_cos, dyn_sin, num_heads, basename):
        # Make nodes for the MRoPE rotation subgraph using RotaryEmbedding op
        #
        # 1. Flatten 3D caches [3, B, S, H] -> [B*S, H/2] (via make_mrope_flattened_caches)
        # 2. Generate linear position IDs [B, S] (0 .. B*S-1)
        # 3. Apply RotaryEmbedding
        #
        #      dyn_cos (3, B, S, H)   dyn_sin (3, B, S, H)
        #              |                      |
        #    make_mrope_flattened_caches (slice, split, gather, concat, flatten)
        #              |                      |
        #        flat_cos               flat_sin
        #      (B*S, H/2)             (B*S, H/2)
        #              |                      |
        #              +-----------+----------+
        #                          |
        #      q_or_k              |              position_ids
        #    (B, S, N*H)           |            (0 .. B*S-1)
        #        |                 |                 |
        #     Reshape              |              Reshape
        #        |                 |                 |
        #    Transpose             |                 |
        #   (B, N, S, H)           |               (B, S)
        #        |                 |                 |
        #        +--------+--------+--------+--------+
        #                 |                 |
        #          RotaryEmbedding (com.microsoft)
        #                 |
        #            output (B, N, S, H)
        #                 |
        #             Transpose
        #                 |
        #              Reshape
        #            (B, S, N*H)

        # 1. Prepare flattened MRoPE caches [B*S, H/2]
        #    This slices, splits, and re-assembles the 3D dynamic caches into the correct per-token layout.
        flat_cos, flat_sin = self.make_mrope_flattened_caches(layer_id, dyn_cos, dyn_sin)

        # 2. Prepare position_ids [B, S] (values 0 to B*S - 1)
        #    RotaryEmbedding will use these indices to access the flattened cache.
        #    Get B*S from q_or_k shape. q_or_k is [B, S, N*H].
        shape_node = f"{basename}/Shape"
        self.make_shape(shape_node, q_or_k_path, [3])

        # Extract B and S
        batch_size_node = f"{basename}/BatchSize/Gather"
        batch_size_out = f"{batch_size_node}/output_0"
        self.make_gather(batch_size_node, [f"{shape_node}/output_0", "/model/constants/INT64/[0]"], ir.DataType.INT64, [], 0)

        seq_len_node = f"{basename}/SeqLen/Gather"
        seq_len_out = f"{seq_len_node}/output_0"
        self.make_gather(seq_len_node, [f"{shape_node}/output_0", "/model/constants/INT64/[1]"], ir.DataType.INT64, [], 0)

        # Calculate Total Tokens = B * S
        mul_len_node = f"{basename}/TotalLen/Mul"
        mul_len_out = f"{mul_len_node}/output_0"
        self.make_mul(mul_len_node, [batch_size_out, seq_len_out], ir.DataType.INT64, [])
        mul_len_out = f"{mul_len_node}/output_0"

        # Range(0, TotalTokens)
        range_node = f"{basename}/Range"
        range_out = f"{range_node}/output_0"
        self.make_range(
            range_node, ["/model/constants/INT64/0", mul_len_out, "/model/constants/INT64/1"], ir.DataType.INT64, ["total_token_count"]
        )
        range_out = f"{range_node}/output_0"

        # Slice Position IDs shape from input shape (take first 2 dims)
        slice_shape_node = f"{basename}/SliceShape"
        slice_shape_out = f"{slice_shape_node}/output_0"
        self.make_slice(
            slice_shape_node,
            [f"{shape_node}/output_0", "/model/constants/INT64/[0]", "/model/constants/INT64/[2]", "/model/constants/INT64/[0]"],
            ir.DataType.INT64,
            [2],
        )

        # Reshape Range output to [B, S]
        pos_ids_reshape_node = f"{basename}/PosIds/Reshape"
        pos_ids_out = f"{pos_ids_reshape_node}/output_0"
        self.make_reshape(pos_ids_reshape_node, [range_out, slice_shape_out], ir.DataType.INT64, ["batch_size", "sequence_length"])

        # 3. Prepare Q/K input [B, N, S, H]
        #    Input is [B, S, N*H]. Reshape -> [B, S, N, H] -> Transpose -> [B, N, S, H]
        reshape_in_node = f"{basename}/Input/Reshape"
        reshape_in_out = f"{reshape_in_node}/output_0"
        self.make_reshape(
            reshape_in_node,
            [q_or_k_path, f"/model/constants/INT64/[0, 0, {num_heads}, {self.head_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, self.head_size],
        )

        transpose_in_node = f"{basename}/Input/Transpose"
        transpose_in_out = f"{transpose_in_node}/output_0"
        target_shape_bnsh = ["batch_size", num_heads, "sequence_length", self.head_size]
        self.make_transpose(transpose_in_node, reshape_in_out, self.io_dtype, target_shape_bnsh, [0, 2, 1, 3])

        # 4. Handle Type Casting
        #    RotaryEmbedding requires input, cos, sin to be same type.
        #    Qwen2.5-VL forces float32 computation.
        force_fp32 = self.rope_attrs.get("cast_to_fp32", False)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype

        rope_input = transpose_in_out
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            cast_in_node = f"{basename}/Input/Cast"
            rope_input = f"{cast_in_node}/output_0"
            self.make_cast(cast_in_node, transpose_in_out, compute_dtype, target_shape_bnsh)

        rope_cos = flat_cos
        rope_sin = flat_sin
        # Note: dyn_cos is Float. flat_cos is Float. If compute_dtype is not Float (e.g. fp16), we must cast cache.
        if compute_dtype != ir.DataType.FLOAT:
            # Cache is Float, we need FP16
            cast_cos_node = f"{basename}/Cos/Cast"
            rope_cos = f"{cast_cos_node}/output_0"
            self.make_cast(cast_cos_node, flat_cos, compute_dtype, ["total_token_count", self.head_size // 2])

            cast_sin_node = f"{basename}/Sin/Cast"
            rope_sin = f"{cast_sin_node}/output_0"
            self.make_cast(cast_sin_node, flat_sin, compute_dtype, ["total_token_count", self.head_size // 2])

        # 5. RotaryEmbedding Node
        rope_node = f"{basename}/RotaryEmbedding"
        rope_output = f"{rope_node}/output_0"
        self.make_node(
            "RotaryEmbedding",
            [rope_input, pos_ids_out, rope_cos, rope_sin],
            [rope_output],
            name=rope_node,
            domain="com.microsoft",
            rotary_embedding_dim=self.head_size,
            num_heads=num_heads,
            interleaved=0,  # False, matches rotate_half logic
        )
        self.make_value(rope_output, compute_dtype, target_shape_bnsh)

        # 6. Post-process Output
        #    Cast back if needed -> Transpose -> Reshape
        final_rope_output = rope_output
        if force_fp32 and self.io_dtype != ir.DataType.FLOAT:
            cast_out_node = f"{basename}/Output/Cast"
            final_rope_output = f"{cast_out_node}/output_0"
            self.make_cast(cast_out_node, rope_output, self.io_dtype, target_shape_bnsh)

        transpose_out_node = f"{basename}/Output/Transpose"
        transpose_out_out = f"{transpose_out_node}/output_0"
        self.make_transpose(
            transpose_out_node, final_rope_output, self.io_dtype, ["batch_size", "sequence_length", num_heads, self.head_size], [0, 2, 1, 3]
        )

        reshape_out_node = f"{basename}/Output/Reshape"
        reshape_out_out = f"{reshape_out_node}/output_0"
        self.make_reshape(
            reshape_out_node,
            [transpose_out_out, f"/model/constants/INT64/[0, 0, {num_heads * self.head_size}]"],
            self.io_dtype,
            q_or_k_shape,
        )

        return reshape_out_out

    def make_attention_qk_subgraph(self, layer_id, attention, root_input, **kwargs):
        # Make nodes for the Attention subgraph (with MRoPE)
        #
        #        q_path    k_path    v_path
        #          |        |        |
        #          |        |        +-----------------+
        #          |        |                          |
        #   (make_dynamic_rope_caches)                 |
        #          |                                   |
        #    +-----+-----+                             |
        #    |           |                             |
        # dyn_cos     dyn_sin                          |
        #    |           |                             |
        #    v           v                             |
        # (apply_mrope_rotation for Q)                 |
        #          |                                   |
        #        Q_Rot                                 |
        #          |     (apply_mrope_rotation for K)  |
        #          |                 |                 |
        #          |               K_Rot               |
        #          |                 |                 |
        #          +--------+--------+                 |
        #                   |                          |
        #           GroupQueryAttention <--------------+
        #                   |

        # 1. Calculate shapes for MRoPE rotation
        q_shape = ["batch_size", "sequence_length", self.num_attn_heads * self.head_size]
        k_shape = ["batch_size", "sequence_length", self.num_kv_heads * self.head_size]

        # 2. Apply 3D RoPE (MRoPE)
        cos_dynamic, sin_dynamic = self.make_dynamic_rope_caches(layer_id, basename=f"/model/layers.{layer_id}/attn/mrope_dynamic_cache")

        # Apply rotation to Q
        self.attention_attrs["q_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["q_path"],
            q_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_attn_heads,
            basename=f"/model/layers.{layer_id}/attn/q_mrope",
        )

        # Apply rotation to K
        self.attention_attrs["k_path"] = self.apply_mrope_rotation(
            layer_id,
            self.attention_attrs["k_path"],
            k_shape,
            cos_dynamic,
            sin_dynamic,
            self.num_kv_heads,
            basename=f"/model/layers.{layer_id}/attn/k_mrope",
        )

        # 3. Call GroupQueryAttention op
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"

        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name,
            q_path=self.attention_attrs["q_path"],
            k_path=self.attention_attrs["k_path"],
            v_path=self.attention_attrs["v_path"],
            past_k=past_k,
            past_v=past_v,
            present_k=present_k,
            present_v=present_v,
            # Pass empty strings for fused caches since we applied RoPE manually
            cos_cache="",
            sin_cache="",
            **kwargs,
        )

    def load_weights(self, input_path):
        # For quantized models (e.g., Quark, AWQ, GPTQ) or GGUF, use base class logic
        # which loads weights directly via QuantModel
        if self.quant_type is not None or input_path.endswith(".gguf"):
            return super().load_weights(input_path)

        # For non-quantized models, load the Hugging Face model
        print("Loading Qwen2_5_VLForConditionalGeneration model...")
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
        )


class Qwen25OmniThinkerModel(Qwen25VLTextModel):
    """Qwen2.5-Omni thinker (text decoder) model builder.

    Qwen2.5-Omni is a multimodal model (audio + vision + text). This builder
    exports only the text (thinker) component.  Like Qwen2.5-VL, the thinker
    text model uses multimodal RoPE (mRoPE) with 3-D position IDs, so this
    class inherits from :class:`Qwen25VLTextModel`.

    The HuggingFace config may be nested in two ways depending on the source:

    * **Full model** (``Qwen2_5OmniConfig``): text attributes live under
      ``config.thinker_config.text_config``.
    * **Thinker-only** (``Qwen2_5OmniThinkerConfig``): text attributes live
      under ``config.text_config``.

    Both cases are normalised by flattening the relevant ``text_config``
    attributes onto the top-level ``config``.  The ``mrope_section`` is read
    from ``rope_parameters`` (where Qwen2.5-Omni stores it) and copied into
    ``rope_scaling`` so that :class:`Qwen25VLTextModel` can locate it via the
    standard :func:`~modelbuilder.builders.base.Model.make_rope_init` path.

    Because the model is multimodal the embedding layer is excluded from the
    ONNX export (``exclude_embeds=True``) by default.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Flatten text_config attributes onto config so the base Model init
        # can read hidden_size, num_attention_heads, etc.
        # Handle two possible nesting levels:
        #   Qwen2_5OmniConfig          → config.thinker_config.text_config
        #   Qwen2_5OmniThinkerConfig   → config.text_config
        if hasattr(config, "thinker_config") and hasattr(config.thinker_config, "text_config"):
            text_config = config.thinker_config.text_config
        elif hasattr(config, "text_config"):
            text_config = config.text_config
        else:
            text_config = None

        if text_config is not None:
            for key in text_config:
                if not hasattr(config, key) or getattr(config, key) is None:
                    setattr(config, key, getattr(text_config, key))

        # Qwen2.5-Omni stores mrope_section inside rope_parameters rather than
        # rope_scaling.  Copy it into rope_scaling so the shared make_rope_init
        # path in the base class picks it up.
        rope_params = getattr(config, "rope_parameters", None)
        if isinstance(rope_params, dict) and "mrope_section" in rope_params:
            if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
                config.rope_scaling = {}
            if "mrope_section" not in config.rope_scaling:
                config.rope_scaling = dict(config.rope_scaling)
                config.rope_scaling["mrope_section"] = rope_params["mrope_section"]
            if "rope_theta" not in config.rope_scaling and "rope_theta" in rope_params:
                config.rope_scaling["rope_theta"] = rope_params["rope_theta"]

        if "exclude_embeds" not in extra_options:
            extra_options["exclude_embeds"] = True
            print("Setting exclude_embeds=True for Qwen2.5-Omni thinker model.")

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # The Qwen2.5-Omni thinker shares the same decoder architecture as
        # Qwen2.5-VL (mRoPE, GQA with QKV bias).  Override model_type so that
        # make_genai_config emits "qwen2_5_vl" – the type string that
        # ORT-GenAI uses for this mRoPE model family.
        self.model_type = "Qwen2_5_VLForConditionalGeneration"

    def load_weights(self, input_path):
        # For quantized models or GGUF use the base class logic.
        if self.quant_type is not None or input_path.endswith(".gguf"):
            return super().load_weights(input_path)

        # Qwen2_5OmniThinkerForConditionalGeneration is not registered with
        # AutoModelForCausalLM, so load it directly.
        print("Loading Qwen2_5OmniThinkerForConditionalGeneration model...")
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
        )

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """Generate genai_config.json for the thinker (text-only) decoder.

        Flattens ``bos_token_id``, ``eos_token_id``, and ``pad_token_id``
        from the nested text config onto the top-level HF config before
        delegating to the base class implementation.
        """
        hf_config = AutoConfig.from_pretrained(model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)

        # Resolve the text config from either nesting level.
        if hasattr(hf_config, "thinker_config") and hasattr(hf_config.thinker_config, "text_config"):
            text_cfg = hf_config.thinker_config.text_config
        elif hasattr(hf_config, "text_config"):
            text_cfg = hf_config.text_config
        else:
            text_cfg = hf_config

        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            val = getattr(text_cfg, attr, None)
            if val is not None:
                setattr(hf_config, attr, val)
        hf_config.save_pretrained(out_dir)

        super().make_genai_config(out_dir, {}, out_dir)


class Qwen25OmniVisionEncoderModel(VisionEncoderModel):
    """ONNX graph builder for the Qwen2.5-Omni vision encoder.

    Exports the vision tower (patch embedding + transformer blocks + patch merger)
    to ``vision_encoder.onnx``.

    Inputs
    ------
    pixel_values : float [n_patches, in_feat_dim]
        Flat image patches pre-processed by the HF image processor, where
        ``in_feat_dim = in_channels * temporal_patch_size * patch_size * patch_size``.
    rotary_pos_emb : float32 [n_patches, head_dim // 2]
        Pre-computed 2-D rotary position embeddings computed by the processor
        from ``image_grid_thw``.

    Outputs
    -------
    image_features : io_dtype [n_merged, out_hidden_size]
        Merged patch features suitable for injection into the text embedding.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        vc = config.vision_config

        # Patch a minimal LLM-style config so that Model.__init__ does not error.
        vis_config = copy.deepcopy(config)
        vis_config.hidden_size = vc.hidden_size
        vis_config.intermediate_size = vc.intermediate_size
        vis_config.num_attention_heads = vc.num_heads
        vis_config.num_key_value_heads = vc.num_heads
        vis_config.num_hidden_layers = vc.depth
        vis_config.head_dim = vc.hidden_size // vc.num_heads
        vis_config.hidden_act = getattr(vc, "hidden_act", "silu")
        vis_config.vocab_size = 1
        vis_config.max_position_embeddings = 1
        vis_config.rms_norm_eps = 1e-6
        vis_config.rope_scaling = None

        extra_options = {**extra_options, "filename": self.FILENAME, "exclude_lm_head": True, "exclude_embeds": True}
        super().__init__(vis_config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        self.graph.name = "qwen25omni_vision_encoder"
        self.config = config
        self.vision_config = vc

        self.vis_hidden_size = vc.hidden_size
        self.vis_intermediate_size = vc.intermediate_size
        self.vis_num_heads = vc.num_heads
        self.vis_head_dim = vc.hidden_size // vc.num_heads
        self.vis_num_layers = vc.depth
        self.spatial_merge_size = vc.spatial_merge_size
        self.out_hidden_size = vc.out_hidden_size
        self.in_channels = vc.in_channels
        self.patch_size = vc.patch_size
        self.temporal_patch_size = vc.temporal_patch_size
        self.in_feat_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

        # Resolve text hidden size from potentially nested configs.
        if hasattr(config, "thinker_config") and hasattr(config.thinker_config, "text_config"):
            self.text_hidden_size = config.thinker_config.text_config.hidden_size
        elif hasattr(config, "text_config"):
            self.text_hidden_size = config.text_config.hidden_size
        else:
            self.text_hidden_size = self.out_hidden_size

    # ------------------------------------------------------------------
    # 2-D RoPE application
    # ------------------------------------------------------------------

    def make_vision_rope(self, name, x, rope_input, n_patches, head_dim):
        """Apply 2-D vision RoPE to Q or K tensor.

        Parameters
        ----------
        x : str
            Name of tensor [n_patches, vis_num_heads, vis_head_dim] in io_dtype.
        rope_input : str
            Name of rotary_pos_emb [n_patches, head_dim // 2] in float32.
        Returns output name with same shape as ``x``.
        """
        half = head_dim // 2
        num_heads = self.vis_num_heads

        # Compute cos/sin from frequencies (always in float32).
        cos_raw = f"{name}/cos_raw/output_0"
        sin_raw = f"{name}/sin_raw/output_0"
        self.make_node("Cos", inputs=[rope_input], outputs=[cos_raw], name=f"{name}/cos_raw")
        self.make_value(cos_raw, ir.DataType.FLOAT, shape=[None, half])
        self.make_node("Sin", inputs=[rope_input], outputs=[sin_raw], name=f"{name}/sin_raw")
        self.make_value(sin_raw, ir.DataType.FLOAT, shape=[None, half])

        # Tile [n_patches, head_dim//2] → [n_patches, head_dim].
        # Use an inline Constant node (not initializer) to avoid external-data serialisation.
        tile_const = f"{name}/tile_repeats"
        _tile_t = ir.Tensor(np.array([1, 2], dtype=np.int64), name=tile_const)
        self.make_node("Constant", inputs=[], outputs=[tile_const], name=f"{tile_const}/Constant", value=_tile_t)
        self.make_value(tile_const, ir.DataType.INT64, shape=[2])
        self.make_tile(f"{name}/cos_full", [cos_raw, tile_const], ir.DataType.FLOAT, [None, head_dim])
        self.make_tile(f"{name}/sin_full", [sin_raw, tile_const], ir.DataType.FLOAT, [None, head_dim])
        cos_full_out = f"{name}/cos_full/output_0"
        sin_full_out = f"{name}/sin_full/output_0"

        # Unsqueeze to [n_patches, 1, head_dim] for broadcasting with [n_patches, num_heads, head_dim].
        ax_1 = f"{name}/ax1_const"
        _ax_t = ir.Tensor(np.array([1], dtype=np.int64), name=ax_1)
        self.make_node("Constant", inputs=[], outputs=[ax_1], name=f"{ax_1}/Constant", value=_ax_t)
        self.make_value(ax_1, ir.DataType.INT64, shape=[1])
        self.make_unsqueeze(f"{name}/cos_3d", [cos_full_out, ax_1], ir.DataType.FLOAT, [None, 1, head_dim])
        self.make_unsqueeze(f"{name}/sin_3d", [sin_full_out, ax_1], ir.DataType.FLOAT, [None, 1, head_dim])
        cos_3d_out = f"{name}/cos_3d/output_0"
        sin_3d_out = f"{name}/sin_3d/output_0"

        # When io_dtype != float32: cast x to fp32 for the rotation computation,
        # and cast the result back.  cos/sin stay in float32 throughout.
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast(f"{name}/x_fp32", x, ir.DataType.FLOAT, [None, num_heads, head_dim])
            x_fp32 = f"{name}/x_fp32/output_0"
        else:
            x_fp32 = x

        # rotate_half: split x into two halves, negate second, concat [-x2, x1].
        split_out_1 = f"{name}/split_1/output_0"
        split_out_2 = f"{name}/split_2/output_0"
        half_const = f"{name}/half_const"
        _half_t = ir.Tensor(np.array([half, half], dtype=np.int64), name=half_const)
        self.make_node("Constant", inputs=[], outputs=[half_const], name=f"{half_const}/Constant", value=_half_t)
        self.make_value(half_const, ir.DataType.INT64, shape=[2])
        self.make_node("Split", inputs=[x_fp32, half_const], outputs=[split_out_1, split_out_2], name=f"{name}/Split", axis=-1)
        self.make_value(split_out_1, ir.DataType.FLOAT, shape=[None, num_heads, half])
        self.make_value(split_out_2, ir.DataType.FLOAT, shape=[None, num_heads, half])

        neg_x2 = self.make_neg(f"{name}/neg_x2", split_out_2, ir.DataType.FLOAT, [None, num_heads, half])
        x_rot = self.make_concat(f"{name}/x_rot", [neg_x2, split_out_1], ir.DataType.FLOAT, [None, num_heads, head_dim], axis=-1)

        # output = x * cos + rotate_half(x) * sin
        xcos = self.make_mul(f"{name}/xcos", [x_fp32, cos_3d_out], ir.DataType.FLOAT, [None, num_heads, head_dim])
        xrot_sin = self.make_mul(f"{name}/xrot_sin", [x_rot, sin_3d_out], ir.DataType.FLOAT, [None, num_heads, head_dim])
        out_fp32 = self.make_add(f"{name}/out_fp32", [xcos, xrot_sin], ir.DataType.FLOAT, [None, num_heads, head_dim])

        # Cast back to io_dtype.
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast(f"{name}/out_cast", out_fp32, self.io_dtype, [None, num_heads, head_dim])
            return f"{name}/out_cast/output_0"
        return out_fp32

    # ------------------------------------------------------------------
    # Vision attention layer (overrides Model.make_attention)
    # ------------------------------------------------------------------

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Build one Qwen2_5OmniVisionAttention block (encoder-style, no KV cache).

        Overrides Model.make_attention for the vision encoder.

        root_input : [n_patches, vis_hidden_size]
        Sets self.layernorm_attrs["skip_input"] to the output name,
        following the base-class convention.
        """
        b = f"/vision/layers.{layer_id}/attn"
        n = None  # n_patches is dynamic
        d = self.vis_hidden_size
        nh = self.vis_num_heads
        hd = self.vis_head_dim
        rope_input = kwargs.get("rotary_pos_emb", "rotary_pos_emb")

        # QKV projections: use base-class make_matmul (handles fp16/int4 automatically).
        q = f"{self.make_matmul(attention.q, f'{b}/q_proj/MatMul', root_input)}/output_0"
        if attention.q.bias is not None:
            self.make_add_bias(attention.q.bias, f"{b}/q_proj/Add", root_input=q)
            q = f"{b}/q_proj/Add/output_0"

        k = f"{self.make_matmul(attention.k, f'{b}/k_proj/MatMul', root_input)}/output_0"
        if attention.k.bias is not None:
            self.make_add_bias(attention.k.bias, f"{b}/k_proj/Add", root_input=k)
            k = f"{b}/k_proj/Add/output_0"

        v = f"{self.make_matmul(attention.v, f'{b}/v_proj/MatMul', root_input)}/output_0"
        if attention.v.bias is not None:
            self.make_add_bias(attention.v.bias, f"{b}/v_proj/Add", root_input=v)
            v = f"{b}/v_proj/Add/output_0"

        # Reshape to [n_patches, num_heads, head_dim].
        q_4d = self.make_reshape(f"{b}/q_reshape", [q, [0, nh, hd]], self.io_dtype, [n, nh, hd])
        k_4d = self.make_reshape(f"{b}/k_reshape", [k, [0, nh, hd]], self.io_dtype, [n, nh, hd])
        v_4d = self.make_reshape(f"{b}/v_reshape", [v, [0, nh, hd]], self.io_dtype, [n, nh, hd])

        # Apply 2-D RoPE to Q and K.
        q_rope = self.make_vision_rope(f"{b}/q_rope", q_4d, rope_input, n, hd)
        k_rope = self.make_vision_rope(f"{b}/k_rope", k_4d, rope_input, n, hd)

        # Transpose to [num_heads, n_patches, head_dim] for batched MatMul.
        q_t = self.make_transpose(f"{b}/q_t", q_rope, self.io_dtype, [nh, n, hd], perm=[1, 0, 2])
        k_t = self.make_transpose(f"{b}/k_t", k_rope, self.io_dtype, [nh, n, hd], perm=[1, 0, 2])
        v_t = self.make_transpose(f"{b}/v_t", v_4d, self.io_dtype, [nh, n, hd], perm=[1, 0, 2])

        # Scaled dot-product attention (full attention, no causal mask).
        k_T = self.make_transpose(f"{b}/k_T", k_t, self.io_dtype, [nh, hd, n], perm=[0, 2, 1])
        attn_w_out = f"{b}/attn_w/MatMul/output_0"
        self.make_node("MatMul", inputs=[q_t, k_T], outputs=[attn_w_out], name=f"{b}/attn_w/MatMul")
        self.make_value(attn_w_out, self.io_dtype, shape=[nh, n, n])

        np_dtype = np.float16 if self.io_dtype == ir.DataType.FLOAT16 else np.float32
        scale_name = f"{b}/attn_scale"
        self.make_initializer(np.array(float(hd**-0.5), dtype=np_dtype), scale_name)
        attn_ws = self.make_mul(f"{b}/attn_scale_mul", [attn_w_out, scale_name], self.io_dtype, [nh, n, n])
        attn_probs = self.make_softmax(f"{b}/attn_softmax", attn_ws, self.io_dtype, [nh, n, n])

        attn_out_t = f"{b}/attn_out_t/MatMul/output_0"
        self.make_node("MatMul", inputs=[attn_probs, v_t], outputs=[attn_out_t], name=f"{b}/attn_out_t/MatMul")
        self.make_value(attn_out_t, self.io_dtype, shape=[nh, n, hd])

        # Transpose back to [n_patches, num_heads, head_dim] and flatten to [n_patches, d].
        attn_back = self.make_transpose(f"{b}/attn_back", attn_out_t, self.io_dtype, [n, nh, hd], perm=[1, 0, 2])
        attn_flat = self.make_reshape(f"{b}/attn_flat", [attn_back, [0, d]], self.io_dtype, [n, d])

        # O projection (no bias in Qwen2.5-Omni vision attention).
        o = f"{self.make_matmul(attention.proj, f'{b}/o_proj/MatMul', attn_flat)}/output_0"

        # Follow base-class convention: store output in layernorm_attrs["skip_input"].
        self.layernorm_attrs["skip_input"] = o

    # ------------------------------------------------------------------
    # Single transformer layer (overrides Model.make_layer)
    # ------------------------------------------------------------------

    def make_layer(self, layer_id, block):
        """Build one Qwen2_5OmniVisionBlock (norm → attn → residual → norm → mlp → residual).

        Overrides Model.make_layer for the vision encoder.
        Reads hidden-states from self.layernorm_attrs["root_input"] and
        stores the output back there, following the base-class convention.
        """
        root = self.layernorm_attrs["root_input"]
        b = f"/vision/layers.{layer_id}"
        n = None
        d = self.vis_hidden_size

        # Pre-attention RMSNorm.
        norm1_out = self.make_rms_norm(f"{b}/norm1", root, block.norm1.weight, shape=[n, d])

        # Attention (override that takes rotary_pos_emb as a kwarg).
        self.make_attention(layer_id, block.attn, norm1_out, rotary_pos_emb="rotary_pos_emb")
        attn_out = self.layernorm_attrs["skip_input"]

        # Residual 1.
        res1 = self.make_add(f"{b}/res1", [root, attn_out], self.io_dtype, [n, d])

        # Pre-MLP RMSNorm.
        norm2_out = self.make_rms_norm(f"{b}/norm2", res1, block.norm2.weight, shape=[n, d])

        # MLP.
        mlp_out = self.make_silu_gated_mlp(layer_id, block.mlp, norm2_out, [None, self.vis_intermediate_size])

        # Residual 2.
        res2 = self.make_add(f"{b}/res2", [res1, mlp_out], self.io_dtype, [n, d])

        self.layernorm_attrs["root_input"] = res2

    # ------------------------------------------------------------------
    # Patch merger
    # ------------------------------------------------------------------

    def make_patch_merger(self, merger, root_input):
        """Build Qwen2_5OmniPatchMerger: RMSNorm → Reshape → Linear+GELU+Linear.

        Uses make_matmul + make_add_bias following the Ministral3 pattern.
        root_input : [n_patches, vis_hidden_size]
        Returns output name [n_merged, out_hidden_size].
        """
        sm2 = self.spatial_merge_size**2
        merge_d = sm2 * self.vis_hidden_size
        n = None

        # RMSNorm on [n_patches, vis_hidden_size].
        ln_out = self.make_rms_norm("/vision/merger/ln_q", root_input, merger.ln_q.weight, shape=[n, self.vis_hidden_size])

        # Reshape [n_patches, vis_hidden_size] → [n_merged, sm2 * vis_hidden_size].
        merged = self.make_reshape("/vision/merger/reshape", [ln_out, [-1, merge_d]], self.io_dtype, [n, merge_d])

        # Linear1 + GELU + Linear2 using base-class make_matmul + make_add_bias.
        lin1 = f"{self.make_matmul(merger.mlp[0], '/vision/merger/mlp_0/MatMul', merged)}/output_0"
        if merger.mlp[0].bias is not None:
            self.make_add_bias(merger.mlp[0].bias, "/vision/merger/mlp_0/Add", root_input=lin1)
            lin1 = "/vision/merger/mlp_0/Add/output_0"

        gelu_out = "/vision/merger/gelu/output_0"
        self.make_node("Gelu", inputs=[lin1], outputs=[gelu_out], name="/vision/merger/gelu", approximate="none")
        self.make_value(gelu_out, self.io_dtype, shape=[n, merge_d])

        out = f"{self.make_matmul(merger.mlp[2], '/vision/merger/mlp_2/MatMul', gelu_out)}/output_0"
        if merger.mlp[2].bias is not None:
            self.make_add_bias(merger.mlp[2].bias, "/vision/merger/mlp_2/Add", root_input=out)
            return "/vision/merger/mlp_2/Add/output_0"
        return out

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_hf_model(self, input_path):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        """Load HF weights and build the vision encoder ONNX graph."""
        hf_model = self.load_hf_model(input_path)
        hf_model.eval()
        vis = hf_model.visual

        # --- Graph inputs ---
        pv_in = self.make_value("pixel_values", ir.DataType.FLOAT, shape=[None, self.in_feat_dim])
        self.graph.inputs.append(pv_in)
        # rotary_pos_emb is always float32 (computed by the image processor).
        rope_in = self.make_value("rotary_pos_emb", ir.DataType.FLOAT, shape=[None, self.vis_head_dim // 2])
        self.graph.inputs.append(rope_in)

        # --- Patch embedding: Linear [n_patches, in_feat_dim] → [n_patches, hidden_size] ---
        pv_cast = "pixel_values"
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast("/vision/pv_cast", "pixel_values", self.io_dtype, [None, self.in_feat_dim])
            pv_cast = "/vision/pv_cast/output_0"

        # Conv3d(in_channels, embed_dim, kernel_size=[T,P,P]) with kernel_size==stride
        # (non-overlapping patches) is equivalent to a Linear layer:
        #   weight shape [embed_dim, in_channels, T, P, P]
        #   → reshape to [embed_dim, in_feat_dim] to collapse all spatial/temporal dims
        #   → make_matmul() transposes to [in_feat_dim, embed_dim] for the ONNX MatMul.
        # No bias: Qwen2_5_VisionPatchEmbed.proj has bias=False.
        patch_proj = torch.nn.Linear(self.in_feat_dim, self.vis_hidden_size, bias=False)
        patch_proj.weight = torch.nn.Parameter(
            vis.patch_embed.proj.weight.detach().reshape(self.vis_hidden_size, self.in_feat_dim), requires_grad=False
        )
        self.make_matmul(patch_proj, "/vision/patch_embed/proj/MatMul", pv_cast)
        patch_emb = "/vision/patch_embed/proj/MatMul/output_0"

        # --- Transformer blocks (make_layer is overridden for vision encoder) ---
        self.layernorm_attrs["root_input"] = patch_emb
        for i, block in enumerate(vis.blocks):
            self.make_layer(i, block)

        # --- Patch merger ---
        hidden = self.layernorm_attrs["root_input"]
        image_features = self.make_patch_merger(vis.merger, hidden)

        # --- Graph output ---
        self.make_node("Identity", inputs=[image_features], outputs=["image_features"], name="/vision/output/Identity")
        out_val = self.make_value("image_features", self.io_dtype, shape=[None, self.out_hidden_size])
        self.graph.outputs.append(out_val)

        self.graph.sort()


class Qwen25OmniAudioEncoderModel(Model):
    """ONNX graph builder for the Qwen2.5-Omni audio encoder.

    Exports the Whisper-style audio tower (two Conv1d layers + sinusoidal
    positional embedding + transformer blocks + average pooling + layer norm +
    linear projection) to ``audio_encoder.onnx``.

    Inputs
    ------
    input_features : float32 [num_mel_bins, n_frames]
        Mel-spectrogram features for a single audio item (variable length).

    Outputs
    -------
    audio_features : io_dtype [n_audio_tokens, output_dim]
        Pooled and projected audio features suitable for injection into the
        text embedding model at ``audio_token_id`` placeholder positions.
    """

    FILENAME = "audio_encoder.onnx"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        ac = config.audio_config

        # Patch a minimal LLM-style config so that Model.__init__ does not error.
        audio_obj_config = copy.deepcopy(config)
        audio_obj_config.hidden_size = ac.d_model
        audio_obj_config.intermediate_size = ac.encoder_ffn_dim
        audio_obj_config.num_attention_heads = ac.encoder_attention_heads
        audio_obj_config.num_key_value_heads = ac.encoder_attention_heads
        audio_obj_config.num_hidden_layers = ac.encoder_layers
        audio_obj_config.head_dim = ac.d_model // ac.encoder_attention_heads
        audio_obj_config.hidden_act = "gelu"
        audio_obj_config.vocab_size = 1
        audio_obj_config.max_position_embeddings = 1
        audio_obj_config.rms_norm_eps = 1e-5
        audio_obj_config.rope_scaling = None

        audio_extra_options = {**extra_options, "filename": self.FILENAME, "exclude_lm_head": True, "exclude_embeds": True}
        super().__init__(audio_obj_config, io_dtype, onnx_dtype, ep, cache_dir, audio_extra_options)

        self.graph.name = "qwen25omni_audio_encoder"
        self.config = config
        self.audio_config = ac

        self.num_mel_bins = ac.num_mel_bins
        self.d_model = ac.d_model
        self.num_heads = ac.encoder_attention_heads
        self.head_dim = ac.d_model // ac.encoder_attention_heads
        self.num_layers = ac.encoder_layers
        self.max_source_positions = ac.max_source_positions
        self.output_dim = ac.output_dim
        self.ffn_dim = ac.encoder_ffn_dim

    # ------------------------------------------------------------------
    # LayerNorm helper (standard LN with weight and bias)
    # ------------------------------------------------------------------

    def _make_audio_layer_norm(self, name, root_input, weight, bias, shape):
        """Build a LayerNormalization node with weight and bias."""
        w_name = f"{name}.weight"
        b_name = f"{name}.bias"
        self.make_initializer(weight.detach(), w_name, to=self.io_dtype)
        self.make_initializer(bias.detach(), b_name, to=self.io_dtype)
        out = f"{name}/output_0"
        self.make_node(
            "LayerNormalization", inputs=[root_input, w_name, b_name], outputs=[out], name=name, axis=-1, epsilon=1e-5, stash_type=1
        )
        self.make_value(out, self.io_dtype, shape=shape)
        return out

    # ------------------------------------------------------------------
    # Audio attention (full bidirectional attention, no RoPE)
    # ------------------------------------------------------------------

    def _make_audio_attention(self, layer_id, attn, root_input):
        """Build the audio encoder self-attention block.

        Input/output: [seq_len, d_model].
        Full bidirectional attention (no causal mask, no rotary embeddings).
        k_proj has no bias; q_proj, v_proj, out_proj have bias.
        """
        b = f"/audio/layers.{layer_id}/self_attn"
        n = None  # seq_len is dynamic
        d = self.d_model
        nh = self.num_heads
        hd = self.head_dim

        # Q projection (with bias)
        q_w = f"{b[1:].replace('/', '.')}.q_proj.MatMul.weight"
        self.make_initializer(attn.q_proj.weight.T.detach(), q_w, to=self.io_dtype)
        q_mm = f"{b}/q_proj/MatMul/output_0"
        self.make_node("MatMul", inputs=[root_input, q_w], outputs=[q_mm], name=f"{b}/q_proj/MatMul")
        self.make_value(q_mm, self.io_dtype, shape=[n, d])
        q_bias = f"{b[1:].replace('/', '.')}.q_proj.Add.bias"
        self.make_initializer(attn.q_proj.bias.detach(), q_bias, to=self.io_dtype)
        q = f"{b}/q_proj/Add/output_0"
        self.make_node("Add", inputs=[q_mm, q_bias], outputs=[q], name=f"{b}/q_proj/Add")
        self.make_value(q, self.io_dtype, shape=[n, d])

        # K projection (no bias)
        k_w = f"{b[1:].replace('/', '.')}.k_proj.MatMul.weight"
        self.make_initializer(attn.k_proj.weight.T.detach(), k_w, to=self.io_dtype)
        k = f"{b}/k_proj/MatMul/output_0"
        self.make_node("MatMul", inputs=[root_input, k_w], outputs=[k], name=f"{b}/k_proj/MatMul")
        self.make_value(k, self.io_dtype, shape=[n, d])

        # V projection (with bias)
        v_w = f"{b[1:].replace('/', '.')}.v_proj.MatMul.weight"
        self.make_initializer(attn.v_proj.weight.T.detach(), v_w, to=self.io_dtype)
        v_mm = f"{b}/v_proj/MatMul/output_0"
        self.make_node("MatMul", inputs=[root_input, v_w], outputs=[v_mm], name=f"{b}/v_proj/MatMul")
        self.make_value(v_mm, self.io_dtype, shape=[n, d])
        v_bias = f"{b[1:].replace('/', '.')}.v_proj.Add.bias"
        self.make_initializer(attn.v_proj.bias.detach(), v_bias, to=self.io_dtype)
        v = f"{b}/v_proj/Add/output_0"
        self.make_node("Add", inputs=[v_mm, v_bias], outputs=[v], name=f"{b}/v_proj/Add")
        self.make_value(v, self.io_dtype, shape=[n, d])

        # Reshape to [seq_len, num_heads, head_dim]
        q_3d = self.make_reshape(f"{b}/q_reshape", [q, [0, nh, hd]], self.io_dtype, [n, nh, hd])
        k_3d = self.make_reshape(f"{b}/k_reshape", [k, [0, nh, hd]], self.io_dtype, [n, nh, hd])
        v_3d = self.make_reshape(f"{b}/v_reshape", [v, [0, nh, hd]], self.io_dtype, [n, nh, hd])

        # Transpose to [num_heads, seq_len, head_dim]
        q_t = self.make_transpose(f"{b}/q_t", q_3d, self.io_dtype, [nh, n, hd], perm=[1, 0, 2])
        k_t = self.make_transpose(f"{b}/k_t", k_3d, self.io_dtype, [nh, n, hd], perm=[1, 0, 2])
        v_t = self.make_transpose(f"{b}/v_t", v_3d, self.io_dtype, [nh, n, hd], perm=[1, 0, 2])

        # Attention scale constant (1/sqrt(head_dim)) in io_dtype
        scale_val = float(hd) ** -0.5
        scale_name = f"{b}/attn_scale"
        np_dtype = np.float32 if self.io_dtype in (ir.DataType.FLOAT, ir.DataType.BFLOAT16) else np.float16
        _scale_t = ir.Tensor(np.array(scale_val, dtype=np_dtype), name=scale_name)
        self.make_node("Constant", inputs=[], outputs=[scale_name], name=f"{scale_name}/Constant", value=_scale_t)
        self.make_value(scale_name, self.io_dtype, shape=[])

        # K^T for MatMul: transpose to [num_heads, head_dim, seq_len]
        k_t2 = self.make_transpose(f"{b}/k_t2", k_t, self.io_dtype, [nh, hd, n], perm=[0, 2, 1])

        # Q @ K^T → [num_heads, seq_len, seq_len]
        attn_w = f"{b}/attn_weights/MatMul/output_0"
        self.make_node("MatMul", inputs=[q_t, k_t2], outputs=[attn_w], name=f"{b}/attn_weights/MatMul")
        self.make_value(attn_w, self.io_dtype, shape=[nh, n, n])

        attn_ws = self.make_mul(f"{b}/attn_scale_mul", [attn_w, scale_name], self.io_dtype, [nh, n, n])
        attn_probs = self.make_softmax(f"{b}/attn_softmax", attn_ws, self.io_dtype, [nh, n, n])

        # Attn @ V → [num_heads, seq_len, head_dim]
        attn_out = f"{b}/attn_out/MatMul/output_0"
        self.make_node("MatMul", inputs=[attn_probs, v_t], outputs=[attn_out], name=f"{b}/attn_out/MatMul")
        self.make_value(attn_out, self.io_dtype, shape=[nh, n, hd])

        # Transpose back to [seq_len, num_heads, head_dim] and flatten to [seq_len, d_model]
        attn_back = self.make_transpose(f"{b}/attn_back", attn_out, self.io_dtype, [n, nh, hd], perm=[1, 0, 2])
        attn_flat = self.make_reshape(f"{b}/attn_flat", [attn_back, [0, d]], self.io_dtype, [n, d])

        # out_proj (with bias)
        out_w = f"{b[1:].replace('/', '.')}.out_proj.MatMul.weight"
        self.make_initializer(attn.out_proj.weight.T.detach(), out_w, to=self.io_dtype)
        out_mm = f"{b}/out_proj/MatMul/output_0"
        self.make_node("MatMul", inputs=[attn_flat, out_w], outputs=[out_mm], name=f"{b}/out_proj/MatMul")
        self.make_value(out_mm, self.io_dtype, shape=[n, d])
        out_bias = f"{b[1:].replace('/', '.')}.out_proj.Add.bias"
        self.make_initializer(attn.out_proj.bias.detach(), out_bias, to=self.io_dtype)
        out_add = f"{b}/out_proj/Add/output_0"
        self.make_node("Add", inputs=[out_mm, out_bias], outputs=[out_add], name=f"{b}/out_proj/Add")
        self.make_value(out_add, self.io_dtype, shape=[n, d])

        return out_add

    # ------------------------------------------------------------------
    # Single transformer layer
    # ------------------------------------------------------------------

    def _make_audio_layer(self, layer_id, layer, root_input):
        """Build one Qwen2_5OmniAudioEncoderLayer.

        Pattern: LN1 → Attention → residual → LN2 → MLP → residual.
        """
        n = None
        d = self.d_model
        ff = self.ffn_dim
        b = f"/audio/layers.{layer_id}"

        # Pre-attention LayerNorm
        ln1_out = self._make_audio_layer_norm(
            f"{b}/self_attn_layer_norm", root_input, layer.self_attn_layer_norm.weight, layer.self_attn_layer_norm.bias, [n, d]
        )

        # Attention
        attn_out = self._make_audio_attention(layer_id, layer.self_attn, ln1_out)

        # Residual 1
        residual1 = self.make_add(f"{b}/residual1/Add", [root_input, attn_out], self.io_dtype, [n, d])

        # Pre-MLP LayerNorm
        ln2_out = self._make_audio_layer_norm(
            f"{b}/final_layer_norm", residual1, layer.final_layer_norm.weight, layer.final_layer_norm.bias, [n, d]
        )

        # MLP: fc1 → GELU → fc2
        fc1_w = f"{b[1:].replace('/', '.')}.fc1.MatMul.weight"
        self.make_initializer(layer.fc1.weight.T.detach(), fc1_w, to=self.io_dtype)
        fc1_mm = f"{b}/fc1/MatMul/output_0"
        self.make_node("MatMul", inputs=[ln2_out, fc1_w], outputs=[fc1_mm], name=f"{b}/fc1/MatMul")
        self.make_value(fc1_mm, self.io_dtype, shape=[n, ff])
        fc1_b = f"{b[1:].replace('/', '.')}.fc1.Add.bias"
        self.make_initializer(layer.fc1.bias.detach(), fc1_b, to=self.io_dtype)
        fc1_out = f"{b}/fc1/Add/output_0"
        self.make_node("Add", inputs=[fc1_mm, fc1_b], outputs=[fc1_out], name=f"{b}/fc1/Add")
        self.make_value(fc1_out, self.io_dtype, shape=[n, ff])

        gelu_out = f"{b}/gelu/output_0"
        self.make_node("Gelu", inputs=[fc1_out], outputs=[gelu_out], name=f"{b}/gelu")
        self.make_value(gelu_out, self.io_dtype, shape=[n, ff])

        fc2_w = f"{b[1:].replace('/', '.')}.fc2.MatMul.weight"
        self.make_initializer(layer.fc2.weight.T.detach(), fc2_w, to=self.io_dtype)
        fc2_mm = f"{b}/fc2/MatMul/output_0"
        self.make_node("MatMul", inputs=[gelu_out, fc2_w], outputs=[fc2_mm], name=f"{b}/fc2/MatMul")
        self.make_value(fc2_mm, self.io_dtype, shape=[n, d])
        fc2_b = f"{b[1:].replace('/', '.')}.fc2.Add.bias"
        self.make_initializer(layer.fc2.bias.detach(), fc2_b, to=self.io_dtype)
        fc2_out = f"{b}/fc2/Add/output_0"
        self.make_node("Add", inputs=[fc2_mm, fc2_b], outputs=[fc2_out], name=f"{b}/fc2/Add")
        self.make_value(fc2_out, self.io_dtype, shape=[n, d])

        # Residual 2
        return self.make_add(f"{b}/residual2/Add", [residual1, fc2_out], self.io_dtype, [n, d])

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_hf_model(self, input_path):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        """Load HF weights and build the audio encoder ONNX graph."""
        hf_model = self.load_hf_model(input_path)
        hf_model.eval()
        audio = hf_model.audio_tower

        n = None  # dynamic sequence length

        # --- Graph input: float32 mel-spectrogram [num_mel_bins, n_frames] ---
        in_val = self.make_value("input_features", ir.DataType.FLOAT, shape=[self.num_mel_bins, n])
        self.graph.inputs.append(in_val)

        # Constant axes tensor for Unsqueeze/Squeeze at batch dim 0.
        _bax = ir.Tensor(np.array([0], dtype=np.int64), name="/audio/batch_ax")
        self.make_node("Constant", inputs=[], outputs=["/audio/batch_ax"], name="/audio/batch_ax/Constant", value=_bax)
        self.make_value("/audio/batch_ax", ir.DataType.INT64, shape=[1])

        # --- Unsqueeze to [1, num_mel_bins, n_frames] for Conv1d ---
        self.make_unsqueeze("/audio/unsqueeze_input", ["input_features", "/audio/batch_ax"], ir.DataType.FLOAT, [1, self.num_mel_bins, n])
        feat = "/audio/unsqueeze_input/output_0"

        # Cast to io_dtype if needed (Conv weights are stored in io_dtype)
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast("/audio/pv_cast", feat, self.io_dtype, [1, self.num_mel_bins, n])
            feat = "/audio/pv_cast/output_0"

        # --- Conv1: [1, num_mel_bins, n_frames] → [1, d_model, n_frames] + GELU ---
        conv1_w = "audio.conv1.Conv.weight"
        conv1_b = "audio.conv1.Conv.bias"
        self.make_initializer(audio.conv1.weight.detach(), conv1_w, to=self.io_dtype)
        self.make_initializer(audio.conv1.bias.detach(), conv1_b, to=self.io_dtype)
        self.make_conv(
            "/audio/conv1",
            [feat, conv1_w, conv1_b],
            self.io_dtype,
            [1, self.d_model, n],
            dilations=[1],
            group=1,
            kernel_shape=[3],
            pads=[1, 1],
            strides=[1],
        )
        conv1_gelu = "/audio/conv1_gelu/output_0"
        self.make_node("Gelu", inputs=["/audio/conv1/output_0"], outputs=[conv1_gelu], name="/audio/conv1_gelu")
        self.make_value(conv1_gelu, self.io_dtype, shape=[1, self.d_model, n])

        # --- Conv2: [1, d_model, n_frames] → [1, d_model, n_conv_out] + GELU (stride=2) ---
        conv2_w = "audio.conv2.Conv.weight"
        conv2_b = "audio.conv2.Conv.bias"
        self.make_initializer(audio.conv2.weight.detach(), conv2_w, to=self.io_dtype)
        self.make_initializer(audio.conv2.bias.detach(), conv2_b, to=self.io_dtype)
        self.make_conv(
            "/audio/conv2",
            [conv1_gelu, conv2_w, conv2_b],
            self.io_dtype,
            [1, self.d_model, n],
            dilations=[1],
            group=1,
            kernel_shape=[3],
            pads=[1, 1],
            strides=[2],
        )
        conv2_gelu = "/audio/conv2_gelu/output_0"
        self.make_node("Gelu", inputs=["/audio/conv2/output_0"], outputs=[conv2_gelu], name="/audio/conv2_gelu")
        self.make_value(conv2_gelu, self.io_dtype, shape=[1, self.d_model, n])

        # --- Transpose: [1, d_model, n_conv_out] → [1, n_conv_out, d_model] ---
        self.make_transpose("/audio/conv_transpose", conv2_gelu, self.io_dtype, [1, n, self.d_model], perm=[0, 2, 1])

        # --- Squeeze batch: [1, n_conv_out, d_model] → [n_conv_out, d_model] ---
        self.make_squeeze("/audio/squeeze_batch", ["/audio/conv_transpose/output_0", "/audio/batch_ax"], self.io_dtype, [n, self.d_model])
        hidden = "/audio/squeeze_batch/output_0"

        # --- Positional embedding (sinusoidal, pre-computed) ---
        # Shape: [max_source_positions, d_model] stored as float32.
        pos_emb_np = audio.positional_embedding.positional_embedding.detach().float().numpy()
        pos_emb_name = "audio.positional_embedding.weight"
        self.make_initializer(pos_emb_np, pos_emb_name)

        # Dynamically slice to [n_conv_out, d_model] using Shape + Gather + Slice.
        self.make_shape("/audio/hidden_shape", hidden, [2])
        _idx = ir.Tensor(np.array(0, dtype=np.int64), name="/audio/seq_len_idx")
        self.make_node("Constant", inputs=[], outputs=["/audio/seq_len_idx"], name="/audio/seq_len_idx/Constant", value=_idx)
        self.make_value("/audio/seq_len_idx", ir.DataType.INT64, shape=[])
        self.make_gather("/audio/seq_len", ["/audio/hidden_shape/output_0", "/audio/seq_len_idx"], ir.DataType.INT64, [], axis=0)

        # Unsqueeze scalar → [1] for use as Slice ends
        self.make_unsqueeze("/audio/seq_len_1d", ["/audio/seq_len/output_0", "/audio/batch_ax"], ir.DataType.INT64, [1])

        _starts = ir.Tensor(np.array([0], dtype=np.int64), name="/audio/pos_emb_starts")
        self.make_node("Constant", inputs=[], outputs=["/audio/pos_emb_starts"], name="/audio/pos_emb_starts/Constant", value=_starts)
        self.make_value("/audio/pos_emb_starts", ir.DataType.INT64, shape=[1])
        _axes = ir.Tensor(np.array([0], dtype=np.int64), name="/audio/pos_emb_axes")
        self.make_node("Constant", inputs=[], outputs=["/audio/pos_emb_axes"], name="/audio/pos_emb_axes/Constant", value=_axes)
        self.make_value("/audio/pos_emb_axes", ir.DataType.INT64, shape=[1])

        pos_sliced = "/audio/pos_emb_sliced/output_0"
        self.make_node(
            "Slice",
            inputs=[pos_emb_name, "/audio/pos_emb_starts", "/audio/seq_len_1d/output_0", "/audio/pos_emb_axes"],
            outputs=[pos_sliced],
            name="/audio/pos_emb_sliced",
        )
        self.make_value(pos_sliced, ir.DataType.FLOAT, shape=[n, self.d_model])

        # Cast positional embedding to io_dtype if needed
        if self.io_dtype != ir.DataType.FLOAT:
            self.make_cast("/audio/pos_emb_cast", pos_sliced, self.io_dtype, [n, self.d_model])
            pos_sliced = "/audio/pos_emb_cast/output_0"

        # Add positional embedding to hidden states
        hidden = self.make_add("/audio/pos_emb_add", [hidden, pos_sliced], self.io_dtype, [n, self.d_model])

        # --- Transformer layers ---
        for i, layer in enumerate(audio.layers):
            hidden = self._make_audio_layer(i, layer, hidden)

        # --- AvgPool1d: [seq_len, d_model] → pool along seq_len → [pooled, d_model] ---
        # Transpose: [seq_len, d_model] → [d_model, seq_len]
        self.make_transpose("/audio/pool_pre_t", hidden, self.io_dtype, [self.d_model, n], perm=[1, 0])
        # Unsqueeze: [d_model, seq_len] → [1, d_model, seq_len]
        self.make_unsqueeze("/audio/pool_unsqueeze", ["/audio/pool_pre_t/output_0", "/audio/batch_ax"], self.io_dtype, [1, self.d_model, n])
        # AveragePool (kernel=2, stride=2)
        pool_out = "/audio/avg_pool/output_0"
        self.make_node(
            "AveragePool",
            inputs=["/audio/pool_unsqueeze/output_0"],
            outputs=[pool_out],
            name="/audio/avg_pool",
            kernel_shape=[2],
            strides=[2],
        )
        self.make_value(pool_out, self.io_dtype, shape=[1, self.d_model, n])
        # Squeeze: [1, d_model, pooled] → [d_model, pooled]
        self.make_squeeze("/audio/pool_squeeze", [pool_out, "/audio/batch_ax"], self.io_dtype, [self.d_model, n])
        # Transpose: [d_model, pooled] → [pooled, d_model]
        self.make_transpose("/audio/pool_post_t", "/audio/pool_squeeze/output_0", self.io_dtype, [n, self.d_model], perm=[1, 0])
        hidden = "/audio/pool_post_t/output_0"

        # --- ln_post (LayerNorm) ---
        hidden = self._make_audio_layer_norm("/audio/ln_post", hidden, audio.ln_post.weight, audio.ln_post.bias, [n, self.d_model])

        # --- Linear projection: [pooled, d_model] → [pooled, output_dim] ---
        proj_w = "audio.proj.MatMul.weight"
        self.make_initializer(audio.proj.weight.T.detach(), proj_w, to=self.io_dtype)
        proj_mm = "/audio/proj/MatMul/output_0"
        self.make_node("MatMul", inputs=[hidden, proj_w], outputs=[proj_mm], name="/audio/proj/MatMul")
        self.make_value(proj_mm, self.io_dtype, shape=[n, self.output_dim])
        proj_b = "audio.proj.Add.bias"
        self.make_initializer(audio.proj.bias.detach(), proj_b, to=self.io_dtype)
        audio_feat = "/audio/proj/Add/output_0"
        self.make_node("Add", inputs=[proj_mm, proj_b], outputs=[audio_feat], name="/audio/proj/Add")
        self.make_value(audio_feat, self.io_dtype, shape=[n, self.output_dim])

        # --- Graph output ---
        self.make_node("Identity", inputs=[audio_feat], outputs=["audio_features"], name="/audio/output/Identity")
        out_val = self.make_value("audio_features", self.io_dtype, shape=[n, self.output_dim])
        self.graph.outputs.append(out_val)

        self.graph.sort()


class Qwen25OmniEmbeddingModel(EmbeddingModel):
    """ONNX embedding model for the Qwen2.5-Omni ``phi3v``-style multimodal pipeline.

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
        self.audio_token_id = extra_options.get("audio_token_id", _QWEN25OMNI_DEFAULT_AUDIO_TOKEN_ID)

    def load_hf_model(self, input_path):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        src = input_path if os.path.isdir(input_path) else self.model_name_or_path
        extra_kwargs = {} if os.path.isdir(input_path) else {"cache_dir": self.cache_dir}
        return Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            src, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
        )

    def get_embed_weight(self, hf_model):
        return hf_model.model.embed_tokens.weight.detach().float().numpy()

    def make_model(self, input_path):
        """Load HF weights and build the embedding+scatter ONNX graph."""
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


class Qwen25OmniConditionalGenerationModel(Model):
    """Orchestrates exporting vision encoder, audio encoder, embedding model,
    and thinker (text decoder) for ``Qwen2_5OmniThinkerForConditionalGeneration``.

    The exported artifacts are:

    * ``vision_encoder.onnx`` – Qwen2.5-Omni vision tower + patch merger.
    * ``audio_encoder.onnx`` – Qwen2.5-Omni audio tower (Conv1d + transformer + projection).
    * ``embedding.onnx`` – token-embedding table + image/audio feature scatter.
    * ``model.onnx`` – Qwen2.5-Omni thinker text decoder.
    * ``genai_config.json`` – ``phi3v``-type VLM config for ORT-GenAI.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # --- Vision encoder ---
        self.vision_encoder = Qwen25OmniVisionEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # --- Audio encoder ---
        self.audio_encoder = Qwen25OmniAudioEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # --- Flatten text config onto a working config copy for remaining models ---
        text_obj_config = copy.deepcopy(config)
        if hasattr(config, "thinker_config") and hasattr(config.thinker_config, "text_config"):
            text_config = config.thinker_config.text_config
        elif hasattr(config, "text_config"):
            text_config = config.text_config
        else:
            text_config = None
        if text_config is not None:
            # HF PretrainedConfig.__iter__ yields attribute names (same as dict-style
            # iteration used in builder.py for Qwen2_5_VLForConditionalGeneration).
            for key in text_config:
                if not hasattr(text_obj_config, key) or getattr(text_obj_config, key) is None:
                    setattr(text_obj_config, key, getattr(text_config, key))

        # image_token_id and audio_token_id are stored at the top-level config.
        image_token_id = getattr(config, "image_token_id", getattr(config, "image_token_index", _QWEN25OMNI_DEFAULT_IMAGE_TOKEN_ID))
        audio_token_id = getattr(config, "audio_token_id", getattr(config, "audio_token_index", _QWEN25OMNI_DEFAULT_AUDIO_TOKEN_ID))

        # --- Embedding model (always float32 for the embedding table) ---
        embed_extra_options = dict(extra_options)
        embed_extra_options["image_token_id"] = image_token_id
        embed_extra_options["audio_token_id"] = audio_token_id
        self.embedding_model = Qwen25OmniEmbeddingModel(text_obj_config, io_dtype, ir.DataType.FLOAT, ep, cache_dir, embed_extra_options)

        # --- Thinker (text decoder) ---
        text_extra_options = dict(extra_options)
        text_extra_options["exclude_embeds"] = True
        self.text_model = Qwen25OmniThinkerModel(text_obj_config, io_dtype, onnx_dtype, ep, cache_dir, text_extra_options)

    # ------------------------------------------------------------------
    # builder.py interface
    # ------------------------------------------------------------------

    def make_model(self, input_path):
        print("Building vision encoder for Qwen2.5-Omni...")
        self.vision_encoder.make_model(input_path)
        print("Building audio encoder for Qwen2.5-Omni...")
        self.audio_encoder.make_model(input_path)
        print("Building embedding model for Qwen2.5-Omni...")
        self.embedding_model.make_model(input_path)
        print("Building thinker text decoder for Qwen2.5-Omni...")
        self.text_model.make_model(input_path)

    def save_model(self, out_dir):
        self.vision_encoder.save_model(out_dir)
        self.audio_encoder.save_model(out_dir)
        self.embedding_model.save_model(out_dir)
        self.text_model.save_model(out_dir)

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        # Write the text model genai_config.json first, then extend it.
        self.text_model.make_genai_config(model_name_or_path, extra_kwargs, out_dir)

        config_path = os.path.join(out_dir, "genai_config.json")
        with open(config_path) as f:
            genai_config = json.load(f)

        spatial_merge_size = self.vision_encoder.vision_config.spatial_merge_size

        # ORT-GenAI uses "phi3v" as the model type for the multimodal pipeline.
        genai_config["model"]["type"] = "phi3v"

        # ORT-GenAI's phi3v parser only accepts "pixel_values" for vision
        # inputs; "rotary_pos_emb" is not a recognised field and would cause
        # a JSON-parsing error at model-load time.  The rotary position
        # embedding must therefore be pre-computed outside ORT-GenAI and
        # baked into the ONNX graph or passed through a separate mechanism.
        genai_config["model"]["vision"] = {
            "filename": self.vision_encoder.FILENAME,
            "spatial_merge_size": spatial_merge_size,
            "inputs": {"pixel_values": "pixel_values"},
            "outputs": {"image_features": "image_features"},
        }

        genai_config["model"]["speech"] = {
            "filename": self.audio_encoder.FILENAME,
            "inputs": {"input_features": "input_features"},
            "outputs": {"audio_features": "audio_features"},
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


class Qwen3VLTextModel(Qwen25VLTextModel):
    """
    Qwen3-VL text model builder. Inherits from Qwen25VLTextModel.

    Key differences from Qwen2.5-VL:
    - Uses interleaved MRoPE layout [THWTHWTHW...TT] instead of chunked [TTT...HHH...WWW]
    - Adds QK normalization (q_norm, k_norm) from Qwen3 base architecture
    - Default mrope_section is [24, 20, 20] (vs [16, 24, 24] in Qwen2.5-VL)
    - Vision encoder uses DeepStack for multi-layer feature injection (handled by vision ONNX model)
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # Fix model_type: HF architecture "Qwen3VLForConditionalGeneration" would produce "qwen3vl"
        # but the C++ runtime expects "qwen3_vl" (with underscore).
        # Intentional override of the superclass attribute (used in genai_config.json).
        self.model_type = "Qwen3_VLForConditionalGeneration"  # overrides Model.model_type on purpose

        # Qwen3 attention uses QK normalization
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True

    def make_attention_qk_subgraph(self, layer_id, attention, root_input, **kwargs):
        # Qwen3-VL adds QK normalization before MRoPE rotation
        # The parent class (Qwen25VLTextModel) skips make_qk_norm since Qwen2.5-VL doesn't use it.
        # We must call it here before proceeding with MRoPE.
        if self.attention_attrs["q_norm"] and self.attention_attrs["k_norm"]:
            self.make_qk_norm(layer_id, attention)

        # Delegate to parent for MRoPE rotation + GQA
        super().make_attention_qk_subgraph(layer_id, attention, root_input, **kwargs)

    def make_mrope_flattened_caches(self, layer_id, dyn_cos, dyn_sin):
        """
        Converts the 3D MRoPE caches [3, B, S, H] into flattened, interleaved caches [B*S, H/2]
        suitable for the RotaryEmbedding operator.

        Qwen3-VL uses interleaved MRoPE layout: [THWTHWTHW...TT]
        This differs from Qwen2.5-VL's chunked layout: [TTT...HHH...WWW]

        The interleaving logic (from HuggingFace Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope):
          freqs_t = freqs[0]  # start with temporal
          for dim, offset in enumerate((1, 2), start=1):  # H=1, W=2
              length = mrope_section[dim] * 3
              idx = slice(offset, length, 3)
              freqs_t[..., idx] = freqs[dim, ..., idx]

        For mrope_section = [24, 20, 20], head_dim/2 = 64:
          - All 64 positions start with Temporal values
          - Height overwrites positions [1, 4, 7, ..., 58] (20 values)
          - Width overwrites positions [2, 5, 8, ..., 59] (20 values)
          - Result pattern: [T,H,W, T,H,W, ..., T,H,W, T,T,T,T] (20 THW groups + 4 T-only)
        """
        basename = f"/model/layers.{layer_id}/attn/mrope_interleaved_cache"
        shared_base = "/model/attn/mrope_interleaved_cache"

        half_head = self.head_size // 2

        # Cache the deterministic index mappings on self so we compute them once
        # and emit shared ONNX Constant nodes that all layers reference.
        if not hasattr(self, "_mrope_cache"):
            # Pre-compute the interleaved index mapping: for each position in H/2,
            # which dimension (0=T, 1=H, 2=W)?
            dim_assignments = [0] * half_head  # Start all positions as Temporal
            for dim_idx, offset in enumerate((1, 2), start=1):  # H=1, W=2
                length = self.mrope_sections[dim_idx] * 3
                for i in range(offset, length, 3):
                    if i < half_head:
                        dim_assignments[i] = dim_idx

            dim_to_positions = {0: [], 1: [], 2: []}
            for pos, dim in enumerate(dim_assignments):
                dim_to_positions[dim].append(pos)

            # Build reorder indices (same for cos and sin, all layers)
            concat_order = []
            for dim_idx in range(3):
                concat_order.extend(dim_to_positions[dim_idx])
            reorder_indices = [0] * half_head
            for concat_idx, orig_pos in enumerate(concat_order):
                reorder_indices[orig_pos] = concat_idx

            # Emit shared position constants (one per dimension, reused across all layers)
            positions_outputs = {}
            for dim_idx in range(3):
                positions = dim_to_positions[dim_idx]
                if not positions:
                    continue
                pname = f"{shared_base}/dim{dim_idx}/Positions/Constant"
                pout = f"{shared_base}/dim{dim_idx}/positions"
                self.make_node("Constant", [], [pout], name=pname, value=ir.tensor(torch.tensor(positions, dtype=torch.int64), name=pout))
                self.make_value(pout, ir.DataType.INT64, [len(positions)])
                positions_outputs[dim_idx] = pout

            # Emit shared reorder constant
            rname = f"{shared_base}/Reorder/Constant"
            rout = f"{shared_base}/reorder"
            self.make_node("Constant", [], [rout], name=rname, value=ir.tensor(torch.tensor(reorder_indices, dtype=torch.int64), name=rout))
            self.make_value(rout, ir.DataType.INT64, [half_head])

            self._mrope_cache = {"dim_to_positions": dim_to_positions, "positions_outputs": positions_outputs, "reorder_output": rout}

        dim_to_positions = self._mrope_cache["dim_to_positions"]
        positions_outputs = self._mrope_cache["positions_outputs"]
        reorder_const_output = self._mrope_cache["reorder_output"]

        def process_cache(input_name, name_suffix):
            # 1. Slice to H/2: [3, B, S, H] -> [3, B, S, H/2]
            slice_name = f"{basename}/{name_suffix}/half/Slice"
            slice_output = f"{slice_name}/output_0"
            self.make_slice(
                slice_name,
                [input_name, "/model/constants/INT64/[0]", f"/model/constants/INT64/[{half_head}]", "/model/constants/INT64/[-1]"],
                ir.DataType.FLOAT,
                [3, "batch_size", "sequence_length", half_head],
            )

            # 2. Build interleaved output by gathering individual positions from appropriate dimensions
            gathered_pieces = []
            for dim_idx in range(3):
                positions = dim_to_positions[dim_idx]
                if not positions:
                    continue

                # Gather this dimension: [3, B, S, H/2] -> [1, B, S, H/2] via index dim_idx on axis 0
                gather_dim_name = f"{basename}/{name_suffix}/dim{dim_idx}/Gather"
                gather_dim_output = f"{gather_dim_name}/output_0"
                self.make_node(
                    "Gather", [slice_output, f"/model/constants/INT64/[{dim_idx}]"], [gather_dim_output], name=gather_dim_name, axis=0
                )

                squeeze_dim_name = f"{basename}/{name_suffix}/dim{dim_idx}/Squeeze"
                squeeze_dim_output = f"{squeeze_dim_name}/output_0"
                self.make_squeeze(
                    squeeze_dim_name,
                    [gather_dim_output, "/model/constants/INT64/[0]"],
                    ir.DataType.FLOAT,
                    ["batch_size", "sequence_length", half_head],
                )

                # Gather specific positions (reuse shared constant node)
                gather_pos_name = f"{basename}/{name_suffix}/dim{dim_idx}/Positions/Gather"
                self.make_gather(
                    gather_pos_name,
                    [squeeze_dim_output, positions_outputs[dim_idx]],
                    ir.DataType.FLOAT,
                    ["batch_size", "sequence_length", len(positions)],
                    axis=-1,
                )
                gather_pos_output = f"{gather_pos_name}/output_0"

                gathered_pieces.append((positions, gather_pos_output))

            # 3. Concatenate all pieces and reorder to interleaved layout
            all_outputs = [(positions, output) for positions, output in gathered_pieces]

            if len(all_outputs) == 1:
                concat_output = all_outputs[0][1]
            else:
                concat_name = f"{basename}/{name_suffix}/AllPieces/Concat"
                concat_output = f"{concat_name}/output_0"
                self.make_concat(
                    concat_name, [out for _, out in all_outputs], ir.DataType.FLOAT, ["batch_size", "sequence_length", half_head], axis=-1
                )

            # Reorder using shared constant
            gather_reorder_name = f"{basename}/{name_suffix}/Reorder/Gather"
            self.make_gather(
                gather_reorder_name,
                [concat_output, reorder_const_output],
                ir.DataType.FLOAT,
                ["batch_size", "sequence_length", half_head],
                axis=-1,
            )
            gather_reorder_output = f"{gather_reorder_name}/output_0"

            # 4. Flatten: -> [B*S, H/2]
            reshape_name = f"{basename}/{name_suffix}_flat/Reshape"
            reshape_output = f"{reshape_name}/output_0"
            self.make_reshape(
                reshape_name,
                [gather_reorder_output, f"/model/constants/INT64/[-1, {half_head}]"],
                ir.DataType.FLOAT,
                ["total_token_count", half_head],
            )
            return reshape_output

        flat_cos = process_cache(dyn_cos, "cos")
        flat_sin = process_cache(dyn_sin, "sin")

        return flat_cos, flat_sin

    def load_weights(self, input_path):
        # Load the Hugging Face model
        print("Loading Qwen3VLForConditionalGeneration model...")
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
        )


class Qwen35TextModel(Model):
    """Qwen3.5 hybrid model builder.

    Qwen3.5 uses a hybrid architecture with two layer types:
    - ``full_attention``: Attention with doubled Q projection (Q + output gate),
      per-head QK RMSNorm, partial rotary embeddings, and output gating
    - ``linear_attention``: GatedDeltaNet recurrent layer with depthwise
      causal conv1d, L2-normalised Q/K, and linear attention recurrence

    The layer type pattern is controlled by ``config.layer_types`` (or
    derived from ``config.full_attention_interval``).

    Both layer types use OffsetRMSNorm (the ``1 + weight`` variant).
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        # Qwen3.5 is a VL model. The decoder takes inputs_embeds.
        if "exclude_embeds" not in extra_options:
            extra_options["exclude_embeds"] = True
            print("Setting exclude_embeds=True for Qwen3.5 VL decoder.")

        # Qwen3.5 is a multimodal model whose HF config nests text config
        # under text_config. Flatten text_config attributes onto config so
        # the base Model init finds them where it expects.
        if hasattr(config, "text_config"):
            text_config = config.text_config
            for key in text_config:
                if not hasattr(config, key) or getattr(config, key) is None:
                    setattr(config, key, getattr(text_config, key))

        # rope_scaling contains the actual rope_theta for Qwen3.5
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            if "rope_theta" in config.rope_scaling:
                config.rope_theta = config.rope_scaling["rope_theta"]
            if "partial_rotary_factor" in config.rope_scaling:
                config.partial_rotary_factor = config.rope_scaling["partial_rotary_factor"]

        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)

        # OffsetRMSNorm: Qwen3.5 uses (1 + weight) * RMSNorm(x).
        # Pre-bake the +1 into the weight initializer so the base class's
        # SkipSimplifiedLayerNormalization can be used directly.
        self.layernorm_attrs["add_offset"] = 1

        # HF Qwen3_5RMSNorm always computes in float32 regardless of model
        # dtype.  Force the builder to cast inputs to fp32 before LayerNorm
        # and cast back after, matching HF behaviour and preventing precision
        # loss that compounds across 36+ layers in fp16/bf16 builds.
        self.layernorm_attrs["cast"]["use_fp32"] = True
        self.layernorm_attrs["cast"]["root_input"] = True
        self.layernorm_attrs["cast"]["skip_input"] = True
        self.layernorm_attrs["cast"]["output_0"] = True
        self.layernorm_attrs["cast"]["output_3"] = True

        # 3D position_ids for mRoPE: [3, batch_size, sequence_length]
        self.input_shapes["position_ids"] = [3, "batch_size", "sequence_length"]
        self.input_names["position_ids"] = "position_ids"

        # mRoPE config
        self.mrope_sections = self.rope_attrs.get("mrope", {}).get("sections", [])
        if not self.mrope_sections:
            raise ValueError("MRoPE sections not found in config.text_config.rope_scaling.mrope_section")
        if len(self.mrope_sections) != 3:
            raise ValueError(f"Expected 3 MRoPE sections [T, H, W], got {len(self.mrope_sections)}: {self.mrope_sections}")
        self.mrope_rotary_dim = int(self.rope_attrs["partial_rotary_factor"] * self.head_size)

        # Force RoPE computation in float32 for numerical stability
        self.rope_attrs["cast_to_fp32"] = True

        # Pre-compute cos/sin cache tables and interleaving masks for mRoPE
        self._make_rotary_caches()

        # Parse layer types
        if hasattr(config, "layer_types") and config.layer_types is not None:
            self.layer_types = list(config.layer_types)
        elif hasattr(config, "full_attention_interval") and config.full_attention_interval is not None:
            interval = config.full_attention_interval
            self.layer_types = ["full_attention" if (i + 1) % interval == 0 else "linear_attention" for i in range(self.num_layers)]
        else:
            self.layer_types = ["full_attention"] * self.num_layers

        # Store linear attention config
        self.linear_key_head_dim = getattr(config, "linear_key_head_dim", 128)
        self.linear_value_head_dim = getattr(config, "linear_value_head_dim", 128)
        self.linear_num_key_heads = getattr(config, "linear_num_key_heads", 16)
        self.linear_num_value_heads = getattr(config, "linear_num_value_heads", 16)
        self.linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

        # Derived dimensions for GatedDeltaNet
        self.linear_key_dim = self.linear_num_key_heads * self.linear_key_head_dim
        self.linear_value_dim = self.linear_num_value_heads * self.linear_value_head_dim
        self.linear_conv_dim = self.linear_key_dim * 2 + self.linear_value_dim

        # Full attention uses QK norm and output gating
        self.attention_attrs["q_norm"] = True
        self.attention_attrs["k_norm"] = True
        # Disable fused RoPE in attention op - we apply mRoPE manually
        self.attention_attrs["use_rope_in_attn"] = False

        # Mixed-precision quantization for linear attention layers.
        # Baseline: whole model INT4. Override linear attention layer nodes
        # to INT8 for better accuracy with modest size increase.
        #
        # Linear attention recurrence accumulates errors across the full sequence,
        # unlike softmax attention which normalizes per-step.
        int8_nodes = {}
        for i, lt in enumerate(self.layer_types):
            if lt == "linear_attention":
                # All linear attention projections: INT8
                for proj in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z", "out_proj"):
                    int8_nodes[f"/model/layers.{i}/linear_attn/{proj}/MatMul"] = {"bits": 8}
                # MLP projections in linear attention layers: INT8
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    int8_nodes[f"/model/layers.{i}/mlp/{proj}/MatMul"] = {"bits": 8}

        if int8_nodes:
            algo_config = self.quant_attrs["int4"].get("algo_config")
            if algo_config is not None and hasattr(algo_config, "customized_weight_config"):
                algo_config.customized_weight_config.update(int8_nodes)
            else:
                algo_config = RTNWeightOnlyQuantConfig(customized_weight_config=int8_nodes)
                self.quant_attrs["int4"]["algo_config"] = algo_config

        # Replace standard KV cache I/O with hybrid cache I/O
        self._setup_hybrid_cache_io()

    def _setup_hybrid_cache_io(self):
        """Set up hybrid cache I/O: KV cache for attention layers,
        conv_state + recurrent_state for linear attention layers."""

        # The base class creates KV cache entries for all num_layers.
        # We rebuild the lists: keep KV entries only for full-attention layers,
        # and add conv/recurrent state entries for linear-attention layers.
        kv_key_inputs = self.input_names["past_key_values.key"]
        kv_value_inputs = self.input_names["past_key_values.value"]
        kv_key_outputs = self.output_names["present.key"]
        kv_value_outputs = self.output_names["present.value"]

        filtered_key_inputs = []
        filtered_value_inputs = []
        filtered_key_outputs = []
        filtered_value_outputs = []

        for i, lt in enumerate(self.layer_types):
            if lt == "full_attention":
                filtered_key_inputs.append(kv_key_inputs[i])
                filtered_value_inputs.append(kv_value_inputs[i])
                filtered_key_outputs.append(kv_key_outputs[i])
                filtered_value_outputs.append(kv_value_outputs[i])
            else:
                # Fused CausalConvWithState + LinearAttention ops use same dtype as activations.
                state_dtype = self.io_dtype

                # linear_attention: add conv_state + recurrent_state
                self.input_names[f"past_state.{i}.conv"] = f"past_key_values.{i}.conv_state"
                self.input_types[f"past_state.{i}.conv"] = state_dtype
                self.input_shapes[f"past_state.{i}.conv"] = ["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim - 1]

                self.input_names[f"past_state.{i}.recurrent"] = f"past_key_values.{i}.recurrent_state"
                self.input_types[f"past_state.{i}.recurrent"] = state_dtype
                self.input_shapes[f"past_state.{i}.recurrent"] = [
                    "batch_size",
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                ]

                self.output_names[f"present_state.{i}.conv"] = f"present.{i}.conv_state"
                self.output_types[f"present_state.{i}.conv"] = state_dtype
                self.output_shapes[f"present_state.{i}.conv"] = ["batch_size", self.linear_conv_dim, self.linear_conv_kernel_dim - 1]

                self.output_names[f"present_state.{i}.recurrent"] = f"present.{i}.recurrent_state"
                self.output_types[f"present_state.{i}.recurrent"] = state_dtype
                self.output_shapes[f"present_state.{i}.recurrent"] = [
                    "batch_size",
                    self.linear_num_value_heads,
                    self.linear_key_head_dim,
                    self.linear_value_head_dim,
                ]

        self.input_names["past_key_values.key"] = filtered_key_inputs
        self.input_names["past_key_values.value"] = filtered_value_inputs
        self.output_names["present.key"] = filtered_key_outputs
        self.output_names["present.value"] = filtered_value_outputs

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        """Dispatch to full attention or GatedDeltaNet based on layer type."""
        if self.layer_types[layer_id] == "linear_attention":
            self._make_linear_attention(layer_id, attention, root_input)
        else:
            self._make_full_attention(layer_id, attention, root_input)

    def make_layer(self, layer_id, layer):
        """Override to pass ``linear_attn`` instead of ``self_attn`` for
        linear-attention layers (the base class assumes ``self_attn``)."""
        attn_module = layer.linear_attn if self.layer_types[layer_id] == "linear_attention" else layer.self_attn
        self.make_layernorm(
            layer_id,
            layer.input_layernorm,
            skip=not self.layernorm_attrs["first_layernorm"],
            simple=self.layernorm_attrs["simple"],
            location="input",
        )
        self.make_attention(layer_id, attn_module, root_input=self.layernorm_attrs["output_0"])
        self.make_layernorm(
            layer_id, layer.post_attention_layernorm, skip=True, simple=self.layernorm_attrs["simple"], location="post_attention"
        )
        self.make_mlp(layer_id, layer.mlp, root_input=self.layernorm_attrs["output_0"])

        self.layernorm_attrs["first_layernorm"] = False
        if layer_id == self.num_layers - 1:
            self.layernorm_attrs["last_layernorm"] = True

    def _make_full_attention(self, layer_id, attn, root_input):
        """Build full attention with output gating.

        Qwen3.5 full attention has a doubled Q projection that produces both
        Q and a gating signal. After attention, the output is multiplied by
        sigmoid(gate) before the output projection.
        """
        # 1. Q projection (doubled: outputs Q and gate)
        q_matmul_name = f"/model/layers.{layer_id}/attn/q_proj/MatMul"
        self.make_matmul(attn.q_proj, q_matmul_name, root_input)
        q_gate_path = f"{q_matmul_name}/output_0"

        # Split Q and gate PER-HEAD: reshape [B,S,N*2H] -> [B,S,N,2H] -> split -> [B,S,N,H] each -> reshape back
        q_size = self.num_attn_heads * self.head_size

        # Reshape to [B, S, N, 2*H]
        rs_qg_name = f"/model/layers.{layer_id}/attn/q_gate_reshape/Reshape"
        rs_qg_output = f"{rs_qg_name}/output_0"
        self.make_reshape(
            rs_qg_name,
            [q_gate_path, f"/model/constants/INT64/[0, 0, {self.num_attn_heads}, {self.head_size * 2}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", self.num_attn_heads, self.head_size * 2],
        )

        # Split per-head: [B, S, N, 2H] -> [B, S, N, H] + [B, S, N, H]
        split_name = f"/model/layers.{layer_id}/attn/q_gate_split/Split"
        q_4d_output = f"{split_name}/output_0"
        gate_4d_output = f"{split_name}/output_1"
        self.make_node(
            "Split",
            [rs_qg_output, f"/model/constants/INT64/[{self.head_size}, {self.head_size}]"],
            [q_4d_output, gate_4d_output],
            name=split_name,
            axis=-1,
        )
        self.make_value(q_4d_output, self.io_dtype, ["batch_size", "sequence_length", self.num_attn_heads, self.head_size])
        self.make_value(gate_4d_output, self.io_dtype, ["batch_size", "sequence_length", self.num_attn_heads, self.head_size])

        # Reshape Q back to [B, S, N*H]
        rs_q_name = f"/model/layers.{layer_id}/attn/q_reshape/Reshape"
        q_output = f"{rs_q_name}/output_0"
        self.make_reshape(
            rs_q_name, [q_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"], self.io_dtype, ["batch_size", "sequence_length", q_size]
        )

        # Reshape gate back to [B, S, N*H]
        rs_g_name = f"/model/layers.{layer_id}/attn/gate_reshape/Reshape"
        gate_output = f"{rs_g_name}/output_0"
        self.make_reshape(
            rs_g_name,
            [gate_4d_output, f"/model/constants/INT64/[0, 0, {q_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", q_size],
        )

        self.attention_attrs["q_path"] = q_output

        # 3. K and V projections
        k_matmul_name = f"/model/layers.{layer_id}/attn/k_proj/MatMul"
        self.make_matmul(attn.k_proj, k_matmul_name, root_input)
        self.attention_attrs["k_path"] = f"{k_matmul_name}/output_0"

        v_matmul_name = f"/model/layers.{layer_id}/attn/v_proj/MatMul"
        self.make_matmul(attn.v_proj, v_matmul_name, root_input)
        self.attention_attrs["v_path"] = f"{v_matmul_name}/output_0"

        # 4. Per-head QK RMSNorm (on Q and K separately)
        #    The base class's make_qk_norm uses SimplifiedLayerNormalization
        #    and pre-bakes the +1 offset via layernorm_attrs["add_offset"].
        self.make_qk_norm(layer_id, attn)

        # 5. Apply interleaved mRoPE to Q and K
        if self.attention_attrs["rope"]:
            q_shape = ["batch_size", "sequence_length", self.num_attn_heads * self.head_size]
            k_shape = ["batch_size", "sequence_length", self.num_kv_heads * self.head_size]

            # Build interleaved cos/sin from pre-computed cache + 3D position_ids
            cos_dyn, sin_dyn = self._make_mrope_cos_sin("/model/rotary_emb")

            # Apply mRoPE rotation to Q
            self.attention_attrs["q_path"] = self._apply_mrope_rotation(
                layer_id,
                self.attention_attrs["q_path"],
                q_shape,
                cos_dyn,
                sin_dyn,
                self.num_attn_heads,
                f"/model/layers.{layer_id}/attn/q_mrope",
            )

            # Apply mRoPE rotation to K
            self.attention_attrs["k_path"] = self._apply_mrope_rotation(
                layer_id,
                self.attention_attrs["k_path"],
                k_shape,
                cos_dyn,
                sin_dyn,
                self.num_kv_heads,
                f"/model/layers.{layer_id}/attn/k_mrope",
            )

        # 6. GroupQueryAttention with per-layer KV cache
        past_k = f"past_key_values.{layer_id}.key"
        past_v = f"past_key_values.{layer_id}.value"
        present_k = f"present.{layer_id}.key"
        present_v = f"present.{layer_id}.value"

        attn_name = f"/model/layers.{layer_id}/attn/{self.attention_attrs['op_type']}"
        self.make_attention_op(
            attn_name,
            q_path=self.attention_attrs["q_path"],
            k_path=self.attention_attrs["k_path"],
            v_path=self.attention_attrs["v_path"],
            past_k=past_k,
            past_v=past_v,
            present_k=present_k,
            present_v=present_v,
            cos_cache="",
            sin_cache="",
        )
        attn_output = f"{attn_name}/output_0"

        # 7. Output gating: attn_output * sigmoid(gate)
        sigmoid_name = f"/model/layers.{layer_id}/attn/gate/Sigmoid"
        self.make_sigmoid(sigmoid_name, gate_output, self.io_dtype, ["batch_size", "sequence_length", q_size])
        sigmoid_output = f"{sigmoid_name}/output_0"

        gated_name = f"/model/layers.{layer_id}/attn/gate/Mul"
        self.make_mul(gated_name, [attn_output, sigmoid_output], self.io_dtype, ["batch_size", "sequence_length", q_size])
        gated_output = f"{gated_name}/output_0"

        # 8. Output projection
        o_matmul_name = f"/model/layers.{layer_id}/attn/o_proj/MatMul"
        self.make_matmul(attn.o_proj, o_matmul_name, gated_output)
        self.layernorm_attrs["skip_input"] = f"{o_matmul_name}/output_0"

    def _make_rotary_caches(self):
        """Pre-compute cos/sin cache table and h/w interleaving masks.

        Matches the reference model's approach:
        - cos_cache [max_len, rdim_half]: pre-computed cos(pos * inv_freq)
        - sin_cache [max_len, rdim_half]: pre-computed sin(pos * inv_freq)
        - h_mask [rdim_half]: bool mask for height positions
        - w_mask [rdim_half]: bool mask for width positions
        """
        rdim = self.mrope_rotary_dim
        rdim_half = rdim // 2
        max_len = self.context_length

        inv_freq = 1.0 / (
            self.rope_attrs["rescale_factors"] * (self.rope_attrs["theta"] ** (torch.arange(0, rdim, 2, dtype=torch.int64).float() / rdim))
        )

        positions = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)  # [max_len, rdim_half]
        cos_cache = torch.cos(freqs)
        sin_cache = torch.sin(freqs)

        self.make_initializer(cos_cache, "model.rotary_emb.cos_cache", to=ir.DataType.FLOAT)
        self.make_initializer(sin_cache, "model.rotary_emb.sin_cache", to=ir.DataType.FLOAT)

        # Build interleaving masks
        dim_assignments = [0] * rdim_half
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = self.mrope_sections[dim_idx] * 3
            for i in range(offset, length, 3):
                if i < rdim_half:
                    dim_assignments[i] = dim_idx

        h_mask = torch.tensor([d == 1 for d in dim_assignments], dtype=torch.bool)
        w_mask = torch.tensor([d == 2 for d in dim_assignments], dtype=torch.bool)

        self.make_initializer(h_mask, "model.rotary_emb.h_mask", to=ir.DataType.BOOL)
        self.make_initializer(w_mask, "model.rotary_emb.w_mask", to=ir.DataType.BOOL)
        print(f"Created rotary caches [{max_len}, {rdim_half}] + h/w masks [{rdim_half}].")

    def _get_shared_q_scale(self, head_dim):
        """Return the name of a shared 1/sqrt(head_dim) constant (created once)."""
        name = "model.constants.q_scale"
        scale_val = float(1.0 / np.sqrt(head_dim))
        self.make_initializer(torch.tensor([scale_val], dtype=torch.float32), name, to=self.io_dtype)
        return name

    def _get_shared_l2_eps(self):
        """Return the name of a shared L2 epsilon constant (created once)."""
        name = "model.constants.l2_eps"
        self.make_initializer(torch.tensor([1e-6], dtype=torch.float32), name, to=self.io_dtype)
        return name

    def _make_mrope_cos_sin(self, basename):
        """Build interleaved mRoPE cos/sin from pre-computed cache + position_ids.

        Input: position_ids [3, B, S]
        Output: cos [B, S, rdim_half], sin [B, S, rdim_half]
        """
        pos_ids = self.input_names["position_ids"]
        cos_cache = "model.rotary_emb.cos_cache"
        sin_cache = "model.rotary_emb.sin_cache"
        h_mask = "model.rotary_emb.h_mask"
        w_mask = "model.rotary_emb.w_mask"
        rdim_half = self.mrope_rotary_dim // 2

        def gather_dim(dim_idx, cache_name, suffix):
            g_name = f"{basename}/{suffix}/dim{dim_idx}/pos/Gather"
            self.make_gather(
                g_name, [pos_ids, f"/model/constants/INT64/[{dim_idx}]"], ir.DataType.INT64, [1, "batch_size", "sequence_length"], axis=0
            )
            sq_name = f"{basename}/{suffix}/dim{dim_idx}/Squeeze"
            self.make_squeeze(
                sq_name, [f"{g_name}/output_0", "/model/constants/INT64/[0]"], ir.DataType.INT64, ["batch_size", "sequence_length"]
            )
            gc_name = f"{basename}/{suffix}/dim{dim_idx}/cache/Gather"
            self.make_gather(
                gc_name, [cache_name, f"{sq_name}/output_0"], ir.DataType.FLOAT, ["batch_size", "sequence_length", rdim_half], axis=0
            )
            return f"{gc_name}/output_0"

        def interleave(suffix, cache_name):
            t = gather_dim(0, cache_name, suffix)
            h = gather_dim(1, cache_name, suffix)
            w = gather_dim(2, cache_name, suffix)
            ww_name = f"{basename}/{suffix}/w/Where"
            self.make_where(ww_name, [w_mask, w, t], ir.DataType.FLOAT, ["batch_size", "sequence_length", rdim_half])
            ww_out = f"{ww_name}/output_0"
            hh_name = f"{basename}/{suffix}/h/Where"
            self.make_where(hh_name, [h_mask, h, ww_out], ir.DataType.FLOAT, ["batch_size", "sequence_length", rdim_half])
            hh_out = f"{hh_name}/output_0"
            return hh_out

        return interleave("cos", cos_cache), interleave("sin", sin_cache)

    def _apply_mrope_rotation(self, layer_id, qk_path, qk_shape, dyn_cos, dyn_sin, num_heads, basename):
        """Apply mRoPE via com.microsoft.RotaryEmbedding (4-input variant).

        cos/sin are pre-gathered [B, S, rdim_half].  We flatten them to
        [B*S, rdim_half] and create synthetic linear position_ids [B, S]
        so the kernel simply gathers row-by-row from the flat cache.

        cos/sin caches are always float32. When io_dtype differs (fp16/bf16),
        cast Q/K to float32 before rotation, then cast back — preserving
        numerical precision in the RoPE computation.
        """
        force_fp32 = self.rope_attrs.get("cast_to_fp32", False)
        compute_dtype = ir.DataType.FLOAT if force_fp32 else self.io_dtype
        rdim_half = self.mrope_rotary_dim // 2

        # --- Flatten cos/sin to [B*S, rdim_half] ---
        flat_cos_name = f"{basename}/cos_flat/Reshape"
        self.make_reshape(
            flat_cos_name, [dyn_cos, f"/model/constants/INT64/[-1, {rdim_half}]"], ir.DataType.FLOAT, ["batch_seq", rdim_half]
        )
        flat_cos = f"{flat_cos_name}/output_0"

        flat_sin_name = f"{basename}/sin_flat/Reshape"
        self.make_reshape(
            flat_sin_name, [dyn_sin, f"/model/constants/INT64/[-1, {rdim_half}]"], ir.DataType.FLOAT, ["batch_seq", rdim_half]
        )
        flat_sin = f"{flat_sin_name}/output_0"

        # Cast flat cos/sin to compute dtype if needed
        rope_cos = flat_cos
        rope_sin = flat_sin
        if compute_dtype != ir.DataType.FLOAT:
            cos_cast_name = f"{basename}/cos/Cast"
            self.make_cast(cos_cast_name, flat_cos, compute_dtype, ["batch_seq", rdim_half])
            rope_cos = f"{cos_cast_name}/output_0"

            sin_cast_name = f"{basename}/sin/Cast"
            self.make_cast(sin_cast_name, flat_sin, compute_dtype, ["batch_seq", rdim_half])
            rope_sin = f"{sin_cast_name}/output_0"

        # --- Build synthetic position_ids [B, S] = Range(0, B*S).reshape(B, S) ---
        # Shape(Q/K input) → [B, S, N*H], Gather dim 0 and 1 → B, S
        shape_name = f"{basename}/qk_shape/Shape"
        self.make_shape(shape_name, qk_path, [3])

        batch_name = f"{basename}/batch/Gather"
        self.make_gather(batch_name, [f"{shape_name}/output_0", "/model/constants/INT64/[0]"], ir.DataType.INT64, [1], axis=0)

        seq_name = f"{basename}/seq/Gather"
        self.make_gather(seq_name, [f"{shape_name}/output_0", "/model/constants/INT64/[1]"], ir.DataType.INT64, [1], axis=0)

        total_name = f"{basename}/total/Mul"
        self.make_mul(total_name, [f"{batch_name}/output_0", f"{seq_name}/output_0"], ir.DataType.INT64, [1])

        range_name = f"{basename}/range/Range"
        self.make_range(
            range_name,
            ["/model/constants/INT64/[0]", f"{total_name}/output_0", "/model/constants/INT64/[1]"],
            ir.DataType.INT64,
            ["batch_seq"],
        )

        # Reshape to [B, S]
        bs_shape_name = f"{basename}/bs_shape/Concat"
        self.make_concat(bs_shape_name, [f"{batch_name}/output_0", f"{seq_name}/output_0"], ir.DataType.INT64, [2], axis=0)

        pos_ids_name = f"{basename}/pos_ids/Reshape"
        self.make_reshape(
            pos_ids_name, [f"{range_name}/output_0", f"{bs_shape_name}/output_0"], ir.DataType.INT64, ["batch_size", "sequence_length"]
        )
        pos_ids = f"{pos_ids_name}/output_0"

        # --- Reshape Q/K to [B, N, S, H] for com.microsoft.RotaryEmbedding ---
        head_size = qk_shape[-1] // num_heads if isinstance(qk_shape[-1], int) else self.head_size
        bnsh_shape = ["batch_size", num_heads, "sequence_length", head_size]

        reshape_in_name = f"{basename}/reshape_in/Reshape"
        self.make_reshape(
            reshape_in_name,
            [qk_path, f"/model/constants/INT64/[0, 0, {num_heads}, {head_size}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", num_heads, head_size],
        )
        transpose_in_name = f"{basename}/transpose_in/Transpose"
        self.make_transpose(transpose_in_name, f"{reshape_in_name}/output_0", self.io_dtype, bnsh_shape, perm=[0, 2, 1, 3])

        rope_input = f"{transpose_in_name}/output_0"
        if compute_dtype != self.io_dtype:
            cast_in_name = f"{basename}/input/Cast"
            self.make_cast(cast_in_name, rope_input, compute_dtype, bnsh_shape)
            rope_input = f"{cast_in_name}/output_0"

        # --- com.microsoft.RotaryEmbedding ---
        rope_name = f"{basename}/RotaryEmbedding"
        rope_out = f"{rope_name}/output_0"
        self.make_node(
            "RotaryEmbedding",
            [rope_input, pos_ids, rope_cos, rope_sin],
            [rope_out],
            name=rope_name,
            domain="com.microsoft",
            num_heads=num_heads,
            rotary_embedding_dim=self.mrope_rotary_dim,
            interleaved=0,
        )
        self.make_value(rope_out, compute_dtype, bnsh_shape)

        # --- Reshape back to [B, S, N*H] ---
        final = rope_out
        if compute_dtype != self.io_dtype:
            cast_out_name = f"{basename}/output/Cast"
            self.make_cast(cast_out_name, rope_out, self.io_dtype, bnsh_shape)
            final = f"{cast_out_name}/output_0"

        transpose_out_name = f"{basename}/transpose_out/Transpose"
        bsnhv_shape = ["batch_size", "sequence_length", num_heads, head_size]
        self.make_transpose(transpose_out_name, final, self.io_dtype, bsnhv_shape, perm=[0, 2, 1, 3])

        reshape_out_name = f"{basename}/reshape_out/Reshape"
        total_dim = num_heads * head_size
        self.make_reshape(
            reshape_out_name, [f"{transpose_out_name}/output_0", f"/model/constants/INT64/[0, 0, {total_dim}]"], self.io_dtype, qk_shape
        )

        return f"{reshape_out_name}/output_0"

    def _make_linear_attention(self, layer_id, linear_attn, root_input):
        """Build GatedDeltaNet using fused CausalConvWithState + LinearAttention ops.

        Uses com.microsoft contrib ops:
        - CausalConvWithState: fused depthwise conv1d + SiLU + carry state
        - LinearAttention: fused 3D-packed linear attention with GQA
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        conv_dim = self.linear_conv_dim
        v_dim = self.linear_value_dim
        n_kv = self.linear_num_value_heads
        n_k = self.linear_num_key_heads
        hk = self.linear_key_head_dim
        hv = self.linear_value_head_dim
        kernel_size = self.linear_conv_kernel_dim

        # Projections, conv weight init, QKV transpose
        z_name, b_name, a_name, qkv_t_output, conv_weight_name = self._make_linear_attention_projections(layer_id, linear_attn, root_input)

        # --- Fused conv: CausalConvWithState (com.microsoft) ---
        conv_bias_name = f"model.layers.{layer_id}.linear_attn.conv1d.bias"
        self.make_initializer(torch.zeros(conv_dim, dtype=torch.float32), conv_bias_name, to=self.io_dtype)

        past_conv = f"past_key_values.{layer_id}.conv_state"
        present_conv = f"present.{layer_id}.conv_state"

        conv_op_name = f"{basename}/CausalConvWithState"
        self.make_causal_conv_with_state(
            conv_op_name,
            root_input=qkv_t_output,
            weight=conv_weight_name,
            bias=conv_bias_name,
            past_conv_state=past_conv,
            present_conv_state=present_conv,
            output_shape=["batch_size", conv_dim, "sequence_length"],
            present_conv_shape=["batch_size", conv_dim, kernel_size - 1],
        )
        silu_output = f"{conv_op_name}/output_0"

        conv_out_t_name = f"{basename}/conv_out/Transpose"
        conv_out_t_output = f"{conv_out_t_name}/output_0"
        self.make_transpose(conv_out_t_name, silu_output, self.io_dtype, ["batch_size", "sequence_length", conv_dim], [0, 2, 1])

        # Split QKV, L2 norm, gates
        q_scaled_output, k_norm_out, v_out, g_output, beta_output = self._make_linear_attention_normalize_and_gate(
            layer_id, linear_attn, conv_out_t_output, b_name, a_name
        )

        # --- Fused recurrence: LinearAttention (com.microsoft) ---
        past_recurrent = f"past_key_values.{layer_id}.recurrent_state"
        present_recurrent = f"present.{layer_id}.recurrent_state"

        la_op_name = f"{basename}/LinearAttention"
        self.make_linear_attention(
            la_op_name,
            q_path=q_scaled_output,
            k_path=k_norm_out,
            v_path=v_out,
            past_recurrent_state=past_recurrent,
            present_recurrent_state=present_recurrent,
            decay=g_output,
            beta=beta_output,
            q_num_heads=n_k,
            kv_num_heads=n_kv,
            update_rule="gated_delta",
            scale=1.0,  # Q is already pre-scaled by 1/sqrt(d_k)
            output_shape=["batch_size", "sequence_length", v_dim],
            present_recurrent_shape=["batch_size", n_kv, hk, hv],
        )
        la_output = f"{la_op_name}/output_0"

        # Gated RMSNorm + output projection
        self._make_linear_attention_output(layer_id, linear_attn, la_output, z_name)

    def _make_linear_attention_projections(self, layer_id, linear_attn, root_input):
        """Build linear projections, conv weight initializer, and QKV transpose.

        Returns:
            (z_name, b_name, a_name, qkv_t_output, conv_weight_name)
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        conv_dim = self.linear_conv_dim

        qkv_name = f"{basename}/in_proj_qkv/MatMul"
        self.make_matmul(linear_attn.in_proj_qkv, qkv_name, root_input)

        z_name = f"{basename}/in_proj_z/MatMul"
        self.make_matmul(linear_attn.in_proj_z, z_name, root_input)

        b_name = f"{basename}/in_proj_b/MatMul"
        self.make_matmul(linear_attn.in_proj_b, b_name, root_input)

        a_name = f"{basename}/in_proj_a/MatMul"
        self.make_matmul(linear_attn.in_proj_a, a_name, root_input)

        qkv_t_name = f"{basename}/qkv_transpose/Transpose"
        qkv_t_output = f"{qkv_t_name}/output_0"
        self.make_transpose(qkv_t_name, f"{qkv_name}/output_0", self.io_dtype, ["batch_size", conv_dim, "sequence_length"], [0, 2, 1])

        conv_weight_name = f"model.layers.{layer_id}.linear_attn.conv1d.weight"
        self.make_initializer(linear_attn.conv1d.weight, conv_weight_name, to=self.io_dtype)

        return z_name, b_name, a_name, qkv_t_output, conv_weight_name

    def _make_linear_attention_normalize_and_gate(self, layer_id, linear_attn, conv_out_3d, b_name, a_name):
        """Split QKV, per-head L2 norm, Q scale, and compute decay/beta gates.

        Args:
            conv_out_3d: Conv output transposed to [B, S, conv_dim].
            b_name: Name of the beta projection MatMul node.
            a_name: Name of the alpha projection MatMul node.

        Returns:
            (q_scaled_output, k_norm_out, v_out, g_output, beta_output)
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        k_dim = self.linear_key_dim
        v_dim = self.linear_value_dim
        n_kv = self.linear_num_value_heads
        n_k = self.linear_num_key_heads
        hk = self.linear_key_head_dim

        # Split into Q, K, V
        split_qkv_name = f"{basename}/split_qkv/Split"
        q_out = f"{split_qkv_name}/output_0"
        k_out = f"{split_qkv_name}/output_1"
        v_out = f"{split_qkv_name}/output_2"
        self.make_node(
            "Split",
            [conv_out_3d, f"/model/constants/INT64/[{k_dim}, {k_dim}, {v_dim}]"],
            [q_out, k_out, v_out],
            name=split_qkv_name,
            axis=-1,
        )
        self.make_value(q_out, self.io_dtype, ["batch_size", "sequence_length", k_dim])
        self.make_value(k_out, self.io_dtype, ["batch_size", "sequence_length", k_dim])
        self.make_value(v_out, self.io_dtype, ["batch_size", "sequence_length", v_dim])

        # Per-head L2 normalize Q and K
        q_norm_out = self._make_per_head_l2_normalize(f"{basename}/q_l2norm", q_out, n_k, hk)
        k_norm_out = self._make_per_head_l2_normalize(f"{basename}/k_l2norm", k_out, n_k, hk)

        # Scale Q by 1/sqrt(head_k_dim)
        scale_name = self._get_shared_q_scale(hk)
        q_scaled_name = f"{basename}/q_scaled/Mul"
        self.make_mul(q_scaled_name, [q_norm_out, scale_name], self.io_dtype, ["batch_size", "sequence_length", k_dim])
        q_scaled_output = f"{q_scaled_name}/output_0"

        # beta = sigmoid(b)
        beta_name = f"{basename}/beta/Sigmoid"
        self.make_sigmoid(beta_name, f"{b_name}/output_0", self.io_dtype, ["batch_size", "sequence_length", n_kv])
        beta_output = f"{beta_name}/output_0"

        # g = -exp(A_log) * softplus(a + dt_bias)
        # The reference model computes this entirely in float32 to prevent
        # precision loss that is exponentially amplified by exp(g) in the
        # recurrence.  Cast inputs to fp32, compute, then cast result back.
        dt_bias_init = f"model.layers.{layer_id}.linear_attn.dt_bias"
        self.make_initializer(linear_attn.dt_bias, dt_bias_init, to=ir.DataType.FLOAT)

        neg_exp_a_name = f"model.layers.{layer_id}.linear_attn.neg_exp_A"
        neg_exp_a = (-linear_attn.A_log.data.exp()).detach()
        self.make_initializer(neg_exp_a, neg_exp_a_name, to=ir.DataType.FLOAT)

        # Cast a projection output to fp32
        a_cast_name = f"{basename}/decay/a_cast/Cast"
        self.make_cast(a_cast_name, f"{a_name}/output_0", ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv])

        a_plus_dt_name = f"{basename}/decay/Add"
        self.make_add(a_plus_dt_name, [f"{a_cast_name}/output_0", dt_bias_init], ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv])
        a_plus_dt_output = f"{a_plus_dt_name}/output_0"

        softplus_name = f"{basename}/decay/Softplus"
        self.make_softplus(softplus_name, a_plus_dt_output, ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv])
        softplus_output = f"{softplus_name}/output_0"

        g_fp32_name = f"{basename}/decay/Mul"
        self.make_mul(g_fp32_name, [neg_exp_a_name, softplus_output], ir.DataType.FLOAT, ["batch_size", "sequence_length", n_kv])
        g_fp32_output = f"{g_fp32_name}/output_0"

        # Cast decay back to io_dtype for the kernel
        g_cast_name = f"{basename}/decay/g_cast/Cast"
        self.make_cast(g_cast_name, g_fp32_output, self.io_dtype, ["batch_size", "sequence_length", n_kv])
        g_output = f"{g_cast_name}/output_0"

        return q_scaled_output, k_norm_out, v_out, g_output, beta_output

    def _make_linear_attention_output(self, layer_id, linear_attn, attn_output_3d, z_name):
        """Build gated RMSNorm and output projection.

        Args:
            attn_output_3d: Attention output [B, S, v_dim] (3D packed).
            z_name: Name of the z-gate projection MatMul node.
        """
        basename = f"/model/layers.{layer_id}/linear_attn"
        z_output = f"{z_name}/output_0"

        gated_norm_output = self._make_gated_rms_norm(f"{basename}/gated_norm", attn_output_3d, z_output, linear_attn.norm, layer_id)

        o_name = f"{basename}/out_proj/MatMul"
        self.make_matmul(linear_attn.out_proj, o_name, gated_norm_output)
        self.layernorm_attrs["skip_input"] = f"{o_name}/output_0"

    def _make_per_head_l2_normalize(self, basename, input_name, n_heads, head_dim):
        """Per-head L2 normalize: reshape [B, S, N*H] -> [B, S, N, H], norm, reshape back.

        Uses [0, 0, N, H] / [0, 0, N*H] reshape targets so all dims are
        constants or copied from the 3D/4D input, avoiding Shape ops that
        would run on CPU and block CUDA graph capture.
        """
        total_dim = n_heads * head_dim

        # Reshape to [B, S, N, H] for per-head normalization
        flat_name = f"{basename}/flat/Reshape"
        flat_out = f"{flat_name}/output_0"
        self.make_reshape(
            flat_name,
            [input_name, f"/model/constants/INT64/[0, 0, {n_heads}, {head_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", n_heads, head_dim],
        )

        # L2 normalize along last dim (head_dim) — input is 4D [B, S, N, H]
        norm_out = self._make_l2_normalize(basename, flat_out, head_dim, leading_dims=["batch_size", "sequence_length", n_heads])

        # Reshape back to [B, S, N*H]
        unflat_name = f"{basename}/unflat/Reshape"
        unflat_out = f"{unflat_name}/output_0"
        self.make_reshape(
            unflat_name,
            [norm_out, f"/model/constants/INT64/[0, 0, {total_dim}]"],
            self.io_dtype,
            ["batch_size", "sequence_length", total_dim],
        )
        return unflat_out

    def _make_l2_normalize(self, basename, input_name, last_dim, leading_dims=None):
        """L2-normalize along last dimension: x * rsqrt(sum(x^2) + eps)

        Matches the FLA library's l2norm used by the PyTorch reference:
            inv_norm = rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)
            return x * inv_norm
        """
        if leading_dims is None:
            leading_dims = ["batch_size", "sequence_length"]
        full_shape = [*leading_dims, last_dim]
        reduced_shape = [*leading_dims, 1]

        # sum(x^2, dim=-1, keepdim=True)
        sq_name = f"{basename}/Square/Mul"
        self.make_mul(sq_name, [input_name, input_name], self.io_dtype, full_shape)

        sum_name = f"{basename}/SumSq/ReduceSum"
        self.make_reduce_sum(sum_name, [f"{sq_name}/output_0", "/model/constants/INT64/[-1]"], self.io_dtype, reduced_shape, keepdims=True)

        # sum(x^2) + eps
        eps_name = self._get_shared_l2_eps()
        add_eps_name = f"{basename}/AddEps/Add"
        self.make_add(add_eps_name, [f"{sum_name}/output_0", eps_name], self.io_dtype, reduced_shape)

        # x * rsqrt(sum(x^2) + eps)
        rsqrt_name = f"{basename}/Rsqrt"
        self.make_rsqrt(rsqrt_name, [f"{add_eps_name}/output_0"], self.io_dtype, reduced_shape)

        norm_name = f"{basename}/Normalize/Mul"
        self.make_mul(norm_name, [input_name, f"{rsqrt_name}/output_0"], self.io_dtype, full_shape)

        return f"{norm_name}/output_0"

    def _make_gated_rms_norm(self, basename, input_name, gate_name, norm_module, layer_id):
        """Gated RMSNorm: RMSNorm(x) * SiLU(z).

        The norm weight is per-head (shape [head_v_dim]).
        Input and gate are [B, S, v_dim]. We reshape to per-head,
        apply per-head norm, gate, and reshape back.
        """
        v_dim = self.linear_value_dim
        hv = self.linear_value_head_dim
        nv = self.linear_num_value_heads

        # Reshape input to [B, S, N, H] for per-head norm (avoids Shape ops
        # that would run on CPU and block CUDA graph capture)
        flat_name = f"{basename}/input_flat/Reshape"
        flat_output = f"{flat_name}/output_0"
        self.make_reshape(
            flat_name, [input_name, f"/model/constants/INT64/[0, 0, {nv}, {hv}]"], self.io_dtype, ["batch_size", "sequence_length", nv, hv]
        )

        # Norm weight (NO offset — Qwen3_5RMSNormGated uses raw weight, not 1+w)
        norm_weight = f"model.layers.{layer_id}.linear_attn.norm.weight"
        self.make_initializer(norm_module.weight, norm_weight, to=self.io_dtype)

        # SimplifiedLayerNormalization (com.microsoft, no offset for gated norm)
        norm_name = f"{basename}/SimplifiedLayerNormalization"
        norm_output = f"{norm_name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            [flat_output, norm_weight],
            [norm_output],
            name=norm_name,
            epsilon=self.layernorm_attrs["epsilon"],
            axis=-1,
            stash_type=1,
        )
        self.make_value(norm_output, self.io_dtype, ["batch_size", "sequence_length", nv, hv])

        # Reshape back to [B, S, v_dim]
        unflat_name = f"{basename}/norm_unflat/Reshape"
        unflat_output = f"{unflat_name}/output_0"
        self.make_reshape(
            unflat_name, [norm_output, f"/model/constants/INT64/[0, 0, {v_dim}]"], self.io_dtype, ["batch_size", "sequence_length", v_dim]
        )

        # SiLU(z) — computed in float32 as in the reference model to preserve
        # gate precision (F.silu(gate.to(torch.float32))).
        z_cast_name = f"{basename}/z_cast/Cast"
        self.make_cast(z_cast_name, gate_name, ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim])
        z_fp32 = f"{z_cast_name}/output_0"

        z_sigmoid_name = f"{basename}/z_sigmoid/Sigmoid"
        self.make_sigmoid(z_sigmoid_name, z_fp32, ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim])
        z_sigmoid_output = f"{z_sigmoid_name}/output_0"

        z_silu_name = f"{basename}/z_silu/Mul"
        self.make_mul(z_silu_name, [z_fp32, z_sigmoid_output], ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim])
        z_silu_output = f"{z_silu_name}/output_0"

        # Cast norm output to fp32 for the multiplication, then cast result back
        norm_cast_name = f"{basename}/norm_cast/Cast"
        self.make_cast(norm_cast_name, unflat_output, ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim])

        # output = norm * silu(z) in fp32
        gated_fp32_name = f"{basename}/gated_fp32/Mul"
        self.make_mul(
            gated_fp32_name, [f"{norm_cast_name}/output_0", z_silu_output], ir.DataType.FLOAT, ["batch_size", "sequence_length", v_dim]
        )

        # Cast back to io_dtype
        gated_name = f"{basename}/gated/Cast"
        self.make_cast(gated_name, f"{gated_fp32_name}/output_0", self.io_dtype, ["batch_size", "sequence_length", v_dim])
        gated_output = f"{gated_name}/output_0"

        return gated_output

    def load_weights(self, input_path):
        # Qwen3_5ForConditionalGeneration is not registered with AutoModelForCausalLM.
        # Load the full multimodal model directly; make_model will find the
        # embedded language-model sub-modules (embed_tokens, Qwen3_5TextModel
        # decoder layers, norm, lm_head) via standard module iteration.
        from transformers import Qwen3_5ForConditionalGeneration

        print("Loading Qwen3_5ForConditionalGeneration model...")
        return Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=self.hf_remote
        )

    def make_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        """Generate genai_config.json for the decoder (text-only) model.

        Temporarily adjusts attributes so the base class produces the correct
        config for Qwen3.5's hybrid architecture (sparse KV cache, nested
        token IDs in ``text_config``).
        """
        # Flatten text_config token IDs onto the HF config so the base class
        # can access them.  Save to out_dir so AutoConfig.from_pretrained
        # picks up the patched version.
        hf_config = AutoConfig.from_pretrained(model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs)
        text_cfg = getattr(hf_config, "text_config", hf_config)
        for attr in ("eos_token_id", "bos_token_id", "pad_token_id"):
            val = getattr(text_cfg, attr, None)
            if val is not None:
                setattr(hf_config, attr, val)
        hf_config.save_pretrained(out_dir)

        # Temporarily restore the KV cache template keys and adjust attributes
        # so the base class generates the right entries.
        saved = {"num_layers": self.num_layers, "model_type": self.model_type}
        self.num_layers = len(self.layer_types)
        self.model_type = "Qwen3_5ForConditionalGeneration"
        self.input_names["past_key_values.key"] = "past_key_values.%d.key"
        self.input_names["past_key_values.value"] = "past_key_values.%d.value"
        self.output_names["present.key"] = "present.%d.key"
        self.output_names["present.value"] = "present.%d.value"

        super().make_genai_config(out_dir, {}, out_dir)

        # Restore
        self.num_layers = saved["num_layers"]
        self.model_type = saved["model_type"]
        del self.input_names["past_key_values.key"]
        del self.input_names["past_key_values.value"]
        del self.output_names["present.key"]
        del self.output_names["present.value"]
