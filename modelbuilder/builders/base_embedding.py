# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import onnx_light.onnx.numpy_helper as numpy_helper
from onnx_light.onnx import TensorProto

from ..helpers.quantization import quantize_ternary_groupwise
from .base import Model


class EmbeddingModel(Model):
    """Base class for ``phi3v``-style ScatterND embedding ONNX models.

    Subclasses implement :meth:`get_embed_weight` to return the token-embedding
    weight matrix (as a float32 NumPy array) from the HF model.  The shared
    :meth:`make_model` builds the identical ONNX graph for both
    ``Ministral3EmbeddingModel`` and ``Qwen25OmniEmbeddingModel``.

    Graph (2-D ``input_ids [1, T]`` from ORT-GenAI's ``EmbeddingState``)::

        text_embeds   = Gather(embed_tokens_weight, input_ids)   # [1, T, H]
        text_2d       = Squeeze(text_embeds, [0])                # [T, H]
        flat_ids      = Squeeze(input_ids, [0])                  # [T]
        is_img        = Equal(flat_ids, image_token_id_const)    # [T] bool
        img_pos       = NonZero(is_img)                          # [1, N]
        img_pos_idx   = Transpose(img_pos, [1, 0])               # [N, 1]
        scattered_2d  = ScatterND(text_2d, img_pos_idx,
                                  image_features)                # [T, H]
        inputs_embeds = Unsqueeze(scattered_2d, [0])             # [1, T, H]
    """

    FILENAME = "embedding.onnx"

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.filename = self.FILENAME
        self.image_token_id = extra_options["image_token_id"]

    def get_embed_weight(self, hf_model):
        """Return the token-embedding weight as a float32 NumPy array.

        Subclasses must override this to extract the embedding table from the
        HF model object returned by :meth:`load_hf_model`.
        """
        raise NotImplementedError

    def make_model(self, input_path):
        """Load HF weights and build the embedding ONNX graph."""
        hf_model = self.load_hf_model(input_path)
        hf_model.eval()
        embed_weight = self.get_embed_weight(hf_model)

        ternary = self.quant_attrs["int4"]["algo_config"]["algorithm"] == "ternary"
        if ternary:
            bits = self.quant_attrs["int4"]["bits"]
            block_size = self.quant_attrs["int4"]["qdq_block_size"]
            if bits != 2:
                raise ValueError(f"Ternary embedding quantization requires 2 bits, not {bits}.")
            if embed_weight.shape[1] % block_size:
                raise ValueError(
                    "Ternary embedding width must be divisible by the block size: "
                    f"{embed_weight.shape[1]} is not divisible by {block_size}."
                )
            packed, scales, zero_points = quantize_ternary_groupwise(embed_weight.T, block_size)
            packed_width = embed_weight.shape[1] * bits // 8
            self.make_initializer(packed.reshape(embed_weight.shape[0], packed_width), name="embed_tokens_weight")
            self.make_initializer(scales, name="embed_tokens_scales")
            self.make_initializer(zero_points, name="embed_tokens_zero_points")
        else:
            self.make_initializer(embed_weight, name="embed_tokens_weight")
        self.make_initializer(np.array(self.image_token_id, dtype=np.int64), name="image_token_id_const")
        # Use a Constant node (always inline) rather than an initializer so that
        # shape inference can read the axes value even when external data is used.
        _squeeze_axes = numpy_helper.from_array(np.array([0], dtype=np.int64), name="squeeze_batch_axes")
        self.make_node(
            "Constant", inputs=[], outputs=["squeeze_batch_axes"], name="/embed/squeeze_batch_axes/Constant", value=_squeeze_axes
        )
        self.make_value("squeeze_batch_axes", TensorProto.INT64, shape=[1])

        # Graph inputs (dynamic shapes).
        # ORT-GenAI passes input_ids as 2D [batch, seq_len].
        self.make_graph_input("input_ids", TensorProto.INT64, [None, None])
        # image_features dtype follows io_dtype so that it matches the vision
        # encoder output (float16 for fp16 models, float32 for fp32/int4).
        self.make_graph_input("image_features", self.io_dtype, [None, self.hidden_size])

        # 1. Embed all tokens: input_ids [1, T] -> text_embeds [1, T, H].
        if ternary:
            self.make_node(
                "GatherBlockQuantized",
                inputs=["embed_tokens_weight", "input_ids", "embed_tokens_scales", "embed_tokens_zero_points"],
                outputs=["text_embeds"],
                name="/embed/GatherBlockQuantized",
                domain="com.microsoft",
                bits=bits,
                block_size=block_size,
                gather_axis=0,
                quantize_axis=1,
            )
        else:
            self.make_node("Gather", inputs=["embed_tokens_weight", "input_ids"], outputs=["text_embeds"], name="/embed/Gather", axis=0)
        # 2. Squeeze batch dim for easier indexing: [1, T, H] → [T, H] (still fp32)
        self.make_node("Squeeze", inputs=["text_embeds", "squeeze_batch_axes"], outputs=["text_2d_fp32"], name="/embed/Squeeze_3d")
        # 3. Cast text embeddings from float32 to io_dtype so that ScatterND
        #    receives tensors of the same dtype (for fp32/int4 this Cast is a
        #    no-op that ORT optimises away at runtime).
        self.make_cast("/embed/Cast_text_2d", "text_2d_fp32", self.io_dtype, [None, self.hidden_size])
        # 4. Flatten input_ids: [1, T] → [T]
        self.make_node("Squeeze", inputs=["input_ids", "squeeze_batch_axes"], outputs=["flat_ids"], name="/embed/Squeeze_ids")
        # 5. Boolean mask where tokens are image placeholders: [T] bool
        self.make_node("Equal", inputs=["flat_ids", "image_token_id_const"], outputs=["is_image"], name="/embed/Equal")
        # 6. Positions of image placeholders: [1, N] int64
        self.make_node("NonZero", inputs=["is_image"], outputs=["img_pos"], name="/embed/NonZero")
        # 7. Transpose to [N, 1] for ScatterND
        self.make_node("Transpose", inputs=["img_pos"], outputs=["img_pos_idx"], name="/embed/Transpose", perm=[1, 0])
        # 8. Scatter image_features into text embeddings at placeholder positions
        self.make_node(
            "ScatterND",
            inputs=["/embed/Cast_text_2d/output_0", "img_pos_idx", "image_features"],
            outputs=["scattered_2d"],
            name="/embed/ScatterND",
        )
        # 9. Re-add batch dimension: [T, H] → [1, T, H]
        self.make_node("Unsqueeze", inputs=["scattered_2d", "squeeze_batch_axes"], outputs=["inputs_embeds"], name="/embed/Unsqueeze")

        # Graph output — dtype matches io_dtype (float16 for fp16 models, float32 for fp32/int4)
        self.make_graph_output("inputs_embeds", self.io_dtype, [1, None, self.hidden_size])
