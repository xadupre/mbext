# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import copy
import os

import torch
from onnx_light.onnx import TensorProto as ir

from .base_embedding import EmbeddingModel
from .base_vision import VisionEncoderModel
from .llama import LlamaModel
from .mistral import Ministral3ConditionalGenerationModel


class SmolLM3Model(LlamaModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.layer_types = config.layer_types
        self.no_rope_layers = config.no_rope_layers

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        # SmolLM3 uses per-layer conditional RoPE and Sliding Window Attention.
        # So, we temporarily modify the model's attributes before calling the
        # base `make_attention` method, then restore them immediately after.
        original_use_rope = self.attention_attrs["use_rope_in_attn"]
        original_rope = self.attention_attrs["rope"]
        original_window_size = self.window_size

        # Enable/disable RoPE for the current layer.
        # no_rope_layers[i] = 1 means layer i uses RoPE; 0 means no RoPE.
        has_rope = bool(self.no_rope_layers[layer_id])
        self.attention_attrs["rope"] = has_rope
        self.attention_attrs["use_rope_in_attn"] = has_rope and original_use_rope

        # Set the sliding window size for the current layer.
        assert self.layer_types[layer_id] in {"sliding_attention", "full_attention"}
        if self.layer_types[layer_id] == "full_attention":
            self.window_size = -1

        # Call the original `make_attention` with the temporarily-modified settings.
        super().make_attention(layer_id, attention, root_input, **kwargs)

        # Restore original values
        self.attention_attrs["use_rope_in_attn"] = original_use_rope
        self.attention_attrs["rope"] = original_rope
        self.window_size = original_window_size


def _smolvlm_text_config(config):
    text_config = copy.deepcopy(config.text_config)
    text_config.architectures = config.architectures
    text_config._name_or_path = config._name_or_path
    return text_config


def _load_smolvlm(builder, input_path):
    from transformers import SmolVLMForConditionalGeneration

    src = input_path if os.path.isdir(input_path) else builder.model_name_or_path
    return SmolVLMForConditionalGeneration.from_pretrained(
        src, cache_dir=builder.cache_dir, token=builder.hf_token, trust_remote_code=builder.hf_remote
    )


class SmolVLMTextModel(LlamaModel):
    """Llama decoder of SmolVLM, optionally consuming external image embeddings."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        for option in ("adapter_path", "reuse_downloaded_weights"):
            if extra_options.get(option):
                raise NotImplementedError(f"SmolVLM does not support {option}.")
        super().__init__(_smolvlm_text_config(config), io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.model_type = "llama"

    def load_weights(self, input_path):
        hf_model = _load_smolvlm(self, input_path)
        model = torch.nn.Module()
        model.model = hf_model.model.text_model
        model.lm_head = hf_model.lm_head
        return model

    def has_lm_head(self, module):
        return module is self.weights.lm_head

    def create_genai_config(self, model_name_or_path, extra_kwargs, out_dir):
        from transformers import GenerationConfig

        config = super().create_genai_config(model_name_or_path, extra_kwargs, out_dir)
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_name_or_path, token=self.hf_token, trust_remote_code=self.hf_remote, **extra_kwargs
            )
        except OSError:
            generation_config = None
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            value = getattr(generation_config, name, None)
            if value is None:
                value = getattr(self._checkpoint_config, name, None)
            if value is not None:
                config["model"][name] = value
        return config


class SmolVLMEmbeddingModel(EmbeddingModel):
    def load_hf_model(self, input_path):
        return _load_smolvlm(self, input_path)

    def get_embed_weight(self, hf_model):
        return hf_model.get_input_embeddings().weight.detach().float().numpy()


class SmolVLMVisionEncoderModel(VisionEncoderModel):
    """SigLIP tower and pixel-shuffle connector for one full, square image crop.

    Image splitting, resizing and normalization remain processor operations.
    Each crop must have the configured image size, without masked padding.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        vc = config.vision_config
        vis_config = _smolvlm_text_config(config)
        for key in ("hidden_size", "intermediate_size", "num_attention_heads", "num_hidden_layers", "hidden_act"):
            setattr(vis_config, key, getattr(vc, key))
        vis_config.num_key_value_heads = vc.num_attention_heads
        vis_config.head_dim = vc.hidden_size // vc.num_attention_heads
        vis_config.max_position_embeddings = (vc.image_size // vc.patch_size) ** 2
        vis_config.rope_scaling = None
        vision_options = {**extra_options, "filename": self.FILENAME, "exclude_lm_head": True, "exclude_embeds": True}
        vision_options.pop("num_hidden_layers", None)
        super().__init__(vis_config, io_dtype, onnx_dtype, ep, cache_dir, vision_options)
        self.config = copy.deepcopy(config)
        self.config.spatial_merge_size = config.scale_factor
        self.vis_hidden_size = vc.hidden_size
        self.vis_rms_norm_eps = vc.layer_norm_eps
        self.n_patches_per_side = vc.image_size // vc.patch_size
        self.n_patches = self.n_patches_per_side**2
        if self.n_patches_per_side % config.scale_factor:
            raise ValueError("SmolVLM patch grid must be divisible by scale_factor.")
        if vc.hidden_act not in ("gelu", "gelu_pytorch_tanh"):
            raise ValueError(f"Unsupported SmolVLM vision activation: {vc.hidden_act}")

    def make_attention(self, layer_id, attention, root_input, **kwargs):
        b = f"/vision/layers.{layer_id}/attn"
        q, k, v = [self.make_vis_proj(getattr(attention, f"{name}_proj"), f"{b}/{name}/MatMul", root_input) for name in ("q", "k", "v")]
        out = self.make_vis_mha(b, q, k, v, self.num_attn_heads, self.head_size**-0.5, [1, self.n_patches, self.vis_hidden_size])
        self.layernorm_attrs["skip_input"] = self.make_vis_proj(attention.out_proj, f"{b}/out/MatMul", out)

    def make_layer(self, layer_id, layer):
        shape = [1, self.n_patches, self.vis_hidden_size]

        def mlp(root):
            b = f"/vision/layers.{layer_id}/mlp"
            out = self.make_vis_proj(layer.mlp.fc1, f"{b}/fc1/MatMul", root)
            act = f"{b}/activation/output_0"
            self.make_node(
                ("FastGelu" if self.config.vision_config.hidden_act == "gelu_pytorch_tanh" else "Gelu"),
                [out],
                [act],
                name=f"{b}/activation",
                domain="com.microsoft",
            )
            self.make_value(act, self.io_dtype, [1, self.n_patches, self.intermediate_size])
            return self.make_vis_proj(layer.mlp.fc2, f"{b}/fc2/MatMul", act)

        self._make_standard_vision_layer(
            layer_id,
            layer.layer_norm1.weight,
            layer.self_attn,
            layer.layer_norm2.weight,
            mlp,
            shape,
            norm1_bias=layer.layer_norm1.bias,
            norm2_bias=layer.layer_norm2.bias,
        )

    def make_model(self, input_path):
        hf_model = _load_smolvlm(self, input_path).eval()
        vision = hf_model.model.vision_model
        vc = self.config.vision_config
        d, n, s = (self.vis_hidden_size, self.n_patches_per_side, self.config.scale_factor)
        self.make_graph_input("pixel_values", self.io_dtype, [1, vc.num_channels, vc.image_size, vc.image_size])
        patch_weight = vision.embeddings.patch_embedding.weight
        self.make_initializer(patch_weight, "patch.weight", to=self.io_dtype)
        self.make_initializer(vision.embeddings.patch_embedding.bias, "patch.bias", to=self.io_dtype)
        self.make_conv(
            "/vision/patch",
            ["pixel_values", "patch.weight", "patch.bias"],
            self.io_dtype,
            [1, d, n, n],
            kernel_shape=[vc.patch_size, vc.patch_size],
            strides=[vc.patch_size, vc.patch_size],
        )
        x = self.make_transpose("/vision/patch/Transpose", "/vision/patch/output_0", self.io_dtype, [1, n, n, d], perm=[0, 2, 3, 1])
        x = self.make_reshape("/vision/patch/Reshape", [x, [1, n * n, d]], self.io_dtype, [1, n * n, d])
        # Match HF's bucketized full-crop positions, including rounding at boundaries.
        buckets = torch.bucketize(torch.arange(n, dtype=torch.float32) * (1.0 / n), torch.arange(1 / n, 1.0, 1 / n), right=True)
        positions = (buckets[:, None] * n + buckets[None, :]).flatten()
        self.make_initializer(vision.embeddings.position_embedding(positions).detach().unsqueeze(0), "position_embedding", to=self.io_dtype)
        x = self.make_add("/vision/positions", [x, "position_embedding"], self.io_dtype, [1, n * n, d])
        self.layernorm_attrs["root_input"] = x
        for i, layer in enumerate(vision.encoder.layers):
            self.make_layer(i, layer)
        norm = vision.post_layernorm
        x = self.make_layer_norm("/vision/post_norm", self.layernorm_attrs["root_input"], norm.weight, norm.bias, [1, n * n, d])
        # Preserve HF's channel ordering; a single spatial transpose is not equivalent.
        for i, (shape, perm) in enumerate([([1, n, n // s, d * s], [0, 2, 1, 3]), ([1, n // s, n // s, d * s * s], [0, 2, 1, 3])]):
            x = self.make_reshape(f"/vision/shuffle.{i}/Reshape", [x, shape], self.io_dtype, shape)
            x = self.make_transpose(f"/vision/shuffle.{i}/Transpose", x, self.io_dtype, [shape[j] for j in perm], perm=perm)
        x = self.make_reshape(
            "/vision/shuffle/Flatten", [x, [1, n * n // (s * s), d * s * s]], self.io_dtype, [1, n * n // (s * s), d * s * s]
        )
        x = self.make_vis_proj(hf_model.model.connector.modality_projection.proj, "/vision/connector/MatMul", x)
        x = self.make_reshape(
            "/vision/connector/Flatten",
            [x, [n * n // (s * s), self.config.text_config.hidden_size]],
            self.io_dtype,
            [n * n // (s * s), self.config.text_config.hidden_size],
        )
        self.make_node("Identity", [x], ["image_features"], name="/vision/output")
        self.make_graph_output("image_features", self.io_dtype, [n * n // (s * s), self.config.text_config.hidden_size])


class SmolVLMConditionalGenerationModel(Ministral3ConditionalGenerationModel):
    """Export SmolVLM using the shared vision/embedding/decoder pipeline."""

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        self.vision_encoder = SmolVLMVisionEncoderModel(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.embedding_model = SmolVLMEmbeddingModel(
            _smolvlm_text_config(config), io_dtype, ir.FLOAT, ep, cache_dir, {**extra_options, "image_token_id": config.image_token_id}
        )
        self.text_model = SmolVLMTextModel(config, io_dtype, onnx_dtype, ep, cache_dir, {**extra_options, "exclude_embeds": True})

    def make_model(self, input_path):
        self.vision_encoder.make_model(input_path)
        self.embedding_model.make_model(input_path)
        self.text_model.make_model(input_path)
