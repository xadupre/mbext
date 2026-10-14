# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Read MLX affine-quantized safetensors without unpacking linear weights."""

import glob
import os

import numpy as np
import torch
from safetensors import safe_open

from .quantization import _pack_zero_points


def repack_mlx(weight, scales, biases, bits=2, group_size=128):
    """Return MatMulNBits packed weights, scales and packed zero points.

    MLX stores output-major uint32 words with the first input in the low bits.
    ONNX Runtime stores output-major blocks of input-major uint8 bytes.
    """
    weight = np.asarray(weight)
    scales = np.asarray(scales)
    biases = np.asarray(biases)
    if bits not in (2, 4) or group_size <= 0 or group_size % 32:
        raise ValueError("MLX requires 2 or 4 bits and a group size divisible by 32.")
    if weight.ndim != 2 or weight.dtype != np.uint32:
        raise ValueError("MLX weight must be a two-dimensional uint32 matrix.")
    outputs, words = weight.shape
    width = words * (32 // bits)
    blocks = width // group_size
    if (
        blocks == 0
        or width % group_size
        or scales.shape != (outputs, blocks)
        or biases.shape != scales.shape
    ):
        raise ValueError(
            "MLX weight, scales and biases have incompatible group shapes."
        )
    if scales.dtype.kind != "f" or biases.dtype != scales.dtype:
        raise ValueError(
            "MLX scales and biases must have the same floating-point dtype."
        )
    scales_f = scales.astype(np.float32)
    biases_f = biases.astype(np.float32)
    if (
        not np.all(np.isfinite(scales_f))
        or not np.all(np.isfinite(biases_f))
        or np.any(scales_f == 0)
    ):
        raise ValueError("MLX scales and biases must be finite and scales nonzero.")
    zeros = np.clip(np.rint(-biases_f / scales_f), 0, (1 << bits) - 1)
    packed = (
        weight.astype("<u4", copy=False)
        .view(np.uint8)
        .reshape(outputs, blocks, group_size * bits // 8)
    )
    zero_points = _pack_zero_points(zeros.astype(np.uint8), bits, outputs, blocks)
    residual = biases_f + zeros * scales_f
    return packed, scales, zero_points, residual


class MLXCheckpoint:
    """Index local safetensors shards and materialize only the requested tensors."""

    def __init__(self, directory):
        self.files = {}
        for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if key in self.files:
                        raise ValueError(f"Duplicate MLX tensor {key!r}.")
                    self.files[key] = path
        if not self.files:
            raise ValueError(f"No safetensors shards found in {directory!r}.")
        for key in self.files:
            if key.endswith((".scales", ".biases")):
                stem = key.rsplit(".", 1)[0]
                if any(
                    stem + suffix not in self.files
                    for suffix in (".weight", ".scales", ".biases")
                ):
                    raise ValueError(f"Incomplete MLX quantized module {stem}.")

    def tensor(self, name):
        with safe_open(self.files[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    def load_model(self, config, component=None):
        """Create a meta Qwen3.5 skeleton and populate one component at a time."""
        from transformers import Qwen3_5ForConditionalGeneration

        with torch.device("meta"):
            model = Qwen3_5ForConditionalGeneration(config)
        if component == "vision":
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5VisionRotaryEmbedding,
            )

            model.model.visual.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(
                config.vision_config
            )
        prefix = {
            "text": "model.language_model.",
            "vision": "model.visual.",
            "embedding": "model.language_model.embed_tokens.",
        }
        wanted = prefix[component] if component else ""
        for name in self.files:
            if not (
                name.startswith(wanted)
                or component == "text"
                and name.startswith("lm_head.")
            ):
                continue
            if name.endswith((".scales", ".biases")):
                continue
            if (
                name.endswith(".weight")
                and name[:-7] + ".biases" in self.files
                and name[:-7] + ".scales" not in self.files
            ):
                raise ValueError(f"Missing MLX scales for {name[:-7]}.")
            if name.endswith(".weight") and name[:-7] + ".scales" in self.files:
                stem = name[:-7]
                if stem + ".biases" not in self.files:
                    raise ValueError(f"Missing MLX biases for {stem}.")
                module = model.get_submodule(stem)
                if not isinstance(
                    module, (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv3d)
                ):
                    raise ValueError(f"Unsupported MLX quantized module {stem}.")
                weight = self.tensor(name).numpy()
                scales = self.tensor(stem + ".scales").float().numpy()
                biases = self.tensor(stem + ".biases").float().numpy()
                width = module.weight.numel() // module.weight.shape[0]
                if weight.shape[1] == 0 or width % weight.shape[1]:
                    raise ValueError(f"MLX quantized weight shape mismatch for {stem}.")
                bits = 32 // (width // weight.shape[1])
                if scales.shape[1] == 0 or width % scales.shape[1]:
                    raise ValueError(f"MLX group shape mismatch for {stem}.")
                group_size = width // scales.shape[1]
                if component in ("text", "embedding") and (
                    bits != 2 or group_size != 128
                ):
                    raise ValueError(
                        f"MLX text tensors require 2 bits and 128-element groups: {stem}."
                    )
                packed, scale, zeros, residual = repack_mlx(
                    weight, scales, biases, bits, group_size
                )
                if module.weight.shape[0] != weight.shape[0] or width != weight.shape[
                    1
                ] * (32 // bits):
                    raise ValueError(f"MLX quantized weight shape mismatch for {stem}.")
                module.qweight = torch.from_numpy(packed.copy())
                module.scales = torch.from_numpy(scale.copy())
                module.qzeros = torch.from_numpy(zeros.copy())
                module.mlx_residual = torch.from_numpy(residual.copy())
                module.bits = bits
                module.group_size = group_size
            else:
                parent, _, field = name.rpartition(".")
                module = model.get_submodule(parent)
                value = self.tensor(name)
                if value.dtype == torch.uint32:
                    raise ValueError(f"Missing MLX scales and biases for {name}.")
                previous = getattr(module, field)
                if previous.shape != value.shape:
                    raise ValueError(f"MLX tensor shape mismatch for {name}.")
                if field in module._parameters:
                    module._parameters[field] = torch.nn.Parameter(
                        value, requires_grad=False
                    )
                elif field in module._buffers:
                    module._buffers[field] = value
                else:
                    raise ValueError(f"Unknown MLX tensor {name}.")
        if component == "text" and "lm_head.weight" not in self.files:
            embedding = model.model.language_model.embed_tokens
            head = model.lm_head
            if getattr(config, "tie_word_embeddings", False):
                if hasattr(embedding, "qweight"):
                    for field in (
                        "qweight",
                        "scales",
                        "qzeros",
                        "mlx_residual",
                        "bits",
                        "group_size",
                    ):
                        setattr(head, field, getattr(embedding, field))
                else:
                    head.weight = embedding.weight
        for name, parameter in model.named_parameters():
            if not (
                name.startswith(wanted)
                or component == "text"
                and name.startswith("lm_head.")
            ):
                continue
            if parameter.is_meta and not (
                name.endswith(".weight")
                and hasattr(model.get_submodule(name[:-7]), "qweight")
            ):
                raise ValueError(f"Missing MLX checkpoint tensor {name}.")
        return model
