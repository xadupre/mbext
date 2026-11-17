# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Checkpoint decoding for DeepSeek-V4.1-Flash."""

import torch

from .checkpoint import SafetensorsCheckpoint
from .weight_reuse import checkpoint_paths

_FP4_VALUES = (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6)


def dequantize_v41_weight(weight, scale, shape, *, block_size=(32, 32), rowwise=False):
    """Decode released FP8 matrices or packed low-nibble-first E2M1 experts."""
    shape = tuple(shape)
    if weight.ndim != 2 or len(shape) != 2:
        raise ValueError("DeepSeek quantized weights must be matrices.")
    if not torch.isfinite(scale.float()).all() or (scale.float() <= 0).any():
        raise ValueError("DeepSeek weight scales must be finite and positive.")
    if weight.dtype in (torch.int8, torch.uint8):
        if tuple(weight.shape) != (shape[0], (shape[1] + 1) // 2):
            raise ValueError(f"Invalid packed FP4 shape {tuple(weight.shape)} for {shape}.")
        packed = weight.view(torch.uint8).long()
        table = torch.tensor(_FP4_VALUES, device=weight.device, dtype=torch.float32)
        weight = torch.stack((table[packed & 15], table[packed >> 4]), -1)
        weight = weight.flatten(1)[:, : shape[1]]
        rowwise = True
    elif weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Unsupported DeepSeek quantized weight dtype {weight.dtype}.")
    elif tuple(weight.shape) != shape:
        raise ValueError(f"Invalid FP8 shape {tuple(weight.shape)} for {shape}.")
    rows, columns = block_size
    if rows <= 0 or columns <= 0:
        raise ValueError("DeepSeek quantization block dimensions must be positive.")
    rows = 1 if rowwise else rows
    expected_scale_shape = ((shape[0] + rows - 1) // rows, (shape[1] + columns - 1) // columns)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(f"Invalid DeepSeek scale shape {tuple(scale.shape)}, " f"expected {expected_scale_shape}.")
    expanded = scale.float().repeat_interleave(rows, 0).repeat_interleave(columns, 1)
    return weight.float() * expanded[: shape[0], : shape[1]]


class DeepseekV41Checkpoint:
    """Index the native released checkpoint without reading its tensor payloads."""

    def __init__(self, directory, *, block_size=(32, 32)):
        self.source = SafetensorsCheckpoint(checkpoint_paths(directory))
        self.block_size = tuple(block_size)

    def _name(self, name):
        candidates = (name, f"model.{name}")
        for candidate in candidates:
            if candidate in self.source.files:
                return candidate
        raise ValueError(f"Missing DeepSeek-V4.1 checkpoint tensor {name!r}.")

    def raw(self, name):
        return self.source.tensor(self._name(name))

    def scale(self, name):
        prefix = name.removesuffix("weight")
        for suffix in ("scale", "weight_scale_inv"):
            for candidate in (prefix + suffix, "model." + prefix + suffix):
                if candidate in self.source.files:
                    return self.source.tensor(candidate)
        raise ValueError(f"Missing DeepSeek-V4.1 quantization scale for {name!r}.")

    def tensor(self, name, shape, *, dtype=torch.float32):
        weight = self.raw(name)
        if weight.dtype in (torch.float8_e4m3fn, torch.int8, torch.uint8):
            if name != "weight" and not name.endswith(".weight"):
                raise ValueError(f"Unexpected quantized non-weight tensor {name!r}.")
            weight = dequantize_v41_weight(weight, self.scale(name), shape, block_size=self.block_size, rowwise=".engram.embed." in name)
        if tuple(weight.shape) != tuple(shape):
            raise ValueError(f"Checkpoint tensor {name!r} has shape {tuple(weight.shape)}, " f"expected {tuple(shape)}.")
        if weight.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
            raise ValueError(f"Checkpoint tensor {name!r} has unsupported dtype {weight.dtype}.")
        return weight.to(dtype=dtype)

    def load(self, model, *, dtype=torch.float32):
        """Assign one decoded parameter at a time; never allocate a random full model."""
        for name, parameter in list(model.named_parameters()):
            module_name, _, leaf = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            if ".engram.embed." in name and hasattr(module, "load_quantized"):
                if leaf != "weight":
                    continue
                raw = self.raw(name)
                if raw.dtype == torch.float8_e4m3fn:
                    scale = self.scale(name)
                    expected_scale = (parameter.shape[0], (parameter.shape[1] + self.block_size[1] - 1) // self.block_size[1])
                    if raw.shape != parameter.shape or tuple(scale.shape) != expected_scale:
                        raise ValueError(f"Invalid quantized Engram table {name!r}.")
                    module.load_quantized(raw, scale)
                    module.output_dtype = dtype
                    continue
            value = self.tensor(name, parameter.shape, dtype=dtype)
            setattr(module, leaf, torch.nn.Parameter(value, requires_grad=False))
            if leaf == "weight" and hasattr(module, "quantized"):
                raw_name = self._name(name)
                quantized = any(
                    candidate in self.source.files
                    for candidate in (raw_name.removesuffix("weight") + "scale", raw_name.removesuffix("weight") + "weight_scale_inv")
                )
                module.quantized = quantized and not module_name.endswith(".wo_a")
        return model
