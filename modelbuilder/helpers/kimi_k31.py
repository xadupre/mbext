# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Native BF16 and compressed-tensors MXFP4 checkpoint loading for K3.1."""

import torch

from .checkpoint import SafetensorsCheckpoint
from .deepseek_v41 import _FP4_VALUES
from .weight_reuse import checkpoint_paths


def decode_mxfp4(packed, scales, shape):
    """Decode low-nibble-first E2M1 with per-32-element E8M0 scales."""
    if len(shape) != 2 or packed.dtype != torch.uint8 or tuple(packed.shape) != (shape[0], shape[1] // 2):
        raise ValueError(f"Invalid MXFP4 packed shape/dtype for {tuple(shape)}.")
    if shape[1] % 32 or scales.dtype != torch.uint8 or tuple(scales.shape) != (shape[0], shape[1] // 32):
        raise ValueError("MXFP4 requires uint8 E8M0 scales for groups of 32 columns.")
    if (scales == 255).any():
        raise ValueError("MXFP4 E8M0 scale contains NaN.")
    table = torch.tensor(_FP4_VALUES, dtype=torch.float32, device=packed.device)
    bits = packed.long()
    values = torch.stack((table[bits & 15], table[bits >> 4]), dim=2).flatten(1)
    scale = torch.exp2(scales.float() - 127).repeat_interleave(32, dim=1)
    return values * scale


def is_fp32_parameter(name):
    return any(suffix in name for suffix in (".A_log", ".dt_bias", "_conv1d.weight", ".o_norm.weight", ".gate."))


class KimiK31Checkpoint:
    def __init__(self, directory):
        self.source = SafetensorsCheckpoint(checkpoint_paths(directory))

    def tensor(self, name, shape, dtype):
        if name in self.source.files:
            result = self.source.tensor(name)
            if result.dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64}:
                raise ValueError(f"Unsupported checkpoint dtype for {name!r}: {result.dtype}.")
        elif name.endswith(".weight") and name.removesuffix("weight") + "weight_packed" in self.source.files:
            prefix = name.removesuffix("weight")
            scale_name = prefix + "weight_scale"
            if scale_name not in self.source.files:
                raise ValueError(f"Missing MXFP4 scale for {name!r}.")
            result = decode_mxfp4(self.source.tensor(prefix + "weight_packed"), self.source.tensor(scale_name), shape)
        else:
            raise ValueError(f"Missing Kimi K3.1 checkpoint tensor {name!r}.")
        if tuple(result.shape) != tuple(shape):
            raise ValueError(f"Invalid Kimi tensor shape for {name!r}: {tuple(result.shape)}, expected {tuple(shape)}.")
        return result.to(dtype=dtype)

    def load(self, module, dtype=torch.float32):
        """Materialize one parameter at a time; the complete export must fit RAM."""
        for name, parameter in list(module.named_parameters()):
            parent, _, leaf = name.rpartition(".")
            target = module.get_submodule(parent)
            precision = torch.float32 if is_fp32_parameter(name) else dtype
            setattr(target, leaf, torch.nn.Parameter(self.tensor(name, parameter.shape, precision), requires_grad=False))
        return module
