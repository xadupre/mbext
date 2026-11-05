# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Load native and legacy ModelOpt Nemotron-H safetensors checkpoints."""

import torch

from .checkpoint import SafetensorsCheckpoint
from .compressed_tensors import QwenCompressedTensorsModel
from .weight_reuse import checkpoint_paths


def load_nemotron_checkpoint(directory, config, num_layers):
    from transformers import NemotronHConfig, NemotronHForCausalLM

    config = NemotronHConfig.from_dict(config.to_dict())
    config.layers_block_type = config.layers_block_type[:num_layers]
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    with torch.device("meta"):
        model = NemotronHForCausalLM(config)
    checkpoint = SafetensorsCheckpoint(checkpoint_paths(directory))
    dtype = getattr(config, "dtype", None) or torch.float32
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)

    def find_name(name):
        candidates = [name]
        if name.startswith("model."):
            candidates.append("backbone." + name[len("model.") :])
        for candidate in candidates:
            if candidate in checkpoint.files:
                return candidate
            packed = candidate.removesuffix(".weight") + ".weight_packed"
            if name.endswith(".weight") and packed in checkpoint.files:
                return packed
        return None

    def read_weight(name, shape, target_dtype=None):
        source = find_name(name)
        if source is None:
            raise ValueError(f"Missing Nemotron checkpoint tensor {name!r}.")
        weight = checkpoint.tensor(source)
        base = source.removesuffix(".weight").removesuffix(".weight_packed")

        def scale(suffix):
            key = base + suffix
            return checkpoint.tensor(key) if key in checkpoint.files else None

        if weight.dtype == torch.uint8:
            output_dtype = target_dtype if target_dtype is not None else dtype
            block_scale = scale(".weight_scale")
            global_scale = scale(".weight_scale_2")
            if global_scale is None:
                inverse_scale = scale(".weight_global_scale")
                if inverse_scale is not None:
                    global_scale = inverse_scale.reciprocal()
            if (
                weight.ndim not in {2, 3}
                or len(shape) != weight.ndim
                or tuple(weight.shape) != (*shape[:-1], shape[-1] // 2)
                or shape[-1] % 16
                or block_scale is None
                or tuple(block_scale.shape) != (*shape[:-1], shape[-1] // 16)
                or block_scale.dtype not in {torch.uint8, torch.float8_e4m3fn}
            ):
                raise ValueError(f"Invalid NVFP4 weight or block scales for {source!r}.")
            scales = block_scale.view(torch.float8_e4m3fn).float()
            if not torch.isfinite(scales).all() or (scales < 0).any():
                raise ValueError(f"Invalid NVFP4 block scales for {source!r}.")
            if weight.ndim == 3:
                if global_scale is None or global_scale.numel() not in {1, shape[0]}:
                    raise ValueError(f"Invalid NVFP4 expert global scales for {source!r}.")
                global_scale = global_scale.flatten().expand(shape[0])
                parts = []
                for i in range(shape[0]):
                    QwenCompressedTensorsModel._validate_positive_scalar(
                        global_scale[i], base + ".weight_scale_2"
                    )
                    parts.append(
                        QwenCompressedTensorsModel._dequantize(
                            weight[i],
                            block_scale[i],
                            global_scale[i],
                            source,
                            dtype=output_dtype,
                        )
                    )
                weight = torch.stack(parts)
            else:
                QwenCompressedTensorsModel._validate_positive_scalar(
                    global_scale, base + ".weight_scale_2"
                )
                weight = QwenCompressedTensorsModel._dequantize(
                    weight, block_scale, global_scale, source, dtype=output_dtype
                )
        elif weight.dtype == torch.float8_e4m3fn:
            weight_scale = scale(".weight_scale")
            QwenCompressedTensorsModel._validate_positive_weight_scale(
                weight_scale, base + ".weight_scale", weight.shape[0]
            )
            output_dtype = target_dtype if target_dtype is not None else dtype
            if weight.ndim == 3:
                scales = weight_scale.float().reshape(-1, 1, 1)
                if scales.shape[0] == 1:
                    scales = scales.expand(weight.shape[0], 1, 1)
                weight = (weight.float() * scales).to(output_dtype)
            else:
                weight = QwenCompressedTensorsModel._dequantize(
                    weight, weight_scale, None, source, dtype=output_dtype
                )
        if tuple(weight.shape) != tuple(shape):
            raise ValueError(
                f"Nemotron tensor {source!r} has shape {tuple(weight.shape)}, "
                f"expected {tuple(shape)}."
            )
        return weight if target_dtype is None else weight.to(target_dtype)

    def read_tensor(name, shape):
        if ".experts." in name and len(shape) == 3 and find_name(name) is None:
            prefix, projection = name.rsplit(".", 1)
            return torch.stack(
                [
                    read_weight(f"{prefix}.{i}.{projection}.weight", shape[1:])
                    for i in range(shape[0])
                ]
            )
        return read_weight(name, shape)

    named_parameters = list(model.named_parameters(remove_duplicate=False))
    parameter_names = {}
    for name, parameter in named_parameters:
        parameter_names.setdefault(id(parameter), []).append(name)

    loaded_parameters = {}
    for name, parameter in named_parameters:
        loaded = loaded_parameters.get(id(parameter))
        if loaded is None:
            source_name = next(
                (
                    alias
                    for alias in parameter_names[id(parameter)]
                    if find_name(alias) is not None
                ),
                name,
            )
            value = read_tensor(source_name, parameter.shape)
            loaded = torch.nn.Parameter(value, requires_grad=False)
            loaded_parameters[id(parameter)] = loaded
        module_name, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(module_name), attribute, loaded)
    for name, buffer in list(model.named_buffers()):
        if find_name(name) is None and name.endswith("e_score_correction_bias"):
            value = torch.zeros(buffer.shape, dtype=buffer.dtype)
        else:
            value = read_weight(name, buffer.shape, target_dtype=buffer.dtype)
        module_name, _, attribute = name.rpartition(".")
        setattr(model.get_submodule(module_name), attribute, value)
    return model.eval()
