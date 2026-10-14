# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Load dense Qwen compressed-tensors checkpoints without onnx-ir."""

import json
import os
from types import SimpleNamespace

import torch
from onnxruntime_genai.models.loaders.base import QuantizedDecoderLayer, QuantizedModel, TensorModule
from safetensors import safe_open


class QwenCompressedTensorsModel(QuantizedModel):
    """Stream native NVFP4 and FP8 tensors into the Qwen builder surface."""

    def __init__(self, input_path, quant_attrs, q_size, kv_size, intermediate_size, num_layers):
        with open(os.path.join(input_path, "config.json")) as f:
            config = json.load(f)
        text_config = config.get("text_config", config)
        num_layers = num_layers or text_config.get("num_hidden_layers")
        super().__init__(
            "compressed-tensors",
            input_path,
            quant_attrs,
            q_size,
            kv_size,
            intermediate_size,
            num_layers,
            load_weights=False,
            lm_head=TensorModule(),
        )
        self.input_path = input_path
        self.handles = {}
        self.handle_keys = {}

        index_path = os.path.join(input_path, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
            self.single_file = None
        else:
            self.weight_map = None
            candidates = sorted(name for name in os.listdir(input_path) if name.endswith(".safetensors"))
            if len(candidates) != 1:
                raise ValueError(
                    f"'{input_path}' has no 'model.safetensors.index.json', so it must "
                    f"contain exactly one .safetensors file, but found {len(candidates)}: "
                    f"{candidates}."
                )
            self.single_file = candidates[0]

        try:
            self.layers.extend(self._make_layer(layer_id) for layer_id in range(num_layers))
            self.embedding.weight = self._get_tensor("model.language_model.embed_tokens.weight")
            self.final_norm.weight = self._get_tensor("model.language_model.norm.weight")
            self._make_linear("lm_head", self.lm_head)
        finally:
            self.close()

    def close(self):
        """Close cached safetensors handles."""
        for handle in self.handles.values():
            handle.__exit__(None, None, None)
        self.handles.clear()
        self.handle_keys.clear()

    def __del__(self):
        if getattr(self, "handles", None):
            self.close()

    def _get_tensor(self, name):
        if self.weight_map is not None:
            filename = self.weight_map.get(name)
            filenames = [filename] if filename is not None else []
        else:
            filenames = [self.single_file]
        for filename in filenames:
            handle = self.handles.get(filename)
            if handle is None:
                handle = safe_open(os.path.join(self.input_path, filename), framework="pt", device="cpu")
                self.handles[filename] = handle
                self.handle_keys[filename] = set(handle.keys())
            if name in self.handle_keys[filename]:
                return handle.get_tensor(name)
        return None

    @staticmethod
    def _validate_positive_scalar(tensor, name):
        if tensor is None or tensor.numel() != 1:
            shape = None if tensor is None else tuple(tensor.shape)
            raise ValueError(f"Compressed tensor '{name}' must be a scalar, got shape {shape}.")
        value = tensor.float()
        if not torch.isfinite(value).item() or value.item() <= 0:
            raise ValueError(f"Compressed tensor '{name}' must be finite and positive, got {value.item()}.")

    @staticmethod
    def _validate_positive_weight_scale(tensor, name, out_features):
        if tensor is None:
            raise ValueError(f"Compressed tensor '{name}' is missing.")
        if tensor.numel() not in {1, out_features}:
            raise ValueError(
                f"Compressed tensor '{name}' must be a scalar or hold {out_features} "
                f"per-channel values, got shape {tuple(tensor.shape)}."
            )
        value = tensor.float()
        if not torch.isfinite(value).all().item() or (value <= 0).any().item():
            raise ValueError(f"Compressed tensor '{name}' must be finite and positive.")

    def _validate_linear(self, module, base):
        if module.weight_scale_2 is not None:
            if module.weight.dtype != torch.uint8 or module.weight.ndim != 2 or module.weight.shape[1] % 8:
                raise ValueError(
                    f"Compressed tensor '{base}.weight' must be packed uint8 [N, K/2] "
                    f"with K divisible by 16, got dtype={module.weight.dtype} "
                    f"shape={tuple(module.weight.shape)}."
                )
            expected_shape = (int(module.weight.shape[0]), int(module.weight.shape[1]) // 8)
            if module.weight_scale is None:
                raise ValueError(f"NVFP4 tensor '{base}' has no weight_scale.")
            if module.weight_scale.dtype not in {torch.uint8, torch.float8_e4m3fn}:
                raise ValueError(f"Compressed tensor '{base}.weight_scale' must contain E4M3 bytes, " f"got {module.weight_scale.dtype}.")
            if tuple(module.weight_scale.shape) != expected_shape:
                raise ValueError(
                    f"Compressed tensor '{base}.weight_scale' has shape " f"{tuple(module.weight_scale.shape)}, expected {expected_shape}."
                )
            self._validate_positive_scalar(module.weight_scale_2, f"{base}.weight_scale_2")
            module.quant_type = "nvfp4"
            module.weight_scale = module.weight_scale.view(torch.uint8).contiguous()
        elif module.weight.dtype == torch.float8_e4m3fn:
            self._validate_positive_weight_scale(module.weight_scale, f"{base}.weight_scale", module.weight.shape[0])
            if module.input_scale is not None:
                self._validate_positive_scalar(module.input_scale, f"{base}.input_scale")
            module.quant_type = "fp8"

    def _make_linear(self, base, module=None):
        weight = self._get_tensor(f"{base}.weight")
        if weight is None:
            weight = self._get_tensor(f"{base}.weight_packed")
        if weight is None:
            return None
        module = module if module is not None else TensorModule()
        module.weight = weight
        module.weight_scale = self._get_tensor(f"{base}.weight_scale")
        module.weight_scale_2 = self._get_tensor(f"{base}.weight_scale_2")
        if module.weight_scale_2 is None:
            global_scale = self._get_tensor(f"{base}.weight_global_scale")
            if global_scale is not None:
                module.weight_scale_2 = torch.reciprocal(global_scale)
        if ".self_attn." in base:
            module.input_scale = self._get_tensor(f"{base}.input_scale")
        self._validate_linear(module, base)
        bias = self._get_tensor(f"{base}.bias")
        if bias is not None:
            module.bias = bias
        return module

    def _make_dense_linear(self, base):
        module = self._make_linear(base)
        if module.quant_type == "none":
            return module
        weight = self._dequantize(module.weight, module.weight_scale, module.weight_scale_2, f"{base}.weight")
        return TensorModule(weight=weight, bias=module.bias)

    def _make_tensor_module(self, name):
        return TensorModule(weight=self._get_tensor(name))

    def _make_layer(self, layer_id):
        prefix = f"model.language_model.layers.{layer_id}"
        layer = QuantizedDecoderLayer(layer_id)
        layer.input_layernorm.weight = self._get_tensor(f"{prefix}.input_layernorm.weight")
        layer.post_attention_layernorm.weight = self._get_tensor(f"{prefix}.post_attention_layernorm.weight")
        layer.self_attn = None
        layer.linear_attn = None

        if self._get_tensor(f"{prefix}.linear_attn.in_proj_qkv.weight") is not None:
            attention = SimpleNamespace()
            attention.in_proj_qkv = self._make_linear(f"{prefix}.linear_attn.in_proj_qkv")
            attention.in_proj_z = self._make_linear(f"{prefix}.linear_attn.in_proj_z")
            attention.in_proj_a = self._make_dense_linear(f"{prefix}.linear_attn.in_proj_a")
            attention.in_proj_b = self._make_dense_linear(f"{prefix}.linear_attn.in_proj_b")
            attention.in_proj_a.exclude_from_quantization = True
            attention.in_proj_b.exclude_from_quantization = True
            attention.out_proj = self._make_linear(f"{prefix}.linear_attn.out_proj")
            attention.conv1d = self._make_tensor_module(f"{prefix}.linear_attn.conv1d.weight")
            attention.A_log = self._get_tensor(f"{prefix}.linear_attn.A_log")
            attention.dt_bias = self._get_tensor(f"{prefix}.linear_attn.dt_bias")
            attention.norm = self._make_tensor_module(f"{prefix}.linear_attn.norm.weight")
            layer.linear_attn = attention
        elif self._get_tensor(f"{prefix}.self_attn.q_proj.weight") is not None:
            attention = SimpleNamespace()
            attention.q_proj = self._make_linear(f"{prefix}.self_attn.q_proj")
            attention.k_proj = self._make_linear(f"{prefix}.self_attn.k_proj")
            attention.v_proj = self._make_linear(f"{prefix}.self_attn.v_proj")
            attention.o_proj = self._make_linear(f"{prefix}.self_attn.o_proj")
            attention.q_norm = self._make_tensor_module(f"{prefix}.self_attn.q_norm.weight")
            attention.k_norm = self._make_tensor_module(f"{prefix}.self_attn.k_norm.weight")
            layer.self_attn = attention
        else:
            raise ValueError(f"Compressed Qwen layer {layer_id} has no supported attention weights.")

        layer.mlp = SimpleNamespace(
            gate_proj=self._make_linear(f"{prefix}.mlp.gate_proj"),
            up_proj=self._make_linear(f"{prefix}.mlp.up_proj"),
            down_proj=self._make_linear(f"{prefix}.mlp.down_proj"),
        )
        return layer

    def _dequantize(self, weight, weight_scale, weight_scale_2, name):
        if weight_scale_2 is not None:
            if weight_scale is None or weight_scale_2.numel() != 1:
                raise ValueError(f"NVFP4 tensor '{name}' has invalid scales.")
            if weight_scale.dtype == torch.uint8:
                weight_scale = weight_scale.view(torch.float8_e4m3fn)
            low = weight & 0x0F
            high = weight >> 4
            codes = torch.stack((low, high), dim=-1).reshape(weight.shape[0], -1).long()
            magnitudes = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])[codes & 0x7]
            values = torch.where((codes & 0x8) > 0, -magnitudes, magnitudes)
            block_scales = weight_scale.float()
            block_scales = block_scales.repeat_interleave(values.shape[1] // block_scales.shape[1], dim=1)
            return (values * block_scales * weight_scale_2.float().item()).to(torch.bfloat16)
        if weight.dtype == torch.float8_e4m3fn:
            scale = weight_scale.float()
            if scale.numel() != 1:
                scale = scale.reshape(-1, 1)
            return (weight.float() * scale).to(torch.bfloat16)
        return weight
