# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Metadata-only checkpoint tensors and dense decoder inputs."""

import json
import math
import os
import struct
from dataclasses import dataclass, replace
from types import SimpleNamespace

import torch
from onnx_light.onnx import TensorProto
from safetensors import safe_open

from .onnx_helper import to_torch_dtype
from .weight_reuse import checkpoint_paths


@dataclass(frozen=True)
class SourceTensor:
    path: str
    location: str
    offset: int
    length: int
    dtype: int
    shape: tuple[int, ...]
    name: str = ""


_SAFETENSORS_ONNX_DTYPES = {
    "BOOL": TensorProto.BOOL,
    "F16": TensorProto.FLOAT16,
    "F32": TensorProto.FLOAT,
    "F64": TensorProto.DOUBLE,
    "BF16": TensorProto.BFLOAT16,
    "I8": TensorProto.INT8,
    "I16": TensorProto.INT16,
    "I32": TensorProto.INT32,
    "I64": TensorProto.INT64,
    "U8": TensorProto.UINT8,
    "U16": TensorProto.UINT16,
    "U32": TensorProto.UINT32,
    "U64": TensorProto.UINT64,
}


def read_checkpoint_metadata(directory):
    """Read only shard headers, validating tensor shapes and byte ranges."""
    tensors = {}
    for path in checkpoint_paths(directory):
        size = os.path.getsize(path)
        with open(path, "rb") as stream:
            length = stream.read(8)
            if len(length) != 8:
                raise ValueError(f"Invalid safetensors header in {path!r}.")
            header_length = struct.unpack("<Q", length)[0]
            if header_length > size - 8:
                raise ValueError(f"Invalid safetensors header length in {path!r}.")
            header = json.loads(stream.read(header_length))
        data_start = 8 + header_length
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if name in tensors:
                raise ValueError(f"Duplicate checkpoint tensor {name!r}.")
            tag = metadata["dtype"]
            dtype = _SAFETENSORS_ONNX_DTYPES.get(tag)
            if dtype is None:
                continue
            shape = tuple(metadata["shape"])
            start, end = metadata["data_offsets"]
            bits = 16 if tag == "BF16" else 8 if tag == "BOOL" else int(tag[1:])
            if (
                any(not isinstance(dim, int) or dim < 0 for dim in shape)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or data_start + end > size
                or end - start != math.prod(shape) * bits // 8
            ):
                raise ValueError(f"Invalid safetensors tensor metadata for {name!r} in {path!r}.")
            tensors[name] = SourceTensor(path, os.path.basename(path), data_start + start, end - start, int(dtype), shape, name)
    return tensors


@dataclass(frozen=True)
class CheckpointTensor:
    """External tensor sources with deferred dtype, concatenation and transpose."""

    sources: tuple[SourceTensor, ...]
    dtype: torch.dtype
    transposed: bool = False

    @property
    def shape(self):
        shape = self.sources[0].shape
        if len(self.sources) > 1:
            shape = (sum(source.shape[0] for source in self.sources), *shape[1:])
        return shape[::-1] if self.transposed else shape

    @property
    def T(self):
        if len(self.shape) != 2:
            raise ValueError("Only matrix checkpoint tensors can be transposed.")
        return replace(self, transposed=not self.transposed)

    def materialize(self, dtype):
        """Read the selected weights and apply their deferred transformations."""
        tensors = []
        for source in self.sources:
            with safe_open(source.path, framework="pt", device="cpu") as checkpoint:
                tensors.append(checkpoint.get_tensor(source.name).to(dtype=dtype))
        tensor = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
        return tensor.T if self.transposed else tensor

    @classmethod
    def concat(cls, tensors):
        first = tensors[0]
        if any(tensor.transposed or tensor.dtype != first.dtype or tensor.shape[1:] != first.shape[1:] for tensor in tensors):
            raise ValueError("Incompatible checkpoint tensors for concatenation.")
        return cls(tuple(source for tensor in tensors for source in tensor.sources), first.dtype)


class SafetensorsCheckpoint:
    """Index checkpoint tensors without constructing PyTorch modules."""

    def __init__(self, paths):
        self.files = {}
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name in self.files:
                        raise ValueError(f"Duplicate checkpoint tensor {name!r}.")
                    self.files[name] = path
        if not self.files:
            raise ValueError("No safetensors checkpoint tensors were found.")

    def tensor(self, name):
        with safe_open(self.files[name], framework="pt", device="cpu") as source:
            return source.get_tensor(name)

    def items(self):
        """Read each shard's tensor views through one safetensors handle."""
        for path in dict.fromkeys(self.files.values()):
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    yield name, source.get_tensor(name)


class DenseCheckpoint:
    """The named tensors consumed by the Llama, Qwen2 and Qwen3 builders."""

    def __init__(self, directory, config, *, materialize=False):
        self.config = config
        self.directory = os.fspath(directory)
        self.materialized = materialize
        self.aliases = {}
        metadata = read_checkpoint_metadata(directory)
        dtype = getattr(config, "dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if dtype is None:
            dtype = next(
                (to_torch_dtype(source.dtype) for source in metadata.values() if to_torch_dtype(source.dtype).is_floating_point), None
            )
            if dtype is None:
                raise ValueError("A dense checkpoint must contain floating-point weights.")
        if materialize:
            self.tensors = {
                name: tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor
                for name, tensor in SafetensorsCheckpoint(checkpoint_paths(directory)).items()
            }
        else:
            self.tensors = {name: CheckpointTensor((source,), dtype) for name, source in metadata.items()}
        if config.tie_word_embeddings:
            name = "lm_head.weight" if "lm_head.weight" in self.tensors else "model.embed_tokens.weight"
            weight = self._tensor(name, (config.vocab_size, config.hidden_size))
            self.tensors["model.embed_tokens.weight"] = weight
            self.tensors["lm_head.weight"] = weight
            self.aliases = {"model.embed_tokens.weight": name, "lm_head.weight": name}

    def _tensor(self, name, shape):
        if name not in self.tensors:
            raise ValueError(f"Missing dense checkpoint tensor {name!r}.")
        tensor = self.tensors[name]
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(f"Dense checkpoint tensor {name!r} has shape {tuple(tensor.shape)}, expected {tuple(shape)}.")
        source_dtypes = (
            [to_torch_dtype(source.dtype) for source in tensor.sources] if isinstance(tensor, CheckpointTensor) else [tensor.dtype]
        )
        if any(dtype not in {torch.float16, torch.bfloat16, torch.float32, torch.float64} for dtype in source_dtypes):
            raise ValueError(f"Dense checkpoint tensor {name!r} has unsupported dtype {tensor.dtype}.")
        return tensor

    def _linear(self, name, out_features, in_features, bias=False):
        return SimpleNamespace(
            weight=self._tensor(name + ".weight", (out_features, in_features)),
            bias=self._tensor(name + ".bias", (out_features,)) if bias else None,
            in_features=in_features,
            out_features=out_features,
        )

    def _norm(self, name, width):
        return SimpleNamespace(weight=self._tensor(name + ".weight", (width,)), bias=None)

    @property
    def embedding(self):
        return self._tensor("model.embed_tokens.weight", (self.config.vocab_size, self.config.hidden_size))

    @property
    def final_norm(self):
        return self._norm("model.norm", self.config.hidden_size)

    @property
    def lm_head(self):
        return self._linear("lm_head", self.config.vocab_size, self.config.hidden_size)

    def layer(self, layer_id):
        config = self.config
        prefix = f"model.layers.{layer_id}"
        hidden = config.hidden_size
        head_dim = getattr(config, "head_dim", None) or hidden // config.num_attention_heads
        q_size = config.num_attention_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim
        attention_bias = getattr(config, "attention_bias", False)
        qkv_bias = config.model_type == "qwen2" or attention_bias
        attention = SimpleNamespace(
            q_proj=self._linear(prefix + ".self_attn.q_proj", q_size, hidden, qkv_bias),
            k_proj=self._linear(prefix + ".self_attn.k_proj", kv_size, hidden, qkv_bias),
            v_proj=self._linear(prefix + ".self_attn.v_proj", kv_size, hidden, qkv_bias),
            o_proj=self._linear(prefix + ".self_attn.o_proj", hidden, q_size, attention_bias and config.model_type != "qwen2"),
        )
        if config.model_type == "qwen3":
            attention.q_norm = self._norm(prefix + ".self_attn.q_norm", head_dim)
            attention.k_norm = self._norm(prefix + ".self_attn.k_norm", head_dim)
        mlp_bias = config.model_type == "llama" and getattr(config, "mlp_bias", False)
        return SimpleNamespace(
            input_layernorm=self._norm(prefix + ".input_layernorm", hidden),
            self_attn=attention,
            post_attention_layernorm=self._norm(prefix + ".post_attention_layernorm", hidden),
            mlp=SimpleNamespace(
                gate_proj=self._linear(prefix + ".mlp.gate_proj", config.intermediate_size, hidden, mlp_bias),
                up_proj=self._linear(prefix + ".mlp.up_proj", config.intermediate_size, hidden, mlp_bias),
                down_proj=self._linear(prefix + ".mlp.down_proj", hidden, config.intermediate_size, mlp_bias),
            ),
        )
