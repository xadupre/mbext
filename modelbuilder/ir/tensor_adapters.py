# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Adapters converting :mod:`torch` tensors into IR tensors.

``torch`` is imported lazily so that the IR can be used without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np

from ._core import Tensor
from ._enums import DataType

if TYPE_CHECKING:
    import torch

_TORCH_DTYPE_TO_ONNX: Optional[Dict[Any, DataType]] = None
_ONNX_DTYPE_TO_TORCH: Optional[Dict[DataType, Any]] = None


def _torch_dtype_map() -> Dict[Any, DataType]:
    global _TORCH_DTYPE_TO_ONNX
    if _TORCH_DTYPE_TO_ONNX is None:
        import torch

        _TORCH_DTYPE_TO_ONNX = {
            torch.bfloat16: DataType.BFLOAT16,
            torch.bool: DataType.BOOL,
            torch.complex128: DataType.COMPLEX128,
            torch.complex64: DataType.COMPLEX64,
            torch.float16: DataType.FLOAT16,
            torch.float32: DataType.FLOAT,
            torch.float64: DataType.DOUBLE,
            torch.float8_e4m3fn: DataType.FLOAT8E4M3FN,
            torch.float8_e4m3fnuz: DataType.FLOAT8E4M3FNUZ,
            torch.float8_e5m2: DataType.FLOAT8E5M2,
            torch.float8_e5m2fnuz: DataType.FLOAT8E5M2FNUZ,
            torch.int16: DataType.INT16,
            torch.int32: DataType.INT32,
            torch.int64: DataType.INT64,
            torch.int8: DataType.INT8,
            torch.uint8: DataType.UINT8,
            torch.uint16: DataType.UINT16,
            torch.uint32: DataType.UINT32,
            torch.uint64: DataType.UINT64,
        }
        if hasattr(torch, "float8_e8m0fnu"):
            _TORCH_DTYPE_TO_ONNX[torch.float8_e8m0fnu] = DataType.FLOAT8E8M0
        if hasattr(torch, "int2"):
            _TORCH_DTYPE_TO_ONNX[torch.int2] = DataType.INT2
        if hasattr(torch, "uint2"):
            _TORCH_DTYPE_TO_ONNX[torch.uint2] = DataType.UINT2
    return _TORCH_DTYPE_TO_ONNX


def from_torch_dtype(dtype: "torch.dtype") -> DataType:
    """Converts a torch dtype into an ONNX data type."""
    mapping = _torch_dtype_map()
    if dtype not in mapping:
        raise TypeError(f"Unsupported torch dtype {dtype!r}.")
    return mapping[dtype]


def to_torch_dtype(dtype: DataType) -> "torch.dtype":
    """Converts an ONNX data type into a torch dtype."""
    global _ONNX_DTYPE_TO_TORCH
    if _ONNX_DTYPE_TO_TORCH is None:
        _ONNX_DTYPE_TO_TORCH = {onnx_dtype: torch_dtype for torch_dtype, onnx_dtype in _torch_dtype_map().items()}
    dtype = DataType(dtype)
    if dtype not in _ONNX_DTYPE_TO_TORCH:
        raise TypeError(f"Unsupported conversion from ONNX data type {dtype!r} to torch.")
    return _ONNX_DTYPE_TO_TORCH[dtype]


class TorchTensor(Tensor):
    """Tensor backed by a :class:`torch.Tensor`."""

    def __init__(self, tensor: "torch.Tensor", name: Optional[str] = None, doc_string: Optional[str] = None):
        super().__init__(tensor, dtype=from_torch_dtype(tensor.dtype), name=name, doc_string=doc_string)

    def numpy(self) -> np.ndarray:
        "Returns the tensor as a numpy array, using ``ml_dtypes`` when needed."
        import torch

        raw: torch.Tensor = self.raw
        if self.dtype == DataType.BFLOAT16:
            return raw.view(torch.uint16).numpy(force=True).view(self.dtype.numpy())
        if self.dtype in {
            DataType.FLOAT8E4M3FN,
            DataType.FLOAT8E4M3FNUZ,
            DataType.FLOAT8E5M2,
            DataType.FLOAT8E5M2FNUZ,
            DataType.FLOAT8E8M0,
            DataType.INT2,
            DataType.UINT2,
        }:
            return raw.view(torch.uint8).numpy(force=True).view(self.dtype.numpy())
        return raw.numpy(force=True)

    def __array__(self, dtype: Any = None, copy: Optional[bool] = None) -> np.ndarray:
        array = self.numpy()
        return array if dtype is None else array.__array__(dtype)

    def _contiguous(self) -> "torch.Tensor":
        return self.raw.detach().cpu().contiguous()

    def tobytes(self) -> bytes:
        "Returns the raw ONNX representation, reading the torch memory directly."
        import ctypes

        tensor = self._contiguous()
        data = (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(tensor.data_ptr())
        return bytes(data)

    def tofile(self, file) -> int:
        "Writes the raw ONNX representation into an opened binary file."
        import ctypes

        tensor = self._contiguous()
        data = (ctypes.c_ubyte * tensor.element_size() * tensor.numel()).from_address(tensor.data_ptr())
        return file.write(data)
