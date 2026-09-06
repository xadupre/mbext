# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Small helpers around the native :mod:`onnx_light.onnx` APIs."""

from __future__ import annotations

from typing import Any

import ml_dtypes
import numpy as np
import onnx_light.onnx as onnx
import torch

import onnx_light.onnx.checker  # noqa: F401,E402
import onnx_light.onnx.external_data_helper  # noqa: F401,E402
import onnx_light.onnx.helper  # noqa: F401,E402
import onnx_light.onnx.numpy_helper  # noqa: F401,E402
import onnx_light.onnx.onnx_pb  # noqa: F401,E402
import onnx_light.onnx.reference  # noqa: F401,E402
import onnx_light.onnx.shape_inference  # noqa: F401,E402

_TORCH_DTYPE_TO_ONNX: dict[torch.dtype, int] = {
    torch.bfloat16: onnx.TensorProto.BFLOAT16,
    torch.bool: onnx.TensorProto.BOOL,
    torch.complex128: onnx.TensorProto.COMPLEX128,
    torch.complex64: onnx.TensorProto.COMPLEX64,
    torch.float16: onnx.TensorProto.FLOAT16,
    torch.float32: onnx.TensorProto.FLOAT,
    torch.float64: onnx.TensorProto.DOUBLE,
    torch.float8_e4m3fn: onnx.TensorProto.FLOAT8E4M3FN,
    torch.float8_e4m3fnuz: onnx.TensorProto.FLOAT8E4M3FNUZ,
    torch.float8_e5m2: onnx.TensorProto.FLOAT8E5M2,
    torch.float8_e5m2fnuz: onnx.TensorProto.FLOAT8E5M2FNUZ,
    torch.int16: onnx.TensorProto.INT16,
    torch.int32: onnx.TensorProto.INT32,
    torch.int64: onnx.TensorProto.INT64,
    torch.int8: onnx.TensorProto.INT8,
    torch.uint8: onnx.TensorProto.UINT8,
    torch.uint16: onnx.TensorProto.UINT16,
    torch.uint32: onnx.TensorProto.UINT32,
    torch.uint64: onnx.TensorProto.UINT64,
}
if hasattr(torch, "float8_e8m0fnu"):
    _TORCH_DTYPE_TO_ONNX[torch.float8_e8m0fnu] = onnx.TensorProto.FLOAT8E8M0
if hasattr(torch, "int2"):
    _TORCH_DTYPE_TO_ONNX[torch.int2] = onnx.TensorProto.INT2
if hasattr(torch, "uint2"):
    _TORCH_DTYPE_TO_ONNX[torch.uint2] = onnx.TensorProto.UINT2

_ONNX_DTYPE_TO_TORCH = {onnx_dtype: torch_dtype for torch_dtype, onnx_dtype in _TORCH_DTYPE_TO_ONNX.items()}
_SPECIAL_NUMPY_DTYPES: dict[int, Any] = {
    onnx.TensorProto.BFLOAT16: ml_dtypes.bfloat16,
    onnx.TensorProto.FLOAT8E4M3FN: ml_dtypes.float8_e4m3fn,
    onnx.TensorProto.FLOAT8E4M3FNUZ: ml_dtypes.float8_e4m3fnuz,
    onnx.TensorProto.FLOAT8E5M2: ml_dtypes.float8_e5m2,
    onnx.TensorProto.FLOAT8E5M2FNUZ: ml_dtypes.float8_e5m2fnuz,
}
if hasattr(ml_dtypes, "float8_e8m0fnu"):
    _SPECIAL_NUMPY_DTYPES[onnx.TensorProto.FLOAT8E8M0] = ml_dtypes.float8_e8m0fnu
if hasattr(ml_dtypes, "int2"):
    _SPECIAL_NUMPY_DTYPES[onnx.TensorProto.INT2] = ml_dtypes.int2
if hasattr(ml_dtypes, "uint2"):
    _SPECIAL_NUMPY_DTYPES[onnx.TensorProto.UINT2] = ml_dtypes.uint2


def from_torch_dtype(dtype: torch.dtype) -> int:
    """Convert a torch dtype to an ONNX TensorProto data type."""
    try:
        return _TORCH_DTYPE_TO_ONNX[dtype]
    except KeyError:
        raise TypeError(f"Unsupported torch dtype {dtype!r}.") from None


def to_torch_dtype(dtype: int) -> torch.dtype:
    """Convert an ONNX TensorProto data type to a torch dtype."""
    try:
        return _ONNX_DTYPE_TO_TORCH[int(dtype)]
    except KeyError:
        raise TypeError(f"Unsupported ONNX tensor data type {dtype!r}.") from None


def torch_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Return a NumPy view with an ONNX-compatible dtype."""
    tensor = tensor.detach().cpu().contiguous()
    onnx_dtype = from_torch_dtype(tensor.dtype)
    if onnx_dtype == onnx.TensorProto.BFLOAT16:
        return tensor.view(torch.uint16).numpy(force=True).view(ml_dtypes.bfloat16)
    if onnx_dtype in _SPECIAL_NUMPY_DTYPES:
        return tensor.view(torch.uint8).numpy(force=True).view(_SPECIAL_NUMPY_DTYPES[onnx_dtype])
    return tensor.numpy(force=True)


#: Opset used when the maximum opset supported by ``onnxruntime`` cannot be
#: determined (for example when ``onnxruntime`` is not installed).
DEFAULT_ONNX_OPSET = 21


def get_default_onnx_opset() -> int:
    """Returns the highest ONNX opset supported by the installed onnxruntime.

    The value is derived from the operator schemas registered by
    :mod:`onnxruntime` for the default (``ai.onnx``) domain.  When
    ``onnxruntime`` is not available, :data:`DEFAULT_ONNX_OPSET` is returned
    instead.
    """
    try:
        from onnxruntime.capi._pybind_state import get_all_operator_schema
    except ImportError:
        return DEFAULT_ONNX_OPSET

    max_opset = 0
    for schema in get_all_operator_schema():
        if schema.domain in ("", "ai.onnx"):
            max_opset = max(max_opset, schema.since_version)
    if not max_opset:
        return DEFAULT_ONNX_OPSET
    # ``onnxruntime`` registers operator schemas for the next opset before it can
    # actually load models using it, so the highest ``since_version`` overshoots
    # the opset ``onnxruntime`` accepts by one (e.g. schemas up to 27 while only
    # 26 is loadable).
    return max_opset - 1
