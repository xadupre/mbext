# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Enumerations mirroring the ONNX specification (``TensorProto`` / ``AttributeProto``)."""

from __future__ import annotations

import enum

import ml_dtypes
import numpy as np


class AttributeType(enum.IntEnum):
    """Types of ONNX attributes, matching ``onnx.AttributeProto.AttributeType``."""

    UNDEFINED = 0
    FLOAT = 1
    INT = 2
    STRING = 3
    TENSOR = 4
    GRAPH = 5
    FLOATS = 6
    INTS = 7
    STRINGS = 8
    TENSORS = 9
    GRAPHS = 10
    SPARSE_TENSOR = 11
    SPARSE_TENSORS = 12
    TYPE_PROTO = 13
    TYPE_PROTOS = 14

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


class DataType(enum.IntEnum):
    """Data types of ONNX tensors, matching ``onnx.TensorProto.DataType``."""

    UNDEFINED = 0
    FLOAT = 1
    UINT8 = 2
    INT8 = 3
    UINT16 = 4
    INT16 = 5
    INT32 = 6
    INT64 = 7
    STRING = 8
    BOOL = 9
    FLOAT16 = 10
    DOUBLE = 11
    UINT32 = 12
    UINT64 = 13
    COMPLEX64 = 14
    COMPLEX128 = 15
    BFLOAT16 = 16
    FLOAT8E4M3FN = 17
    FLOAT8E4M3FNUZ = 18
    FLOAT8E5M2 = 19
    FLOAT8E5M2FNUZ = 20
    UINT4 = 21
    INT4 = 22
    FLOAT4E2M1 = 23
    FLOAT8E8M0 = 24
    UINT2 = 25
    INT2 = 26

    @classmethod
    def from_numpy(cls, dtype) -> "DataType":
        """Returns the ONNX data type matching a numpy dtype."""
        dtype = np.dtype(dtype)
        if dtype in _NP_TYPE_TO_DATA_TYPE:
            return cls(_NP_TYPE_TO_DATA_TYPE[dtype])
        if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_):
            return DataType.STRING
        raise TypeError(f"Unsupported numpy data type: {dtype}")

    @classmethod
    def from_short_name(cls, short_name: str) -> "DataType":
        """Returns the ONNX data type matching a short name such as ``f32``."""
        if short_name not in _SHORT_NAME_TO_DATA_TYPE:
            raise TypeError(f"Unknown short name: {short_name}")
        return cls(_SHORT_NAME_TO_DATA_TYPE[short_name])

    def numpy(self) -> np.dtype:
        """Returns the numpy dtype matching this ONNX data type."""
        if self not in _DATA_TYPE_TO_NP_TYPE:
            raise TypeError(f"Numpy does not support ONNX data type: {self}")
        return _DATA_TYPE_TO_NP_TYPE[self]

    def short_name(self) -> str:
        """Returns a compact name for the data type such as ``f32`` or ``i64``."""
        if self not in _DATA_TYPE_TO_SHORT_NAME:
            raise TypeError(f"Short name not available for ONNX data type: {self}")
        return _DATA_TYPE_TO_SHORT_NAME[self]

    @property
    def bitwidth(self) -> int:
        """Returns the number of bits used by a single element."""
        if self not in _BITWIDTH_MAP:
            raise TypeError(f"Bit width not available for ONNX data type: {self}")
        return _BITWIDTH_MAP[self]

    @property
    def itemsize(self) -> float:
        """Returns the size of one element in bytes (may be fractional for sub-byte types)."""
        return self.bitwidth / 8

    def __repr__(self) -> str:
        return self.name

    def __str__(self) -> str:
        return self.__repr__()


_BITWIDTH_MAP = {
    DataType.FLOAT: 32,
    DataType.UINT8: 8,
    DataType.INT8: 8,
    DataType.UINT16: 16,
    DataType.INT16: 16,
    DataType.INT32: 32,
    DataType.INT64: 64,
    DataType.BOOL: 8,
    DataType.FLOAT16: 16,
    DataType.DOUBLE: 64,
    DataType.UINT32: 32,
    DataType.UINT64: 64,
    DataType.COMPLEX64: 64,
    DataType.COMPLEX128: 128,
    DataType.BFLOAT16: 16,
    DataType.FLOAT8E4M3FN: 8,
    DataType.FLOAT8E4M3FNUZ: 8,
    DataType.FLOAT8E5M2: 8,
    DataType.FLOAT8E5M2FNUZ: 8,
    DataType.UINT4: 4,
    DataType.INT4: 4,
    DataType.FLOAT4E2M1: 4,
    DataType.FLOAT8E8M0: 8,
    DataType.INT2: 2,
    DataType.UINT2: 2,
}

_NP_TYPE_TO_DATA_TYPE = {
    np.dtype("bool"): DataType.BOOL,
    np.dtype("complex128"): DataType.COMPLEX128,
    np.dtype("complex64"): DataType.COMPLEX64,
    np.dtype("float16"): DataType.FLOAT16,
    np.dtype("float32"): DataType.FLOAT,
    np.dtype("float64"): DataType.DOUBLE,
    np.dtype("int16"): DataType.INT16,
    np.dtype("int32"): DataType.INT32,
    np.dtype("int64"): DataType.INT64,
    np.dtype("int8"): DataType.INT8,
    np.dtype("object"): DataType.STRING,
    np.dtype("uint16"): DataType.UINT16,
    np.dtype("uint32"): DataType.UINT32,
    np.dtype("uint64"): DataType.UINT64,
    np.dtype("uint8"): DataType.UINT8,
    np.dtype(ml_dtypes.bfloat16): DataType.BFLOAT16,
    np.dtype(ml_dtypes.float8_e4m3fn): DataType.FLOAT8E4M3FN,
    np.dtype(ml_dtypes.float8_e4m3fnuz): DataType.FLOAT8E4M3FNUZ,
    np.dtype(ml_dtypes.float8_e5m2): DataType.FLOAT8E5M2,
    np.dtype(ml_dtypes.float8_e5m2fnuz): DataType.FLOAT8E5M2FNUZ,
    np.dtype(ml_dtypes.float8_e8m0fnu): DataType.FLOAT8E8M0,
    np.dtype(ml_dtypes.int4): DataType.INT4,
    np.dtype(ml_dtypes.uint4): DataType.UINT4,
    np.dtype(ml_dtypes.float4_e2m1fn): DataType.FLOAT4E2M1,
    np.dtype(ml_dtypes.int2): DataType.INT2,
    np.dtype(ml_dtypes.uint2): DataType.UINT2,
}

_DATA_TYPE_TO_NP_TYPE = {v: k for k, v in _NP_TYPE_TO_DATA_TYPE.items()}

_DATA_TYPE_TO_SHORT_NAME = {
    DataType.UNDEFINED: "undefined",
    DataType.BFLOAT16: "bf16",
    DataType.DOUBLE: "f64",
    DataType.FLOAT: "f32",
    DataType.FLOAT16: "f16",
    DataType.FLOAT8E4M3FN: "f8e4m3fn",
    DataType.FLOAT8E5M2: "f8e5m2",
    DataType.FLOAT8E4M3FNUZ: "f8e4m3fnuz",
    DataType.FLOAT8E5M2FNUZ: "f8e5m2fnuz",
    DataType.FLOAT8E8M0: "f8e8m0",
    DataType.FLOAT4E2M1: "f4e2m1",
    DataType.COMPLEX64: "c64",
    DataType.COMPLEX128: "c128",
    DataType.INT2: "i2",
    DataType.INT4: "i4",
    DataType.INT8: "i8",
    DataType.INT16: "i16",
    DataType.INT32: "i32",
    DataType.INT64: "i64",
    DataType.BOOL: "b8",
    DataType.UINT2: "u2",
    DataType.UINT4: "u4",
    DataType.UINT8: "u8",
    DataType.UINT16: "u16",
    DataType.UINT32: "u32",
    DataType.UINT64: "u64",
    DataType.STRING: "s",
}

_SHORT_NAME_TO_DATA_TYPE = {v: k for k, v in _DATA_TYPE_TO_SHORT_NAME.items()}
