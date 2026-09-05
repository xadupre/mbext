# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Minimal ONNX graph IR used to build models.

This package implements the subset of the ``onnx_ir`` (ir-py) API mbext relies
on, on top of the protos exposed by ``onnx_light``.  It is used as::

    from modelbuilder import ir

    value = ir.Value(name="x", type=ir.TensorType(ir.DataType.FLOAT), shape=ir.Shape([1, 2]))
"""

from . import _serde as serde  # noqa: F401
from . import tensor_adapters  # noqa: F401
from ._core import (  # noqa: F401
    Attr,
    ExternalTensor,
    Function,
    Graph,
    LazyTensor,
    Model,
    Node,
    PackedTensor,
    Shape,
    StringTensor,
    Tensor,
    TensorBase,
    TensorProtocol,
    TensorType,
    Value,
    node,
    tensor,
)
from ._enums import AttributeType, DataType  # noqa: F401
from ._io import CallbackInfo, load, save  # noqa: F401
from ._serde import from_proto, to_proto  # noqa: F401

__all__ = [
    "Attr",
    "AttributeType",
    "CallbackInfo",
    "DataType",
    "ExternalTensor",
    "Function",
    "Graph",
    "LazyTensor",
    "Model",
    "Node",
    "PackedTensor",
    "Shape",
    "StringTensor",
    "Tensor",
    "TensorBase",
    "TensorProtocol",
    "TensorType",
    "Value",
    "from_proto",
    "load",
    "node",
    "save",
    "serde",
    "tensor",
    "tensor_adapters",
    "to_proto",
]
