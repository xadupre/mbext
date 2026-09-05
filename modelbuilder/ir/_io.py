# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Loads and saves ONNX models, with support for external data."""

from __future__ import annotations

import dataclasses
import os
from typing import Callable, Optional

from modelbuilder.helpers.onnx_helper import onnx

from . import _serde
from ._core import Model, TensorBase

# Offsets of large tensors are aligned so the data can be memory mapped.
_ALIGN_THRESHOLD = 1048576  # 1MB
_ALIGNMENT_FACTOR = 4096


@dataclasses.dataclass
class CallbackInfo:
    """Information given to the callback of :func:`save` for every saved tensor.

    Attributes:
        total: total number of tensors written to the external data file.
        offset: offset of the tensor in the external data file.
        location: name of the external data file.
    """

    total: int
    offset: int
    location: str


def _external_tensor_count(model: Model, size_threshold_bytes: int) -> int:
    """Counts the initializers written to the external data file."""
    total = 0
    for graph in model.graphs():
        for value in graph.initializers.values():
            tensor = value.const_value
            if tensor is not None and tensor.nbytes > size_threshold_bytes:
                total += 1
    return total


def save(
    model: Model,
    path: str | os.PathLike,
    format: Optional[str] = None,  # noqa: A002
    external_data: str | os.PathLike | None = None,
    size_threshold_bytes: int = 256,
    callback: Optional[Callable[[TensorBase, CallbackInfo], None]] = None,
) -> None:
    """Saves a model on disk.

    Args:
        model: model to save.
        path: path of the ONNX file.
        format: format of the file, inferred from the extension when ``None``.
        external_data: relative path of the external data file. When specified,
            every initializer bigger than ``size_threshold_bytes`` is written
            into that file instead of the ONNX proto.
        size_threshold_bytes: minimum size (in bytes) for a tensor to be written
            into the external data file.
        callback: called for every tensor written into the external data file.

    Raises:
        ValueError: if ``external_data`` is an absolute path.
    """
    if external_data is None:
        proto = _serde.serialize_model(model)
        onnx.save_model(proto, os.fspath(path), format=format)
        return

    if os.path.isabs(external_data):
        raise ValueError(f"The external data path must be relative to the ONNX file path, not {external_data!r}.")

    base_dir = os.path.dirname(os.fspath(path))
    location = os.fspath(external_data)
    data_path = os.path.join(base_dir, location)
    total = _external_tensor_count(model, size_threshold_bytes)

    with open(data_path, "wb") as f:
        offset = 0

        def writer(tensor: TensorBase):
            nonlocal offset
            length = tensor.nbytes
            if length <= size_threshold_bytes:
                return None
            if length > _ALIGN_THRESHOLD and offset % _ALIGNMENT_FACTOR:
                padding = _ALIGNMENT_FACTOR - (offset % _ALIGNMENT_FACTOR)
                f.write(b"\0" * padding)
                offset += padding
            if callback is not None:
                callback(tensor, CallbackInfo(total=total, offset=offset, location=location))
            tofile = getattr(tensor, "tofile", None)
            if tofile is not None:
                written = tofile(f)
            else:
                written = f.write(tensor.tobytes())
            if written != length:
                raise RuntimeError(f"Tensor {tensor.name!r} wrote {written} bytes instead of {length}.")
            tensor_offset = offset
            offset += length
            return (location, tensor_offset, length)

        proto = _serde.serialize_model(model, external_data_writer=writer)

    onnx.save_model(proto, os.fspath(path), format=format)


def load(path: str | os.PathLike, format: Optional[str] = None) -> Model:  # noqa: A002
    """Loads a model from disk. External data is read lazily.

    Args:
        path: path of the ONNX file.
        format: format of the file, inferred from the extension when ``None``.

    Returns:
        The model.
    """
    kwargs = {} if format is None else {"format": format}
    proto = onnx.load_model(os.fspath(path), load_external_data=False, **kwargs)
    return _serde.deserialize_model(proto, base_dir=os.path.dirname(os.fspath(path)))
