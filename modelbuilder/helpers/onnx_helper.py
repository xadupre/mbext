# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Resolves the :mod:`onnx` module, backed by ``onnx-light``.

mbext only depends on ``onnx-light``: the lightweight :mod:`onnx_light.onnx`
module implements the same Python API as the :mod:`onnx` package, so the rest
of the code uses the exported ``onnx`` object transparently::

    from modelbuilder.helpers.onnx_helper import onnx

    TensorProto = onnx.TensorProto
"""

import sys

import onnx_light.onnx as onnx  # noqa: F401

# ``onnx_light.onnx`` does not import its submodules automatically while the
# ``onnx`` package does. They are imported here so that ``onnx.checker``,
# ``onnx.helper``, ... are available on the exported module.
import onnx_light.onnx.checker  # noqa: F401,E402
import onnx_light.onnx.external_data_helper  # noqa: F401,E402
import onnx_light.onnx.helper  # noqa: F401,E402
import onnx_light.onnx.numpy_helper  # noqa: F401,E402
import onnx_light.onnx.onnx_pb  # noqa: F401,E402
import onnx_light.onnx.reference  # noqa: F401,E402
import onnx_light.onnx.shape_inference  # noqa: F401,E402


def _add_repeated_field_compatibility() -> None:
    """Add protobuf-style mutation methods missing from onnx-light fields."""

    def deepcopy(self, memo):
        copied = type(self)()
        copied.ParseFromString(self.SerializeToString())
        memo[id(self)] = copied
        return copied

    def remove(self, value) -> None:
        values = list(self)
        for index, item in enumerate(values):
            if item is value or item == value:
                self.clear()
                self.extend(values[:index])
                self.extend(values[index + 1 :])
                return
        raise ValueError(f"{value!r} is not in the repeated field")

    def insert(self, index, value) -> None:
        values = list(self)
        values.insert(index, value)
        self.clear()
        self.extend(values)

    graph = onnx.GraphProto()
    if not hasattr(onnx.Message, "__deepcopy__"):
        onnx.Message.__deepcopy__ = deepcopy
    for field in (graph.node, graph.input, graph.output, graph.initializer):
        cls = type(field)
        if not hasattr(cls, "remove"):
            cls.remove = remove
        if not hasattr(cls, "insert"):
            cls.insert = insert


def enable_onnxruntime_quantization() -> None:
    """Expose onnx-light compatibility modules required by ORT quantization.

    ``onnxruntime.quantization`` imports the legacy module names even though it
    only needs APIs implemented by onnx-light and :mod:`modelbuilder.ir`.
    Registering these aliases keeps quantization usable without installing
    either legacy package.
    """
    from modelbuilder import ir

    _add_repeated_field_compatibility()
    modules = {
        "onnx": onnx,
        "onnx.external_data_helper": onnx.external_data_helper,
        "onnx.helper": onnx.helper,
        "onnx.numpy_helper": onnx.numpy_helper,
        "onnx.onnx_pb": onnx.onnx_pb,
        "onnx.reference": onnx.reference,
        "onnx.shape_inference": onnx.shape_inference,
        "onnx_ir": ir,
    }
    sys.modules.update(modules)


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
