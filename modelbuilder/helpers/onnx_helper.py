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

import onnx_light.onnx as onnx  # noqa: F401

# ``onnx_light.onnx`` does not import its submodules automatically while the
# ``onnx`` package does. They are imported here so that ``onnx.checker``,
# ``onnx.helper``, ... are available on the exported module.
import onnx_light.onnx.checker  # noqa: F401,E402
import onnx_light.onnx.external_data_helper  # noqa: F401,E402
import onnx_light.onnx.helper  # noqa: F401,E402
import onnx_light.onnx.numpy_helper  # noqa: F401,E402
import onnx_light.onnx.reference  # noqa: F401,E402
import onnx_light.onnx.shape_inference  # noqa: F401,E402

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
