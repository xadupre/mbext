# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Resolves the :mod:`onnx` module, optionally backed by ``onnx-light``.

When the environment variable ``USE_ONNX_LIGHT`` is set to a truthy value
(``1``, ``true`` or ``True``), the lightweight :mod:`onnx_light.onnx` module is
imported in place of the regular :mod:`onnx` package.  Both modules expose the
same Python API, so the rest of the code can use the exported ``onnx`` object
transparently::

    from modelbuilder.helpers.onnx_helper import onnx

    TensorProto = onnx.TensorProto
"""

import os


def use_onnx_light() -> bool:
    """Returns ``True`` if mbext should use ``onnx_light.onnx`` instead of ``onnx``.

    The choice is controlled by the ``USE_ONNX_LIGHT`` environment variable.
    """
    return os.environ.get("USE_ONNX_LIGHT", "") in (1, "1", "True", "true")


if use_onnx_light():
    import onnx_light.onnx as onnx  # noqa: F401
else:
    import onnx  # noqa: F401


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
