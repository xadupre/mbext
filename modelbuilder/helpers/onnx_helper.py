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
