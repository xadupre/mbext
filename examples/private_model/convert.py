# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Converter file for the ``private_model`` example.

This is the ``convert-file`` of the ``--private`` option. It defines the ONNX
builder used to convert the tiny private model (see ``modeling.py``) to ONNX.

Because the ``PrivateTiny`` architecture reuses the Qwen3 backbone, the builder
simply subclasses :class:`modelbuilder.builders.qwen.Qwen3Model`. A real private
model would override the relevant ``make_*`` methods here.

:func:`modelbuilder.builder.load_private_model_builder` selects the builder from
the module-level ``MODEL_BUILDER`` attribute if present, otherwise it uses the
single :class:`~modelbuilder.builders.Model` subclass defined in this file.
"""

from modelbuilder.builders.qwen import Qwen3Model


class PrivateTinyModel(Qwen3Model):
    """ONNX builder for the ``PrivateTiny`` architecture."""


MODEL_BUILDER = PrivateTinyModel
