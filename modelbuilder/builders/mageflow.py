# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from .qwen import Qwen3Model


class MageFlowModel(Qwen3Model):
    """Builder for the text backbone of ``microsoft/Mage-Flow``.

    Mage-Flow is a native-resolution text-to-image / image-editing foundation
    model whose prompt/instruction encoder is a Qwen3-VL text decoder.  When the
    model is exported for causal-LM style prompt encoding, its text component is
    a Qwen3-style decoder (RMSNorm, SwiGLU MLP and per-head QK normalisation), so
    this builder reuses :class:`Qwen3Model` unchanged.
    """

    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
