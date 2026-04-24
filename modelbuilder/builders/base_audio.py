# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import numpy as np
import onnx_ir as ir

from .base import Model


class AudioEncoderModel(Model):
    """Base class for audio-encoder ONNX graph builders.

    Provides shared graph-construction helpers used by both
    ``Phi4MultimodalAudioEncoderModel`` (Conformer-based) and
    ``Qwen25OmniAudioEncoderModel`` (Whisper-style):

    * :meth:`make_audio_layer_norm` — ``LayerNormalization`` with weight and
      bias (standard LN, no simplified variant).
    * :meth:`_make_audio_linear` — ``MatMul`` + optional ``Add`` (bias)
      projection.
    * :meth:`_make_swish` — ``sigmoid(x) * x`` (SiLU activation).
    * :meth:`_make_constant_i64` — scalar ``int64`` ``Constant`` node.
    * :meth:`_make_constant_i64_1d` — 1-D ``int64`` ``Constant`` node.

    Subclasses implement :meth:`load_hf_model` and :meth:`make_model` with
    architecture-specific logic.
    """

    FILENAME = "audio_encoder.onnx"

    #: LayerNorm epsilon; subclasses may override.
    audio_ln_eps = 1e-5

    # ------------------------------------------------------------------
    # Shared graph-construction helpers
    # ------------------------------------------------------------------

    def make_audio_layer_norm(self, name, root_input, weight, bias, shape):
        """Build a ``LayerNormalization`` node with weight and bias.

        Parameters
        ----------
        name : str
            Base name for the ONNX node and its output value.  The output
            value is named ``{name}/output_0``.
        root_input : str
            Name of the input value.
        weight : torch.Tensor
            Layer-norm weight (gamma), 1-D.
        bias : torch.Tensor
            Layer-norm bias (beta), 1-D.
        shape : list
            Shape annotation for the output value.

        Returns
        -------
        str
            Output value name (``"{name}/output_0"``).
        """
        w_name = f"{name}.weight"
        b_name = f"{name}.bias"
        self.make_initializer(weight.detach(), w_name, to=self.io_dtype)
        self.make_initializer(bias.detach(), b_name, to=self.io_dtype)
        out = f"{name}/output_0"
        self.make_node(
            "LayerNormalization",
            inputs=[root_input, w_name, b_name],
            outputs=[out],
            name=name,
            axis=-1,
            epsilon=self.audio_ln_eps,
            stash_type=1,
        )
        self.make_value(out, self.io_dtype, shape=shape)
        return out

    def _make_audio_linear(self, name, linear, root_input, shape):
        """Build a ``MatMul`` + optional ``Add`` (bias) projection.

        Parameters
        ----------
        name : str
            ONNX node-name prefix (e.g. ``"/audio/proj/up"``).
        linear : nn.Linear
            Source HF linear layer.  ``weight.T`` is stored as an initializer;
            bias (if present) is added via a separate ``Add`` node.
        root_input : str
            Name of the input value.
        shape : list
            Shape annotation for the output value.

        Returns
        -------
        str
            Output value name.
        """
        prefix = name[1:].replace("/", ".")
        w_name = f"{prefix}.weight"
        self.make_initializer(linear.weight.T.detach(), w_name, to=self.io_dtype)
        mm_out = f"{name}/MatMul/output_0"
        self.make_node("MatMul", inputs=[root_input, w_name], outputs=[mm_out], name=f"{name}/MatMul")
        self.make_value(mm_out, self.io_dtype, shape=shape)
        if linear.bias is not None:
            b_name = f"{prefix}.bias"
            self.make_initializer(linear.bias.detach(), b_name, to=self.io_dtype)
            add_out = f"{name}/Add/output_0"
            self.make_node("Add", inputs=[mm_out, b_name], outputs=[add_out], name=f"{name}/Add")
            self.make_value(add_out, self.io_dtype, shape=shape)
            return add_out
        return mm_out

    def _make_swish(self, name, root_input, shape):
        """Build a SiLU activation: ``sigmoid(x) * x``.

        Parameters
        ----------
        name : str
            ONNX node-name prefix.
        root_input : str
            Name of the input value.
        shape : list
            Shape annotation for the output value.

        Returns
        -------
        str
            Output value name.
        """
        self.make_sigmoid(f"{name}/Sigmoid", root_input, self.io_dtype, shape)
        return self.make_mul(f"{name}/Mul", [root_input, f"{name}/Sigmoid/output_0"], self.io_dtype, shape)

    def _make_constant_i64(self, name, value):
        """Create a scalar ``int64`` ``Constant`` node.

        Parameters
        ----------
        name : str
            Output name for the constant value.
        value : int
            Scalar integer value.

        Returns
        -------
        str
            Output value name (same as *name*).
        """
        t = ir.Tensor(np.array(value, dtype=np.int64), name=name)
        self.make_node("Constant", inputs=[], outputs=[name], name=f"{name}/Constant", value=t)
        self.make_value(name, ir.DataType.INT64, shape=[])
        return name

    def _make_constant_i64_1d(self, name, values):
        """Create a 1-D ``int64`` ``Constant`` node.

        Parameters
        ----------
        name : str
            Output name for the constant value.
        values : list[int]
            List of integer values.

        Returns
        -------
        str
            Output value name (same as *name*).
        """
        t = ir.Tensor(np.array(values, dtype=np.int64), name=name)
        self.make_node("Constant", inputs=[], outputs=[name], name=f"{name}/Constant", value=t)
        self.make_value(name, ir.DataType.INT64, shape=[len(values)])
        return name
