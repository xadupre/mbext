# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from .base import Model


class VisionEncoderModel(Model):
    """Base class for ViT-style vision-encoder ONNX graph builders.

    Provides shared graph-construction helpers used by both
    ``Ministral3VisionEncoderModel`` (Pixtral) and
    ``Qwen25OmniVisionEncoderModel`` (Qwen2.5-Omni visual tower):

    * :meth:`make_rms_norm` — ``SimplifiedLayerNormalization`` node.
    * :meth:`make_silu_gated_mlp` — ``SiLU(gate) * up → down`` block.

    Subclasses override :meth:`make_attention`, :meth:`make_layer`, and
    :meth:`make_model` with architecture-specific logic.
    """

    FILENAME = "vision_encoder.onnx"

    #: RMSNorm epsilon; subclasses may override (e.g. Ministral3 uses 1e-5).
    vis_rms_norm_eps = 1e-6

    # ------------------------------------------------------------------
    # Shared graph-construction helpers
    # ------------------------------------------------------------------

    def make_rms_norm(self, name, root_input, weight, shape, weight_name=None):
        """Build a ``SimplifiedLayerNormalization`` node (encoder RMSNorm).

        Parameters
        ----------
        name : str
            Base name for the ONNX node and its output value.  The output
            value is named ``{name}/output_0``.
        root_input : str
            Name of the input value.
        weight : torch.Tensor
            RMSNorm weight tensor (1-D, length = last dim of ``root_input``).
        shape : list
            Shape annotation for the output value.
        weight_name : str, optional
            Initializer name for the weight tensor.  Defaults to
            ``"{name}.weight"``.
        """
        w_name = weight_name if weight_name is not None else f"{name}.weight"
        self.make_initializer(weight, w_name, to=self.io_dtype)
        out = f"{name}/output_0"
        self.make_node(
            "SimplifiedLayerNormalization",
            inputs=[root_input, w_name],
            outputs=[out],
            name=name,
            axis=-1,
            epsilon=self.vis_rms_norm_eps,
            stash_type=1,
        )
        self.make_value(out, self.io_dtype, shape=shape)
        return out

    def make_layer_norm(self, name, root_input, weight, bias, shape, weight_name=None, bias_name=None):
        """Build a ``LayerNormalization`` node (full LN with weight and bias).

        Unlike :meth:`make_rms_norm` (``SimplifiedLayerNormalization``, no bias),
        this emits a standard ``LayerNormalization`` that includes both a
        multiplicative weight (gamma) and an additive bias (beta).

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
        weight_name : str, optional
            Initializer name for the weight tensor.  Defaults to
            ``"{name}.weight"``.
        bias_name : str, optional
            Initializer name for the bias tensor.  Defaults to
            ``"{name}.bias"``.
        """
        w_name = weight_name if weight_name is not None else f"{name}.weight"
        b_name = bias_name if bias_name is not None else f"{name}.bias"
        self.make_initializer(weight, w_name, to=self.io_dtype)
        self.make_initializer(bias, b_name, to=self.io_dtype)
        out = f"{name}/output_0"
        self.make_node(
            "LayerNormalization",
            inputs=[root_input, w_name, b_name],
            outputs=[out],
            name=name,
            axis=-1,
            epsilon=self.vis_rms_norm_eps,
            stash_type=1,
        )
        self.make_value(out, self.io_dtype, shape=shape)
        return out

    def make_gelu_mlp(self, linear1, linear2, root_input, shape, basename):
        """Build a ``Linear + GELU + Linear`` block with optional biases.

        A common projector pattern in multimodal vision encoders (e.g. Phi-4-multimodal).

        Parameters
        ----------
        linear1 : nn.Linear
            First linear layer.
        linear2 : nn.Linear
            Second linear layer.
        root_input : str
            Input tensor name.
        shape : list
            Shape annotation for intermediate and output tensors.
        basename : str
            Base name prefix for ONNX nodes (e.g. ``"/vision/proj"``).

        Returns
        -------
        str
            Output value name.
        """
        # Linear 1 + optional bias.
        out = f"{self.make_matmul(linear1, f'{basename}/linear_1/MatMul', root_input)}/output_0"
        if linear1.bias is not None:
            bias_name = f"{basename[1:].replace('/', '.')}.linear_1.bias"
            self.make_initializer(linear1.bias, bias_name, to=self.io_dtype)
            out = self.make_add(f"{basename}/linear_1/Add", [out, bias_name], self.io_dtype, shape)

        # GELU activation.
        gelu_out = f"{basename}/gelu/output_0"
        self.make_node("Gelu", inputs=[out], outputs=[gelu_out], name=f"{basename}/gelu/Gelu", domain="com.microsoft")
        self.make_value(gelu_out, self.io_dtype, shape=shape)

        # Linear 2 + optional bias.
        out = f"{self.make_matmul(linear2, f'{basename}/linear_2/MatMul', gelu_out)}/output_0"
        if linear2.bias is not None:
            bias_name = f"{basename[1:].replace('/', '.')}.linear_2.bias"
            self.make_initializer(linear2.bias, bias_name, to=self.io_dtype)
            out = self.make_add(f"{basename}/linear_2/Add", [out, bias_name], self.io_dtype, shape)

        return out

    def make_silu_gated_mlp(self, layer_id, mlp, root_input, intermediate_shape):
        """Build a SiLU-gated MLP: ``SiLU(gate_proj) * up_proj → down_proj``.

        Handles optional bias on each linear layer.

        Parameters
        ----------
        layer_id : int
            Index of the transformer layer (used for node naming).
        mlp : nn.Module
            Module with ``gate_proj``, ``up_proj``, and ``down_proj``
            attributes (each an ``nn.Linear``).
        root_input : str
            Name of the input value tensor.
        intermediate_shape : list
            Shape annotation for intermediate tensors (e.g. ``[1, n, ff]``
            for static-batch models or ``[None, ff]`` for dynamic).
        """
        b = f"/vision/layers.{layer_id}/mlp"

        gate = f"{self.make_matmul(mlp.gate_proj, f'{b}/gate_proj/MatMul', root_input)}/output_0"
        if mlp.gate_proj.bias is not None:
            self.make_add_bias(mlp.gate_proj.bias, f"{b}/gate_proj/Add", root_input=gate)
            gate = f"{b}/gate_proj/Add/output_0"

        up = f"{self.make_matmul(mlp.up_proj, f'{b}/up_proj/MatMul', root_input)}/output_0"
        if mlp.up_proj.bias is not None:
            self.make_add_bias(mlp.up_proj.bias, f"{b}/up_proj/Add", root_input=up)
            up = f"{b}/up_proj/Add/output_0"

        # SiLU(gate) = gate * Sigmoid(gate)
        self.make_sigmoid(f"{b}/act/Sigmoid", gate, self.io_dtype, intermediate_shape)
        silu = self.make_mul(f"{b}/act/Mul_silu", [gate, f"{b}/act/Sigmoid/output_0"], self.io_dtype, intermediate_shape)
        gate_up = self.make_mul(f"{b}/gate_up/Mul", [silu, up], self.io_dtype, intermediate_shape)

        down = f"{self.make_matmul(mlp.down_proj, f'{b}/down_proj/MatMul', gate_up)}/output_0"
        if mlp.down_proj.bias is not None:
            self.make_add_bias(mlp.down_proj.bias, f"{b}/down_proj/Add", root_input=down)
            return f"{b}/down_proj/Add/output_0"
        return down
