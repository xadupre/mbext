# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""ONNX local-function fallbacks built with onnx-light."""

from __future__ import annotations

import numpy as np
import onnx_light.onnx.helper as onnx_helper
import onnx_light.onnx.numpy_helper as numpy_helper
from onnx_light.onnx import FunctionProto, TensorProto
from onnx_light.onnx_core.graph_builder import GraphBuilder

_ONNX_LARGE_SLICE_END = 2**62


def _constant(builder: GraphBuilder, name: str, data, dtype=np.int64) -> str:
    tensor = numpy_helper.from_array(np.asarray(data, dtype=dtype), name=name)
    builder.make_node("Constant", [], [name], attributes={"value": tensor})
    return name


def _input(builder: GraphBuilder, name: str, dtype: int, shape) -> str:
    return builder.make_input(onnx_helper.make_tensor_value_info(name, dtype, shape))


def _constant_node(name: str, data, dtype=np.int64):
    tensor = numpy_helper.from_array(np.asarray(data, dtype=dtype), name=name)
    return onnx_helper.make_node("Constant", [], [name], value=tensor)


def normalize_function_opsets(function: FunctionProto) -> FunctionProto:
    """Use the canonical empty domain for standard ONNX function nodes."""
    imports = [opset for opset in function.opset_import if opset.domain != function.domain]
    function.opset_import.clear()
    function.opset_import.extend(imports)
    for opset in function.opset_import:
        if opset.domain == "ai.onnx":
            opset.domain = ""
    return function


class LocalFunctionsMixin:
    """Mixin adding standard-operator fallbacks for newer contrib operators."""

    @staticmethod
    def _ort_version() -> tuple[int, ...]:
        try:
            import onnxruntime

            return tuple(int(x) for x in onnxruntime.__version__.split(".")[:3])
        except (ImportError, ValueError):
            return (99, 99, 0)

    @staticmethod
    def _build_causal_conv_local_function(builder: GraphBuilder, kernel_size: int, io_dtype: int) -> None:
        builder.set_opset_version("", 21)
        _input(builder, "X", io_dtype, ["batch", "channels", "sequence"])
        _input(builder, "W", io_dtype, ["channels", 1, kernel_size])
        _input(builder, "bias", io_dtype, ["channels"])
        _input(builder, "past_state", io_dtype, ["batch", "channels", kernel_size - 1])

        builder.make_node("Concat", ["past_state", "X"], ["padded"], attributes={"axis": 2})
        builder.make_node("Shape", ["X"], ["shape_x"])
        _constant(builder, "idx2", 2)
        builder.make_node("Gather", ["shape_x", "idx2"], ["s_scalar"], attributes={"axis": 0})
        _constant(builder, "idx0", [0])
        builder.make_node("Unsqueeze", ["s_scalar", "idx0"], ["s_1d"])
        _constant(builder, "axes2", [2])
        _constant(builder, "large_end", [_ONNX_LARGE_SLICE_END])
        _constant(builder, "one_1d", [1])
        builder.make_node("Shape", ["bias"], ["shape_bias"])
        builder.make_node("Concat", ["one_1d", "shape_bias", "one_1d"], ["w_shp"], attributes={"axis": 0})
        builder.make_node("Reshape", ["bias", "w_shp"], ["bias_r"])

        contributions = []
        for index in range(kernel_size):
            _constant(builder, f"starts_{index}", [index])
            if index == 0:
                end = "s_1d"
            else:
                _constant(builder, f"k_offset_{index}", [index])
                end = f"ends_{index}"
                builder.make_node("Add", ["s_1d", f"k_offset_{index}"], [end])
            builder.make_node("Slice", ["padded", f"starts_{index}", end, "axes2"], [f"slice_{index}"])
            _constant(builder, f"ws_{index}", [index])
            _constant(builder, f"we_{index}", [index + 1])
            builder.make_node("Slice", ["W", f"ws_{index}", f"we_{index}", "axes2"], [f"wk_sl_{index}"])
            builder.make_node("Reshape", [f"wk_sl_{index}", "w_shp"], [f"wk_r_{index}"])
            contribution = f"contrib_{index}"
            builder.make_node("Mul", [f"wk_r_{index}", f"slice_{index}"], [contribution])
            contributions.append(contribution)

        total = contributions[0]
        for index, contribution in enumerate(contributions[1:], 1):
            output = f"sum_{index}"
            builder.make_node("Add", [total, contribution], [output])
            total = output

        builder.make_node("Add", [total, "bias_r"], ["pre_silu"])
        builder.make_node("Sigmoid", ["pre_silu"], ["sig"])
        builder.make_node("Mul", ["pre_silu", "sig"], ["Y"])
        builder.make_node("Slice", ["padded", "s_1d", "large_end", "axes2"], ["present_state"])
        builder.make_output("Y")
        builder.make_output("present_state")

    @classmethod
    def _make_causal_conv_local_function(cls, kernel_size: int, io_dtype: int) -> FunctionProto:
        root = GraphBuilder("local_functions", schema_lookup=None)
        function = root.make_local_function("CausalConvWithState", "com.microsoft")
        cls._build_causal_conv_local_function(function, kernel_size, io_dtype)
        return normalize_function_opsets(root.to_model(ir_version=10).functions[0])

    def _register_causal_conv_local_function(self, kernel_size: int) -> None:
        key = ("com.microsoft", "CausalConvWithState", "")
        if self._ort_version() < (1, 26) and key not in self.local_functions:
            self.local_functions[key] = self._make_causal_conv_local_function(kernel_size, self.io_dtype)

    @classmethod
    def make_linear_attention_local_function(
        cls, q_num_heads: int, kv_num_heads: int, key_head_size: int, value_head_size: int, io_dtype: int
    ) -> FunctionProto:
        """Build a standard-operator GatedDeltaNet fallback function."""
        nq = q_num_heads
        nkv = kv_num_heads
        nq_per_kv = nq // nkv

        nodes = [
            onnx_helper.make_node("Transpose", ["K"], ["K_T"], perm=[1, 0, 2]),
            onnx_helper.make_node("Transpose", ["V"], ["V_T"], perm=[1, 0, 2]),
            onnx_helper.make_node("Transpose", ["Q"], ["Q_T"], perm=[1, 0, 2]),
            onnx_helper.make_node("Transpose", ["decay"], ["g_T"], perm=[1, 0, 2]),
            onnx_helper.make_node("Transpose", ["beta"], ["beta_T"], perm=[1, 0, 2]),
            onnx_helper.make_node("Shape", ["K"], ["K_shape"]),
            _constant_node("seq_dim_idx", 1),
            onnx_helper.make_node("Gather", ["K_shape", "seq_dim_idx"], ["S_scalar"], axis=0),
            _constant_node("cond_true_init", True, dtype=np.bool_),
        ]

        body_nodes = [
            onnx_helper.make_node("Gather", ["K_T", "iter_count"], ["k_t_flat"], axis=0),
            onnx_helper.make_node("Gather", ["V_T", "iter_count"], ["v_t_flat"], axis=0),
            onnx_helper.make_node("Gather", ["Q_T", "iter_count"], ["q_t_flat"], axis=0),
            onnx_helper.make_node("Gather", ["g_T", "iter_count"], ["g_t_flat"], axis=0),
            onnx_helper.make_node("Gather", ["beta_T", "iter_count"], ["beta_t_flat"], axis=0),
            _constant_node("lb_k_shp", [0, nkv, key_head_size]),
            onnx_helper.make_node("Reshape", ["k_t_flat", "lb_k_shp"], ["k_t"]),
            _constant_node("lb_v_shp", [0, nkv, value_head_size]),
            onnx_helper.make_node("Reshape", ["v_t_flat", "lb_v_shp"], ["v_t"]),
            _constant_node("lb_q_shp", [0, nkv, nq_per_kv, key_head_size]),
            onnx_helper.make_node("Reshape", ["q_t_flat", "lb_q_shp"], ["q_t"]),
            _constant_node("lb_g_shp", [0, nkv, 1, 1]),
            onnx_helper.make_node("Reshape", ["g_t_flat", "lb_g_shp"], ["g_t"]),
            _constant_node("lb_beta_shp", [0, nkv, 1, 1]),
            onnx_helper.make_node("Reshape", ["beta_t_flat", "lb_beta_shp"], ["beta_t"]),
            _constant_node("lb_k_unsq_ax", [2]),
            onnx_helper.make_node("Unsqueeze", ["k_t", "lb_k_unsq_ax"], ["k_unsq"]),
            onnx_helper.make_node("MatMul", ["k_unsq", "loop_state"], ["kS_4d"]),
            _constant_node("lb_sq2_ax", [2]),
            onnx_helper.make_node("Squeeze", ["kS_4d", "lb_sq2_ax"], ["kS"]),
            onnx_helper.make_node("Sub", ["v_t", "kS"], ["v_prime"]),
            _constant_node("lb_k_outer_ax", [3]),
            onnx_helper.make_node("Unsqueeze", ["k_t", "lb_k_outer_ax"], ["k_outer"]),
            _constant_node("lb_vp_outer_ax", [2]),
            onnx_helper.make_node("Unsqueeze", ["v_prime", "lb_vp_outer_ax"], ["vp_outer"]),
            onnx_helper.make_node("Mul", ["k_outer", "vp_outer"], ["outer_prod"]),
            onnx_helper.make_node("Mul", ["g_t", "loop_state"], ["gS"]),
            onnx_helper.make_node("Mul", ["beta_t", "outer_prod"], ["beta_outer"]),
            onnx_helper.make_node("Add", ["gS", "beta_outer"], ["state_new"]),
            onnx_helper.make_node("MatMul", ["q_t", "state_new"], ["y_t_4d"]),
            _constant_node("lb_y_shp", [0, nq, value_head_size]),
            onnx_helper.make_node("Reshape", ["y_t_4d", "lb_y_shp"], ["y_t"]),
            _constant_node("cond_out", True, dtype=np.bool_),
        ]
        body = onnx_helper.make_graph(
            body_nodes,
            "LinearAttention_loop_body",
            [
                onnx_helper.make_tensor_value_info("iter_count", TensorProto.INT64, []),
                onnx_helper.make_tensor_value_info("cond_in", TensorProto.BOOL, []),
                onnx_helper.make_tensor_value_info("loop_state", io_dtype, ["batch", nkv, key_head_size, value_head_size]),
            ],
            [
                onnx_helper.make_tensor_value_info("cond_out", TensorProto.BOOL, []),
                onnx_helper.make_tensor_value_info("state_new", io_dtype, ["batch", nkv, key_head_size, value_head_size]),
                onnx_helper.make_tensor_value_info("y_t", io_dtype, ["batch", nq, value_head_size]),
            ],
        )
        nodes.extend(
            [
                onnx_helper.make_node("Loop", ["S_scalar", "cond_true_init", "past_state"], ["state_final", "scan_y"], body=body),
                onnx_helper.make_node("Transpose", ["scan_y"], ["scan_y_t"], perm=[1, 0, 2, 3]),
                _constant_node("y_out_shp", [0, 0, nq * value_head_size]),
                onnx_helper.make_node("Reshape", ["scan_y_t", "y_out_shp"], ["output"]),
                onnx_helper.make_node("Identity", ["state_final"], ["present_state"]),
            ]
        )
        return onnx_helper.make_function(
            "com.microsoft",
            "LinearAttention",
            ["Q", "K", "V", "past_state", "decay", "beta"],
            ["output", "present_state"],
            nodes,
            [onnx_helper.make_opsetid("", 21)],
        )

    def register_linear_attention_local_function(
        self, q_num_heads: int, kv_num_heads: int, key_head_size: int, value_head_size: int
    ) -> None:
        key = ("com.microsoft", "LinearAttention", "")
        if self._ort_version() < (1, 26) and key not in self.local_functions:
            self.local_functions[key] = self.make_linear_attention_local_function(
                q_num_heads, kv_num_heads, key_head_size, value_head_size, self.io_dtype
            )
