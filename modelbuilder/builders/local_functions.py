# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""ONNX local-function fallbacks for ``com.microsoft`` contrib ops.

When ``onnxruntime < 1.26`` is installed the native kernel for some contrib
ops may not be registered.  :class:`LocalFunctionsMixin` provides class
methods that build ONNX local-function bodies using only standard opset-21
primitives, and registers them into the ONNX model whenever the installed ORT
version requires the fallback.

Usage::

    class Model(LocalFunctionsMixin, ...):
        ...
"""

from __future__ import annotations

import numpy as np
import onnx_ir as ir

# Sentinel used as the "end-of-tensor" value for ONNX Slice nodes.  We pick
# 2**62 rather than sys.maxsize (2**63-1) to stay safely within the signed
# INT64 range that ONNX Slice accepts on all platforms.
_ONNX_LARGE_SLICE_END: int = 2**62


class LocalFunctionsMixin:
    """Mixin that adds ONNX local-function helpers to model-builder classes.

    Concrete subclasses are expected to expose ``self.model`` (an
    :class:`onnx_ir.Model`) and ``self.io_dtype`` (an
    :class:`onnx_ir.DataType`) — both of which are set by
    :class:`~modelbuilder.builders.base.Model`.
    """

    # ------------------------------------------------------------------
    # ORT version detection
    # ------------------------------------------------------------------

    @staticmethod
    def _ort_version() -> tuple[int, ...]:
        """Return the installed onnxruntime version as an integer tuple.

        For example ``(1, 24, 4)``.

        Returns ``(99, 99, 0)`` when onnxruntime is not installed **or** when
        the version string cannot be parsed (e.g. a dev build with a
        non-numeric suffix).  Both cases are treated as "current enough" so
        that no local-function fallback is added.
        """
        try:
            import onnxruntime

            return tuple(int(x) for x in onnxruntime.__version__.split(".")[:3])
        except (ImportError, ValueError):
            return (99, 99, 0)

    # ------------------------------------------------------------------
    # CausalConvWithState local function
    # ------------------------------------------------------------------

    @classmethod
    def _make_causal_conv_local_function(cls, K: int, io_dtype: ir.DataType) -> ir.Function:
        """Build an ONNX local function that implements ``com.microsoft:CausalConvWithState``.

        The function body uses only standard ONNX opset-21 primitives so that
        runtimes without the native ``CausalConvWithState`` kernel (e.g. ORT < 1.26)
        can fall back to this definition automatically.

        The convolution kernel width *K* is unrolled at function-build time.
        The channel count *C* is inferred dynamically from the weight shape, so the
        same function definition handles any *C* value.

        Formal inputs
        -------------
        ``X``           – input tensor ``[B, C, S]``
        ``W``           – depthwise-conv weight ``[C, 1, K]``
        ``bias``        – conv bias ``[C]``
        ``past_state``  – causal-conv carry state ``[B, C, K-1]``

        Formal outputs
        --------------
        ``Y``              – SiLU-activated output ``[B, C, S]``
        ``present_state``  – updated carry state ``[B, C, K-1]``
        """

        def mkv(name):
            return ir.Value(name=name)

        # Formal inputs / outputs
        X = mkv("X")
        X.dtype = io_dtype
        W = mkv("W")
        W.dtype = io_dtype
        bias = mkv("bias")
        bias.dtype = io_dtype
        past = mkv("past_state")
        past.dtype = io_dtype
        Y = mkv("Y")
        Y.dtype = io_dtype
        present = mkv("present_state")
        present.dtype = io_dtype

        nodes: list[ir.Node] = []

        def ci(name: str, data) -> ir.Value:
            """Create an INT64 Constant node; return its output Value."""
            t = ir.tensor(np.array(data, dtype=np.int64), name=name)
            v = mkv(name)
            v.dtype = ir.DataType.INT64
            nodes.append(ir.node("Constant", inputs=[], outputs=[v], attributes={"value": t}))
            return v

        # ---- 1. padded = Concat([past_state, X], axis=2) → [B, C, S+K-1] ----
        padded = mkv("padded")
        padded.dtype = io_dtype
        nodes.append(ir.node("Concat", inputs=[past, X], outputs=[padded], attributes={"axis": 2}))

        # ---- 2. S = Shape(X)[2] stored as a 1-D tensor for Slice inputs ----
        shape_x = mkv("shape_x")
        shape_x.dtype = ir.DataType.INT64
        nodes.append(ir.node("Shape", inputs=[X], outputs=[shape_x]))

        idx2 = ci("idx2", 2)  # scalar index 2
        s_scalar = mkv("s_scalar")
        s_scalar.dtype = ir.DataType.INT64
        nodes.append(ir.node("Gather", inputs=[shape_x, idx2], outputs=[s_scalar], attributes={"axis": 0}))

        idx0 = ci("idx0", [0])  # 1-D index [0] for Unsqueeze
        s_1d = mkv("s_1d")
        s_1d.dtype = ir.DataType.INT64
        nodes.append(ir.node("Unsqueeze", inputs=[s_scalar, idx0], outputs=[s_1d]))

        axes2 = ci("axes2", [2])  # slice along axis 2
        large_end = ci("large_end", [_ONNX_LARGE_SLICE_END])  # effectively +∞ for end-of-tensor slice
        one_1d = ci("one_1d", [1])  # constant [1] for shape construction

        # ---- 3. Dynamic reshape shape [1, C, 1] built from bias dimensions ----
        # Shape(bias) → 1-element int64 vector [C]; concat with [1] on each side.
        shape_bias = mkv("shape_bias")
        shape_bias.dtype = ir.DataType.INT64
        nodes.append(ir.node("Shape", inputs=[bias], outputs=[shape_bias]))

        w_shp = mkv("w_shp")  # [1, C, 1] as a shape vector
        w_shp.dtype = ir.DataType.INT64
        nodes.append(ir.node("Concat", inputs=[one_1d, shape_bias, one_1d], outputs=[w_shp], attributes={"axis": 0}))

        # Reshape bias from [C] to [1, C, 1] for broadcasting
        bias_r = mkv("bias_r")
        bias_r.dtype = io_dtype
        nodes.append(ir.node("Reshape", inputs=[bias, w_shp], outputs=[bias_r]))

        # ---- 4. Unrolled depthwise conv: contrib_k = W[:,k] * padded[:,:, k:k+S] ----
        contribs: list[ir.Value] = []
        for k in range(K):
            starts_k = ci(f"starts_{k}", [k])

            # ends_k = s_1d + k  (dynamic)
            if k == 0:
                ends_k = s_1d
            else:
                k_off = ci(f"k_offset_{k}", [k])
                ends_k = mkv(f"ends_{k}")
                ends_k.dtype = ir.DataType.INT64
                nodes.append(ir.node("Add", inputs=[s_1d, k_off], outputs=[ends_k]))

            # slice_k = padded[:, :, k : k+S]  → [B, C, S]
            slice_k = mkv(f"slice_{k}")
            slice_k.dtype = io_dtype
            nodes.append(ir.node("Slice", inputs=[padded, starts_k, ends_k, axes2], outputs=[slice_k]))

            # W_k = W[:, :, k:k+1]  → [C, 1, 1]
            ws = ci(f"ws_{k}", [k])
            we = ci(f"we_{k}", [k + 1])
            wk_sl = mkv(f"wk_sl_{k}")
            wk_sl.dtype = io_dtype
            nodes.append(ir.node("Slice", inputs=[W, ws, we, axes2], outputs=[wk_sl]))

            # Reshape W_k from [C, 1, 1] to [1, C, 1] for broadcast multiplication
            wk_r = mkv(f"wk_r_{k}")
            wk_r.dtype = io_dtype
            nodes.append(ir.node("Reshape", inputs=[wk_sl, w_shp], outputs=[wk_r]))

            contrib = mkv(f"contrib_{k}")
            contrib.dtype = io_dtype
            nodes.append(ir.node("Mul", inputs=[wk_r, slice_k], outputs=[contrib]))
            contribs.append(contrib)

        # Sum all K contributions
        total: ir.Value = contribs[0]
        for i in range(1, K):
            new_total = mkv(f"sum_{i}")
            new_total.dtype = io_dtype
            nodes.append(ir.node("Add", inputs=[total, contribs[i]], outputs=[new_total]))
            total = new_total

        # Add bias (broadcast [1, C, 1] + [B, C, S] → [B, C, S])
        pre_silu = mkv("pre_silu")
        pre_silu.dtype = io_dtype
        nodes.append(ir.node("Add", inputs=[total, bias_r], outputs=[pre_silu]))

        # SiLU activation: Y = x * sigmoid(x)
        sig = mkv("sig")
        sig.dtype = io_dtype
        nodes.append(ir.node("Sigmoid", inputs=[pre_silu], outputs=[sig]))
        nodes.append(ir.node("Mul", inputs=[pre_silu, sig], outputs=[Y]))

        # ---- 5. present_state = padded[:, :, S:] ----
        nodes.append(ir.node("Slice", inputs=[padded, s_1d, large_end, axes2], outputs=[present]))

        body = ir.Graph(
            inputs=(X, W, bias, past), outputs=(Y, present), nodes=nodes, opset_imports={"": 21}, name="CausalConvWithState_body"
        )
        return ir.Function("com.microsoft", "CausalConvWithState", "", graph=body, attributes={})

    def _register_causal_conv_local_function(self, K: int) -> None:
        """Register the ``CausalConvWithState`` local function if ORT < 1.26.

        The function is added to ``self.model.functions`` at most once per
        model (guarded by the function key lookup).

        Args:
            K: Convolution kernel width.  Derived from ``present_conv_shape[-1] + 1``.
        """
        if self._ort_version() < (1, 26):
            func_key = ("com.microsoft", "CausalConvWithState", "")
            if func_key not in self.model.functions:
                func = self._make_causal_conv_local_function(K, self.io_dtype)
                self.model.functions[func_key] = func

    # ------------------------------------------------------------------
    # LinearAttention local function  (GatedDeltaNet recurrence)
    # ------------------------------------------------------------------

    @classmethod
    def _make_linear_attention_local_function(
        cls, q_num_heads: int, kv_num_heads: int, hk: int, hv: int, io_dtype: ir.DataType
    ) -> ir.Function:
        """Build an ONNX local function implementing ``com.microsoft:LinearAttention``.

        Implements the GatedDeltaNet update rule (``update_rule="gated_delta"``)
        using a standard ONNX opset-21 Loop over the sequence dimension.  The
        architecture parameters *q_num_heads*, *kv_num_heads*, *hk*, *hv* are
        embedded as integer constants so the function body can be compiled
        without declaring formal ONNX attributes.

        GatedDeltaNet recurrence (per KV-head group, time step *t*)::

            kS      = k_t @ S_{t-1}                            # [B, nkv, hv]
            v_prime = v_t - kS                                  # [B, nkv, hv]
            S_t     = g_t * S_{t-1} + beta_t * outer(k_t, v_prime)
            y_t     = q_t @ S_t                                 # [B, nq, hv]

        Formal inputs
        -------------
        ``Q``          – ``[B, S, q_num_heads * hk]``
        ``K``          – ``[B, S, kv_num_heads * hk]``
        ``V``          – ``[B, S, kv_num_heads * hv]``
        ``past_state`` – ``[B, kv_num_heads, hk, hv]``
        ``decay``      – ``[B, S, kv_num_heads]``
        ``beta``       – ``[B, S, kv_num_heads]``

        Formal outputs
        --------------
        ``output``        – ``[B, S, q_num_heads * hv]``
        ``present_state`` – ``[B, kv_num_heads, hk, hv]``
        """
        nq = q_num_heads
        nkv = kv_num_heads
        nq_per_kv = nq // nkv

        def mkv(name: str) -> ir.Value:
            return ir.Value(name=name)

        def ci(node_list: list, name: str, data) -> ir.Value:
            """Create an INT64 Constant; return its output Value."""
            t = ir.tensor(np.array(data, dtype=np.int64), name=name)
            v = mkv(name)
            v.dtype = ir.DataType.INT64
            node_list.append(ir.node("Constant", inputs=[], outputs=[v], attributes={"value": t}))
            return v

        # ── Formal inputs of the local function ──────────────────────────
        Q = mkv("Q")
        Q.dtype = io_dtype
        K = mkv("K")
        K.dtype = io_dtype
        V = mkv("V")
        V.dtype = io_dtype
        past_state = mkv("past_state")
        past_state.dtype = io_dtype
        decay = mkv("decay")
        decay.dtype = io_dtype
        beta = mkv("beta")
        beta.dtype = io_dtype

        # ── Formal outputs ────────────────────────────────────────────────
        output = mkv("output")
        output.dtype = io_dtype
        present_state = mkv("present_state")
        present_state.dtype = io_dtype

        outer_nodes: list = []

        # ── Transpose [B, S, C] → [S, B, C] for use as loop scan inputs ──
        K_T = mkv("K_T")
        K_T.dtype = io_dtype
        outer_nodes.append(ir.node("Transpose", inputs=[K], outputs=[K_T], attributes={"perm": [1, 0, 2]}))

        V_T = mkv("V_T")
        V_T.dtype = io_dtype
        outer_nodes.append(ir.node("Transpose", inputs=[V], outputs=[V_T], attributes={"perm": [1, 0, 2]}))

        Q_T = mkv("Q_T")
        Q_T.dtype = io_dtype
        outer_nodes.append(ir.node("Transpose", inputs=[Q], outputs=[Q_T], attributes={"perm": [1, 0, 2]}))

        g_T = mkv("g_T")
        g_T.dtype = io_dtype
        outer_nodes.append(ir.node("Transpose", inputs=[decay], outputs=[g_T], attributes={"perm": [1, 0, 2]}))

        beta_T = mkv("beta_T")
        beta_T.dtype = io_dtype
        outer_nodes.append(ir.node("Transpose", inputs=[beta], outputs=[beta_T], attributes={"perm": [1, 0, 2]}))

        # ── Trip count = S (sequence length) ─────────────────────────────
        K_shape = mkv("K_shape")
        K_shape.dtype = ir.DataType.INT64
        outer_nodes.append(ir.node("Shape", inputs=[K], outputs=[K_shape]))

        seq_idx = ci(outer_nodes, "seq_dim_idx", 1)  # scalar index for dim 1 of K
        S_scalar = mkv("S_scalar")
        S_scalar.dtype = ir.DataType.INT64
        outer_nodes.append(ir.node("Gather", inputs=[K_shape, seq_idx], outputs=[S_scalar], attributes={"axis": 0}))

        cond_true_init = mkv("cond_true_init")
        cond_true_init.dtype = ir.DataType.BOOL
        outer_nodes.append(
            ir.node(
                "Constant",
                inputs=[],
                outputs=[cond_true_init],
                attributes={"value": ir.tensor(np.array(True, dtype=bool), name="cond_init_val")},
            )
        )

        # ════════════════════════════════════════════════════════════════
        # Loop body: one time-step of GatedDeltaNet
        # ════════════════════════════════════════════════════════════════
        loop_nodes: list = []

        # Loop body formal inputs
        iter_count = mkv("iter_count")  # 0-d int64 scalar
        iter_count.dtype = ir.DataType.INT64
        cond_in = mkv("cond_in")  # 0-d bool
        cond_in.dtype = ir.DataType.BOOL
        loop_state = mkv("loop_state")  # [B, nkv, hk, hv]
        loop_state.dtype = io_dtype

        # ── Gather slice at iter_count from transposed K_T/V_T/Q_T/g_T/beta_T ──
        # K_T: [S, B, nkv*hk] → k_t_flat: [B, nkv*hk]
        k_t_flat = mkv("k_t_flat")
        k_t_flat.dtype = io_dtype
        loop_nodes.append(ir.node("Gather", inputs=[K_T, iter_count], outputs=[k_t_flat], attributes={"axis": 0}))

        v_t_flat = mkv("v_t_flat")
        v_t_flat.dtype = io_dtype
        loop_nodes.append(ir.node("Gather", inputs=[V_T, iter_count], outputs=[v_t_flat], attributes={"axis": 0}))

        q_t_flat = mkv("q_t_flat")
        q_t_flat.dtype = io_dtype
        loop_nodes.append(ir.node("Gather", inputs=[Q_T, iter_count], outputs=[q_t_flat], attributes={"axis": 0}))

        g_t_flat = mkv("g_t_flat")
        g_t_flat.dtype = io_dtype
        loop_nodes.append(ir.node("Gather", inputs=[g_T, iter_count], outputs=[g_t_flat], attributes={"axis": 0}))

        beta_t_flat = mkv("beta_t_flat")
        beta_t_flat.dtype = io_dtype
        loop_nodes.append(ir.node("Gather", inputs=[beta_T, iter_count], outputs=[beta_t_flat], attributes={"axis": 0}))

        # ── Reshape slices; use 0 as "copy batch dimension" per ONNX Reshape ──
        # k_t: [B, nkv*hk] → [B, nkv, hk]
        k_shp = ci(loop_nodes, "lb_k_shp", [0, nkv, hk])
        k_t = mkv("k_t")
        k_t.dtype = io_dtype
        loop_nodes.append(ir.node("Reshape", inputs=[k_t_flat, k_shp], outputs=[k_t]))

        # v_t: [B, nkv*hv] → [B, nkv, hv]
        v_shp = ci(loop_nodes, "lb_v_shp", [0, nkv, hv])
        v_t = mkv("v_t")
        v_t.dtype = io_dtype
        loop_nodes.append(ir.node("Reshape", inputs=[v_t_flat, v_shp], outputs=[v_t]))

        # q_t: [B, nq*hk] → [B, nkv, nq_per_kv, hk]  (handles GQA)
        q_shp = ci(loop_nodes, "lb_q_shp", [0, nkv, nq_per_kv, hk])
        q_t = mkv("q_t")
        q_t.dtype = io_dtype
        loop_nodes.append(ir.node("Reshape", inputs=[q_t_flat, q_shp], outputs=[q_t]))

        # g_t: [B, nkv] → [B, nkv, 1, 1]
        g_shp = ci(loop_nodes, "lb_g_shp", [0, nkv, 1, 1])
        g_t = mkv("g_t")
        g_t.dtype = io_dtype
        loop_nodes.append(ir.node("Reshape", inputs=[g_t_flat, g_shp], outputs=[g_t]))

        # beta_t: [B, nkv] → [B, nkv, 1, 1]
        beta_shp = ci(loop_nodes, "lb_beta_shp", [0, nkv, 1, 1])
        beta_t = mkv("beta_t")
        beta_t.dtype = io_dtype
        loop_nodes.append(ir.node("Reshape", inputs=[beta_t_flat, beta_shp], outputs=[beta_t]))

        # ── kS = k_t.unsqueeze(2) @ loop_state  →  [B, nkv, 1, hv] ──────
        k_unsq_ax = ci(loop_nodes, "lb_k_unsq_ax", [2])
        k_unsq = mkv("k_unsq")
        k_unsq.dtype = io_dtype
        loop_nodes.append(ir.node("Unsqueeze", inputs=[k_t, k_unsq_ax], outputs=[k_unsq]))

        kS_4d = mkv("kS_4d")
        kS_4d.dtype = io_dtype
        loop_nodes.append(ir.node("MatMul", inputs=[k_unsq, loop_state], outputs=[kS_4d]))

        # Squeeze axis=2: [B, nkv, 1, hv] → [B, nkv, hv]
        sq2_ax = ci(loop_nodes, "lb_sq2_ax", [2])
        kS = mkv("kS")
        kS.dtype = io_dtype
        loop_nodes.append(ir.node("Squeeze", inputs=[kS_4d, sq2_ax], outputs=[kS]))

        # ── v_prime = v_t - kS  →  [B, nkv, hv] ─────────────────────────
        v_prime = mkv("v_prime")
        v_prime.dtype = io_dtype
        loop_nodes.append(ir.node("Sub", inputs=[v_t, kS], outputs=[v_prime]))

        # ── outer = k_t.unsqueeze(3) * v_prime.unsqueeze(2)  [B,nkv,hk,hv]
        k_outer_ax = ci(loop_nodes, "lb_k_outer_ax", [3])
        k_outer = mkv("k_outer")
        k_outer.dtype = io_dtype
        loop_nodes.append(ir.node("Unsqueeze", inputs=[k_t, k_outer_ax], outputs=[k_outer]))

        vp_outer_ax = ci(loop_nodes, "lb_vp_outer_ax", [2])
        vp_outer = mkv("vp_outer")
        vp_outer.dtype = io_dtype
        loop_nodes.append(ir.node("Unsqueeze", inputs=[v_prime, vp_outer_ax], outputs=[vp_outer]))

        outer_prod = mkv("outer_prod")
        outer_prod.dtype = io_dtype
        loop_nodes.append(ir.node("Mul", inputs=[k_outer, vp_outer], outputs=[outer_prod]))

        # ── state_new = g_t * loop_state + beta_t * outer_prod ───────────
        gS = mkv("gS")
        gS.dtype = io_dtype
        loop_nodes.append(ir.node("Mul", inputs=[g_t, loop_state], outputs=[gS]))

        beta_outer = mkv("beta_outer")
        beta_outer.dtype = io_dtype
        loop_nodes.append(ir.node("Mul", inputs=[beta_t, outer_prod], outputs=[beta_outer]))

        state_new = mkv("state_new")
        state_new.dtype = io_dtype
        loop_nodes.append(ir.node("Add", inputs=[gS, beta_outer], outputs=[state_new]))

        # ── y_t = q_t @ state_new  →  [B, nkv, nq_per_kv, hv] ──────────
        # Then reshape to [B, nq, hv].
        y_t_4d = mkv("y_t_4d")
        y_t_4d.dtype = io_dtype
        loop_nodes.append(ir.node("MatMul", inputs=[q_t, state_new], outputs=[y_t_4d]))

        y_shp = ci(loop_nodes, "lb_y_shp", [0, nq, hv])
        y_t = mkv("y_t")
        y_t.dtype = io_dtype
        loop_nodes.append(ir.node("Reshape", inputs=[y_t_4d, y_shp], outputs=[y_t]))

        # ── cond_out = True (continue looping) ───────────────────────────
        cond_out = mkv("cond_out")
        cond_out.dtype = ir.DataType.BOOL
        loop_nodes.append(
            ir.node(
                "Constant", inputs=[], outputs=[cond_out], attributes={"value": ir.tensor(np.array(True, dtype=bool), name="cond_out_val")}
            )
        )

        # ── Assemble loop body graph ──────────────────────────────────────
        # Body inputs:  (iter_count, cond_in, loop_state)
        # Body outputs: (cond_out, state_new, y_t)   ← scan output: y_t
        loop_body = ir.Graph(
            inputs=(iter_count, cond_in, loop_state),
            outputs=(cond_out, state_new, y_t),
            nodes=loop_nodes,
            opset_imports={"": 21},
            name="LinearAttention_loop_body",
        )

        # ════════════════════════════════════════════════════════════════
        # Loop node in the outer function body
        # ════════════════════════════════════════════════════════════════
        # Inputs:  trip_count, cond_init, loop-carried state
        # Outputs: loop-carried state_final, scan_output y_accumulated [S,B,nq,hv]
        state_final = mkv("state_final")
        state_final.dtype = io_dtype
        scan_y = mkv("scan_y")  # [S, B, nq, hv]
        scan_y.dtype = io_dtype
        outer_nodes.append(
            ir.node("Loop", inputs=[S_scalar, cond_true_init, past_state], outputs=[state_final, scan_y], attributes={"body": loop_body})
        )

        # ── Transpose scan_y: [S, B, nq, hv] → [B, S, nq, hv] ──────────
        scan_y_t = mkv("scan_y_t")
        scan_y_t.dtype = io_dtype
        outer_nodes.append(ir.node("Transpose", inputs=[scan_y], outputs=[scan_y_t], attributes={"perm": [1, 0, 2, 3]}))

        # ── Reshape [B, S, nq, hv] → [B, S, nq*hv] ──────────────────────
        # Shape [0, 0, nq*hv]: 0 copies B and S from the input dimensions.
        y_out_shp = ci(outer_nodes, "y_out_shp", [0, 0, nq * hv])
        outer_nodes.append(ir.node("Reshape", inputs=[scan_y_t, y_out_shp], outputs=[output]))

        # ── present_state = state_final ──────────────────────────────────
        outer_nodes.append(ir.node("Identity", inputs=[state_final], outputs=[present_state]))

        func_body = ir.Graph(
            inputs=(Q, K, V, past_state, decay, beta),
            outputs=(output, present_state),
            nodes=outer_nodes,
            opset_imports={"": 21},
            name="LinearAttention_func_body",
        )
        return ir.Function("com.microsoft", "LinearAttention", "", graph=func_body, attributes={})

    def _register_linear_attention_local_function(self, q_num_heads: int, kv_num_heads: int, hk: int, hv: int) -> None:
        """Register the ``LinearAttention`` local function if ORT < 1.26.

        Embeds a GatedDeltaNet fallback body (ONNX Loop) into the model so
        that runtimes without the native ``LinearAttention`` kernel can
        execute hybrid Qwen3.5 / Nemotron-H models.

        The function is added to ``self.model.functions`` at most once per
        model (guarded by the function key lookup).

        Args:
            q_num_heads: Number of query attention heads.
            kv_num_heads: Number of key/value attention heads.
            hk: Per-head key dimension.
            hv: Per-head value dimension.
        """
        if self._ort_version() < (1, 26):
            func_key = ("com.microsoft", "LinearAttention", "")
            if func_key not in self.model.functions:
                func = self._make_linear_attention_local_function(q_num_heads, kv_num_heads, hk, hv, self.io_dtype)
                self.model.functions[func_key] = func
