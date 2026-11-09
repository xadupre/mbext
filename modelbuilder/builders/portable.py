# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Standard ONNX building blocks for the provider-independent export."""

import numpy as np
from onnx_light.onnx import TensorProto, defs, numpy_helper


def validate_portable_model(model):
    """Check nested graphs as well as top-level nodes without reading weights."""
    standard_domains = {"", "ai.onnx"}
    if model.functions or any(item.domain not in standard_domains for item in model.opset_import):
        raise NotImplementedError("--ep any cannot export custom functions or nonstandard opset imports.")
    version = next(item.version for item in model.opset_import if item.domain in standard_domains)
    checked = set()
    graphs = [model.graph]
    while graphs:
        for node in graphs.pop().node:
            if node.domain not in standard_domains:
                raise NotImplementedError(f"--ep any requires standard ONNX operators; {node.domain}::{node.op_type} at {node.name!r}.")
            if node.op_type not in checked:
                defs.get_schema(node.op_type, version)
                checked.add(node.op_type)
            for attribute in node.attribute:
                if attribute.HasField("g"):
                    graphs.append(attribute.g)
                graphs.extend(attribute.graphs)


def make_portable_node(model, op_type, inputs, outputs, name, domain, attributes):
    """Lower known contrib operations; reject everything else explicitly."""
    if op_type in {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "SkipLayerNormalization"} and domain in {
        "",
        None,
        "com.microsoft",
    }:
        if any(outputs[1:3]):
            raise NotImplementedError(f"--ep any does not support normalization statistics: {name}")
        simple = "Simplified" in op_type
        norm_inputs = list(inputs)
        if op_type.startswith("Skip"):
            summed = outputs[3] if len(outputs) > 3 and outputs[3] else f"{name}/sum"
            bias_index = 3 if simple else 4
            has_bias = len(inputs) > bias_index and bool(inputs[bias_index])
            addition = f"{name}/unbiased_sum" if has_bias else summed
            model.make_node("Add", inputs[:2], [addition], name=f"{name}/Add")
            if has_bias:
                model.make_node("Add", [addition, inputs[bias_index]], [summed], name=f"{name}/Bias")
            norm_inputs = [summed, *inputs[2 : 3 if simple else 4]]
        model.make_node(
            "RMSNormalization" if simple else "LayerNormalization",
            norm_inputs,
            outputs[:1],
            name=f"{name}/Normalization",
            axis=attributes.get("axis", -1),
            epsilon=attributes.get("epsilon", 1e-5),
            stash_type=attributes.get("stash_type", 1),
        )
        return True
    if domain in {"", "ai.onnx", None}:
        return False
    if domain == "com.microsoft" and op_type == "RotaryEmbedding":
        model.make_node("RotaryEmbedding", [inputs[0], inputs[2], inputs[3], inputs[1]], outputs, name=name, **attributes)
        return True
    if domain == "com.microsoft" and op_type in {"Gelu", "FastGelu"}:
        if len(inputs) > 1 and inputs[1]:
            biased = f"{name}/biased"
            model.make_node("Add", inputs[:2], [biased], name=f"{name}/Bias")
            inputs = [biased]
        model.make_node("Gelu", inputs, outputs, name=name, approximate="tanh" if op_type == "FastGelu" else "none")
        return True
    if domain == "com.microsoft" and op_type == "QuickGelu":
        scaled = f"{name}/scaled"
        sigmoid = f"{name}/sigmoid"
        alpha = f"{name}/alpha"
        model.make_node(
            "Constant",
            [],
            [alpha],
            name=f"{name}/Alpha",
            value=numpy_helper.from_array(np.asarray(attributes.get("alpha", 1.702), dtype=np.float32)),
        )
        model.make_node("Cast", [alpha], [f"{alpha}/cast"], name=f"{name}/AlphaCast", to=model.io_dtype)
        model.make_node("Mul", [inputs[0], f"{alpha}/cast"], [scaled], name=f"{name}/Scale")
        model.make_node("Sigmoid", [scaled], [sigmoid], name=f"{name}/Sigmoid")
        model.make_node("Mul", [inputs[0], sigmoid], outputs, name=name)
        return True
    raise NotImplementedError(f"--ep any requires standard ONNX operators; {domain}::{op_type} at {name!r} has no portable lowering.")


def make_portable_attention(model, name, **kwargs):
    """Build causal grouped-query attention with compact, growing KV caches."""

    def node(op_type, inputs, suffix, *, output=None, **attributes):
        output = output or f"{name}/{suffix}/output"
        model.make_node(op_type, inputs, [output], name=f"{name}/{suffix}", **attributes)
        return output

    def constant(value, suffix, dtype=np.int64):
        return node("Constant", [], suffix, value=numpy_helper.from_array(np.asarray(value, dtype=dtype)))

    zero = constant(0, "zero")
    one = constant(1, "one")
    two = constant(2, "two")
    axes_2 = constant([2], "axes_2")
    axes_01 = constant([0, 1], "axes_01")
    axes_12 = constant([1, 2], "axes_12")
    q_shape = constant([0, 0, model.num_attn_heads, model.head_size], "q_shape")
    kv_shape = constant([0, 0, model.num_kv_heads, model.head_size], "kv_shape")
    q = node("Reshape", [kwargs["q_path"], q_shape], "q_reshape")
    q = node("Transpose", [q], "q_transpose", perm=[0, 2, 1, 3])
    cache = {}
    for label in ("k", "v"):
        current = node("Reshape", [kwargs[f"{label}_path"], kv_shape], f"{label}_reshape")
        current = node("Transpose", [current], f"{label}_transpose", perm=[0, 2, 1, 3])
        past = kwargs.get(f"past_{label}", "")
        present = kwargs.get(f"present_{label}", "")
        if past:
            current = node("Concat", [past, current], f"{label}_cache", output=present or None, axis=2)
        elif present:
            current = node("Identity", [current], f"{label}_cache", output=present)
        cache[label] = current
    k, v = cache["k"], cache["v"]
    query_length = node("Gather", [node("Shape", [q], "q_size"), two], "q_length", axis=0)
    total_length = node("Gather", [node("Shape", [k], "k_size"), two], "total_length", axis=0)
    past_length = node("Sub", [total_length, query_length], "past_length")
    query_positions = node("Range", [past_length, total_length, one], "query_positions")
    query_positions = node("Unsqueeze", [query_positions, constant([1], "axis_1")], "query_column")
    key_positions = node("Range", [zero, total_length, one], "key_positions")
    allowed = node("LessOrEqual", [key_positions, query_positions], "causal")
    if model.window_size is not None and model.window_size > 0:
        lower = node("Sub", [query_positions, constant(model.window_size, "window")], "window_start")
        local = node("Greater", [key_positions, lower], "local")
        allowed = node("And", [allowed, local], "causal_local")
    allowed = node("Unsqueeze", [allowed, axes_01], "causal_4d")
    if model.input_names.get("attention_mask"):
        padding = node("Cast", [model.input_names["attention_mask"]], "padding_bool", to=TensorProto.BOOL)
        padding = node("Unsqueeze", [padding, axes_12], "padding_4d")
        allowed = node("And", [allowed, padding], "allowed")

    if model.num_attn_heads != model.num_kv_heads:
        repeats = constant([1, 1, model.num_attn_heads // model.num_kv_heads, 1, 1], "kv_repeats")
        shape = constant([0, model.num_attn_heads, -1, model.head_size], "repeated_shape")

        def repeat(value, label):
            expanded = node("Unsqueeze", [value, axes_2], f"{label}_unsqueeze")
            tiled = node("Tile", [expanded, repeats], f"{label}_tile")
            return node("Reshape", [tiled, shape], f"{label}_repeat")

        k, v = repeat(k, "k"), repeat(v, "v")
    k = node("Transpose", [k], "k_transposed", perm=[0, 1, 3, 2])
    q = node("Cast", [q], "q_float", to=TensorProto.FLOAT)
    k = node("Cast", [k], "k_float", to=TensorProto.FLOAT)
    scores = node("MatMul", [q, k], "qk")
    scale = constant(model.attention_attrs["scale"], "scale", np.float32)
    scores = node("Mul", [scores, scale], "scaled")
    softcap = model.attention_attrs["softcap"]
    if softcap:
        cap = constant(softcap, "softcap", np.float32)
        scores = node("Div", [scores, cap], "uncapped")
        scores = node("Tanh", [scores], "tanh")
        scores = node("Mul", [scores, cap], "capped")
    scores = node("Where", [allowed, scores, constant(float("-inf"), "masked_value", np.float32)], "masked")
    sinks = kwargs.get("sinks", "")
    if sinks:
        sink = node("Cast", [sinks], "sinks_float", to=TensorProto.FLOAT)
        sink = node("Unsqueeze", [sink, constant([0, 2, 3], "sinks_axes")], "sinks_4d")
        score_shape = node("Shape", [scores], "score_shape")
        prefix = node("Slice", [score_shape, constant([0], "prefix_start"), constant([3], "prefix_end")], "score_prefix")
        sink_shape = node("Concat", [prefix, constant([1], "sink_one")], "sink_shape", axis=0)
        sink = node("Expand", [sink, sink_shape], "sinks_expanded")
        scores = node("Concat", [scores, sink], "scores_with_sinks", axis=-1)
    probabilities = node("Softmax", [scores], "softmax", axis=-1)
    if sinks:
        probabilities = node(
            "Slice",
            [
                probabilities,
                constant([0], "probability_start"),
                node("Unsqueeze", [total_length, constant([0], "axis_0")], "probability_end"),
                constant([3], "probability_axis"),
            ],
            "without_sinks",
        )
    # A fully padded query must contribute zero rather than propagate NaNs.
    probabilities = node("Where", [allowed, probabilities, constant(0, "zero_float", np.float32)], "valid_probabilities")
    probabilities = node("Cast", [probabilities], "probability_cast", to=model.io_dtype)
    context = node("MatMul", [probabilities, v], "context")
    context = node("Transpose", [context], "context_transpose", perm=[0, 2, 1, 3])
    node("Reshape", [context, constant([0, 0, model.q_size], "output_shape")], "output", output=f"{name}/output_0")
    model.make_value(f"{name}/output_0", model.io_dtype, ["batch_size", "sequence_length", model.q_size])
