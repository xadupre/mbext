# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Weight-only quantization implemented directly on onnx-light protos."""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np
import onnx_light.onnx.helper as onnx_helper
import onnx_light.onnx.numpy_helper as numpy_helper
from onnx_light.onnx import AttributeProto, GraphProto, ModelProto, TensorProto


def _default_quantize(weight, bits, block_size, is_symmetric, use_qdq):
    from onnxruntime.capi._pybind_state import (
        quantize_matmul_2bits,
        quantize_matmul_4bits,
        quantize_matmul_8bits,
        quantize_qdq_matmul_4bits,
    )

    rows, cols = weight.shape
    if use_qdq:
        if bits != 4:
            raise ValueError("QDQ weight-only quantization only supports 4 bits.")
        packed = np.zeros((rows * cols + 1) // 2, dtype=np.uint8)
        zero_points = np.zeros((cols * math.ceil(rows / block_size) + 1) // 2, dtype=np.uint8)
        scales = np.zeros((math.ceil(rows / block_size), cols), dtype=weight.dtype)
        quantize_qdq_matmul_4bits(packed, weight, scales, zero_points, block_size, cols, rows, is_symmetric)
        return packed, scales, zero_points

    pack = 8 // bits
    blocks = math.ceil(rows / block_size)
    blob_size = math.ceil(block_size / pack)
    padded_rows = blocks * block_size
    if padded_rows != rows:
        weight = np.pad(weight, ((0, padded_rows - rows), (0, 0)))
    packed = np.zeros((cols, blocks, blob_size), dtype=np.uint8)
    zero_points = np.zeros((cols, math.ceil(blocks / pack)), dtype=np.uint8)
    scales = np.zeros((cols, blocks), dtype=weight.dtype)
    functions = {2: quantize_matmul_2bits, 4: quantize_matmul_4bits, 8: quantize_matmul_8bits}
    try:
        quantize = functions[bits]
    except KeyError:
        raise ValueError(f"MatMulNBits only supports 2, 4, or 8 bits, not {bits}.") from None
    quantize(packed, weight, scales, zero_points, block_size, cols, rows, is_symmetric)
    return packed, scales, zero_points


def _quantize_rtn(weight, bits, block_size, symmetric):
    columns = weight.shape[1]
    blocks = weight.shape[0] // block_size
    values = weight.T.reshape(-1, block_size)
    if symmetric:
        qmax = 2**bits - 1
        bound = np.maximum(np.abs(values.min(axis=1, keepdims=True)), np.abs(values.max(axis=1, keepdims=True)))
        scales = np.ones_like(bound)
        mask = bound > 0
        scales[mask] = bound[mask] * 2.0 / qmax
        zero_points = np.full(bound.shape, 1 << (bits - 1), dtype=np.uint8)
    else:
        qmax = 2**bits - 1
        minimum = values.min(axis=1, keepdims=True)
        maximum = values.max(axis=1, keepdims=True)
        scales = np.ones_like(maximum)
        mask = minimum != maximum
        scales[mask] = (maximum[mask] - minimum[mask]) / qmax
        zero_points = np.clip(np.rint(-minimum / scales), 0, qmax).astype(np.uint8)
    quantized = np.clip(np.rint(values / scales + zero_points), 0, 2**bits - 1).astype(np.uint8)
    scales = scales.reshape(columns, blocks)
    if symmetric:
        return quantized, scales, None
    return quantized, scales, zero_points.reshape(columns, blocks)


def _quantize_k(weight, bits, block_size):
    columns = weight.shape[1]
    blocks = weight.shape[0] // block_size
    values = weight.T.reshape(-1, block_size).astype(np.float32)
    qmax = 2**bits - 1
    sum_x2 = np.sum(values**2, axis=1, keepdims=True)
    weights = np.sqrt(sum_x2 / block_size) + np.abs(values)
    minimum = values.min(axis=1, keepdims=True)
    maximum = values.max(axis=1, keepdims=True)
    sum_w = weights.sum(axis=1, keepdims=True)
    sum_x = (weights * values).sum(axis=1, keepdims=True)
    inverse_scale = np.ones_like(maximum)
    mask = minimum != maximum
    inverse_scale[mask] = qmax / (maximum[mask] - minimum[mask])
    scales = 1.0 / inverse_scale
    quantized = np.clip(np.rint(inverse_scale * (values - minimum)), 0, qmax)
    error = scales * quantized + minimum - values
    best_error = (weights * error**2).sum(axis=1, keepdims=True)

    for step in range(20):
        factor = -1.0 + 0.1 * step + qmax
        candidate_inverse = np.ones_like(maximum)
        candidate_inverse[mask] = factor / (maximum[mask] - minimum[mask])
        candidate = np.clip(np.rint(candidate_inverse * (values - minimum)), 0, qmax)
        weighted_candidate = weights * candidate
        sum_l = weighted_candidate.sum(axis=1, keepdims=True)
        sum_l2 = (weighted_candidate * candidate).sum(axis=1, keepdims=True)
        sum_xl = (weighted_candidate * values).sum(axis=1, keepdims=True)
        denominator = sum_w * sum_l2 - sum_l**2
        valid = denominator != 0
        candidate_scale = scales.copy()
        candidate_minimum = minimum.copy()
        candidate_scale[valid] = ((sum_w * sum_xl - sum_x * sum_l) / denominator)[valid]
        candidate_minimum[valid] = ((sum_l2 * sum_x - sum_l * sum_xl) / denominator)[valid]
        error = candidate_scale * candidate + candidate_minimum - values
        candidate_error = (weights * error**2).sum(axis=1, keepdims=True)
        better = candidate_error[:, 0] < best_error[:, 0]
        quantized[better] = candidate[better]
        best_error[better] = candidate_error[better]
        scales[better] = candidate_scale[better]
        minimum[better] = candidate_minimum[better]

    zero_points = np.clip(np.rint(-minimum / scales), 0, qmax).astype(np.uint8)
    return (quantized.astype(np.uint8), scales.reshape(columns, blocks), zero_points.reshape(columns, blocks))


def _quantize_ternary(weight, bits, block_size):
    if bits != 2:
        raise ValueError(f"Ternary quantization requires 2 bits, not {bits}.")
    columns = weight.shape[1]
    blocks = weight.shape[0] // block_size
    values = weight.T.reshape(-1, block_size).astype(np.float32)
    scales = np.max(np.abs(values), axis=1, keepdims=True)
    safe_scales = np.where(scales == 0, 1.0, scales)
    levels = np.rint(values / safe_scales)
    reconstructed = levels * scales
    if not np.allclose(values, reconstructed, rtol=1e-3, atol=1e-6):
        error = float(np.max(np.abs(values - reconstructed)))
        raise ValueError(
            "Ternary quantization requires every group to contain only " f"{{-scale, 0, +scale}} values; maximum error is {error}."
        )
    quantized = (levels + 1).astype(np.uint8)
    zero_points = np.ones((columns, blocks), dtype=np.uint8)
    return quantized, scales.reshape(columns, blocks), zero_points


def _pack_groupwise(quantized, bits, columns, blocks, block_size):
    if bits == 8:
        packed = quantized
    elif bits == 4:
        packed = quantized[:, ::2] | (quantized[:, 1::2] << 4)
    elif bits == 2:
        packed = quantized[:, ::4] | (quantized[:, 1::4] << 2) | (quantized[:, 2::4] << 4) | (quantized[:, 3::4] << 6)
    else:
        raise ValueError(f"MatMulNBits only supports 2, 4, or 8 bits, not {bits}.")
    return packed.reshape(columns, blocks, math.ceil(block_size * bits / 8))


def _pack_zero_points(zero_points, bits, columns, blocks):
    pack = 8 // bits
    flat = zero_points.reshape(columns, blocks)
    padded = np.full((columns, math.ceil(blocks / pack) * pack), 1 << (bits - 1), dtype=np.uint8)
    padded[:, :blocks] = flat
    packed = np.zeros((columns, padded.shape[1] // pack), dtype=np.uint8)
    for index in range(pack):
        packed |= padded[:, index::pack] << (index * bits)
    return packed


def quantize_ternary_groupwise(weight, block_size):
    """Pack a ``[K, N]`` ternary matrix for 2-bit ``MatMulNBits``."""
    if weight.ndim != 2:
        raise ValueError(f"Ternary quantization expects a matrix, not shape {weight.shape}.")
    rows, columns = weight.shape
    blocks = math.ceil(rows / block_size)
    padded_rows = blocks * block_size
    padded = np.pad(weight, ((0, padded_rows - rows), (0, 0))) if padded_rows != rows else weight
    quantized, scales, zero_points = _quantize_ternary(padded, 2, block_size)
    packed = _pack_groupwise(quantized, 2, columns, blocks, block_size)
    packed_zero_points = _pack_zero_points(zero_points, 2, columns, blocks)
    return packed, scales, packed_zero_points


def _make_raw_tensor(name, data_type, dims, array):
    return onnx_helper.make_tensor(name, data_type, dims, np.ascontiguousarray(array).tobytes(), raw=True)


def _effective_config(source_name, algorithm, bits, block_size, is_symmetric, use_qdq):
    return (source_name, algorithm, bits, block_size, is_symmetric, use_qdq)


def _config_suffix(config):
    _, algorithm, bits, block_size, is_symmetric, use_qdq = config
    quant_format = "qdq" if use_qdq else "qop"
    symmetry = "sym" if is_symmetric else "asym"
    return f"{algorithm}_{quant_format}_{symmetry}_Q{bits}G{block_size}"


def _quantizable_node(node, initializers, nodes_to_exclude, op_types_to_quantize):
    return (
        node.op_type == "MatMul"
        and node.op_type in op_types_to_quantize
        and node.name not in nodes_to_exclude
        and len(node.input) >= 2
        and node.input[1] in initializers
    )


def _quantize_graph(
    graph: GraphProto,
    *,
    bits: int,
    block_size: int,
    is_symmetric: bool,
    accuracy_level: int,
    nodes_to_exclude: set[str],
    op_types_to_quantize: set[str],
    use_qdq: bool,
    algorithm: str,
    customized_weight_config: dict[str, dict[str, int]],
) -> None:
    initializers = {tensor.name: tensor for tensor in graph.initializer}
    new_initializers = []
    new_nodes = []
    quantized_cache = {}
    claimed_names = {name: ("original", name) for name in initializers}
    primary_configs = {}

    for node in graph.node:
        if not _quantizable_node(node, initializers, nodes_to_exclude, op_types_to_quantize):
            continue
        source_name = node.input[1]
        node_bits = int(customized_weight_config.get(node.name, {}).get("bits", bits))
        config = _effective_config(source_name, algorithm, node_bits, block_size, is_symmetric, use_qdq)
        primary_configs.setdefault(source_name, config)
        if node.name == "/lm_head/MatMul":
            primary_configs[source_name] = config

    def claim_name(base_name, config):
        if primary_configs[config[0]] == config:
            candidate = base_name
        else:
            candidate = f"{base_name}_{_config_suffix(config)}"
        owner = claimed_names.get(candidate)
        if owner is None or owner == config:
            claimed_names[candidate] = config
            return candidate
        index = 2
        while True:
            unique = f"{candidate}_{index}"
            owner = claimed_names.get(unique)
            if owner is None or owner == config:
                claimed_names[unique] = config
                return unique
            index += 1

    for node in graph.node:
        for attribute in node.attribute:
            if attribute.type == AttributeProto.GRAPH:
                _quantize_graph(
                    attribute.g,
                    bits=bits,
                    block_size=block_size,
                    is_symmetric=is_symmetric,
                    accuracy_level=accuracy_level,
                    nodes_to_exclude=nodes_to_exclude,
                    op_types_to_quantize=op_types_to_quantize,
                    use_qdq=use_qdq,
                    algorithm=algorithm,
                    customized_weight_config=customized_weight_config,
                )
            elif attribute.type == AttributeProto.GRAPHS:
                for subgraph in attribute.graphs:
                    _quantize_graph(
                        subgraph,
                        bits=bits,
                        block_size=block_size,
                        is_symmetric=is_symmetric,
                        accuracy_level=accuracy_level,
                        nodes_to_exclude=nodes_to_exclude,
                        op_types_to_quantize=op_types_to_quantize,
                        use_qdq=use_qdq,
                        algorithm=algorithm,
                        customized_weight_config=customized_weight_config,
                    )

        if not _quantizable_node(node, initializers, nodes_to_exclude, op_types_to_quantize):
            new_nodes.append(node)
            continue

        source_name = node.input[1]
        node_config = customized_weight_config.get(node.name, {})
        node_bits = int(node_config.get("bits", bits))
        config = _effective_config(source_name, algorithm, node_bits, block_size, is_symmetric, use_qdq)
        cached = quantized_cache.get(config)

        if cached is None:
            weight_proto = initializers[source_name]
            weight = np.asarray(numpy_helper.to_array(weight_proto))
            if weight.ndim != 2:
                new_nodes.append(node)
                continue
            scales_dtype = weight.dtype
            if weight.dtype.name == "bfloat16":
                scales_dtype = np.dtype(ml_dtypes.bfloat16)
                weight = weight.astype(np.float32)

            rows, columns = weight.shape
            blocks = math.ceil(rows / block_size)
            padded_rows = blocks * block_size
            padded = np.pad(weight, ((0, padded_rows - rows), (0, 0))) if padded_rows != rows else weight

            if use_qdq:
                packed, scales, zero_points = _default_quantize(weight, node_bits, block_size, is_symmetric, True)
                qtype = TensorProto.INT4 if is_symmetric else TensorProto.UINT4
                quant_name = claim_name(f"{source_name}_DQ_Q{node_bits}", config)
                scales_name = claim_name(f"{source_name}_DQ_scales", config)
                quant_tensor = _make_raw_tensor(quant_name, qtype, weight.shape, packed)
                scales_tensor = numpy_helper.from_array(scales.astype(scales_dtype), name=scales_name)
                zero_name = None
                if not is_symmetric:
                    zero_name = claim_name(f"{source_name}_DQ_zero_points", config)
                    new_initializers.append(_make_raw_tensor(zero_name, qtype, scales.shape, zero_points))
                new_initializers.extend([quant_tensor, scales_tensor])
                cached = {"rows": rows, "columns": columns, "quant_name": quant_name, "scales_name": scales_name, "zero_name": zero_name}
                quantized_cache[config] = cached
            else:
                if algorithm == "k_quant":
                    quantized, scales, zero_points = _quantize_k(padded, node_bits, block_size)
                    packed = _pack_groupwise(quantized, node_bits, columns, blocks, block_size)
                elif algorithm == "rtn":
                    quantized, scales, zero_points = _quantize_rtn(padded, node_bits, block_size, is_symmetric)
                    packed = _pack_groupwise(quantized, node_bits, columns, blocks, block_size)
                elif algorithm == "ternary":
                    quantized, scales, zero_points = _quantize_ternary(padded, node_bits, block_size)
                    packed = _pack_groupwise(quantized, node_bits, columns, blocks, block_size)
                else:
                    packed, scales, zero_points = _default_quantize(weight, node_bits, block_size, is_symmetric, False)

                grouped_name = algorithm in {"rtn", "k_quant", "ternary"}
                quant_base = f"{source_name}_Q{node_bits}G{block_size}" if grouped_name else f"{source_name}_Q{node_bits}"
                scales_base = f"{source_name}_scale" if grouped_name else f"{source_name}_scales"
                quant_name = claim_name(quant_base, config)
                scales_name = claim_name(scales_base, config)
                zero_name = None
                new_initializers.extend(
                    [
                        numpy_helper.from_array(np.asarray(packed, dtype=np.uint8), name=quant_name),
                        numpy_helper.from_array(scales.astype(scales_dtype), name=scales_name),
                    ]
                )
                if zero_points is not None and (not is_symmetric or algorithm in {"k_quant", "ternary"}):
                    zero_base = f"{source_name}_zp" if grouped_name else f"{source_name}_zero_points"
                    zero_name = claim_name(zero_base, config)
                    packed_zero_points = _pack_zero_points(zero_points, node_bits, columns, blocks) if grouped_name else zero_points
                    new_initializers.append(numpy_helper.from_array(np.asarray(packed_zero_points, dtype=np.uint8), name=zero_name))
                cached = {"rows": rows, "columns": columns, "quant_name": quant_name, "scales_name": scales_name, "zero_name": zero_name}
                quantized_cache[config] = cached

        rows = cached["rows"]
        columns = cached["columns"]
        quant_name = cached["quant_name"]
        scales_name = cached["scales_name"]
        zero_name = cached["zero_name"]

        if use_qdq:
            dq_inputs = [quant_name, scales_name]
            if zero_name is not None:
                dq_inputs.append(zero_name)
            dq_output = f"{node.output[0]}_dequantized_weight"
            new_nodes.append(
                onnx_helper.make_node(
                    "DequantizeLinear", dq_inputs, [dq_output], name=f"{node.name}_DQ_Q{node_bits}", axis=0, block_size=block_size
                )
            )
            new_nodes.append(
                onnx_helper.make_node("MatMul", [node.input[0], dq_output], list(node.output), name=f"{node.name}_matmul_Q{node_bits}")
            )
            continue

        inputs = [node.input[0], quant_name, scales_name]
        if zero_name is not None:
            inputs.append(zero_name)
        attributes = {"K": rows, "N": columns, "bits": node_bits, "block_size": block_size}
        if accuracy_level:
            attributes["accuracy_level"] = accuracy_level
        new_nodes.append(
            onnx_helper.make_node(
                "MatMulNBits",
                inputs,
                list(node.output),
                name=f"{node.name}_Q{node_bits}" if node.name else f"_Q{node_bits}",
                domain="com.microsoft",
                **attributes,
            )
        )

    graph.node.clear()
    graph.node.extend(new_nodes)
    graph.initializer.extend(new_initializers)
    used = {name for node in graph.node for name in node.input if name}
    kept = [tensor for tensor in graph.initializer if tensor.name in used]
    graph.initializer.clear()
    graph.initializer.extend(kept)


def quantize_matmul_nbits(
    model: ModelProto,
    *,
    bits: int,
    block_size: int,
    is_symmetric: bool,
    accuracy_level: int,
    nodes_to_exclude,
    op_types_to_quantize,
    use_qdq: bool,
    algorithm_config: dict | None,
) -> ModelProto:
    """Quantize constant MatMul weights without importing onnx or onnx-ir."""
    algorithm_config = algorithm_config or {}
    algorithm = algorithm_config.get("algorithm", "default")
    customized_weight_config = algorithm_config.get("customized_weight_config", {})
    _quantize_graph(
        model.graph,
        bits=bits,
        block_size=block_size,
        is_symmetric=is_symmetric,
        accuracy_level=accuracy_level,
        nodes_to_exclude=set(nodes_to_exclude),
        op_types_to_quantize=set(op_types_to_quantize),
        use_qdq=use_qdq,
        algorithm=algorithm,
        customized_weight_config=customized_weight_config,
    )
    if not any(opset.domain == "com.microsoft" for opset in model.opset_import):
        model.opset_import.append(onnx_helper.make_opsetid("com.microsoft", 1))
    return model
