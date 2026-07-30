# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Utilities to compute distribution statistics for the weight tensors that are
quantized when exporting a model.

The statistics are meant to help understand how each weight tensor is
distributed (min, max, mean, median, quantiles) and how far the distribution is
from a normal distribution (Kolmogorov-Smirnov distance to a fitted normal).
They are written to a separate file next to the ONNX model.
"""

import json

import numpy as np
import onnx_ir as ir
from scipy import stats

# Quantiles reported for every weight tensor.
DEFAULT_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)

# Upper bound on the number of samples used to compute the distance to a normal
# distribution. Large weight tensors are subsampled to keep the computation fast.
_MAX_NORMAL_SAMPLES = 100_000


def _tensor_statistics(name: str, array: np.ndarray, quantiles=DEFAULT_QUANTILES) -> dict:
    """Compute distribution statistics for a single weight tensor."""
    shape = list(array.shape)
    values = array.astype(np.float64, copy=False).ravel()

    mean = float(np.mean(values))
    std = float(np.std(values))

    quantile_values = np.quantile(values, quantiles)
    quantiles_dict = {str(q): float(v) for q, v in zip(quantiles, quantile_values)}

    # Distance to a normal distribution fitted on the tensor (mean/std): the
    # Kolmogorov-Smirnov statistic lies in [0, 1], 0 meaning a perfect fit.
    if std > 0:
        sample = values
        if sample.size > _MAX_NORMAL_SAMPLES:
            rng = np.random.default_rng(0)
            sample = rng.choice(sample, size=_MAX_NORMAL_SAMPLES, replace=False)
        normal_distance = float(stats.kstest(sample, stats.norm(loc=mean, scale=std).cdf).statistic)
    else:
        # A constant tensor is degenerate; report the maximal distance.
        normal_distance = 1.0

    return {
        "name": name,
        "shape": shape,
        "size": int(values.size),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": mean,
        "median": float(np.median(values)),
        "std": std,
        "quantiles": quantiles_dict,
        "normal_distance": normal_distance,
    }


def compute_weight_statistics(model: ir.Model, op_types=("MatMul",), nodes_to_exclude=()) -> list[dict]:
    """Compute distribution statistics for every quantized weight tensor.

    Only the initializer inputs of the nodes whose ``op_type`` is in ``op_types``
    are considered, matching the tensors that the int2/int4/int8/int16 quantizer
    targets.
    Nodes listed in ``nodes_to_exclude`` are skipped.
    """
    op_types = set(op_types)
    nodes_to_exclude = set(nodes_to_exclude)
    initializers = model.graph.initializers

    stats_list = []
    seen = set()
    for node in model.graph.all_nodes():
        if node.op_type not in op_types:
            continue
        if node.name in nodes_to_exclude:
            continue
        for value in node.inputs:
            if value is None:
                continue
            name = value.name
            if name is None or name in seen or name not in initializers:
                continue
            const_value = initializers[name].const_value
            if const_value is None:
                continue
            seen.add(name)
            array = np.asarray(const_value.numpy())
            if not np.issubdtype(array.dtype, np.floating):
                array = array.astype(np.float64)
            stats_list.append(_tensor_statistics(name, array))

    return stats_list


def save_weight_statistics(model: ir.Model, path: str, op_types=("MatMul",), nodes_to_exclude=()) -> list[dict]:
    """Compute and write weight statistics to ``path`` as JSON."""
    stats_list = compute_weight_statistics(model, op_types=op_types, nodes_to_exclude=nodes_to_exclude)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats_list, f, indent=2)
    return stats_list
