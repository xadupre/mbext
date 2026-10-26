# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Compare initializer export time and peak RSS in isolated processes.

Run with ``python examples/benchmark_zero_copy_initializers.py --size 2048``.
Each process builds and serializes a model with a large in-memory CPU weight.
"""

import argparse
import json
import resource
import subprocess
import sys
import time

import torch
from onnx_light.onnx.numpy_helper import from_array
from onnx_light.onnx_core.graph_builder import GraphBuilder

from modelbuilder.builders.base import Model
from modelbuilder.helpers.onnx_helper import torch_tensor_to_numpy


def peak_rss_mib(ru_maxrss: int, platform: str) -> float:
    divisor = 1024 * 1024 if platform == "darwin" else 1024
    return round(ru_maxrss / divisor, 1)


def run(mode: str, size: int) -> dict:
    weight = torch.ones((size, size), dtype=torch.float32)
    builder = GraphBuilder("weights")
    start = time.perf_counter()
    if mode == "legacy":
        builder.make_initializer(from_array(torch_tensor_to_numpy(weight), name="weight"))
    else:
        model = Model.__new__(Model)
        model.graph = builder
        model._initialize_graph_state()
        model.reuse_downloaded_weights = False
        model.make_initializer(weight, "weight")
    serialized = builder.to_model().SerializeToString()
    return {
        "mode": mode,
        "size": size,
        "seconds": round(time.perf_counter() - start, 3),
        "peak_rss_mib": peak_rss_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform),
        "serialized_bytes": len(serialized),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096, help="Side length of the float32 weight matrix")
    parser.add_argument("--mode", choices=("legacy", "zero_copy"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.mode:
        print(json.dumps(run(args.mode, args.size)))
    else:
        for mode in ("legacy", "zero_copy"):
            subprocess.run([sys.executable, __file__, "--mode", mode, "--size", str(args.size)], check=True)
