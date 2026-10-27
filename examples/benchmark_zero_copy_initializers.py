# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Compare rewriting weights with reusing an existing safetensors checkpoint.

Run with ``python examples/benchmark_zero_copy_initializers.py --size 2048``.
Checkpoint creation and each export run in separate isolated processes.
Exports use onnx-light's file writer, never an in-memory protobuf byte string.
"""

import argparse
import contextlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from onnx_light.onnx import TensorProto, load_model
from onnx_light.onnx_core.graph_builder import GraphBuilder
from safetensors.torch import load_file, save_file

from modelbuilder.builders.base import Model
from modelbuilder.helpers.weight_reuse import WeightReuse


def peak_rss_mib(ru_maxrss: int, platform: str) -> float:
    divisor = 1024 * 1024 if platform == "darwin" else 1024
    return round(ru_maxrss / divisor, 1)


def run(mode: str, checkpoint_path: str, output_dir: str) -> dict:
    checkpoint = Path(checkpoint_path)
    original_stat = checkpoint.stat()
    weight = load_file(str(checkpoint))["weight"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    reuse = WeightReuse(checkpoint.parent, output) if mode == "zero_copy" else None
    model = Model.__new__(Model)
    model.graph = GraphBuilder("weights", opsets=21)
    model._initialize_graph_state()
    model.reuse_downloaded_weights = reuse is not None
    model.extra_options = {"_weight_reuse": reuse} if reuse is not None else {}
    model.source_model_path = str(checkpoint.parent)
    model.external_data_base_dir = str(output)
    model._source_tensors_by_data_ptr = {}
    model._used_source_weight_files = set()
    model.local_functions = {}
    model.filename = "model.onnx"
    model.cache_dir = str(output / "cache")
    model.quant_type = None
    model.onnx_dtype = TensorProto.FLOAT
    model.weights = torch.nn.Module()
    model.weights.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    model._initialize_source_tensor_tracking()
    model.make_initializer(model.weights.weight, "weight")
    model.graph.inp("input", TensorProto.FLOAT, [1, weight.shape[0]])
    model.graph.out(model.graph.op.MatMul("input", "weight"))
    built = time.perf_counter()
    model.save_model(str(output))
    saved = time.perf_counter()
    if reuse is not None:
        reuse.relocate()
    finished = time.perf_counter()

    path = output / model.filename
    exported = load_model(str(path), load_external_data=False)
    reused_checkpoint = Path(reuse.destinations[str(checkpoint.resolve())]) if reuse is not None else checkpoint
    reused_stat = reused_checkpoint.stat()
    reused_bytes = 0
    rewritten_bytes = 0
    rewritten_files = set()
    for initializer in exported.graph.initializer:
        rewritten_bytes += len(initializer.raw_data)
        if initializer.data_location != TensorProto.EXTERNAL:
            continue
        reference = {entry.key: entry.value for entry in initializer.external_data}
        external_path = (output / reference["location"]).resolve()
        if reuse is not None and external_path == reused_checkpoint.resolve():
            reused_bytes += int(reference["length"])
        else:
            rewritten_files.add(external_path)
    rewritten_bytes += sum(file.stat().st_size for file in rewritten_files)
    return {
        "mode": mode,
        "size": weight.shape[0],
        "seconds": round(finished - start, 6),
        "construction_seconds": round(built - start, 6),
        "save_seconds": round(saved - built, 6),
        "relocation_seconds": round(finished - saved, 6),
        "peak_rss_mib": peak_rss_mib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, sys.platform),
        "onnx_bytes": path.stat().st_size,
        "reused_weight_bytes": reused_bytes,
        "rewritten_weight_bytes": rewritten_bytes,
        "reused_checkpoint_same_inode": reuse is not None
        and (original_stat.st_dev, original_stat.st_ino) == (reused_stat.st_dev, reused_stat.st_ino),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=4096, help="Side length of the float32 weight matrix")
    parser.add_argument("--mode", choices=("legacy", "zero_copy"), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", help=argparse.SUPPRESS)
    parser.add_argument("--create-checkpoint", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("--size must be positive")
    if args.create_checkpoint:
        if not args.checkpoint:
            parser.error("--create-checkpoint requires --checkpoint")
        save_file({"weight": torch.ones((args.size, args.size), dtype=torch.float32)}, args.checkpoint)
    elif args.checkpoint:
        if not args.mode or not args.output_dir:
            parser.error("--checkpoint requires --mode and --output-dir")
        with contextlib.redirect_stdout(sys.stderr):
            result = run(args.mode, args.checkpoint, args.output_dir)
        print(json.dumps(result))
    else:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.safetensors")
            subprocess.run(
                [sys.executable, __file__, "--create-checkpoint", "--checkpoint", checkpoint, "--size", str(args.size)], check=True
            )
            for mode in (args.mode,) if args.mode else ("legacy", "zero_copy"):
                subprocess.run(
                    [sys.executable, __file__, "--mode", mode, "--checkpoint", checkpoint, "--output-dir", os.path.join(directory, mode)],
                    check=True,
                )
