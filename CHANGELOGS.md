# Changelog

All notable changes to this project are documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Model support

- Adds Nemotron 3 Nano 4B BF16, Nano 30B-A3B NVFP4, and Nemotron 3.5
  Lightning 30B-A3B NVFP4 conversion, with offline hybrid-model parity and
  generation tests. ModelOpt NVFP4/FP8 weights are dequantized for export;
  activation and KV-cache quantization are not reproduced.
- Adds Qwen3.8 conversion with native compressed-tensors NVFP4/FP8 weights and
  floating-point, INT2, INT4, and INT8 exports.
- Adds exact INT2 conversion for Ternary Bonsai and direct conversion of packed
  MLX Qwen3.5 multimodal checkpoints without expanding language-model weights
  into BF16.
- Adds Qwen2.5 Omni video inference support.
- Corrects DeepSeek grouped sigmoid routing, including selection-only correction
  biases, normalization, and routed scaling.
- Supports real Transformers PhiMoE packed experts and inference SparseMixer
  routing, recognizing both PhiMoE architecture names. Honors requested export
  options; the legacy expert layout requires symmetric INT4 weights and CUDA.
- Enables CPU BF16 group-query attention.

### Export & memory usage

- Uses onnx-light for fast ONNX graph construction, loading, saving, and
  external-data sharding.
- Borrows eligible CPU weight storage for zero-copy initializer construction,
  with fallback handling for unsupported inputs.
- Adds `--reuse-weights` to relocate checkpoint safetensors into the export and
  reference their bytes directly. This transfers ownership and removes the
  original weight files; generated weights are shared across exported components.
- Adds `--max-shard-size` to control generated ONNX external-data file sizes.
  Reused checkpoint files are not split, and individual tensors can exceed the
  requested limit.
- Automatically emits a shared-weight `pipeline.onnx` PipelineProto for
  multi-model CLI exports, preserving standalone models and reporting components
  not supported by the pipeline schema.
- Shares sparse selected-token expert execution between DeepSeek and PhiMoE,
  using static expert weights and a single token/slot scatter. Preserves expert
  accumulation order and FP16 SiLU rounding with FP32 activation intermediates.
- Preserves DeepSeek router checkpoint reuse unless the router is eligible for
  explicitly requested INT4 quantization.

### Tests & fixes

- Adds offline ONNX Runtime parity and ONNX Runtime GenAI generation coverage for
  Qwen3.8 and packed MLX exports, plus generation tests with onnx-light-cpu.
- Enables offline native onnx-light five-token greedy generation parity for
  DeepSeek and real PhiMoE, with routing, cached-decode, FP16 saturation, and
  legacy export-option regressions.
- Preserves standalone `pipeline.onnx` exports when cleaning up stale pipeline
  artifacts, including equivalent output paths.
- Corrects macOS peak RSS units in the zero-copy initializer benchmark.

### Documentation & CI

- Describes mbext as a "Fast Experimental Exporter for transformers" in package
  metadata and documentation, emphasizing onnx-light as the fast export backend.
- Builds a universal Python wheel and source distribution for every pushed tag
  and publishes them to the corresponding GitHub Release.
- Supports rebuilding an existing tag through a manual workflow run, replacing
  matching release assets without changing existing release notes.
- Schedules weekly documentation builds and opens issues when they fail.
- Refreshes CI coverage for PyTorch 2.14.1, Transformers 5.20.2, and
  huggingface-hub 2.2.0 while preserving compatibility with older Transformers.
- Reduces onnx-light wheel CI checks to Python 3.14 and adds weekly Dependabot
  updates.
