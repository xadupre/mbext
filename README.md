# ModelBuilder for onnxruntime-genai

[![codecov](https://codecov.io/gh/xadupre/mbext/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/mbext)

The code base comes from https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models.
It adds fast unit tests checking discrepancies, end to end test with the trained model.
It supports more architectures.

## Convert a model

Example converting ``Qwen/Qwen3-8B`` to ONNX for CPU with int4 precision:

```bash
python -m modelbuilder.builder \
    -m Qwen/Qwen3-8B \
    -o qwen3-8b-cpu-int4 \
    -p int4 \
    -e cpu \
    -c cache_dir
```

The arguments are:

- ``-m/--model_name``: model name on Hugging Face (use ``-i/--input`` instead for a
  local folder).
- ``-o/--output``: folder where the ONNX model and additional files are written.
- ``-p/--precision``: precision of the model (``int4``, ``bf16``, ``fp16`` or ``fp32``).
- ``-e/--execution_provider``: execution provider to target (``cpu`` here).
- ``-c/--cache_dir``: cache directory for Hugging Face files and temporary ONNX
  external data files.

## Style

```bash
black . && ruff check .
```

## Development

```bash
pip install -e .[dev]
```

Converting a model downloads a full Hugging Face model into `./cache_dir` (the
default `--cache_dir`) and writes large ONNX artifacts (`*.onnx` / `*.onnx.data`).
These directories are git-ignored, and `.vscode/settings.json` excludes them from
the VS Code file watcher so the editor does not run out of file watches and
repeatedly prompt to reload the window. The conversion also drops a
`.vscode/settings.json` inside the output model folder with the same exclusions,
so opening that folder directly in VS Code stays stable too.

## Fast Unit tests

```bash
pytest tests/fast
```

## Long Unit tests

```bash
python tests/trained/test_trained_tiny_llm.py
```

With a better machine:

```bash
LONGTEST=1 pytest tests/trained
```

## Llama.cpp tests

`tests/fast_llama_cpp` compares `llama.cpp` and `onnxruntime-genai` on the same
model. The conversion script `convert_hf_to_gguf.py` comes from the
[llama.cpp](https://github.com/ggml-org/llama.cpp) repository and its
requirements pin a specific `torch` version (for example `torch==2.11.0`).
Installing them downgrades `torch` from the version installed for the rest of
the test suite.

`torchaudio` (and `torchvision`) compiled against the previous `torch` build is
then incompatible with the downgraded `torch`. Because `transformers` imports
`torchaudio` lazily, the mismatch surfaces while loading the model as:

```
ModuleNotFoundError: Could not import module 'LlamaForCausalLM'.
```

Uninstalling `torchaudio` (and `torchvision`) removes the mismatch, since these
tests only need `torch` itself:

```bash
pip uninstall -y torchaudio torchvision
```

You can see the results in ``stats/end2end_results.json``. Example:

```
{'first_diff': 0, 'delta_length': 4, 'expected_length': 16, 'total_diff': 16, 'precision': 'fp32', 'model_id': 'HuggingFaceTB/SmolLM3-3B', 'experiment': 'generate', 'provider': 'cpu'}
{'max_abs_err': 1.6875, '%_gt_0.1': np.float64(0.5278832959081836), '%_gt_0.01': np.float64(0.962624750499002), 'avg_abs_discrepancy': 0.1749267578125, 'shape': (1, 5, 128256), 'dtype': dtype('float16'), 'precision': 'fp16', 'model_id': 'HuggingFaceTB/SmolLM3-3B', 'experiment': 'forward'}
```
