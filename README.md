# ModelBuilder for onnxruntime-genai

The code base comes from https://github.com/microsoft/onnxruntime-genai/tree/main/src/python/py/models.
It adds fast unit tests checking discrepancies, end to end test with the trained model.
It supports more architectures.

## Style

```bash
black . && ruff check .
```

## Development

```bash
pip install -e .[dev]
```

## Supported Models

The table below shows which models have test coverage across the three test
tiers.  *Fast tests* use randomly-initialised weights and run entirely offline.
*Trained tests* download the real model weights and check numerical
discrepancies against PyTorch.  *Genai tests* additionally verify end-to-end
token generation via ``onnxruntime-genai``.

| Model | Architecture | Fast tests | Trained tests | Genai tests |
|-------|-------------|:---------:|:------------:|:-----------:|
| [arnir0/Tiny-LLM](https://huggingface.co/arnir0/Tiny-LLM) | `LlamaForCausalLM` | ✓ | ✓ CPU | ✓ CPU |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | `GptOssForCausalLM` | ✓ | | ✓ CPU |
| [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | `MistralNeMoForCausalLM` | ✓ | ✓ CUDA | ✓ CUDA, CPU |
| [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B) | `Olmo2ForCausalLM` | ✓ | | |
| [allenai/OLMo-3-7B-Instruct](https://huggingface.co/allenai/OLMo-3-7B-Instruct) | `Olmo3ForCausalLM` | ✓ | ✓ CUDA | ✓ CUDA, CPU |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | `SmolLM3ForCausalLM` | ✓ | ✓ CUDA | ✓ CUDA, CPU |

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

You can see the results in ``stats/end2end_results.json``. Example:

```
{'first_diff': 0, 'delta_length': 4, 'expected_length': 16, 'total_diff': 16, 'precision': 'fp32', 'model_id': 'HuggingFaceTB/SmolLM3-3B', 'experiment': 'generate', 'provider': 'cpu'}
{'max_abs_err': 1.6875, '%_gt_0.1': np.float64(0.5278832959081836), '%_gt_0.01': np.float64(0.962624750499002), 'avg_abs_discrepancy': 0.1749267578125, 'shape': (1, 5, 128256), 'dtype': dtype('float16'), 'precision': 'fp16', 'model_id': 'HuggingFaceTB/SmolLM3-3B', 'experiment': 'forward'}
```

