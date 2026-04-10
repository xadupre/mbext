# ModelBuilder for onnxruntime-genai

[![codecov](https://codecov.io/gh/xadupre/mbext/branch/main/graph/badge.svg)](https://codecov.io/gh/xadupre/mbext)

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
| [baidu/ERNIE-4.5-0.3B-PT](https://huggingface.co/baidu/ERNIE-4.5-0.3B-PT) | `Ernie4_5ForCausalLM` | ✓ | | |
| [google/gemma-2b](https://huggingface.co/google/gemma-2b) | `GemmaForCausalLM` | ✓ | | |
| [google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b) | `Gemma2ForCausalLM` | ✓ | | |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | `Gemma3ForCausalLM` (text-only) | ✓ | | |
| [google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) | `Gemma3ForConditionalGeneration` (multimodal) | ✓ | | |
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | `GptOssForCausalLM` | ✓ | | |
| [ibm-granite/granite-3.3-2b-instruct](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct) | `GraniteForCausalLM` | ✓ | | |
| [internlm/internlm2-7b](https://huggingface.co/internlm/internlm2-7b) | `InternLM2ForCausalLM` | ✓ | | |
| [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | `Ministral3ForCausalLM` (text-only) | ✓ | | |
| [mistralai/Ministral-3-3B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512) | `Mistral3ForConditionalGeneration` (multimodal) | ✓ | ✓ CPU | ✓ CPU |
| [mistralai/Mistral-Nemo-Instruct-2407](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) | `MistralNeMoForCausalLM` | ✓ | ✓ CPU | ✓ CPU |
| [nvidia/Minitron-4B-Base](https://huggingface.co/nvidia/Minitron-4B-Base) | `NemotronForCausalLM` | ✓ | | |
| [nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16) | `NemotronHForCausalLM` | ✓ | | |
| [allenai/OLMo-7B](https://huggingface.co/allenai/OLMo-7B) | `OlmoForCausalLM` | ✓ | | |
| [allenai/OLMo-2-1124-7B](https://huggingface.co/allenai/OLMo-2-1124-7B) | `Olmo2ForCausalLM` | ✓ | | |
| [allenai/OLMo-3-7B-Instruct](https://huggingface.co/allenai/OLMo-3-7B-Instruct) | `Olmo3ForCausalLM` | ✓ | ✓ CPU | ✓ CPU |
| [microsoft/phi-2](https://huggingface.co/microsoft/phi-2) | `PhiForCausalLM` | ✓ | | |
| [microsoft/Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | `Phi3ForCausalLM` | ✓ | | |
| [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) | `Phi3ForCausalLM` (LongRoPE) | ✓ | | |
| [microsoft/Phi-3-small-8k-instruct](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) | `Phi3SmallForCausalLM` | ✓ | | |
| [microsoft/Phi-3-vision-128k-instruct](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | `Phi3VForCausalLM` | ✓ | | |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | `Phi4MMForCausalLM` | ✓ | | |
| [microsoft/Phi-3.5-MoE-instruct](https://huggingface.co/microsoft/Phi-3.5-MoE-instruct) | `PhiMoEForCausalLM` | ✓ | | |
| [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | `Qwen2_5_VLForConditionalGeneration` | ✓ | | |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | `Qwen3ForCausalLM` | ✓ | ✓ CPU | ✓ CPU |
| [Qwen/Qwen3.5-3B](https://huggingface.co/Qwen/Qwen3.5-3B) | `Qwen3_5ForConditionalGeneration` | ✓ | | |
| [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | `Qwen3VLForConditionalGeneration` | ✓ | | |
| [HuggingFaceTB/SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | `SmolLM3ForCausalLM` | ✓ | ✓ CPU | ✓ CPU |
| [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) | `WhisperForConditionalGeneration` | ✓ | | |
| [THUDM/chatglm3-6b](https://huggingface.co/THUDM/chatglm3-6b) | `ChatGLMForConditionalGeneration` | ✓ | | |
| [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b) | `ChatGLMModel` | ✓ | | |

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

Or in markdown:

