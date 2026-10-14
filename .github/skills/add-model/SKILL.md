---
name: add-model
description: Add support for a model architecture in mbext, including offline ONNX Runtime and ONNX Runtime GenAI tests with discrepancy checks for every supported model type.
---

# Add a model

Use this skill when adding or extending a Hugging Face model architecture.

1. Find the closest builder in `modelbuilder/builders/` and the corresponding
   `tests/fast/test_random_*.py` test. Implement only the architecture-specific
   differences, then register each supported `config.architectures[0]` in
   `modelbuilder/builder.py` (`create_model`). Check that the exported graph inputs,
   outputs, and `genai_config.json` agree for every supported variant.
2. Add offline fast tests in `tests/fast/` for **every model type** supported by the
   new builder (for example, text-only and multimodal variants, and each supported
   input modality). Exercise all supported precision types (`fp32`, `fp16`,
   `bf16`, and applicable integer precisions) and execution providers, including
   CPU and CUDA where supported; gate CUDA tests with `@requires_cuda()`.
   Use tiny, seeded, randomly initialized Hugging Face models and local tokenizers,
   not downloaded weights. Extend `ExtTestCase` and use `@hide_stdout()` like the
   neighboring tests.
3. Test the exported model with **onnxruntime**: run prefill and a cached decode
   step, comparing ONNX logits (and modality-specific outputs or intermediate
   tensors where applicable) against the same PyTorch model and inputs. Prefer
   `run_random_weights_test` for text or `run_vl_random_weights_test` for
   vision-language models in `modelbuilder/ext_test_case.py`; follow nearby
   modality-specific tests for other graph layouts. Assert numerical discrepancies
   with precision-appropriate `atol`/`rtol`, not just successful execution, output
   shapes, or printed statistics. Investigate failures rather than disabling the
   comparison or raising tolerances without justification.
4. Test **onnxruntime-genai** for every supported model type too: load the exported
   directory and exercise generation with representative inputs for that type,
   including the appropriate modality inputs. Use `@requires_genai()` and
   `run_genai_generation_test` for standard text generation, comparing greedy
   tokens against PyTorch; for multimodal types follow an existing modality-specific
   GenAI test and compare generated tokens and available intermediate outputs
   against the reference. A GenAI smoke test alone does not check discrepancies.
5. Run the new tests first with `pytest tests/fast/test_random_<model>.py -v`,
   then run the applicable fast suite (`pytest tests/fast -v`). Run the repository
   style checks (`black --check .` and `ruff check .`) for Python changes. Ensure
   each supported type has both runtime coverage and real discrepancy assertions;
   document any unsupported type explicitly instead of silently omitting it.

See `docs/architectures.rst` for the dispatch convention and
`tests/fast/test_random_olmo3.py` for ONNX Runtime discrepancy and GenAI
generation test patterns.
