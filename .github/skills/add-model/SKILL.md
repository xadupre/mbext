---
name: add-model
description: Add a model with offline fast tests for every supported type in ONNX Runtime and ONNX Runtime GenAI, checking discrepancies.
---

# Add a model

1. Extend the closest builder in `modelbuilder/builders/` and register each
   supported architecture in `modelbuilder/builder.py` (`create_model`).
2. For **every supported model type** (including modalities), add offline
   `tests/fast/` tests with tiny seeded random weights and local tokenizers;
   cover supported precisions/providers. Follow neighboring `ExtTestCase` tests.
3. With **onnxruntime**, compare prefill and cached-decode outputs against
   PyTorch using precision-appropriate numerical tolerances. Prefer
   `run_random_weights_test` or `run_vl_random_weights_test` where applicable.
4. With **onnxruntime-genai**, exercise generation for each type (including
   modality inputs) and compare generated tokens against the reference; use
   `run_genai_generation_test` for text. Smoke tests alone are insufficient.
5. Run `pytest tests/fast/test_random_<model>.py -v`, then `pytest tests/fast -v`.

See `tests/fast/test_random_olmo3.py` for both runtime test patterns.
