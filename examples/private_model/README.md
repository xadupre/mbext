# `--private` option example

This example shows how to convert a **custom model implemented outside this
package** with the `--private` option of `modelbuilder.builder`.

The private model, `PrivateMoE`, is a tiny **Mixture-of-Experts (MoE)** causal
language model. It reuses the Mistral attention stack (RMSNorm + rotary GQA
attention) but the custom builder replaces each dense MLP with a sparse MoE layer
that routes every token to the top-k of `num_local_experts` experts. It is
exposed under a private architecture name (`PrivateMoEForCausalLM`) and the
conversion goes through the `--private` option so the custom builder is used.

## Files

The `--private` value is made of up to three `;`-separated file paths
(`modeling-file;convert-file;fast-test-file`):

- [`modeling.py`](modeling.py) — the **modeling-file**. Imported before the
  Hugging Face config is loaded so a custom architecture can register itself with
  `transformers`. Here it also exposes the helpers (`make_config`, `make_model`,
  `make_tokenizer`) used to build the config (a two-layer MoE `MixtralConfig`),
  tokenizer and PyTorch reference.
- [`convert.py`](convert.py) — the **convert-file**. Defines the ONNX builder
  (`PrivateMoEModel`, selected through the module-level `MODEL_BUILDER`
  attribute). It implements a **custom decoder layer with a MoE MLP** by
  overriding `make_layer` / `make_moe` and emitting a `com.microsoft:MoE` op via
  the shared `make_fused_moe` helper.
- [`test.py`](test.py) — the **fast-test-file**. Validates the custom builder
  with a discrepancy test (PyTorch vs ONNX) and a genai generation test (PyTorch
  vs `onnxruntime-genai`). The dummy model uses **two** decoder layers.

## Convert a model

Point `-i/--input` at a local Hugging Face checkpoint of the private model:

```bash
python -m modelbuilder.builder \
    -i my-private-model \
    -o my-private-model-cpu-fp32 \
    -p fp32 \
    -e cpu \
    --private "examples/private_model/modeling.py;examples/private_model/convert.py;examples/private_model/test.py"
```

## Run the fast tests

When no model id is given (both `-m/--model_name` and `-i/--input` are omitted),
the `fast-test-file` is executed as a script instead of converting a model:

```bash
python -m modelbuilder.builder \
    --private "examples/private_model/modeling.py;examples/private_model/convert.py;examples/private_model/test.py"
```

The test file can also be run directly or with `pytest`:

```bash
python examples/private_model/test.py
pytest examples/private_model/test.py
```
