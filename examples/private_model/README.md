# `--private` option example

This example shows how to convert a **custom model implemented outside this
package** with the `--private` option of `modelbuilder.builder`.

The private model, `PrivateMoE`, is a tiny **Mixture-of-Experts (MoE)** causal
language model implemented **from scratch** in [`modeling.py`](modeling.py)
(`PrivateDecoderLayer`, `PrivateModel`, `PrivateModelForCausalLM`). Each decoder
layer uses a Mistral-style attention stack (RMSNorm + rotary GQA attention) but
replaces the dense MLP with a sparse MoE layer that routes every token to the
top-k of `num_local_experts` experts. It is exposed under a private architecture
name (`PrivateModelForCausalLM`) and the conversion goes through the `--private`
option so the custom builder is used.

## Files

The `--private` value is made of up to three `;`-separated file paths
(`modeling-file;convert-file;fast-test-file`):

- [`modeling.py`](modeling.py) — the **modeling-file**. Imported before the
  Hugging Face config is loaded so the custom architecture can register itself
  with `transformers`. It implements the model from scratch
  (`PrivateDecoderLayer`, `PrivateModel`, `PrivateModelForCausalLM`) and exposes
  the helpers (`make_config`, `make_model`, `make_tokenizer`) used to build the
  config, tokenizer and PyTorch reference.
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
DONTCLEAN=1 python -m modelbuilder.builder --private "examples/private_model/modeling.py;examples/private_model/convert.py;examples/private_model/test.py"
```

The test file can also be run directly or with `pytest`:

```bash
python examples/private_model/test.py
pytest examples/private_model/test.py
```

## Current results

|    | model_id           | experiment   | precision   | provider   | input_type   | step    |   max_abs_err |   avg_abs_discrepancy |   dnan | next_token   |   next_token_id_tch |   next_token_id_ort | test                                     | kind   |    %>0.1 |   %>0.01 |
|---:|:-------------------|:-------------|:------------|:-----------|:-------------|:--------|--------------:|----------------------:|-------:|:-------------|--------------------:|--------------------:|:-----------------------------------------|:-------|---------:|---------:|
|  0 | private/PrivateMoE | forward      | fp16        | cpu        | text         | prefill |   0.00167513  |           0.000294997 |      0 | OK           |               12086 |               12086 | test_discrepancies_private_moe_fp16_cpu  | random | 0        | 0        |
|  1 | private/PrivateMoE | forward      | fp16        | cpu        | text         | decode  |   0.00150967  |           0.000279989 |      0 | OK           |               15728 |               15728 | test_discrepancies_private_moe_fp16_cpu  | random | 0        | 0        |
|  2 | private/PrivateMoE | forward      | fp16        | cuda       | text         | prefill |               |                       |      0 | CRASH        |                  -1 |                  -1 | test_discrepancies_private_moe_fp16_cuda | random |          |          |
|  3 | private/PrivateMoE | forward      | fp32        | cpu        | text         | prefill |   1.65403e-06 |           2.48252e-07 |      0 | OK           |               29190 |               29190 | test_discrepancies_private_moe_fp32_cpu  | random | 0        | 0        |
|  4 | private/PrivateMoE | forward      | fp32        | cpu        | text         | decode  |   1.3113e-06  |           2.54275e-07 |      0 | OK           |               29190 |               29190 | test_discrepancies_private_moe_fp32_cpu  | random | 0        | 0        |
|  5 | private/PrivateMoE | forward      | fp16        | cpu        | text         | prefill |   0.00180542  |           0.000313571 |      0 | OK           |               14494 |               14494 | test_discrepancies_private_moe_fp16_cpu  | random | 0        | 0        |
|  6 | private/PrivateMoE | forward      | fp16        | cpu        | text         | decode  |   0.0018189   |           0.000326457 |      0 | OK           |               30800 |               30800 | test_discrepancies_private_moe_fp16_cpu  | random | 0        | 0        |
|  7 | private/PrivateMoE | forward      | fp16        | cuda       | text         | prefill |               |                       |      0 | CRASH        |                  -1 |                  -1 | test_discrepancies_private_moe_fp16_cuda | random |          |          |
|  8 | private/PrivateMoE | forward      | fp32        | cuda       | text         | prefill |   2.0112      |           0.324257    |      0 | FAIL         |                8574 |               27866 | test_discrepancies_private_moe_fp32_cuda | random | 0.804406 | 0.980219 |
|  9 | private/PrivateMoE | forward      | fp32        | cuda       | text         | decode  |   1.95624     |           0.366622    |      0 | FAIL         |               22411 |               27866 | test_discrepancies_private_moe_fp32_cuda | random | 0.826187 | 0.982219 |
| 10 | private/PrivateMoE | forward      | fp32        | cpu        | text         | prefill |   1.65403e-06 |           2.48252e-07 |      0 | OK           |               29190 |               29190 | test_discrepancies_private_moe_fp32_cpu  | random | 0        | 0        |
| 11 | private/PrivateMoE | forward      | fp32        | cpu        | text         | decode  |   1.3113e-06  |           2.54275e-07 |      0 | OK           |               29190 |               29190 | test_discrepancies_private_moe_fp32_cpu  | random | 0        | 0        |
| 12 | private/PrivateMoE | forward      | int4        | cpu        | text         | prefill |   0.598127    |           0.0915725   |      0 | OK           |               29190 |               29190 | test_discrepancies_private_moe_int4_cpu  | random | 0.375437 | 0.930031 |
| 13 | private/PrivateMoE | forward      | int4        | cpu        | text         | decode  |   0.409553    |           0.0756651   |      0 | FAIL         |               29190 |               25888 | test_discrepancies_private_moe_int4_cpu  | random | 0.290812 | 0.916281 |
| 14 | private/PrivateMoE | forward      | fp32        | cuda       | text         | prefill |   2.0112      |           0.324257    |      0 | FAIL         |                8574 |               27866 | test_discrepancies_private_moe_fp32_cuda | random | 0.804406 | 0.980219 |
| 15 | private/PrivateMoE | forward      | fp32        | cuda       | text         | decode  |   1.95624     |           0.366622    |      0 | FAIL         |               22411 |               27866 | test_discrepancies_private_moe_fp32_cuda | random | 0.826187 | 0.982219 |
| 16 | private/PrivateMoE | forward      | int4        | cuda       | text         | prefill |               |                       |      0 | CRASH        |                  -1 |                  -1 | test_discrepancies_private_moe_int4_cuda | random |          |          |
| 17 | private/PrivateMoE | forward      | int4        | cpu        | text         | prefill |   0.598127    |           0.0915725   |      0 | OK           |               29190 |               29190 | test_discrepancies_private_moe_int4_cpu  | random | 0.375437 | 0.930031 |
| 18 | private/PrivateMoE | forward      | int4        | cpu        | text         | decode  |   0.409553    |           0.0756651   |      0 | FAIL         |               29190 |               25888 | test_discrepancies_private_moe_int4_cpu  | random | 0.290812 | 0.916281 |
| 19 | private/PrivateMoE | forward      | int4        | cuda       | text         | prefill |               |                       |      0 | CRASH        |                  -1 |                  -1 | test_discrepancies_private_moe_int4_cuda | random |          |          |
