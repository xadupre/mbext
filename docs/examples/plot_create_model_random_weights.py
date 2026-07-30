# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Convert a random-weight model with ``create_model``
===================================================

This example shows the core workflow of mbext on a **tiny, randomly
initialised** model. Nothing is downloaded from the Hugging Face hub: a small
:class:`~transformers.LlamaConfig` is built with a single hidden layer and random
weights, converted to ONNX with :func:`modelbuilder.builder.create_model`, and
the ONNX logits are compared to the PyTorch reference.

This is exactly how the fast test suite validates every architecture, which is
why the tests run offline and in seconds (see the *Design* page).
"""

import os
import tempfile

import numpy as np

# %%
# Build a tiny random-weight model
# --------------------------------
# A minimal ``LlamaForCausalLM`` with small dimensions and a single hidden layer.
# The weights are randomly initialised (seeded for reproducibility).
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

num_hidden_layers = 1
config = LlamaConfig(
    architectures=["LlamaForCausalLM"],
    hidden_size=512,
    intermediate_size=1376,
    max_position_embeddings=2048,
    num_attention_heads=8,
    num_key_value_heads=4,
    num_hidden_layers=num_hidden_layers,
    vocab_size=1024,
    bos_token_id=1,
    eos_token_id=2,
)
head_size = config.hidden_size // config.num_attention_heads

torch.manual_seed(42)
model = AutoModelForCausalLM.from_config(config)
model.eval()

# %%
# A minimal word-level tokenizer is enough for ``create_model`` to write the
# processing files next to the ONNX model.
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=Tokenizer(WordLevel(vocab={"<unk>": 0, "<s>": 1, "</s>": 2}, unk_token="<unk>")),
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
)

# %%
# Convert to ONNX with ``create_model``
# -------------------------------------
# The PyTorch model and tokenizer are saved to a folder, then converted to ONNX
# for the CPU execution provider in ``fp32`` precision.
from modelbuilder.builder import create_model  # noqa: E402

work_dir = tempfile.mkdtemp(prefix="mbext_example_")
model_dir = os.path.join(work_dir, "model")
output_dir = os.path.join(work_dir, "onnx")
cache_dir = os.path.join(work_dir, "cache")

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)

create_model(
    model_name="tiny-llama",
    input_path=model_dir,
    output_dir=output_dir,
    precision="fp32",
    execution_provider="cpu",
    cache_dir=cache_dir,
    num_hidden_layers=num_hidden_layers,
)

onnx_path = os.path.join(output_dir, "model.onnx")
print("ONNX model written to:", onnx_path)

# %%
# Compare the ONNX and PyTorch logits
# -----------------------------------
# The exported graph expects ``input_ids``, ``attention_mask``, ``position_ids``
# and an (empty) KV cache. We run a single prefill step through OnnxRuntime and
# compare the logits with the PyTorch model.
import onnxruntime  # noqa: E402

sess = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_input_names = {i.name for i in sess.get_inputs()}

batch_size, seq_len = 1, 5
torch.manual_seed(0)
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

feed = {
    "input_ids": input_ids.numpy().astype(np.int64),
    "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
    "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
}
for i in range(num_hidden_layers):
    feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, config.num_key_value_heads, 0, head_size), dtype=np.float32)
    feed[f"past_key_values.{i}.value"] = np.zeros((batch_size, config.num_key_value_heads, 0, head_size), dtype=np.float32)
feed = {k: v for k, v in feed.items() if k in onnx_input_names}

onnx_logits = sess.run(None, feed)[0]

with torch.no_grad():
    torch_logits = model(input_ids).logits.numpy()

max_abs_diff = np.abs(onnx_logits - torch_logits).max()
print("logits shape:", onnx_logits.shape)
print("max absolute discrepancy (ONNX vs PyTorch):", float(max_abs_diff))

# %%
# The discrepancy is tiny, confirming the ONNX model matches the PyTorch
# reference. This is the check that runs for every supported architecture in the
# fast test suite.
