# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Modeling file for the ``private_model`` example.

This is the ``modeling-file`` of the ``--private`` option. It is imported by
:func:`modelbuilder.builder.create_model` *before* the Hugging Face config is
loaded so that a custom architecture can register itself with ``transformers``.

The example defines a tiny custom causal language model, ``PrivateTiny``, that
reuses the Qwen3 backbone but is exposed under a private architecture name
(``PrivateTinyForCausalLM``) that is **not** part of the built-in dispatch in
:func:`modelbuilder.builder.create_model`. The conversion therefore has to go
through the ``--private`` option (see ``convert.py``).

Besides the architecture name, this file provides the helpers used to build the
model config, the (word-level) tokenizer and the PyTorch reference model. They
are shared by the fast test (``test.py``).
"""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen3Config

# Name of the private architecture. It is deliberately different from every
# architecture handled by the built-in dispatch so the conversion requires the
# custom builder loaded through ``--private``.
ARCHITECTURE = "PrivateTinyForCausalLM"

# Fake model id. It is only used for logging/metadata: the actual weights come
# from the local checkpoint directory passed with ``-i/--input``.
MODEL_NAME = "private/PrivateTiny"


def make_config(num_hidden_layers: int = 1) -> Qwen3Config:
    """Return the configuration of the tiny private model.

    The dimensions are intentionally small so the fast test stays completely
    offline and quick. ``head_dim=64`` matches ``hidden_size // num_attention_heads``
    (``512 // 8``).
    """
    return Qwen3Config(
        architectures=[ARCHITECTURE],
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=512,
        intermediate_size=1376,
        max_position_embeddings=2048,
        num_attention_heads=8,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=4,
        head_dim=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        vocab_size=32000,
        use_sliding_window=False,
    )


def make_model(config: Qwen3Config = None):
    """Return a PyTorch reference model with random weights for *config*."""
    if config is None:
        config = make_config()
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model


def make_tokenizer(bos_token_id: int = 1, eos_token_id: int = 2) -> PreTrainedTokenizerFast:
    """Return a minimal word-level tokenizer for the tiny private model.

    The vocabulary contains exactly three tokens: ``<unk>`` at id 0 plus the
    ``<s>`` (bos) and ``</s>`` (eos) tokens at their respective ids.
    """
    vocab = {"<unk>": 0, "<s>": bos_token_id, "</s>": eos_token_id}
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
    )
