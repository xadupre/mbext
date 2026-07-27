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

The example defines a tiny **Mixture-of-Experts** causal language model,
``PrivateMoE``. It reuses the Mixtral backbone (RMSNorm + rotary GQA attention
with a sparse MoE MLP) but is exposed under a private architecture name
(``PrivateMoEForCausalLM``). The conversion is driven through the ``--private``
option so the custom builder in ``convert.py`` — which implements the MoE decoder
layer — is used instead of the built-in dispatch.

This file also provides the helpers used to build the model config, the
(word-level) tokenizer and the PyTorch reference model. They are shared by the
fast test (``test.py``). The dummy model is intentionally tiny and uses **two**
decoder layers so the MoE routing is exercised across more than one layer while
staying fast and completely offline.
"""

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
from transformers.models.mixtral.configuration_mixtral import MixtralConfig

# Name of the private architecture. ``--private`` forces the custom builder, so
# the conversion does not depend on this name being in the built-in dispatch.
ARCHITECTURE = "PrivateMoEForCausalLM"

# Fake model id. It is only used for logging/metadata: the actual weights come
# from the local checkpoint directory passed with ``-i/--input``.
MODEL_NAME = "private/PrivateMoE"


def make_config(num_hidden_layers: int = 2) -> MixtralConfig:
    """Return the configuration of the tiny private MoE model.

    The dimensions are intentionally small so the fast test stays completely
    offline and quick. ``head_dim=64`` matches ``hidden_size // num_attention_heads``
    (``512 // 8``). The model has ``num_local_experts`` experts and routes each
    token to ``num_experts_per_tok`` of them. It defaults to **two** decoder
    layers.
    """
    return MixtralConfig(
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
        sliding_window=None,
        # MoE-specific settings (kept small for CI).
        num_local_experts=4,
        num_experts_per_tok=2,
    )


def make_model(config: MixtralConfig = None):
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
