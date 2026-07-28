# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Modeling file for the ``private_model`` example.

This is the ``modeling-file`` of the ``--private`` option. It is imported by
:func:`modelbuilder.builder.create_model` *before* the Hugging Face config is
loaded so that the custom architecture defined here can register itself with
``transformers`` (``AutoConfig`` / ``AutoModelForCausalLM``).

The example implements a tiny **Mixture-of-Experts** causal language model
**from scratch** (it does not reuse any ``transformers`` model class). The
PyTorch reference is built out of three hand-written modules:

* :class:`PrivateDecoderLayer` — one decoder block: RMSNorm, rotary GQA
  attention (:class:`PrivateAttention`) and a sparse MoE MLP
  (:class:`PrivateMoE`).
* :class:`PrivateModel` — the backbone: token embedding, a stack of
  :class:`PrivateDecoderLayer` and a final RMSNorm.
* :class:`PrivateModelForCausalLM` — the backbone plus the language-model head.

The module and parameter names (``self_attn.{q,k,v,o}_proj``, ``mlp.gate``,
``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``, ...) match what the
custom ONNX builder in ``convert.py`` reads, so the conversion — driven through
the ``--private`` option — maps this private model to ONNX.

This file also provides the helpers used to build the model config, the
(word-level) tokenizer and the PyTorch reference model. They are shared by the
fast test (``test.py``). The dummy model is intentionally tiny and uses **two**
decoder layers so the MoE routing is exercised across more than one layer while
staying fast and completely offline.
"""

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, GenerationMixin, PretrainedConfig, PreTrainedModel, PreTrainedTokenizerFast
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import CausalLMOutputWithPast, MoeModelOutputWithPast

# Name of the private architecture. ``--private`` forces the custom builder, so
# the conversion does not depend on this name being in the built-in dispatch.
ARCHITECTURE = "PrivateModelForCausalLM"

# Fake model id. It is only used for logging/metadata: the actual weights come
# from the local checkpoint directory passed with ``-i/--input``.
MODEL_NAME = "private/PrivateMoE"


class PrivateConfig(PretrainedConfig):
    """Configuration of the tiny private Mixture-of-Experts model.

    It carries the standard decoder hyper-parameters plus the MoE settings
    (``num_local_experts`` / ``num_experts_per_tok``). It is registered with
    ``AutoConfig`` under the ``private_moe`` model type so that a checkpoint
    saved with :meth:`~transformers.PreTrainedModel.save_pretrained` can be
    reloaded through ``AutoModelForCausalLM.from_pretrained`` (which the
    converter uses to read the weights).
    """

    model_type = "private_moe"

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 1376,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int = 64,
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        num_local_experts: int = 4,
        num_experts_per_tok: int = 2,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.initializer_range = initializer_range
        # This model has no sliding-window attention; the flag is read both by
        # the PyTorch mask helper and by the ONNX builder.
        self.sliding_window = None
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class PrivateRMSNorm(nn.Module):
    """RMSNorm (root-mean-square layer normalization) without bias."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PrivateRotaryEmbedding(nn.Module):
    """Standard (non-interleaved) rotary position embedding."""

    inv_freq: torch.Tensor

    def __init__(self, config: PrivateConfig):
        super().__init__()
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, config.head_dim, 2, dtype=torch.int64).float() / config.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Apply the rotary position embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand key/value heads to match the number of query heads (GQA)."""
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class PrivateAttention(nn.Module):
    """Grouped-query attention with rotary position embeddings."""

    def __init__(self, config: PrivateConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class PrivateExperts(nn.Module):
    """The routed experts of one MoE layer, stored as fused 3-D tensors.

    ``gate_up_proj`` packs the gate and up projections of every expert as
    ``(num_experts, 2 * intermediate, hidden)`` and ``down_proj`` holds the
    down projection as ``(num_experts, hidden, intermediate)``. This is the
    fused layout the ONNX builder reads to emit a single ``com.microsoft:MoE``
    op.
    """

    def __init__(self, config: PrivateConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size, self.hidden_size))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            if token_idx.numel() == 0:
                continue
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


class PrivateMoE(nn.Module):
    """Sparse Mixture-of-Experts MLP: a top-k router plus routed experts."""

    def __init__(self, config: PrivateConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(config.hidden_size, config.num_local_experts, bias=False)
        self.experts = PrivateExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_index = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(hidden_states.dtype)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        return hidden_states.view(batch_size, sequence_length, hidden_dim)


class PrivateDecoderLayer(nn.Module):
    """One decoder block: RMSNorm, GQA attention, RMSNorm and a MoE MLP."""

    def __init__(self, config: PrivateConfig, layer_idx: int):
        super().__init__()
        self.self_attn = PrivateAttention(config, layer_idx)
        self.mlp = PrivateMoE(config)
        self.input_layernorm = PrivateRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = PrivateRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, cache_position=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PrivatePreTrainedModel(PreTrainedModel):
    """Common base wiring the private config and weight initialization."""

    config_class = PrivateConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["PrivateDecoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, PrivateRMSNorm):
            module.weight.data.fill_(1.0)
        elif isinstance(module, PrivateExperts):
            module.gate_up_proj.data.normal_(mean=0.0, std=std)
            module.down_proj.data.normal_(mean=0.0, std=std)


class PrivateModel(PrivatePreTrainedModel):
    """Backbone: token embedding, a stack of decoder layers and a final norm."""

    def __init__(self, config: PrivateConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([PrivateDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = PrivateRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = PrivateRotaryEmbedding(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache is None:
            use_cache = getattr(self.config, "use_cache", True)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
            )

        hidden_states = self.norm(hidden_states)
        return MoeModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class PrivateModelForCausalLM(PrivatePreTrainedModel, GenerationMixin):
    """The private Mixture-of-Experts model with a language-model head."""

    def __init__(self, config: PrivateConfig):
        super().__init__(config)
        self.model = PrivateModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=outputs.past_key_values)


# Register the private architecture with transformers so a checkpoint saved with
# ``save_pretrained`` can be reloaded through ``AutoModelForCausalLM`` (which the
# converter uses to read the weights). The modeling file is imported before the
# config is loaded, so the registration happens in time.
AutoConfig.register(PrivateConfig.model_type, PrivateConfig, exist_ok=True)
AutoModelForCausalLM.register(PrivateConfig, PrivateModelForCausalLM, exist_ok=True)


def make_config(num_hidden_layers: int = 2) -> PrivateConfig:
    """Return the configuration of the tiny private MoE model.

    The dimensions are intentionally small so the fast test stays completely
    offline and quick. ``head_dim=64`` matches ``hidden_size // num_attention_heads``
    (``512 // 8``). The model has ``num_local_experts`` experts and routes each
    token to ``num_experts_per_tok`` of them. It defaults to **two** decoder
    layers.
    """
    return PrivateConfig(architectures=[ARCHITECTURE], num_hidden_layers=num_hidden_layers)


def make_model(config: PrivateConfig = None) -> PrivateModelForCausalLM:
    """Return a PyTorch reference model with random weights for *config*."""
    if config is None:
        config = make_config()
    model = PrivateModelForCausalLM(config)
    model.eval()
    return model


def make_trained_model(
    config: PrivateConfig = None, num_steps: int = 40, batch_size: int = 4, seq_len: int = 16, learning_rate: float = 1e-3, seed: int = 0
) -> PrivateModelForCausalLM:
    """Return a **trained** private MoE model.

    The fast test (``test.py``) exercises the converter with *random* weights.
    This helper instead trains the whole model for a handful of steps on a tiny
    synthetic next-token dataset so the checkpoint carries meaningful (trained)
    weights. It is used by the trained test (``tests/trained``) which converts
    and validates the whole trained model end to end.

    The training loop is intentionally small, fully deterministic (seeded) and
    completely offline. The returned model is set to eval mode.
    """
    if config is None:
        config = make_config()

    torch.manual_seed(seed)
    model = PrivateModelForCausalLM(config)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    # Keep token ids away from the reserved special tokens (0/1/2).
    low = 3
    for _ in range(num_steps):
        input_ids = torch.randint(low, config.vocab_size, (batch_size, seq_len), generator=generator)
        optimizer.zero_grad()
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()

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
