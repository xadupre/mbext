# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""CPU-capable Kimi K3.1 reference with explicit inference state.

The published configuration and checkpoint schema are at
https://huggingface.co/moonshotai/Kimi-K3.1-Pro.
Architecture equations can also be found in NVIDIA-NeMo/Automodel's
``components/models/kimi_k3/{model,situ,vision}.py`` and FLA's
``ops/kda/naive.py``. No fused, distributed, or remote-code kernels are needed.

One forward call consumes one token per batch row. Prefill is deliberately
sequential: this keeps the KDA recurrence exact without tracing a fixed-length
Python loop. Padding, packed documents, speculative heads and training are not
implemented. MLA is positional-encoding-free, including its shared key branch.
"""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig


class KimiK31Config(PretrainedConfig):
    model_type = "kimi_k31"
    is_composition = True

    def __init__(self, text_config=None, vision_config=None, media_placeholder_token_id=163605, **kwargs):
        kwargs.setdefault("architectures", ["KimiK31ForConditionalGeneration"])
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        text = dict(
            vocab_size=163840,
            hidden_size=7168,
            intermediate_size=33792,
            num_hidden_layers=93,
            num_attention_heads=96,
            num_key_value_heads=96,
            max_position_embeddings=1048576,
            rms_norm_eps=1e-5,
            hidden_act="situ",
            initializer_range=0.02,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            mla_use_nope=True,
            mla_use_output_gate=True,
            num_experts=896,
            num_experts_per_token=16,
            num_shared_experts=2,
            num_expert_group=1,
            topk_group=1,
            moe_renormalize=True,
            moe_router_activation_func="sigmoid",
            routed_scaling_factor=1.0,
            moe_intermediate_size=3072,
            routed_expert_hidden_size=3584,
            latent_moe_use_norm=True,
            first_k_dense_replace=1,
            moe_layer_freq=1,
            attn_res_block_size=12,
            activation_situ_beta=4.0,
            activation_situ_linear_beta=25.0,
            linear_attn_config=dict(
                head_dim=128,
                num_heads=96,
                short_conv_kernel_size=4,
                use_full_rank_gate=True,
                gate_lower_bound=-5.0,
                full_attn_layers=list(range(4, 93, 4)) + [93],
                kda_layers=[i for i in range(1, 94) if i % 4 and i != 93],
            ),
            dsa_config=dict(
                dsa_layers=list(range(4, 93, 4)) + [93],
                index_topk=2048,
                index_head_dim=128,
                index_num_heads=32,
                index_rope_dim=0,
                index_k_norm_type="layernorm",
                index_topk_freq=1,
                index_skip_topk_offset=1,
            ),
        )
        text.update(text_config.to_dict() if isinstance(text_config, PretrainedConfig) else text_config or {})
        self.text_config = PretrainedConfig(**text)
        self.vision_config = None if vision_config is None else dict(vision_config)
        self.media_placeholder_token_id = media_placeholder_token_id
        self._validate()

    def _validate(self):
        c = self.text_config
        if c.hidden_act != "situ" or not c.mla_use_nope:
            raise ValueError("Kimi K3.1 requires SiTU and positional-encoding-free MLA.")
        if c.num_key_value_heads != c.num_attention_heads:
            raise ValueError("Kimi K3.1 requires equal MLA query and KV head counts.")
        if c.attn_res_block_size is not None and c.attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be positive or None.")
        linear, dsa = c.linear_attn_config, c.dsa_config
        kda, full = set(linear["kda_layers"]), set(linear["full_attn_layers"])
        active = set(range(1, c.num_hidden_layers + 1))
        if kda & full or (kda | full) & active != active:
            raise ValueError("The one-based KDA/MLA layer schedules must partition the decoder.")
        if set(dsa["dsa_layers"]) & active != full & active:
            raise ValueError("Every K3.1 MLA layer must have a DSA indexer.")
        if dsa["index_rope_dim"] != 0 or dsa["index_topk_freq"] != 1 or dsa["index_skip_topk_offset"] != 1:
            raise ValueError("Only the released K3.1 DSA indexer schedule is supported.")
        if dsa["index_topk"] <= 0 or linear["short_conv_kernel_size"] < 2:
            raise ValueError("DSA top-k must be positive and the KDA convolution needs at least two taps.")
        if c.moe_router_activation_func not in {"sigmoid", "softmax"}:
            raise ValueError("Unsupported Kimi router activation.")
        if c.num_experts % c.num_expert_group or not 1 <= c.topk_group <= c.num_expert_group:
            raise ValueError("Invalid grouped-expert routing configuration.")
        eligible = c.topk_group * (c.num_experts // c.num_expert_group)
        if not 1 <= c.num_experts_per_token <= eligible:
            raise ValueError("The selected groups cannot supply the requested number of experts.")
        if c.num_expert_group > 1 and c.num_experts // c.num_expert_group < 2:
            raise ValueError("Grouped routing requires at least two experts per group.")


class RMSNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        value = x.float()
        eps = torch.finfo(x.dtype).eps if self.eps is None else self.eps
        return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype) * self.weight.to(x.dtype)


def situ(gate, up, beta, linear_beta):
    dtype = gate.dtype
    gate, up = gate.float(), up.float()
    value = beta * torch.tanh(gate / beta) * gate.sigmoid()
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (value * up).to(dtype)


class MLP(nn.Module):
    def __init__(self, c, hidden=None, intermediate=None, *, expert=False):
        super().__init__()
        hidden = c.hidden_size if hidden is None else hidden
        intermediate = c.intermediate_size if intermediate is None else intermediate
        self.expert = expert
        names = ("w1", "w3", "w2") if expert else ("gate_proj", "up_proj", "down_proj")
        for name, dims in zip(names, ((hidden, intermediate), (hidden, intermediate), (intermediate, hidden))):
            setattr(self, name, nn.Linear(*dims, bias=False))
        self.beta, self.linear_beta = c.activation_situ_beta, c.activation_situ_linear_beta

    def forward(self, x):
        gate, up, down = (self.w1, self.w3, self.w2) if self.expert else (self.gate_proj, self.up_proj, self.down_proj)
        return down(situ(gate(x), up(x), self.beta, self.linear_beta))


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.gate = nn.Linear(c.hidden_size, c.num_experts, bias=False)
        self.gate.e_score_correction_bias = nn.Parameter(torch.zeros(c.num_experts))
        latent = c.routed_expert_hidden_size
        self.routed_expert_down_proj = nn.Linear(c.hidden_size, latent, bias=False)
        self.routed_expert_up_proj = nn.Linear(latent, c.hidden_size, bias=False)
        self.routed_expert_norm = RMSNorm(latent, c.rms_norm_eps) if c.latent_moe_use_norm else nn.Identity()
        self.experts = nn.ModuleList([MLP(c, latent, c.moe_intermediate_size, expert=True) for _ in range(c.num_experts)])
        self.shared_experts = MLP(c, intermediate=c.num_shared_experts * c.moe_intermediate_size) if c.num_shared_experts else None

    def route(self, x):
        c = self.c
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = logits.sigmoid() if c.moe_router_activation_func == "sigmoid" else logits.softmax(-1)
        choice = scores + self.gate.e_score_correction_bias.float()
        if c.num_expert_group > 1 and c.topk_group < c.num_expert_group:
            groups = choice.reshape(-1, c.num_expert_group, c.num_experts // c.num_expert_group)
            group_scores = groups.topk(2, dim=-1).values.sum(-1)
            selected = group_scores.topk(c.topk_group, dim=-1).indices
            mask = torch.zeros_like(group_scores).scatter(1, selected, 1)
            choice = choice.masked_fill(~mask.bool().unsqueeze(-1).expand_as(groups).reshape_as(choice), -float("inf"))
        indices = choice.topk(c.num_experts_per_token, dim=-1).indices
        weights = scores.gather(1, indices)
        if c.num_experts_per_token > 1 and c.moe_renormalize:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return indices, weights * c.routed_scaling_factor

    def forward(self, hidden):
        x = hidden.reshape(-1, self.c.hidden_size)
        indices, weights = self.route(x)
        latent = self.routed_expert_down_proj(x)
        result = torch.zeros_like(latent, dtype=torch.float32)
        for index, expert in enumerate(self.experts):
            rows, slots = torch.where(indices == index)
            output = expert(latent[rows]).float() * weights[rows, slots, None]
            result = result.index_add(0, rows, output)
        result = self.routed_expert_up_proj(self.routed_expert_norm(result.to(latent.dtype)))
        if self.shared_experts is not None:
            result = result + self.shared_experts(x)
        return result.reshape_as(hidden)


class KDA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        linear = c.linear_attn_config
        self.heads, self.dim = linear["num_heads"], linear["head_dim"]
        width, taps = self.heads * self.dim, linear["short_conv_kernel_size"]
        for name in ("q", "k", "v"):
            setattr(self, f"{name}_proj", nn.Linear(c.hidden_size, width, bias=False))
            setattr(self, f"{name}_conv1d", nn.Conv1d(width, width, taps, groups=width, bias=False))
        self.A_log = nn.Parameter(torch.zeros(self.heads))
        self.dt_bias = nn.Parameter(torch.zeros(width))
        self.f_a_proj = nn.Linear(c.hidden_size, self.dim, bias=False)
        self.f_b_proj = nn.Linear(self.dim, width, bias=False)
        self.b_proj = nn.Linear(c.hidden_size, self.heads, bias=False)
        self.full_rank = linear["use_full_rank_gate"]
        if self.full_rank:
            self.g_proj = nn.Linear(c.hidden_size, width, bias=False)
        else:
            self.g_a_proj = nn.Linear(c.hidden_size, self.dim, bias=False)
            self.g_b_proj = nn.Linear(self.dim, width, bias=False)
        self.o_norm = RMSNorm(self.dim, c.rms_norm_eps)
        self.o_proj = nn.Linear(width, c.hidden_size, bias=False)

    def initial_state(self, batch, device, dtype):
        taps = self.c.linear_attn_config["short_conv_kernel_size"]
        return [torch.zeros(batch, self.heads * self.dim, taps - 1, device=device, dtype=torch.float32) for _ in range(3)] + [
            torch.zeros(batch, self.heads, self.dim, self.dim, device=device, dtype=torch.float32)
        ]

    def forward(self, x, *state):
        convolved, next_state = [], []
        for name, history in zip(("q", "k", "v"), state[:3]):
            projected = getattr(self, f"{name}_proj")(x).transpose(1, 2).float()
            window = torch.cat((history, projected), dim=2)
            weight = getattr(self, f"{name}_conv1d").weight.float().squeeze(1)
            value = F.silu((window * weight).sum(2))
            convolved.append(value.reshape(-1, self.heads, self.dim))
            next_state.append(window[:, :, 1:])
        q, k, v = convolved
        # FLA's in-kernel L2 normalization uses rsqrt(sum(x*x) + 1e-6).
        q = q * torch.rsqrt(q.square().sum(2, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(2, keepdim=True) + 1e-6)
        raw = self.f_b_proj(self.f_a_proj(x)).float().reshape(-1, self.heads, self.dim)
        raw = raw + self.dt_bias.float().reshape(self.heads, self.dim)
        rate = self.A_log.float().exp().reshape(self.heads, 1)
        lower = self.c.linear_attn_config["gate_lower_bound"]
        decay = -rate * F.softplus(raw) if lower is None else lower * torch.sigmoid(rate * raw)
        matrix = state[3] * decay.exp().unsqueeze(3)
        beta = self.b_proj(x).float().sigmoid().reshape(-1, self.heads, 1)
        delta = (v - (k.unsqueeze(3) * matrix).sum(2)) * beta
        matrix = matrix + k.unsqueeze(3) * delta.unsqueeze(2)
        output = ((q * self.dim**-0.5).unsqueeze(3) * matrix).sum(2)
        gate = self.g_proj(x) if self.full_rank else self.g_b_proj(self.g_a_proj(x))
        gate = gate.float().reshape(-1, self.heads, self.dim)
        output = self.o_norm(output) * gate.sigmoid()
        output = output.to(x.dtype).reshape(-1, 1, self.heads * self.dim)
        return self.o_proj(output), [*next_state, matrix]


class _TopKIndices(torch.autograd.Function):
    """Use ONNX's lower-index-first rule for exact score ties in eager mode."""

    @staticmethod
    def forward(ctx, scores, count):
        return torch.argsort(scores, dim=1, descending=True, stable=True)[:, : int(count)].clone()

    @staticmethod
    def symbolic(g, scores, count):
        shape = g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
        count = g.op("Reshape", count, shape)
        _, indices = g.op("TopK", scores, count, axis_i=1, largest_i=1, sorted_i=1, outputs=2)
        return indices.setType(scores.type().with_dtype(torch.int64).with_sizes([None, None]))


class DSA(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        h, qdim = c.num_attention_heads, c.qk_nope_head_dim + c.qk_rope_head_dim
        if c.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(c.q_lora_rank)
            self.q_b_proj = nn.Linear(c.q_lora_rank, h * qdim, bias=False)
        else:
            self.q_proj = nn.Linear(c.hidden_size, h * qdim, bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank)
        self.kv_b_proj = nn.Linear(c.kv_lora_rank, h * (c.qk_nope_head_dim + c.v_head_dim), bias=False)
        self.g_proj = nn.Linear(c.hidden_size, h * c.v_head_dim, bias=False) if c.mla_use_output_gate else None
        self.o_proj = nn.Linear(h * c.v_head_dim, c.hidden_size, bias=False)
        d = c.dsa_config
        self.indexer = nn.Module()
        self.indexer.wq_b = nn.Linear(c.q_lora_rank or c.hidden_size, d["index_num_heads"] * d["index_head_dim"], bias=False)
        self.indexer.wk = nn.Linear(c.hidden_size, d["index_head_dim"], bias=False)
        self.indexer.weights_proj = nn.Linear(c.hidden_size, d["index_num_heads"], bias=False)
        norm = d["index_k_norm_type"]
        if norm not in {"layernorm", "rmsnorm"}:
            raise ValueError(f"Unsupported DSA key normalization {norm!r}.")
        self.indexer.k_norm = (
            nn.LayerNorm(d["index_head_dim"], eps=c.rms_norm_eps) if norm == "layernorm" else RMSNorm(d["index_head_dim"], c.rms_norm_eps)
        )

    def initial_state(self, batch, device, dtype):
        c = self.c
        return [
            torch.zeros(batch, 1, size, device=device, dtype=dtype)
            for size in (c.kv_lora_rank, c.qk_rope_head_dim, c.dsa_config["index_head_dim"])
        ]

    def forward(self, x, positions, *state):
        c, d = self.c, self.c.dsa_config
        batch, h = x.shape[0], c.num_attention_heads
        qdim = c.qk_nope_head_dim + c.qk_rope_head_dim
        q_input = self.q_a_layernorm(self.q_a_proj(x)) if c.q_lora_rank is not None else x
        q = self.q_b_proj(q_input) if c.q_lora_rank is not None else self.q_proj(x)
        q = q.reshape(batch, h, qdim)
        compressed = self.kv_a_proj_with_mqa(x)
        latent = torch.cat((state[0], self.kv_a_layernorm(compressed[:, :, : c.kv_lora_rank])), dim=1)
        shared = torch.cat((state[1], compressed[:, :, c.kv_lora_rank :]), dim=1)
        keys = torch.cat((state[2], self.indexer.k_norm(self.indexer.wk(x))), dim=1)
        iq = self.indexer.wq_b(q_input).reshape(batch, d["index_num_heads"], d["index_head_dim"])
        weights = self.indexer.weights_proj(x).float().reshape(batch, d["index_num_heads"], 1)
        scores = (torch.matmul(iq.float(), keys.float().transpose(1, 2)).relu() * weights).sum(1)
        scores = scores * (d["index_head_dim"] * d["index_num_heads"]) ** -0.5
        valid = positions >= 0
        scores = scores.masked_fill(~valid, -float("inf"))
        count = torch.minimum(torch._shape_as_tensor(scores)[1], torch.tensor(d["index_topk"], device=x.device))
        selected = _TopKIndices.apply(scores, count)

        def gather(value):
            return value.gather(1, selected.unsqueeze(2).expand(-1, -1, value.shape[2]))

        expanded = self.kv_b_proj(gather(latent)).reshape(batch, -1, h, c.qk_nope_head_dim + c.v_head_dim).transpose(1, 2)
        k_nope, value = expanded[:, :, :, : c.qk_nope_head_dim], expanded[:, :, :, c.qk_nope_head_dim :]
        key = torch.cat((k_nope, gather(shared).unsqueeze(1).expand(-1, h, -1, -1)), dim=3)
        attention = (q.unsqueeze(2).float() * key.float()).sum(3) * qdim**-0.5
        attention = attention.masked_fill(~valid.gather(1, selected).unsqueeze(1), -float("inf"))
        weighted = attention.softmax(2).to(value.dtype).unsqueeze(3) * value
        output = weighted.float().sum(2).to(value.dtype).reshape(batch, 1, h * c.v_head_dim)
        if self.g_proj is not None:
            output = output * self.g_proj(x).sigmoid()
        return self.o_proj(output), [latent, shared, keys]


class DecoderLayer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.linear = index + 1 in c.linear_attn_config["kda_layers"]
        self.self_attn = KDA(c) if self.linear else DSA(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.moe = index >= c.first_k_dense_replace and index % c.moe_layer_freq == 0
        if self.moe:
            self.block_sparse_moe = MoE(c)
        else:
            self.mlp = MLP(c)
        self.block_size = c.attn_res_block_size
        if self.block_size is not None:
            self.self_attention_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
            self.mlp_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
            self.self_attention_res_proj = nn.Linear(c.hidden_size, 1, bias=False)
            self.mlp_res_proj = nn.Linear(c.hidden_size, 1, bias=False)


def mix_residual(prefix, blocks, norm, projection):
    values = torch.stack([*blocks, prefix], dim=2).float()
    keys = values * torch.rsqrt(values.square().mean(3, keepdim=True) + norm.eps)
    scores = (keys * norm.weight.float() * projection.weight.float()).sum(3).softmax(2)
    return (values * scores.unsqueeze(3)).sum(2).to(prefix.dtype)


class KimiK31ForConditionalGeneration(nn.Module):
    """Native checkpoint names, one-token decoder and optional vision tower."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config.text_config
        self.language_model = nn.Module()
        model = self.language_model.model = nn.Module()
        model.embed_tokens = nn.Embedding(c.vocab_size, c.hidden_size)
        model.layers = nn.ModuleList([DecoderLayer(c, i) for i in range(c.num_hidden_layers)])
        model.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        if c.attn_res_block_size is not None:
            model.output_attn_res_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
            model.output_attn_res_proj = nn.Linear(c.hidden_size, 1, bias=False)
        self.language_model.lm_head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        if config.vision_config is not None:
            from .kimi_k31_vision import VisionTower, VisionProjector

            vision = SimpleNamespace(**config.vision_config)
            self.vision_tower = VisionTower(vision)
            self.mm_projector = VisionProjector(vision, c.hidden_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d, nn.Conv2d)):
            nn.init.normal_(module.weight, std=self.config.text_config.initializer_range)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    @property
    def state_names(self):
        names = ["positions"]
        for i, layer in enumerate(self.language_model.model.layers):
            suffixes = ("conv_q", "conv_k", "conv_v", "recurrent") if layer.linear else ("latent", "shared_key", "index_key")
            names.extend(f"layer_{i}_{suffix}" for suffix in suffixes)
        return names

    def initial_state(self, batch_size):
        weight = self.language_model.model.embed_tokens.weight
        result = [torch.full((batch_size, 1), -1, dtype=torch.int64, device=weight.device)]
        for layer in self.language_model.model.layers:
            result.extend(layer.self_attn.initial_state(batch_size, weight.device, weight.dtype))
        return result

    def forward(self, input_ids, position_ids, token_types, inputs_embeds, *state):
        if not torch.jit.is_tracing():
            if input_ids.ndim != 2 or input_ids.shape[1] != 1:
                raise ValueError("The Kimi decoder consumes [batch, 1] tokens; use KimiK31Session for prefill.")
            if len(state) != len(self.state_names):
                raise ValueError("Kimi decoder state does not match the configured layer schedule.")
        model = self.language_model.model
        x = model.embed_tokens(input_ids)
        x = torch.where((token_types == 1).unsqueeze(2), inputs_embeds, x)
        positions = torch.cat((state[0], position_ids), dim=1)
        present, blocks, offset = [positions], [], 1
        for i, layer in enumerate(model.layers):
            prefix = x
            if blocks:
                x = mix_residual(prefix, blocks, layer.self_attention_res_norm, layer.self_attention_res_proj)
            if layer.block_size is not None and i % layer.block_size == 0:
                blocks.append(prefix)
                prefix = None
            length = 4 if layer.linear else 3
            attention = layer.self_attn
            normalized = layer.input_layernorm(x)
            args = (normalized,) if layer.linear else (normalized, positions)
            output, next_state = attention(*args, *state[offset : offset + length])
            offset += length
            present.extend(next_state)
            prefix = output if prefix is None else prefix + output
            mlp_input = mix_residual(prefix, blocks, layer.mlp_res_norm, layer.mlp_res_proj) if blocks else prefix
            mlp = layer.block_sparse_moe if layer.moe else layer.mlp
            x = prefix + mlp(layer.post_attention_layernorm(mlp_input))
        if blocks:
            x = mix_residual(x, blocks, model.output_attn_res_norm, model.output_attn_res_proj)
        return (self.language_model.lm_head(model.norm(x)), *present)

    def encode_media(self, patches, grid_shape):
        return self.mm_projector(self.vision_tower(patches, grid_shape))
