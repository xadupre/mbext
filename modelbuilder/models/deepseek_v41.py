# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Independent, single-device DeepSeek-V4.1-Flash reference.

Architecture and numerics follow the MIT-licensed public reference at
https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/main/inference.
Unlike that implementation, all persistent inference state is explicit, ordinary
floating point tensors work on CPU, and no distributed or custom kernels are used.

``forward(ids, positions, types, image_embeddings, *state)`` returns
``(logits, *present)``. Positions are contiguous, zero-based, and synchronized
across the batch; padding is not supported. Types use the official convention:
-1=text, 0=image start, 1=image feature, 2=newline, 3=image end. Only feature
positions read image_embeddings. Delimiters use learned checkpoint parameters.
State tensors must be passed unchanged between calls (except for beam reordering
along their batch axis). SWA and incomplete groups are bounded; only completed
compressed latents/index keys grow with context. MTP/speculative draft heads are
not part of this causal language-model reference.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, PretrainedConfig
from transformers import PreTrainedModel


class DeepseekV41TextConfig(PretrainedConfig):
    model_type = "deepseek_v41_text"
    # Keep the published schema rather than Transformers' newer RoPE alias,
    # which rewrites these fields before custom dimensions have been assigned.
    rope_scaling = None

    def __init__(self, **kwargs):
        defaults = dict(
            vocab_size=129280,
            hidden_size=5120,
            moe_intermediate_size=2304,
            num_hidden_layers=40,
            num_attention_heads=64,
            num_key_value_heads=1,
            head_dim=512,
            qk_rope_head_dim=64,
            q_lora_rank=1280,
            o_lora_rank=1024,
            o_groups=8,
            hidden_act="silu",
            swiglu_limit=10.0,
            rms_norm_eps=1e-20,
            attention_bias=False,
            attention_dropout=0.0,
            initializer_range=0.02,
            use_cache=True,
            tie_word_embeddings=False,
            max_position_embeddings=1048576,
            rope_theta=10000.0,
            rope_scaling=dict(rope_type="yarn", factor=16, beta_fast=32, beta_slow=1, original_max_position_embeddings=65536),
            n_routed_experts=384,
            n_shared_experts=1,
            num_experts_per_tok=6,
            scoring_func="sqrtsoftplus",
            topk_method="noaux_tc",
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            gate_temp=1.0,
            sliding_window=128,
            compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
            compress_rope_theta=160000.0,
            kv_source_layer_ids=[2, 8, 14, 20],
            index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
            index_n_heads=32,
            index_head_dim=128,
            index_topk=512,
            candidate_source_layer_id=20,
            candidate_topk_blocks=2048,
            candidate_block_size=8,
            hc_mult=4,
            hc_sinkhorn_iters=20,
            hc_eps=1e-6,
            engram_layer_ids=[1, 14],
            engram_num_embeddings=[384006168, 384016682],
            engram_max_ngram_size=4,
            engram_vocab_size=16000000,
            engram_n_heads=8,
            engram_head_dim=256,
            engram_pad_token_id=2,
            engram_compressed_vocab_size=99092,
            num_nextn_predict_layers=3,
            dspark_block_size=5,
            dspark_noise_token_id=128799,
            dspark_target_layer_ids=[37, 38, 39],
            dspark_markov_rank=256,
            dspark_n_routed_experts=128,
            dspark_num_experts_per_tok=3,
        )
        defaults.update(kwargs)
        self.rope_scaling = defaults.pop("rope_scaling")
        self.rope_theta = defaults.pop("rope_theta")
        super().__init__(**defaults)


class DeepseekV41VisionConfig(PretrainedConfig):
    model_type = "deepseek_v41_vision"

    def __init__(self, **kwargs):
        defaults = dict(
            num_hidden_layers=32,
            hidden_size=1024,
            num_attention_heads=16,
            intermediate_size=2816,
            patch_size=14,
            rope_theta=10000.0,
            downsample_ratio=3,
            max_image_tokens=1024,
            min_pixels=295936,
            max_wh_ratio=None,
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


class DeepseekV41Config(PretrainedConfig):
    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig, "vision_config": DeepseekV41VisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        self.text_config = text_config if isinstance(text_config, DeepseekV41TextConfig) else DeepseekV41TextConfig(**(text_config or {}))
        self.vision_config = (
            vision_config if isinstance(vision_config, DeepseekV41VisionConfig) else DeepseekV41VisionConfig(**(vision_config or {}))
        )
        defaults = dict(
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=2,
            image_token_id=129264,
            tie_word_embeddings=False,
            architectures=["DeepseekV41ForCausalLM"],
        )
        defaults.update(kwargs)
        super().__init__(**defaults)


def _fp8_values(x):
    """E4M3FN, round-to-nearest-even, including subnormals; no float8 operators."""
    a = x.float().abs().clamp(max=448)
    step = torch.pow(2.0, torch.floor(torch.log2(a.clamp_min(2**-126))).clamp_min(-6) - 3)
    return torch.sign(x) * torch.round(a / step) * step


def _fp4_values(x):
    """E2M1FN's eight magnitudes (0,.5,1,1.5,2,3,4,6), ties to even."""
    a = x.float().abs().clamp(max=6)
    step = torch.pow(2.0, torch.floor(torch.log2(a.clamp_min(2**-126))).clamp_min(0) - 1)
    return torch.sign(x) * torch.round(a / step) * step


class _Reshape(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shape):
        return torch._reshape_from_tensor(x, shape)

    @staticmethod
    def symbolic(g, x, shape):
        result = g.op("Reshape", x, shape, allowzero_i=1)
        return result.setType(x.type().with_sizes([None] * shape.type().sizes()[0]))


def _reshape(x, *shape):
    # Unlike ONNX's default reshape, these dimensions preserve empty caches.
    if not torch.jit.is_tracing():
        return x.reshape(shape)
    dimensions = torch.stack([(size if isinstance(size, torch.Tensor) else torch.scalar_tensor(size, dtype=torch.int64)) for size in shape])
    return _Reshape.apply(x, dimensions)


def _split_last(x, first, second):
    # Explicit reshape preserves rank in the legacy exporter's symbolic path,
    # including zero-length compressed banks; aten::unflatten does not.
    if second == -1:
        second = x.shape[-1] // first
    elif first == -1:
        first = x.shape[-1] // second
    return _reshape(x, *x.shape[:-1], first, second)


def quantize_roundtrip(x, bits=8, block_size=32, scale_format="ue8m0"):
    """Official cache/activation quantization, including the two FP4 scale formats.

    Partial blocks are zero-padded, allowing tiny reference dimensions without
    changing full-block checkpoint numerics. The returned dtype matches ``x``.
    """
    width = x.shape[-1]
    blocks = _split_last(F.pad(x.float(), (0, -width % block_size)), -1, block_size)
    amax = blocks.abs().amax(blocks.ndim - 1, keepdim=True)
    if bits == 8:
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(1e-4) / 448)))
        values = _fp8_values(blocks / scale)
    elif bits == 4:
        if scale_format == "e4m3":
            scale = _fp8_values(amax.clamp_min(6 * 2**-9) / 6)
        else:
            scale = torch.pow(2.0, torch.ceil(torch.log2(amax.clamp_min(6 * 2**-126) / 6)))
        values = _fp4_values(blocks / scale)
    else:
        raise ValueError(f"Unsupported quantization width {bits}")
    restored = _reshape(values * scale, *x.shape[:-1], width + (-width % block_size))
    return restored[..., :width].to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        z = x.float()
        return (z * torch.rsqrt(z.square().mean(z.ndim - 1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class Linear(nn.Linear):
    """Floating checkpoint weight, optionally with official MXFP8 activations.

    Checkpoint loaders set ``quantized=True`` after dequantizing FP8/FP4 weights.
    Grouped ``wo_a`` is used directly, with its input quantized by Attention.
    """

    quantized = False

    def forward(self, x):
        if self.quantized:
            x = quantize_roundtrip(x, 8, 32)
        return F.linear(x, self.weight, self.bias)


class EngramEmbedding(nn.Embedding):
    """Rowwise embedding lookup; optional FP8 storage avoids dense dequantization.

    A loader may assign a frozen FP8 ``weight`` and its [rows, dim/32] ``scale``.
    Scale values are decoded floating values, not raw E8M0 exponent bytes.
    """

    def __init__(self, rows, dim):
        super().__init__(rows, dim)
        self.register_parameter("scale", None)
        self.output_dtype = torch.float32
        self.register_buffer(
            "fp8_lut",
            torch.arange(256, device="cpu", dtype=torch.int64).to(torch.uint8).view(torch.float8_e4m3fn).float(),
            persistent=False,
        )

    def load_quantized(self, weight, scale):
        """Install checkpoint FP8 rows and E8M0 scales without expanding the table."""
        self.output_dtype = self.weight.dtype if self.weight.is_floating_point() else torch.float32
        if weight.dtype == torch.float8_e4m3fn:
            weight = weight.view(torch.uint8)
        if scale.dtype == torch.float8_e8m0fnu:
            scale = scale.view(torch.uint8)
        if weight.ndim != 2 or scale.ndim != 2:
            raise ValueError("Engram weights/scales must be rank two")
        if weight.shape[0] != scale.shape[0] or weight.shape[1] % scale.shape[1]:
            raise ValueError("Engram scales must evenly partition each embedding row")
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.scale = nn.Parameter(scale, requires_grad=False)

    def forward(self, indices):
        if self.scale is None:
            return F.embedding(indices, self.weight)
        values = F.embedding(indices, self.weight)
        if self.weight.dtype == torch.uint8:
            values = self.fp8_lut[values.long()]
        else:
            values = values.float()
        scales = F.embedding(indices, self.scale).float()
        if self.scale.dtype == torch.uint8:
            scales = torch.where(scales == 255, torch.full_like(scales, torch.nan), torch.pow(2.0, scales - 127))
        values = _split_last(values, scales.shape[-1], -1)
        return (values * scales.unsqueeze(-1)).reshape(*indices.shape, self.embedding_dim).to(self.output_dtype)


class _RotaryApply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, positions, inv_freq, dim, inverse):
        angle = positions.float().unsqueeze(-1) * inv_freq
        if x.ndim == 4:
            angle = angle.unsqueeze(-2)
        cos, sin = angle.cos(), angle.sin()
        if inverse:
            sin = -sin
        tail = _split_last(x[..., -dim:].float(), dim // 2, 2)
        re, im = tail[..., 0], tail[..., 1]
        tail = torch.stack((re * cos - im * sin, re * sin + im * cos), -1)
        return torch.cat((x[..., :-dim], tail.reshape(*x.shape[:-1], dim).to(x.dtype)), -1)

    @staticmethod
    def symbolic(graph, x, positions, inv_freq, dim, inverse):
        # Explicit axes and output type avoid the legacy exporter's incorrect
        # inferred Einsum ranks propagating into stack/negative-axis flatten.
        def const(value):
            return graph.op("Constant", value_t=torch.tensor(value, dtype=torch.int64))

        def unsqueeze(value, axis):
            return graph.op("Unsqueeze", value, const([axis]))

        shape = graph.op("Shape", x)
        prefix_shape = graph.op("Slice", shape, const([0]), const([-1]), const([0]))
        angle = graph.op("Mul", unsqueeze(graph.op("Cast", positions, to_i=1), -1), graph.op("Cast", inv_freq, to_i=1))
        if x.type().dim() == 4:
            angle = unsqueeze(angle, -2)
        cos, sin = graph.op("Cos", angle), graph.op("Sin", angle)
        if inverse:
            sin = graph.op("Neg", sin)
        tail = graph.op("Slice", graph.op("Cast", x, to_i=1), const([-dim]), const([9223372036854775807]), const([-1]))
        pair_shape = graph.op("Concat", prefix_shape, const([dim // 2, 2]), axis_i=0)
        tail = graph.op("Reshape", tail, pair_shape, allowzero_i=1)
        re = graph.op("Gather", tail, const(0), axis_i=-1)
        im = graph.op("Gather", tail, const(1), axis_i=-1)
        real = graph.op("Sub", graph.op("Mul", re, cos), graph.op("Mul", im, sin))
        imaginary = graph.op("Add", graph.op("Mul", re, sin), graph.op("Mul", im, cos))
        paired = graph.op("Concat", unsqueeze(real, -1), unsqueeze(imaginary, -1), axis_i=-1)
        tail_shape = graph.op("Concat", prefix_shape, const([dim]), axis_i=0)
        tail = graph.op("Reshape", paired, tail_shape, allowzero_i=1)
        tail = graph.op("CastLike", tail, x)
        prefix = graph.op("Slice", x, const([0]), const([-dim]), const([-1]))
        return graph.op("Concat", prefix, tail, axis_i=-1).setType(x.type())


class Rotary(nn.Module):
    def __init__(self, config, compressed=False):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        base = config.compress_rope_theta if compressed else config.rope_theta
        freq = base ** (-torch.arange(0, self.dim, 2, device="cpu").float() / self.dim)
        scaling = config.rope_scaling if compressed else None
        if scaling and scaling.get("original_max_position_embeddings", 0) > 0:
            original = scaling["original_max_position_embeddings"]

            def correction(rotations):
                return self.dim * math.log(original / (rotations * 2 * math.pi)) / (2 * math.log(base))

            low = max(math.floor(correction(scaling.get("beta_fast", 32))), 0)
            high = min(math.ceil(correction(scaling.get("beta_slow", 1))), self.dim - 1)
            ramp = ((torch.arange(self.dim // 2, device="cpu") - low) / max(high - low, 1e-3)).clamp(0, 1)
            freq = freq * (1 - ramp) + freq / scaling["factor"] * ramp
        self.register_buffer("inv_freq", freq, persistent=False)

    def forward(self, x, positions, inverse=False):
        return _RotaryApply.apply(x, positions, self.inv_freq, self.dim, inverse)


def sinkhorn_mixes(x, weight, scale, base, hc_mult, iterations, eps, norm_eps):
    flat = x.flatten(-2).float()
    mixes = F.linear(flat, weight.float()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    pre = torch.sigmoid(mixes[..., :hc_mult] * scale[0] + base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc_mult : 2 * hc_mult] * scale[1] + base[hc_mult : 2 * hc_mult])
    comb = _split_last(mixes[..., 2 * hc_mult :] * scale[2] + base[2 * hc_mult :], hc_mult, hc_mult)
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def _stable_topk(scores, k):
    """Position-stable top-k, without perturbing distinct float32 scores.

    ReLU index scores frequently tie at zero. Native topk breaks those ties
    differently when a later prefill adds masked columns, changing causal
    outputs. Add less than half a float32 ULP in float64, preferring older
    positions. This remains exportable as ordinary arithmetic and TopK.
    """
    value = scores.double()
    step = value.abs().clamp_min(2**-126) * 2**-25
    width = scores.shape[-1]
    index = torch.arange(width, device=scores.device, dtype=torch.float64)
    ranked = value + step * ((width - index) / (width + 1))
    ranked = torch.where(torch.isfinite(value), ranked, value)
    indices = F.pad(ranked, (0, k), value=-torch.inf).topk(k, dim=-1, sorted=False).indices
    values = F.pad(scores, (0, k), value=-torch.inf).gather(-1, indices)
    shape = (*scores.shape[:-1], k)
    return values.reshape(shape), indices.reshape(shape)


def _topk_mask(scores, k):
    # Padding makes zero-length caches and short prefills legal with a static k.
    padded = F.pad(scores, (0, k), value=-torch.inf)
    values, indices = _stable_topk(scores, k)
    mask = torch.zeros_like(padded, dtype=torch.int64).scatter(-1, indices, (values > -torch.inf).to(torch.int64))
    return mask[..., : scores.shape[-1]].bool().reshape(scores.shape)


def select_candidate_blocks(scores, compress_lens, topk_blocks, block_size):
    width = scores.shape[-1]
    blocks = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    blocks = _split_last(blocks, -1, block_size)
    if blocks.dtype == torch.bfloat16:
        # CUDA ReduceMax has no BF16 kernel; widening is exact for selection.
        blocks = blocks.float()
    blocks = blocks.amax(blocks.ndim - 1)
    newest = (compress_lens - 1) // block_size
    indices = torch.arange(blocks.shape[-1], device=scores.device)
    blocks = torch.where(indices == newest, torch.full_like(blocks, torch.inf), blocks)
    # Legacy ONNX lowering mutates a shared negative-axis constant in place.
    return _topk_mask(blocks, topk_blocks).repeat_interleave(block_size, scores.ndim - 1)[..., :width]


def build_compressed_token_map(tokenizer):
    """Training normalization, including whitespace and partial UTF-8 tokens."""
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalize = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    keys, mapping = {}, []
    for i in range(len(tokenizer)):
        text = backend.decode([i], skip_special_tokens=False)
        key = backend.id_to_token(i) if "\ufffd" in text else normalize.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        mapping.append(keys[key])
    return mapping, len(keys)


def _next_prime(start, seen):
    candidate = max(2, start + 1)
    while candidate in seen or any(candidate % divisor == 0 for divisor in range(2, math.isqrt(candidate) + 1)):
        candidate += 1
    return candidate


class _BitwiseXor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second):
        return torch.bitwise_xor(first, second)

    @staticmethod
    def symbolic(graph, first, second):
        return graph.op("BitwiseXor", first, second).setType(first.type())


class NgramHash(nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.size = config.engram_max_ngram_size
        self.layers = tuple(config.engram_layer_ids)
        mapping, vocab = build_compressed_token_map(tokenizer)
        if vocab != config.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary is {vocab}, expected " f"{config.engram_compressed_vocab_size}")
        self.pad_id = mapping[config.engram_pad_token_id]
        seen, primes, offsets, multipliers = set(), [], [], []
        for layer in self.layers:
            row, current_offsets, offset = [], [], 0
            for _ in range(self.size - 1):
                current = config.engram_vocab_size - 1
                group = []
                for _ in range(config.engram_n_heads):
                    current = _next_prime(current, seen)
                    seen.add(current)
                    group.append(current)
                    current_offsets.append(offset)
                    offset += current
                row.append(group)
            primes.append(row)
            offsets.append(current_offsets)
            rng = np.random.default_rng(10007 * layer)
            bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
            multipliers.append(rng.integers(0, bound, size=self.size, dtype=np.int64) * 2 + 1)
        for name, value in (("token_map", mapping), ("primes", primes), ("offsets", offsets), ("multipliers", np.array(multipliers))):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.int64, device="cpu"), persistent=False)
        self.required_rows = tuple(sum(sum(group) for group in row) for row in primes)

    def forward(self, ids, types, history):
        tokens = torch.where(types < 0, self.token_map[ids], -1)
        joined = torch.cat((history, tokens), 1)
        positions = torch.arange(ids.shape[1], device=ids.device) + self.size - 1
        blocked = torch.zeros_like(ids, dtype=torch.bool)
        lookback = []
        for shift in range(self.size):
            source = joined[:, positions - shift]
            blocked = blocked | (source == -1)
            lookback.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(lookback, -1).unsqueeze(-2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.size):
            rolling = _BitwiseXor.apply(rolling, products[..., shift])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, shift - 1])
        hashes = torch.cat(hashes, -1) + self.offsets
        return hashes, joined[:, -(self.size - 1) :]


class Engram(nn.Module):
    def __init__(self, config, rows):
        super().__init__()
        c = config
        self.dim, self.hc_mult = c.hidden_size, c.hc_mult
        self.eps = c.rms_norm_eps
        self.embed = EngramEmbedding(rows, c.engram_head_dim)
        columns = (c.engram_max_ngram_size - 1) * c.engram_n_heads
        self.wkv = Linear(columns * c.engram_head_dim, self.dim * (self.hc_mult + 1), bias=False)
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.dim))

    def forward(self, x, hashes, mask):
        kv = self.wkv(self.embed(hashes).flatten(-2))
        key, value = kv.split((self.hc_mult * self.dim, self.dim), -1)
        key = _split_last(key.float(), self.hc_mult, self.dim)
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * self.q_weight * self.k_weight * key).sum(-1)
        dot = dot * rstd * self.dim**-0.5
        signed = torch.sign(dot) * dot.abs().clamp_min(1e-6).sqrt()
        # copysign(+sqrt(eps), +0) is positive, rather than sign(0)==0.
        signed = torch.where(dot == 0, torch.full_like(dot, 1e-3), signed)
        gate = torch.sigmoid(signed) * mask.unsqueeze(-1)
        return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


class Compressor(nn.Module):
    def __init__(self, config, ratio):
        super().__init__()
        self.ratio = ratio
        self.wkv = Linear(config.hidden_size, config.head_dim, bias=False)
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        if ratio > 1:
            self.wgate = Linear(config.hidden_size, config.head_dim, bias=False)

    def forward(self, x, positions, old_kv, old_score):
        ratio = self.ratio
        if ratio == 1:
            return self.norm(self.wkv(x)), positions, old_kv, old_score
        kv = F.linear(x.float(), self.wkv.weight.float())
        scores = F.linear(x.float(), self.wgate.weight.float())
        values = torch.cat((old_kv, kv), 1)
        gates = torch.cat((old_score, scores), 1)
        start = positions[0, 0]
        first = torch.div(start, ratio, rounding_mode="floor")
        count = torch.div(start + x.shape[1], ratio, rounding_mode="floor") - first
        groups = torch.arange(count, device=x.device)
        offsets = groups[:, None] * ratio - start.remainder(ratio) + torch.arange(ratio, device=x.device) + ratio - 1
        # ORT 1.31 preserves the wrong rank for empty reductions with negative axes.
        pooled = (values[:, offsets] * gates[:, offsets].softmax(-2)).sum(2)
        pos = ((groups + first) * ratio).unsqueeze(0).expand(x.shape[0], -1)
        return (self.norm(pooled.to(x.dtype)), pos, values[:, -(ratio - 1) :], gates[:, -(ratio - 1) :])


class Indexer(nn.Module):
    def __init__(self, config, owns_k):
        super().__init__()
        c = config
        self.heads, self.dim = c.index_n_heads, c.index_head_dim
        self.wq_b = Linear(c.q_lora_rank, self.heads * self.dim, bias=False)
        self.weights_proj = Linear(c.hidden_size, self.heads, bias=False)
        if owns_k:
            self.wk = Linear(c.head_dim, self.dim, bias=False)
            self.k_norm = RMSNorm(self.dim, c.rms_norm_eps)

    def keys(self, latent, positions, rotary):
        k = rotary(self.k_norm(self.wk(latent)), positions)
        return quantize_roundtrip(k, 4, 32)

    def forward(self, x, qr, keys, positions, rotary):
        q = _split_last(self.wq_b(qr), self.heads, self.dim)
        q = quantize_roundtrip(rotary(q, positions), 4, 32)
        weights = self.weights_proj(x) * (self.dim**-0.5 * self.heads**-0.5)
        scores = torch.matmul(q.flatten(1, 2), keys.transpose(1, 2))
        scores = _reshape(scores, x.shape[0], x.shape[1], self.heads, keys.shape[1]).relu()
        return (scores * weights.unsqueeze(-1)).sum(2)


class Gate(nn.Module):
    def __init__(self, config, vision_enabled):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.n_routed_experts, config.hidden_size))
        self.bias = nn.Parameter(torch.zeros(config.n_routed_experts))
        if vision_enabled:
            self.bias_vl = nn.Parameter(torch.zeros(config.n_routed_experts))
        else:
            self.register_parameter("bias_vl", None)

    def forward(self, x, image_mask):
        c = self.config
        scores = F.linear(x.float(), self.weight.float()) / c.gate_temp
        if c.scoring_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif c.scoring_func == "sigmoid":
            scores = scores.sigmoid()
        elif c.scoring_func == "softmax":
            scores = scores.softmax(-1)
        else:
            raise ValueError(f"Unknown routing score {c.scoring_func}")
        bias = self.bias
        if self.bias_vl is not None:
            bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, self.bias)
        indices = (scores + bias).topk(c.num_experts_per_tok, -1).indices
        weights = scores.gather(-1, indices)
        if c.norm_topk_prob and c.num_experts_per_tok > 1:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return weights * c.routed_scaling_factor, indices


class Expert(nn.Module):
    def __init__(self, dim, intermediate, limit):
        super().__init__()
        self.w1 = Linear(dim, intermediate, bias=False)
        self.w2 = Linear(intermediate, dim, bias=False)
        self.w3 = Linear(dim, intermediate, bias=False)
        self.limit = limit

    def forward(self, x, weights=None):
        gate, up = self.w1(x).float(), self.w3(x).float()
        if self.limit > 0:
            gate = gate.clamp(max=self.limit)
            up = up.clamp(-self.limit, self.limit)
        h = F.silu(gate) * up
        if weights is not None:
            h = h * weights
        return self.w2(h.to(x.dtype))


class MoE(nn.Module):
    def __init__(self, config, vision_enabled):
        super().__init__()
        c = config
        self.gate = Gate(c, vision_enabled)
        self.experts = nn.ModuleList([Expert(c.hidden_size, c.moe_intermediate_size, c.swiglu_limit) for _ in range(c.n_routed_experts)])
        self.shared_experts = Expert(c.hidden_size, c.moe_intermediate_size * c.n_shared_experts, c.swiglu_limit)

    def forward(self, x, image_mask):
        shape = x.shape
        x = x.flatten(0, -2)
        weights, indices = self.gate(x, image_mask.flatten())
        result = self.shared_experts(x).float()
        # Empty expert batches are legal. Tensor-only gather/scatter avoids both
        # evaluating every expert on every token and data-dependent Python.
        for index, expert in enumerate(self.experts):
            tokens, slots = torch.where(indices == index)
            output = expert(x[tokens], weights[tokens, slots, None]).float()
            result = result.scatter_add(0, tokens[:, None].expand(-1, x.shape[-1]), output)
        return result.to(x.dtype).reshape(shape)


class Attention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        c = config
        self.config, self.layer_id = c, layer_id
        self.ratio = c.compress_ratios[layer_id]
        self.is_kv_source = layer_id in c.kv_source_layer_ids
        self.is_index_source = layer_id in c.index_source_layer_ids
        self.attn_sink = nn.Parameter(torch.zeros(c.num_attention_heads))
        self.wq_a = Linear(c.hidden_size, c.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.wq_b = Linear(c.q_lora_rank, c.num_attention_heads * c.head_dim, bias=False)
        self.wkv = Linear(c.hidden_size, c.head_dim, bias=False)
        self.kv_norm = RMSNorm(c.head_dim, c.rms_norm_eps)
        self.wo_a = Linear(c.num_attention_heads * c.head_dim // c.o_groups, c.o_groups * c.o_lora_rank, bias=False)
        self.wo_b = Linear(c.o_groups * c.o_lora_rank, c.hidden_size, bias=False)
        self.rotary = Rotary(c, compressed=self.ratio > 0)
        if self.is_kv_source:
            self.compressor = Compressor(c, self.ratio)
        if self.is_index_source:
            self.indexer = Indexer(c, self.is_kv_source)

    def forward(self, x, positions, states, shared):
        c = self.config
        window, window_pos = states[:2]
        qr = self.q_norm(self.wq_a(x))
        q = _split_last(self.wq_b(qr), c.num_attention_heads, c.head_dim)
        q = self.rotary(q, positions)
        kv = quantize_roundtrip(self.rotary(self.kv_norm(self.wkv(x)), positions), 8, 32)
        window = torch.cat((window, kv), 1)
        window_pos = torch.cat((window_pos, positions), 1)
        allowed = (
            (window_pos[:, None, :] >= positions[:, :, None] - c.sliding_window + 1)
            & (window_pos[:, None, :] <= positions[:, :, None])
            & (window_pos[:, None, :] >= 0)
        )
        present = [window[:, -c.sliding_window :], window_pos[:, -c.sliding_window :]]
        if self.ratio:
            if self.is_kv_source:
                compressed, keys, partial, scores = states[2:]
                latent, latent_pos, partial, scores = self.compressor(x, positions, partial, scores)
                keys = torch.cat((keys, self.indexer.keys(latent, latent_pos, self.rotary)), 1)
                new_kv = quantize_roundtrip(self.rotary(latent, latent_pos), 4, 16, "e4m3")
                compressed = torch.cat((compressed, new_kv), 1)
                present.extend((compressed, keys, partial, scores))
                shared["compressed"], shared["keys"] = compressed, keys
            compressed, keys = shared["compressed"], shared["keys"]
            visible = torch.div(positions + 1, self.ratio, rounding_mode="floor").unsqueeze(-1)
            indices = torch.arange(compressed.shape[1], device=x.device)
            reachable = indices < visible
            if self.is_index_source:
                index_scores = self.indexer(x, qr, keys, positions, self.rotary)
                index_scores = index_scores.masked_fill(~reachable, -torch.inf)
                if self.layer_id == c.candidate_source_layer_id:
                    shared["candidates"] = select_candidate_blocks(index_scores, visible, c.candidate_topk_blocks, c.candidate_block_size)
                elif 0 <= c.candidate_source_layer_id < self.layer_id:
                    index_scores = index_scores.masked_fill(~shared["candidates"], -torch.inf)
                values, selected = _stable_topk(index_scores, c.index_topk)
                order = selected.argsort(-1)
                shared["selected"] = (selected.gather(-1, order), values.gather(-1, order) > -torch.inf)
            selected, selected_valid = shared["selected"]
            padded = F.pad(compressed, (0, 0, 0, c.index_topk))
            bank = padded[:, None].expand(-1, x.shape[1], -1, -1)
            selected_kv = bank.gather(2, selected.unsqueeze(-1).expand(-1, -1, -1, c.head_dim))
        logits = torch.matmul(q.float().flatten(1, 2), window.float().transpose(1, 2))
        logits = _reshape(logits, x.shape[0], x.shape[1], c.num_attention_heads, window.shape[1])
        logits = (logits * c.head_dim**-0.5).masked_fill(~allowed.unsqueeze(-2), -torch.inf)
        if self.ratio:
            sparse_logits = torch.matmul(q.float(), selected_kv.float().transpose(-1, -2))
            sparse_logits = sparse_logits.reshape(x.shape[0], x.shape[1], c.num_attention_heads, c.index_topk)
            selected_valid = selected_valid.reshape(x.shape[0], x.shape[1], c.index_topk)
            sparse_logits = (sparse_logits * c.head_dim**-0.5).masked_fill(~selected_valid.unsqueeze(-2), -torch.inf)
            logits = torch.cat((logits, sparse_logits), -1)
        sink = self.attn_sink.view(1, 1, -1, 1).expand(x.shape[0], x.shape[1], -1, -1)
        probability = torch.cat((logits, sink), -1).softmax(-1)[..., :-1]
        output = torch.matmul(probability[..., : window.shape[1]].flatten(1, 2), window.float()).reshape(
            x.shape[0], x.shape[1], c.num_attention_heads, c.head_dim
        )
        if self.ratio:
            output = output + torch.matmul(probability[..., window.shape[1] :], selected_kv.float())
        output = output.reshape(x.shape[0], x.shape[1], c.num_attention_heads, c.head_dim)
        output = self.rotary(output.to(x.dtype), positions, inverse=True)
        output = _split_last(output.flatten(-2), c.o_groups, -1)
        output = quantize_roundtrip(output, 8, 32)
        low_rank = self.wo_a.weight.reshape(c.o_groups, c.o_lora_rank, -1)
        output = torch.matmul(output.transpose(1, 2), low_rank.transpose(1, 2)).transpose(1, 2)
        output = output.reshape(x.shape[0], x.shape[1], c.o_groups * c.o_lora_rank)
        return self.wo_b(output), tuple(present)


class Block(nn.Module):
    def __init__(self, config, layer_id, vision_enabled, engram_rows=None):
        super().__init__()
        self.config = config
        c = config
        self.attn = Attention(c, layer_id)
        self.ffn = MoE(c, vision_enabled)
        self.attn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.ffn_norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        size = (2 + c.hc_mult) * c.hc_mult
        for kind in ("attn", "ffn"):
            self.register_parameter(f"hc_{kind}_fn", nn.Parameter(torch.empty(size, c.hc_mult * c.hidden_size)))
            self.register_parameter(f"hc_{kind}_base", nn.Parameter(torch.zeros(size)))
            self.register_parameter(f"hc_{kind}_scale", nn.Parameter(torch.full((3,), 0.01)))
        self.engram = None if engram_rows is None else Engram(c, engram_rows)

    def mixes(self, x, kind):
        c = self.config
        return sinkhorn_mixes(
            x,
            getattr(self, f"hc_{kind}_fn"),
            getattr(self, f"hc_{kind}_scale"),
            getattr(self, f"hc_{kind}_base"),
            c.hc_mult,
            c.hc_sinkhorn_iters,
            c.hc_eps,
            c.rms_norm_eps,
        )

    @staticmethod
    def collapse(x, pre):
        return (x.float() * pre.unsqueeze(-1)).sum(-2).to(x.dtype)

    @staticmethod
    def expand(output, residual, post, comb):
        # Checkpoint comb is [source copy, destination copy], not its transpose.
        return (post.unsqueeze(-1) * output.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual.float())).to(output.dtype)

    def forward(self, x, pre, positions, image_mask, states, shared):
        attn_pre, post, comb = self.mixes(x, "attn")
        output, present = self.attn(self.attn_norm(self.collapse(x, pre)), positions, states, shared)
        x = self.expand(output, x, post, comb)
        ffn_pre, post, comb = self.mixes(x, "ffn")
        output = self.ffn(self.ffn_norm(self.collapse(x, attn_pre)), image_mask)
        return self.expand(output, x, post, comb), ffn_pre, present


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.scale = (config.hidden_size // self.heads) ** -0.5
        self.wqkv = nn.Linear(config.hidden_size, config.hidden_size * 3)
        self.wo = nn.Linear(config.hidden_size, config.hidden_size)

    @staticmethod
    def rotate(x, angle):
        first, second = x.float().chunk(2, -1)
        cos, sin = angle.cos().unsqueeze(-2), angle.sin().unsqueeze(-2)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), -1).to(x.dtype)

    def forward(self, x, angle):
        q, k, v = [_split_last(t, self.heads, -1) for t in self.wqkv(x).chunk(3, -1)]
        q, k = self.rotate(q, angle), self.rotate(k, angle)
        scores = torch.matmul(q.transpose(0, 1), k.permute(1, 2, 0)) * self.scale
        output = torch.matmul(scores.softmax(-1), v.transpose(0, 1)).transpose(0, 1)
        return self.wo(output.flatten(-2))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.w1(x).chunk(2, -1)
        return self.w2(F.silu(gate) * up)


class VisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size)
        self.attn = VisionAttention(config)
        self.norm2 = RMSNorm(config.hidden_size)
        self.mlp = VisionMLP(config)

    def forward(self, x, angle):
        x = x + self.attn(self.norm1(x), angle)
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = nn.Linear(3 * config.patch_size**2, config.hidden_size)

    def forward(self, patches):
        return self.proj(patches.flatten(1))


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_embed = PatchEmbed(config)
        self.blocks = nn.ModuleList([VisionBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size)
        half = config.hidden_size // config.num_attention_heads // 2
        self.register_buffer("inv_freq", config.rope_theta ** (-torch.arange(0, half, 2, device="cpu").float() / half), persistent=False)

    def forward(self, patches, position_ids_2d):
        angle = (position_ids_2d.float().unsqueeze(-1) * self.inv_freq).flatten(-2)
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x, angle)
        return self.norm(x)


class Aligner(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.ratio = vision_config.downsample_ratio
        self.w1 = nn.Linear(vision_config.hidden_size * self.ratio**2, text_config.hidden_size)
        self.w2 = nn.Linear(text_config.hidden_size, text_config.hidden_size)

    def forward(self, features, positions, grid_shape):
        # Gather is equivalent to channel-major F.unfold, but permits tensor
        # grid sizes and avoids legacy ONNX's dynamic unfold limitation.
        ratio = self.ratio
        h, w = grid_shape[0], grid_shape[1]
        oh, ow = (h + ratio - 1) // ratio, (w + ratio - 1) // ratio
        out = torch.arange(oh * ow, device=features.device)
        inner = torch.arange(ratio**2, device=features.device)
        rows = (out // ow)[:, None] * ratio + (inner // ratio)[None, :]
        cols = (out % ow)[:, None] * ratio + (inner % ratio)[None, :]
        valid = (rows < h) & (cols < w)
        indices = (rows * w + cols).clamp(max=features.shape[0] - 1)
        gathered = features[indices] * valid.unsqueeze(-1).to(features.dtype)
        gathered = gathered.transpose(-2, -1).flatten(-2)
        return self.w2(F.gelu(self.w1(gathered)))


class VisionWrapper(nn.Module):
    """Tensor-only image export: patches [N,3,P,P], positions [N,2], grid [2]."""

    def __init__(self, model):
        super().__init__()
        if model.vision is None:
            raise ValueError("Vision is disabled")
        self.vision, self.aligner = model.vision, model.aligner

    def forward(self, patches, position_ids_2d, grid_shape):
        return self.aligner(self.vision(patches, position_ids_2d), position_ids_2d, grid_shape)


class DeepseekV41ForCausalLM(PreTrainedModel):
    config_class = DeepseekV41Config
    base_model_prefix = ""
    main_input_name = "input_ids"
    _no_split_modules = ["Block", "VisionBlock"]

    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        c = config.text_config
        self._validate_config(c)
        self.embed = nn.Embedding(c.vocab_size, c.hidden_size)
        self.engram_hash = None
        rows = {}
        if c.engram_layer_ids:
            if tokenizer is None:
                raise ValueError("Engram requires the checkpoint's tokenizer")
            self.engram_hash = NgramHash(c, tokenizer)
            for i, layer in enumerate(c.engram_layer_ids):
                if c.engram_num_embeddings[i] < self.engram_hash.required_rows[i]:
                    raise ValueError("Engram embedding table is smaller than its hash buckets")
                rows[layer] = c.engram_num_embeddings[i]
        vision_enabled = config.vision_config.num_hidden_layers > 0
        self.layers = nn.ModuleList([Block(c, layer, vision_enabled, rows.get(layer)) for layer in range(c.num_hidden_layers)])
        self.norm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.head = nn.Linear(c.hidden_size, c.vocab_size, bias=False)
        self.vision = None
        if vision_enabled:
            self.vision = VisionTransformer(config.vision_config)
            self.aligner = Aligner(config.vision_config, c)
            for name in ("image_start", "image_end", "image_newline"):
                self.register_parameter(name, nn.Parameter(torch.empty(c.hidden_size)))
        self.state_names = self._make_state_names()
        self.post_init()

    @staticmethod
    def _validate_config(c):
        if len(c.compress_ratios) < c.num_hidden_layers:
            raise ValueError("compress_ratios must cover every backbone layer")
        if c.num_attention_heads % c.o_groups:
            raise ValueError("Attention heads must divide evenly into output groups")
        if c.qk_rope_head_dim % 2 or not 0 < c.qk_rope_head_dim <= min(c.head_dim, c.index_head_dim):
            raise ValueError("RoPE dimension must be positive, even, and fit both heads")
        source = None
        for layer, ratio in enumerate(c.compress_ratios[: c.num_hidden_layers]):
            if layer in c.kv_source_layer_ids:
                if ratio < 1 or layer not in c.index_source_layer_ids:
                    raise ValueError("A KV source needs compression and its own indexer")
                source = ratio
            if ratio and source != ratio:
                raise ValueError("Compressed layers need a preceding same-ratio KV source")

    def _init_weights(self, module):
        std = self.config.text_config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, Gate):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, Block):
            nn.init.normal_(module.hc_attn_fn, std=std)
            nn.init.normal_(module.hc_ffn_fn, std=std)
        elif isinstance(module, DeepseekV41ForCausalLM) and module.vision is not None:
            for name in ("image_start", "image_end", "image_newline"):
                nn.init.normal_(getattr(module, name), std=std)

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def get_output_embeddings(self):
        return self.head

    def _make_state_names(self):
        names = ["engram_history"] if self.engram_hash is not None else []
        for i, layer in enumerate(self.layers):
            names.extend((f"layer_{i}_window", f"layer_{i}_window_positions"))
            if layer.attn.is_kv_source:
                names.extend(f"layer_{i}_{name}" for name in ("compressed", "index_keys", "partial_kv", "partial_scores"))
        return tuple(names)

    def initial_state(self, batch_size, device=None):
        c = self.config.text_config
        device = self.embed.weight.device if device is None else device
        dtype = self.embed.weight.dtype
        result = []
        if self.engram_hash is not None:
            result.append(torch.full((batch_size, c.engram_max_ngram_size - 1), -1, device=device, dtype=torch.int64))
        for layer in self.layers:
            result.extend(
                (
                    torch.zeros(batch_size, c.sliding_window, c.head_dim, device=device, dtype=dtype),
                    torch.full((batch_size, c.sliding_window), -1, device=device, dtype=torch.int64),
                )
            )
            if layer.attn.is_kv_source:
                ratio = layer.attn.ratio
                result.extend(
                    (
                        torch.zeros(batch_size, 0, c.head_dim, device=device, dtype=dtype),
                        torch.zeros(batch_size, 0, c.index_head_dim, device=device, dtype=dtype),
                        torch.zeros(batch_size, ratio - 1, c.head_dim, device=device, dtype=torch.float32),
                        torch.zeros(batch_size, ratio - 1, c.head_dim, device=device, dtype=torch.float32),
                    )
                )
        return tuple(result)

    def encode_image(self, patches, grid_h, grid_w):
        positions = torch.stack(
            torch.meshgrid(torch.arange(grid_h, device=patches.device), torch.arange(grid_w, device=patches.device), indexing="ij"), -1
        ).reshape(-1, 2)
        grid = torch.stack(
            [
                (
                    value.to(device=patches.device, dtype=torch.int64)
                    if isinstance(value, torch.Tensor)
                    else torch.tensor(value, device=patches.device, dtype=torch.int64)
                )
                for value in (grid_h, grid_w)
            ]
        )
        return VisionWrapper(self)(patches, positions, grid)

    def forward(self, input_ids, position_ids, token_types, inputs_embeds, *state):
        if len(state) != len(self.state_names):
            raise ValueError("Pass all tensors returned by initial_state or a preceding forward")
        image_mask = token_types >= 0
        x = self.embed(input_ids)
        if self.vision is not None:
            x = torch.where((token_types == 1).unsqueeze(-1), inputs_embeds, x)
            for kind, embedding in ((0, self.image_start), (2, self.image_newline), (3, self.image_end)):
                x = torch.where((token_types == kind).unsqueeze(-1), embedding, x)
        x = x.unsqueeze(-2).expand(-1, -1, self.config.text_config.hc_mult, -1)
        pre = F.one_hot(torch.zeros_like(input_ids), self.config.text_config.hc_mult).float()
        present, offset = [], 0
        hashes = None
        if self.engram_hash is not None:
            hashes, history = self.engram_hash(input_ids, token_types, state[0])
            present.append(history)
            offset = 1
        shared = {}
        engram_index = 0
        for layer in self.layers:
            if layer.engram is not None:
                x = layer.engram(x, hashes[:, :, engram_index], ~image_mask)
                engram_index += 1
            count = 6 if layer.attn.is_kv_source else 2
            x, pre, cache = layer(x, pre, position_ids, image_mask, state[offset : offset + count], shared)
            present.extend(cache)
            offset += count
        logits = self.head(self.norm(Block.collapse(x, pre))).float()
        return (logits, *present)


for _config in (DeepseekV41Config, DeepseekV41TextConfig, DeepseekV41VisionConfig):
    AutoConfig.register(_config.model_type, _config, exist_ok=True)
AutoModelForCausalLM.register(DeepseekV41Config, DeepseekV41ForCausalLM, exist_ok=True)
