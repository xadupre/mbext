# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Offline numerical tests for the independent V4.1 text and vision reference."""

import math
import unittest
import warnings

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout
from modelbuilder.models.deepseek_v41 import (
    Aligner,
    Attention,
    Block,
    Compressor,
    DeepseekV41Config,
    DeepseekV41ForCausalLM,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
    Engram,
    EngramEmbedding,
    Expert,
    Gate,
    Linear,
    NgramHash,
    Rotary,
    VisionAttention,
    VisionWrapper,
    _fp4_values,
    _fp8_values,
    build_compressed_token_map,
    quantize_roundtrip,
    select_candidate_blocks,
    sinkhorn_mixes,
)


def local_tokenizer():
    words = ["[UNK]", "[PAD]", "[BOS]", "Hello", "hello", "CAFÉ", "cafe", "A", "B", "C", "D", "é", "the", "THE", "image", "z"]
    backend = Tokenizer(WordLevel(dict(zip(words, range(len(words)))), "[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")


def tiny_config(tokenizer, vision=True, engram=True):
    _, compressed = build_compressed_token_map(tokenizer)
    text = DeepseekV41TextConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        moe_intermediate_size=24,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=32,
        qk_rope_head_dim=8,
        q_lora_rank=12,
        o_lora_rank=8,
        o_groups=2,
        n_routed_experts=3,
        num_experts_per_tok=2,
        sliding_window=3,
        compress_ratios=[0, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 5],
        index_n_heads=2,
        index_head_dim=32,
        index_topk=2,
        candidate_source_layer_id=3,
        candidate_topk_blocks=1,
        candidate_block_size=2,
        hc_mult=2,
        hc_sinkhorn_iters=4,
        rms_norm_eps=1e-6,
        engram_layer_ids=[0, 4] if engram else [],
        engram_num_embeddings=[64, 128] if engram else [],
        engram_max_ngram_size=3,
        engram_vocab_size=5,
        engram_n_heads=2,
        engram_head_dim=4,
        engram_compressed_vocab_size=compressed,
        engram_pad_token_id=1,
        rope_scaling=dict(factor=4, beta_fast=4, beta_slow=1, original_max_position_embeddings=32),
    )
    visual = DeepseekV41VisionConfig(
        num_hidden_layers=2 if vision else 0, hidden_size=16, num_attention_heads=2, intermediate_size=24, patch_size=2, downsample_ratio=2
    )
    return DeepseekV41Config(text_config=text, vision_config=visual, image_token_id=14)


class TestDeepseekV41Reference(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)
        super().tearDownClass()

    def make_model(self, vision=True, engram=True):
        torch.manual_seed(602)
        tokenizer = local_tokenizer()
        config = tiny_config(tokenizer, vision, engram)
        return DeepseekV41ForCausalLM(config, tokenizer).eval(), tokenizer

    def inputs(self, model, length=11, batch=2):
        ids = torch.arange(length).unsqueeze(0).expand(batch, -1) % 12 + 3
        positions = torch.arange(length).unsqueeze(0).expand(batch, -1)
        types = torch.full_like(ids, -1)
        embeds = torch.zeros(batch, length, model.config.text_config.hidden_size)
        return ids, positions, types, embeds

    @hide_stdout()
    def test_registered_config_and_model(self):
        tokenizer = local_tokenizer()
        config = tiny_config(tokenizer)
        rebuilt = AutoConfig.for_model("deepseek_v41", **{k: v for k, v in config.to_dict().items() if k != "model_type"})
        self.assertIsInstance(rebuilt.text_config, DeepseekV41TextConfig)
        self.assertIsInstance(AutoConfig.for_model("deepseek_v41_text"), DeepseekV41TextConfig)
        self.assertIsInstance(AutoConfig.for_model("deepseek_v41_vision"), DeepseekV41VisionConfig)
        model = AutoModelForCausalLM.from_config(rebuilt, tokenizer=tokenizer)
        names = model.state_dict()
        self.assertIn("layers.1.attn.compressor.wgate.weight", names)
        self.assertIn("layers.5.attn.indexer.wq_b.weight", names)
        self.assertNotIn("layers.2.attn.indexer.wq_b.weight", names)
        self.assertNotIn("layers.5.attn.indexer.wk.weight", names)
        self.assertIn("vision.blocks.0.attn.wqkv.weight", names)

    def test_fp8_and_fp4_nearest_even(self):
        # Native float8 is an independent CPU oracle, not used by the reference.
        values = torch.cat((torch.linspace(-448, 448, 4001), torch.tensor([2**-10, 3 * 2**-10, 1.0625, 1.1875, 0])))
        expected = values.to(torch.float8_e4m3fn).float()
        torch.testing.assert_close(_fp8_values(values), expected, atol=0, rtol=0)
        x = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.75])
        torch.testing.assert_close(_fp4_values(x), torch.tensor([0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -1.0]), atol=0, rtol=0)

    def test_quantization_scale_formats(self):
        x = torch.tensor([[6.0, 1.1, -2.7, 0.2], [9.0, -1.6, 3.6, 0.1], [0.0, 0.0, 0.0, 0.0]])
        # E8M0 ceil scales: 1 and 2. Compressed KV's E4M3 scale: 1 and 1.5.
        e8 = torch.tensor([[6.0, 1.0, -3.0, 0.0], [8.0, -2.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        e4 = torch.tensor([[6.0, 1.0, -3.0, 0.0], [9.0, -1.5, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        torch.testing.assert_close(quantize_roundtrip(x, 4, 4), e8, atol=0, rtol=0)
        torch.testing.assert_close(quantize_roundtrip(x, 4, 4, "e4m3"), e4, atol=0, rtol=0)
        fp8_scale = 2 ** math.ceil(math.log2(9 / 448))
        expected = (x[1] / fp8_scale).to(torch.float8_e4m3fn).float() * fp8_scale
        torch.testing.assert_close(quantize_roundtrip(x, 8, 4)[1], expected, atol=0, rtol=0)

    def test_checkpoint_activation_quantization_and_embedding(self):
        layer = Linear(4, 2, bias=False)
        x = torch.tensor([[1.1, 2.3, -4.7, 6.0]])
        layer.quantized = True
        expected = F.linear(quantize_roundtrip(x), layer.weight)
        torch.testing.assert_close(layer(x), expected)
        table = EngramEmbedding(2, 4)
        table.load_quantized(
            torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]]).to(torch.float8_e4m3fn),
            torch.tensor([[128], [126]], dtype=torch.uint8),
        )
        self.assertEqual(table.weight.dtype, torch.uint8)
        self.assertEqual(table.scale.dtype, torch.uint8)
        expected = torch.tensor([[-0.5, -1.0, -1.5, -2.0], [2.0, 4.0, 6.0, 8.0]])
        torch.testing.assert_close(table(torch.tensor([1, 0])), expected)

    def test_rotary_inverse_and_yarn(self):
        c = tiny_config(local_tokenizer()).text_config
        c.qk_rope_head_dim = 4
        c.compress_rope_theta = 100
        c.rope_scaling = None
        rotary = Rotary(c, compressed=True)
        x = torch.tensor([[[2.0, 3.0, 1.0, 0.0, 1.0, 0.0]]])
        pos = torch.tensor([[2]])
        result = rotary(x, pos)
        expected = torch.tensor([[[2.0, 3.0, math.cos(2), math.sin(2), math.cos(0.2), math.sin(0.2)]]])
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(rotary(result, pos, inverse=True), x)
        c.rope_scaling = dict(factor=4, original_max_position_embeddings=32, beta_fast=4, beta_slow=1)
        yarn = Rotary(c, compressed=True)
        # low=0, high=1: first frequency unchanged, second divided by four.
        torch.testing.assert_close(yarn.inv_freq, torch.tensor([1.0, 0.025]))

    def test_mhc_sinkhorn_coefficients(self):
        x = torch.ones(1, 1, 2, 3)
        pre, post, comb = sinkhorn_mixes(x, torch.zeros(8, 6), torch.ones(3), torch.zeros(8), 2, 3, 1e-6, 1e-6)
        torch.testing.assert_close(pre, torch.full((1, 1, 2), 0.500001))
        torch.testing.assert_close(post, torch.ones_like(post))
        torch.testing.assert_close(comb.sum(-1), torch.ones(1, 1, 2), atol=2e-6, rtol=0)
        # A non-symmetric comb verifies the checkpoint's source/destination order.
        mixed = Block.expand(
            torch.zeros(1, 1, 1), torch.tensor([[[[2.0], [3.0]]]]), torch.zeros(1, 1, 2), torch.tensor([[[[1.0, 2.0], [4.0, 8.0]]]])
        )
        torch.testing.assert_close(mixed.flatten(), torch.tensor([14.0, 28.0]))

    def test_compressor_independent_pooling_and_partial_groups(self):
        c = tiny_config(local_tokenizer()).text_config
        c.hidden_size, c.head_dim = 2, 2
        compressor = Compressor(c, 2)
        with torch.no_grad():
            compressor.wkv.weight.copy_(torch.eye(2))
            compressor.wgate.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, -1.0]]))
        x = torch.tensor([[[1.0, 3.0], [2.0, 4.0], [5.0, 6.0]]])
        tail = torch.zeros(1, 1, 2)
        latent, pos, kv, score = compressor(x, torch.arange(3)[None], tail, tail)
        p = math.e / (1 + math.e)
        pooled = torch.tensor([[[1 * (1 - p) + 2 * p, 3 * p + 4 * (1 - p)]]])
        expected = pooled / (pooled.square().mean(-1, keepdim=True) + c.rms_norm_eps).sqrt()
        torch.testing.assert_close(latent, expected)
        torch.testing.assert_close(pos, torch.tensor([[0]]))
        self.assertEqual(kv.shape, (1, 1, 2))
        next_x = torch.tensor([[[7.0, 8.0]]])
        decoded = compressor(next_x, torch.tensor([[3]]), kv, score)[0]
        whole = compressor(torch.cat((x, next_x), 1), torch.arange(4)[None], tail, tail)[0][:, 1:]
        torch.testing.assert_close(decoded, whole)

    def test_routing_image_bias_and_weighted_swiglu(self):
        c = tiny_config(local_tokenizer()).text_config
        c.hidden_size, c.n_routed_experts, c.num_experts_per_tok = 2, 3, 1
        gate = Gate(c, True)
        with torch.no_grad():
            gate.weight.zero_()
            gate.bias.copy_(torch.tensor([5.0, 0.0, 0.0]))
            gate.bias_vl.copy_(torch.tensor([0.0, 0.0, 5.0]))
        weights, indices = gate(torch.ones(2, 2), torch.tensor([False, True]))
        torch.testing.assert_close(indices, torch.tensor([[0], [2]]))
        expected = math.sqrt(math.log(2)) * c.routed_scaling_factor
        torch.testing.assert_close(weights, torch.full_like(weights, expected))
        expert = Expert(2, 2, 2.0)
        with torch.no_grad():
            expert.w1.weight.copy_(torch.eye(2))
            expert.w3.weight.copy_(torch.eye(2))
            expert.w2.weight.copy_(torch.eye(2))
        x = torch.tensor([[-3.0, 4.0]])
        expected = F.silu(torch.tensor([[-3.0, 2.0]])) * torch.tensor([[-2.0, 2.0]]) * 0.25
        torch.testing.assert_close(expert(x, torch.tensor([[0.25]])), expected)
        expert.w2.quantized = True
        activation = F.silu(torch.tensor([[-3.0, 2.0]])) * torch.tensor([[-2.0, 2.0]])
        expected = quantize_roundtrip(activation * 0.7)
        torch.testing.assert_close(expert(x, torch.tensor([[0.7]])), expected)

    def test_engram_normalization_hash_oracle_and_image_boundary(self):
        tokenizer = local_tokenizer()
        c = tiny_config(tokenizer).text_config
        mapping, _ = build_compressed_token_map(tokenizer)
        self.assertEqual(mapping[3], mapping[4])
        self.assertEqual(mapping[5], mapping[6])
        self.assertEqual(mapping[12], mapping[13])
        hasher = NgramHash(c, tokenizer)
        ids = torch.tensor([[3, 7, 8, 9]])
        types = torch.tensor([[-1, -1, 1, -1]])
        history = torch.full((1, 2), -1)
        actual, tail = hasher(ids, types, history)
        expected = torch.empty_like(actual)
        mapped = [mapping[3], mapping[7], -1, mapping[9]]
        for t in range(4):
            for layer in range(2):
                block, rolling, col = False, 0, 0
                for shift in range(3):
                    source = mapped[t - shift] if t >= shift else -1
                    block = block or source == -1
                    token = hasher.pad_id if block else source
                    rolling ^= token * int(hasher.multipliers[layer, shift])
                    if shift:
                        for head in range(2):
                            prime = int(hasher.primes[layer, shift - 1, head])
                            expected[0, t, layer, col] = rolling % prime + int(hasher.offsets[layer, col])
                            col += 1
        torch.testing.assert_close(actual, expected)
        cached = hasher(ids[:, 3:], types[:, 3:], tail.new_tensor([[mapping[7], -1]]))[0]
        torch.testing.assert_close(cached, actual[:, 3:])

    def test_engram_gated_update(self):
        c = tiny_config(local_tokenizer()).text_config
        c.hidden_size, c.hc_mult = 2, 2
        layer = Engram(c, 10)
        with torch.no_grad():
            layer.embed.weight.fill_(1)
            layer.wkv.weight.zero_()
            layer.wkv.weight[0].fill_(1 / 16)
            layer.wkv.weight[3].fill_(1 / 16)
            layer.wkv.weight[4:].fill_(2 / 16)
        x = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]])
        out = layer(x, torch.zeros(1, 2, 4, dtype=torch.int64), torch.tensor([[True, False]]))
        dot = 1 / (0.5 + c.rms_norm_eps) / math.sqrt(2)
        update = 2 * torch.sigmoid(torch.tensor(math.sqrt(dot)))
        torch.testing.assert_close(out[:, :1], x[:, :1] + update)
        torch.testing.assert_close(out[:, 1:], x[:, 1:])

    def test_candidate_blocks_pin_newest(self):
        scores = torch.tensor([[[9.0, 8.0, 7.0, 6.0, 1.0]]])
        keep = select_candidate_blocks(scores, torch.tensor([[[5]]]), 1, 2)
        torch.testing.assert_close(keep, torch.tensor([[[False, False, False, False, True]]]))
        scores = torch.tensor([[[9.0, 8.0, -torch.inf, -torch.inf, -torch.inf]]])
        keep = select_candidate_blocks(scores, torch.tensor([[[2]]]), 3, 2)
        torch.testing.assert_close(keep, torch.tensor([[[True, True, False, False, False]]]))

    def test_attention_sink_grouped_projection_and_inverse_rope(self):
        torch.manual_seed(7)
        c = tiny_config(local_tokenizer()).text_config
        attention = Attention(c, 0)
        x = torch.randn(1, 4, c.hidden_size)
        positions = torch.arange(4)[None]
        with torch.no_grad():
            attention.attn_sink.copy_(torch.tensor([0.0, 1.0, -1.0, 0.5]))

        def norm(value, module):
            return value / (value.square().mean(-1, keepdim=True) + c.rms_norm_eps).sqrt() * module.weight

        def rotate(value, position, inverse=False):
            tail = value[..., -8:].unflatten(-1, (4, 2)).contiguous()
            z = torch.view_as_complex(tail)
            angle = position * (c.rope_theta ** (-torch.arange(0, 8, 2) / 8))
            z = z * torch.polar(torch.ones_like(angle), -angle if inverse else angle)
            return torch.cat((value[..., :-8], torch.view_as_real(z).flatten(-2)), -1)

        def fp8(value):
            grouped = value.unflatten(-1, (-1, 32))
            scale = 2 ** torch.ceil(torch.log2(grouped.abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448))
            return ((grouped / scale).to(torch.float8_e4m3fn).float() * scale).flatten(-2)

        qr = norm(F.linear(x, attention.wq_a.weight), attention.q_norm)
        q = F.linear(qr, attention.wq_b.weight).unflatten(-1, (4, 32))
        raw_kv = norm(F.linear(x, attention.wkv.weight), attention.kv_norm)
        kv = torch.stack([fp8(rotate(raw_kv[0, t], t)) for t in range(4)])
        expected = []
        for t in range(4):
            query = rotate(q[0, t], t)
            bank = kv[max(0, t - c.sliding_window + 1) : t + 1]
            logits = query @ bank.T / math.sqrt(32)
            probabilities = torch.cat((logits, attention.attn_sink[:, None]), -1).softmax(-1)[:, :-1]
            output = rotate(probabilities @ bank, t, inverse=True).flatten().reshape(2, 64)
            output = fp8(output)
            grouped = attention.wo_a.weight.reshape(2, 8, 64)
            output = torch.stack([grouped[g] @ output[g] for g in range(2)]).flatten()
            expected.append(F.linear(output, attention.wo_b.weight))
        state = (torch.zeros(1, 3, 32), torch.full((1, 3), -1))
        actual = attention(x, positions, state, {})[0]
        torch.testing.assert_close(actual, torch.stack(expected)[None], atol=2e-6, rtol=2e-5)

    @hide_stdout()
    def test_mhc_uses_previous_sublayer_coefficients(self):
        model, _ = self.make_model(vision=False)
        block = model.layers[0]
        x = torch.randn(1, 2, 2, 16)
        pre = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        positions = torch.tensor([[0, 1]])
        types = torch.zeros(1, 2, dtype=torch.bool)
        state = model.initial_state(1)[1:3]
        shared = {}
        with torch.no_grad():
            original, original_pre, _ = block(x, pre, positions, types, state, shared)
            block.hc_ffn_base[:2].copy_(torch.tensor([3.0, -3.0]))
            changed, changed_pre, _ = block(x, pre, positions, types, state, {})
        # FFN pre coefficients are for the NEXT attention, never its own input.
        torch.testing.assert_close(changed, original, atol=0, rtol=0)
        self.assertGreater((changed_pre - original_pre).abs().max().item(), 0.1)

    @hide_stdout()
    def test_nonempty_decode_trace_accepts_empty_prefill_and_new_shapes(self):
        model, _ = self.make_model()
        inputs = self.inputs(model, length=4, batch=1)
        with torch.no_grad():
            state = model(*inputs, *model.initial_state(1))[1:]
            sample = tuple(t[:, :1] for t in inputs)
            sample = (sample[0], torch.tensor([[4]]), *sample[2:])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                traced = torch.jit.trace(model, (*sample, *state), check_trace=False)
            trace_warnings = [w for w in caught if isinstance(w.message, torch.jit.TracerWarning)]
            self.assertEqual(trace_warnings, [])
            inputs = self.inputs(model, length=7, batch=2)
            expected = model(*inputs, *model.initial_state(2))
            actual = traced(*inputs, *model.initial_state(2))
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)

    @hide_stdout()
    def test_text_prefill_decode_and_cache_bounds(self):
        model, _ = self.make_model(vision=False)
        inputs = self.inputs(model)
        with torch.no_grad():
            whole = model(*inputs, *model.initial_state(2))
            for split in (1, 3, 6, 9):
                first = model(*(t[:, :split] for t in inputs), *model.initial_state(2))
                chunks = [first[0]]
                state = first[1:]
                for i in range(split, inputs[0].shape[1]):
                    result = model(*(t[:, i : i + 1] for t in inputs), *state)
                    chunks.append(result[0])
                    state = result[1:]
                torch.testing.assert_close(torch.cat(chunks, 1), whole[0], atol=2e-6, rtol=2e-5)
                for actual, expected in zip(state, whole[1:]):
                    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=2e-5)
        self.assertEqual(len(state), len(model.state_names))
        state = dict(zip(model.state_names, state))
        self.assertEqual(state["engram_history"].shape[1], 2)
        self.assertEqual(state["layer_1_partial_kv"].shape[1], 1)
        self.assertEqual(state["layer_1_compressed"].shape[1], 5)
        self.assertEqual(state["layer_3_compressed"].shape[1], 11)
        for name, value in state.items():
            if name.endswith("_window"):
                self.assertEqual(value.shape[1], 3)

    def test_vision_2d_rope_and_unshuffle_oracle(self):
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        angle = torch.tensor([[math.pi / 2, 0.0]])
        torch.testing.assert_close(VisionAttention.rotate(x, angle), torch.tensor([[[-3.0, 2.0, 1.0, 4.0]]]))
        config = tiny_config(local_tokenizer())
        aligner = Aligner(config.vision_config, config.text_config)
        features = torch.arange(3 * 5 * 16).float().reshape(15, 16) / 100
        grid = torch.tensor([3, 5])
        raw = features.reshape(3, 5, 16).permute(2, 0, 1)
        expected = F.unfold(F.pad(raw, (0, 1, 0, 1))[None], 2, stride=2)[0].T
        expected = aligner.w2(F.gelu(aligner.w1(expected)))
        torch.testing.assert_close(aligner(features, None, grid), expected)

    @hide_stdout()
    def test_encode_image_trace_dynamic_grid(self):
        model, _ = self.make_model()

        class Encoder(torch.nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, patches, grid):
                return self.module.encode_image(patches, grid[0], grid[1])

        wrapper = Encoder(model)
        with torch.no_grad(), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            traced = torch.jit.trace(wrapper, (torch.randn(4, 3, 2, 2), torch.tensor([2, 2])), check_trace=False)
        self.assertEqual([w for w in caught if isinstance(w.message, torch.jit.TracerWarning)], [])
        patches, grid = torch.randn(15, 3, 2, 2), torch.tensor([3, 5])
        with torch.no_grad():
            torch.testing.assert_close(traced(patches, grid), wrapper(patches, grid), atol=2e-6, rtol=2e-5)

    @hide_stdout()
    def test_image_prefill_decode_delimiters_and_differences(self):
        model, _ = self.make_model()
        ids, positions, types, embeds = self.inputs(model, length=10, batch=1)
        patches = torch.randn(4, 3, 2, 2)
        with torch.no_grad():
            features = model.encode_image(patches, 2, 2)
            self.assertEqual(features.shape, (1, 16))
            positions2d = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
            projected = VisionWrapper(model)(patches, positions2d, torch.tensor([2, 2]))
            torch.testing.assert_close(features, projected)
            types[:, 1:5] = torch.tensor([0, 1, 2, 3])
            ids[:, 1:5] = model.config.image_token_id
            embeds[:, 2] = features
            inputs = ids, positions, types, embeds
            whole = model(*inputs, *model.initial_state(1))[0]
            first = model(*(t[:, :6] for t in inputs), *model.initial_state(1))
            rest = model(*(t[:, 6:] for t in inputs), *first[1:])
            torch.testing.assert_close(whole, torch.cat((first[0], rest[0]), 1), atol=2e-6, rtol=2e-5)
            # Text and delimiters ignore externally supplied embeddings.
            ignored = embeds.clone()
            ignored[types != 1] = torch.randn_like(ignored[types != 1]) * 100
            identical = model(ids, positions, types, ignored, *model.initial_state(1))[0]
            torch.testing.assert_close(whole, identical, atol=0, rtol=0)
            changed = embeds.clone()
            changed[:, 2] = model.encode_image(patches * -2, 2, 2)
            different = model(ids, positions, types, changed, *model.initial_state(1))[0]
            self.assertGreater((whole - different).abs().max().item(), 1e-6)

    @hide_stdout()
    def test_models_do_not_share_mutable_state_and_meta_buffers(self):
        model, tokenizer = self.make_model()
        other, _ = self.make_model()
        inputs = self.inputs(model, length=4, batch=1)
        initial = model.initial_state(1)
        copies = tuple(x.clone() for x in initial)
        with torch.no_grad():
            expected = model(*inputs, *initial)
            other(*inputs, *other.initial_state(1))
            actual = model(*inputs, *initial)
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        for left, right in zip(initial, copies):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
        with torch.device("meta"):
            empty = DeepseekV41ForCausalLM(tiny_config(tokenizer), tokenizer)
        self.assertTrue(all(not b.is_meta for b in empty.buffers()))
        self.assertTrue(all(p.is_meta for p in empty.parameters()))

    @hide_stdout()
    def test_float16_and_bfloat16_text_and_image_cached_decode(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model, _ = self.make_model()
                model = model.to(dtype=dtype)
                inputs = self.inputs(model, length=5, batch=1)
                ids, positions, types, embeds = inputs
                embeds = embeds.to(dtype)
                patches = torch.randn(4, 3, 2, 2).to(dtype)
                with torch.no_grad():
                    embeds[:, 1] = model.encode_image(patches, 2, 2)
                    types[:, :3] = torch.tensor([0, 1, 3])
                    ids[:, :3] = model.config.image_token_id
                    inputs = ids, positions, types, embeds
                    expected = model(*inputs, *model.initial_state(1))[0]
                    first = model(*(t[:, :3] for t in inputs), *model.initial_state(1))
                    rest = model(*(t[:, 3:] for t in inputs), *first[1:])
                self.assertTrue(torch.isfinite(expected).all())
                torch.testing.assert_close(torch.cat((first[0], rest[0]), 1), expected, atol=2e-3, rtol=2e-2)


if __name__ == "__main__":
    unittest.main()
