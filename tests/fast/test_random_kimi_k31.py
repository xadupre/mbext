# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Offline K3.1 text, image and video logits/state/generated-token parity.

Native GenAI cannot update these named heterogeneous states reliably (#604).
All three direct ONNX backends are tested instead of aliasing K3.1 to Kimi K2.
Full-checkpoint memory consumption and throughput are not inferred from these
tiny random-weight tests.
"""

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from onnx_light.onnx import TensorProto

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda


def make_tiny_kimi(vision=True):
    from modelbuilder.models.kimi_k31 import KimiK31Config, KimiK31ForConditionalGeneration

    config = KimiK31Config(
        text_config=dict(
            vocab_size=32,
            hidden_size=32,
            # The released model's widths are block aligned. Partial-block
            # native QDQ dequantization is tracked in xadupre/onnx-light#6009.
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
            q_lora_rank=16,
            kv_lora_rank=8,
            qk_nope_head_dim=8,
            qk_rope_head_dim=4,
            v_head_dim=8,
            num_experts=4,
            num_experts_per_token=2,
            num_shared_experts=1,
            moe_intermediate_size=32,
            routed_expert_hidden_size=32,
            attn_res_block_size=2,
            linear_attn_config=dict(
                head_dim=8,
                num_heads=2,
                short_conv_kernel_size=4,
                use_full_rank_gate=True,
                gate_lower_bound=-5.0,
                kda_layers=[1, 3],
                full_attn_layers=[2, 4],
            ),
            dsa_config=dict(
                dsa_layers=[2, 4],
                index_topk=2,
                index_head_dim=4,
                index_num_heads=2,
                index_rope_dim=0,
                index_k_norm_type="layernorm",
                index_topk_freq=1,
                index_skip_topk_offset=1,
            ),
        ),
        vision_config=(
            dict(
                vt_hidden_size=8,
                mm_hidden_size=8,
                qkv_hidden_size=16,
                vt_num_attention_heads=2,
                vt_num_hidden_layers=1,
                vt_intermediate_size=16,
                patch_size=2,
                init_pos_emb_height=4,
                init_pos_emb_width=4,
                init_pos_emb_time=4,
                norm_type="rmsnorm",
                mlp_type="mlp2",
                merge_type="sd2_tpool",
                mm_projector_type="patchmergerv2",
                pos_emb_type="divided_fixed",
                pos_emb_interpolation_mode="bilinear",
                attn_bias=False,
                linear_bias=False,
                patch_embed_proj_bias=False,
                merge_kernel_size=[2, 2],
                projector_ln_eps=1e-5,
            )
            if vision
            else None
        ),
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        media_placeholder_token_id=31,
    )
    torch.manual_seed(42)
    module = KimiK31ForConditionalGeneration(config).eval()
    return config, module


class TestKimiK31(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)
        super().tearDownClass()

    def test_kda_numpy_oracle(self):
        from modelbuilder.models.kimi_k31 import KDA

        config, _ = make_tiny_kimi(False)
        torch.manual_seed(13)
        layer = KDA(config.text_config).eval()
        with torch.no_grad():
            layer.A_log.copy_(torch.tensor([0.3, -0.5]))
            layer.dt_bias.copy_(torch.linspace(-1, 1, 16))
        rng = np.random.default_rng(42)
        history = [rng.normal(size=(2, 16, 3)).astype(np.float32) for _ in range(3)]
        matrix = rng.normal(size=(2, 2, 8, 8)).astype(np.float32)
        expected_state = [*history, matrix]
        actual_state = [torch.from_numpy(a.copy()) for a in expected_state]

        def linear(x, module):
            return x @ module.weight.detach().numpy().T

        for _ in range(5):
            x = rng.normal(size=(2, 1, 32)).astype(np.float32)
            qkv, histories = [], []
            for name, old in zip(("q", "k", "v"), expected_state[:3]):
                current = linear(x, getattr(layer, name + "_proj")).transpose(0, 2, 1)
                window = np.concatenate((old, current), axis=2)
                value = (window * getattr(layer, name + "_conv1d").weight.detach().numpy()[:, 0]).sum(2)
                qkv.append((value / (1 + np.exp(-value))).reshape(2, 2, 8))
                histories.append(window[:, :, 1:])
            q, k, v = qkv
            q = q / np.sqrt((q * q).sum(2, keepdims=True) + 1e-6) / np.sqrt(8)
            k = k / np.sqrt((k * k).sum(2, keepdims=True) + 1e-6)
            raw = linear(linear(x, layer.f_a_proj), layer.f_b_proj).reshape(2, 2, 8) + layer.dt_bias.detach().numpy().reshape(2, 8)
            decay = -5 / (1 + np.exp(-np.exp(layer.A_log.detach().numpy())[:, None] * raw))
            decayed = expected_state[3] * np.exp(decay)[..., None]
            beta = 1 / (1 + np.exp(-linear(x, layer.b_proj).reshape(2, 2, 1)))
            residual = v - np.einsum("bhk,bhkv->bhv", k, decayed)
            matrix = decayed + np.einsum("bhk,bhv->bhkv", k, residual * beta)
            output = np.einsum("bhk,bhkv->bhv", q, matrix)
            output = output / np.sqrt((output * output).mean(2, keepdims=True) + config.text_config.rms_norm_eps)
            output *= layer.o_norm.weight.detach().numpy()
            output *= 1 / (1 + np.exp(-linear(x, layer.g_proj).reshape(2, 2, 8)))
            expected = linear(output.reshape(2, 1, 16), layer.o_proj)
            with torch.no_grad():
                actual, actual_state = layer(torch.from_numpy(x), *actual_state)
            expected_state = [*histories, matrix]
            np.testing.assert_allclose(actual.numpy(), expected, atol=1e-6, rtol=2e-5)
            for reference, value in zip(expected_state, actual_state):
                np.testing.assert_allclose(value.numpy(), reference, atol=1e-6, rtol=2e-5)

    @torch.no_grad()
    def test_situ_routing_and_residual_oracles(self):
        from modelbuilder.models.kimi_k31 import MoE, RMSNorm, mix_residual, situ

        gate = torch.tensor([-40.0, -4.0, 0.0, 4.0, 40.0])
        up = gate.flip(0)
        expected = 4 * np.tanh(gate.numpy() / 4) / (1 + np.exp(-gate.numpy())) * 25 * np.tanh(up.numpy() / 25)
        np.testing.assert_allclose(situ(gate, up, 4, 25).numpy(), expected, atol=1e-6)
        config, _ = make_tiny_kimi(False)
        c = config.text_config
        c.num_expert_group, c.topk_group = 2, 1
        module = MoE(c)
        with torch.no_grad():
            module.gate.weight.zero_()
            module.gate.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 3.0, 2.0]))
        x = torch.randn(3, 32)
        indices, weights = module.route(x)
        self.assertEqual(set(indices.flatten().tolist()), {2, 3})
        self.assertEqualArray(torch.full((3, 2), 0.5), weights)
        latent = module.routed_expert_down_proj(x)
        expected = (module.experts[2](latent).float() + module.experts[3](latent).float()) / 2
        expected = module.routed_expert_up_proj(module.routed_expert_norm(expected)) + module.shared_experts(x)
        self.assertEqualArray(expected, module(x), atol=1e-6)
        norm, projection = RMSNorm(32, 1e-5), torch.nn.Linear(32, 1, bias=False)
        prefix, previous = torch.randn(2, 1, 32), torch.randn(2, 1, 32)
        values = torch.stack([previous, prefix], dim=2).double()
        keys = values / torch.sqrt(values.square().mean(3, keepdim=True) + 1e-5)
        scores = F.linear(keys * norm.weight.double(), projection.weight.double()).softmax(2)
        expected = (scores * values).sum(2)
        self.assertEqualArray(expected.float(), mix_residual(prefix, [previous], norm, projection), atol=1e-6)

    def test_mxfp4_and_native_checkpoint(self):
        from safetensors.torch import save_file

        from modelbuilder.helpers.kimi_k31 import KimiK31Checkpoint, decode_mxfp4

        values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]).repeat(2).reshape(1, 32)
        packed = torch.tensor([[a | ((a + 1) << 4) for a in range(0, 16, 2)] * 2], dtype=torch.uint8)
        self.assertEqualArray(values * 2, decode_mxfp4(packed, torch.tensor([[128]], dtype=torch.uint8), (1, 32)))
        with self.assertRaisesRegex(ValueError, "NaN"):
            decode_mxfp4(packed, torch.tensor([[255]], dtype=torch.uint8), (1, 32))
        config, original = make_tiny_kimi()
        tensors = {name: value.detach().clone() for name, value in original.state_dict().items()}
        key = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"
        weight = tensors.pop(key)
        tensors[key.removesuffix("weight") + "weight_packed"] = packed.expand(weight.shape[0], -1).contiguous()
        tensors[key.removesuffix("weight") + "weight_scale"] = torch.full((weight.shape[0], 1), 128, dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as directory:
            save_file(tensors, os.path.join(directory, "model.safetensors"))
            with torch.device("meta"):
                from modelbuilder.models.kimi_k31 import KimiK31ForConditionalGeneration

                loaded = KimiK31ForConditionalGeneration(config)
            KimiK31Checkpoint(directory).load(loaded)
        for name, value in loaded.state_dict().items():
            expected = (values * 2).expand(weight.shape[0], -1) if name == key else tensors[name]
            self.assertEqualArray(expected, value)

    @torch.no_grad()
    def test_dsa_dense_selection_oracle(self):
        from modelbuilder.models.kimi_k31 import DSA

        config, _ = make_tiny_kimi(False)
        c = config.text_config
        torch.manual_seed(7)
        attention = DSA(c)

        def norm(value, module):
            x = value.double()
            return (x / torch.sqrt(x.square().mean(-1, keepdim=True) + module.eps) * module.weight.double()).float()

        for past in (1, 5):
            x = torch.randn(2, 1, 32)
            state = [torch.randn(2, past, size) for size in (8, 4, 4)]
            positions = torch.arange(-1, past).expand(2, -1)
            qi = norm(attention.q_a_proj(x), attention.q_a_layernorm)
            query = attention.q_b_proj(qi).view(2, 2, 12)
            raw = attention.kv_a_proj_with_mqa(x)
            latent = torch.cat((state[0], norm(raw[:, :, :8], attention.kv_a_layernorm)), 1)
            shared = torch.cat((state[1], raw[:, :, 8:]), 1)
            keys = torch.cat(
                (
                    state[2],
                    F.layer_norm(
                        attention.indexer.wk(x), (4,), attention.indexer.k_norm.weight, attention.indexer.k_norm.bias, c.rms_norm_eps
                    ),
                ),
                1,
            )
            iq = attention.indexer.wq_b(qi).view(2, 2, 4).double()
            scores = torch.einsum("bhd,btd->bht", iq, keys.double()).clamp_min(0)
            scores *= attention.indexer.weights_proj(x).double().view(2, 2, 1) / np.sqrt(8)
            scores = scores.sum(1)
            scores[:, 0] = -float("inf")
            selected = scores.argsort(dim=1, descending=True, stable=True)[:, :2]
            expanded = attention.kv_b_proj(latent).view(2, past + 1, 2, 16)
            full_keys = torch.cat((expanded[:, :, :, :8], shared[:, :, None].expand(-1, -1, 2, -1)), 3)
            values = expanded[:, :, :, 8:]
            outputs = []
            for batch in range(2):
                indices = selected[batch]
                logits = torch.einsum("hd,thd->ht", query[batch].double(), full_keys[batch, indices].double()) / np.sqrt(12)
                logits[:, indices == 0] = -float("inf")
                outputs.append(torch.einsum("ht,thv->hv", logits.softmax(1), values[batch, indices].double()))
            output = torch.stack(outputs).float().view(2, 1, 16) * attention.g_proj(x).sigmoid()
            expected = attention.o_proj(output)
            actual, present = attention(x, positions, *state)
            self.assertEqualArray(expected, actual, atol=1e-6)
            for reference, value in zip((latent, shared, keys), present):
                self.assertEqualArray(reference, value, atol=1e-6)

    @torch.no_grad()
    def test_moonvit_sdpa_oracle(self):
        _, model = make_tiny_kimi()
        tower, projector = model.vision_tower, model.mm_projector
        for frames, height, width in ((1, 2, 4), (3, 4, 2)):
            patches = torch.randn(frames * height * width, 3, 2, 2)
            grid = torch.tensor([frames, height, width])
            spatial = (
                F.interpolate(
                    tower.patch_embed.pos_emb.weight.permute(2, 0, 1)[None], size=(height, width), mode="bilinear", align_corners=False
                )[0]
                .permute(1, 2, 0)
                .flatten(0, 1)
            )
            freq = 10000.0 ** (-torch.arange(4).float() / 4)
            time = torch.arange(frames).float()[:, None] * freq
            temporal = torch.cat((time.sin(), time.cos()), 1)
            positions = spatial[None].expand(frames, -1, -1)
            if frames != 1:
                positions = positions + temporal[:, None]
            x = F.conv2d(patches, tower.patch_embed.proj.weight, stride=2).flatten(1) + positions.reshape(-1, 8)

            def rms(value, module):
                eps = torch.finfo(value.dtype).eps if module.eps is None else module.eps
                return F.rms_norm(value, (value.shape[-1],), module.weight, eps)

            coordinates = [(col, row) for _ in range(frames) for row in range(height) for col in range(width)]
            phase = torch.tensor(
                [[axis * (10000.0 ** (-4 * band / 8)) for band in range(2) for axis in (col, row)] for col, row in coordinates]
            )
            rotation = torch.polar(torch.ones_like(phase), phase)[:, None]
            for block in tower.encoder.blocks:
                q, k, v = block.wqkv(rms(x, block.norm0)).reshape(-1, 3, 2, 8).unbind(1)
                q, k = [
                    torch.view_as_real(torch.view_as_complex(a.reshape(-1, 2, 4, 2).contiguous()) * rotation).flatten(2) for a in (q, k)
                ]
                output = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))
                x = x + block.wo(output.transpose(0, 1).reshape(-1, 16))
                x = x + block.mlp.fc1(F.gelu(block.mlp.fc0(rms(x, block.norm1)), approximate="tanh"))
            x = rms(x, tower.encoder.final_layernorm).reshape(frames, height, width, 8)
            pooled = x.mean(0)
            merged = torch.stack([pooled[i : i + 2, j : j + 2].flatten() for i in range(0, height, 2) for j in range(0, width, 2)])
            expected = rms(projector.proj(merged), projector.post_norm)
            self.assertEqualArray(expected, model.encode_media(patches, grid), atol=1e-6)

    @hide_stdout()
    @torch.no_grad()
    def test_text_only_portable_chunks_and_validation(self):
        from modelbuilder.helpers.kimi_k31_runtime import KimiK31Session
        from modelbuilder.models.kimi_k31 import KimiK31Config

        with tempfile.TemporaryDirectory() as directory:
            config, module, output = self.export(directory, vision=False, provider="any")
            session = KimiK31Session(output)
            ids = np.array([[4, 5, 6, 7, 8], [8, 7, 6, 5, 4]])
            positions = np.tile(np.arange(5), (2, 1))
            whole, state = session.run(ids, position_ids=positions)
            first, partial = session.run(ids[:, :2], position_ids=positions[:, :2])
            saved = {name: value.copy() for name, value in partial.items()}
            last, resumed = session.run(ids[:, 2:], position_ids=positions[:, 2:], state=partial)
            np.testing.assert_allclose(whole, np.concatenate((first, last), 1), atol=1e-6)
            for name in state:
                np.testing.assert_array_equal(state[name], resumed[name])
                np.testing.assert_array_equal(partial[name], saved[name])
            pt_state = module.initial_state(2)
            expected = []
            for index in range(5):
                logits, *pt_state = module(
                    torch.tensor(ids[:, index : index + 1]),
                    torch.full((2, 1), index),
                    torch.full((2, 1), -1),
                    torch.zeros(2, 1, 32),
                    *pt_state,
                )
                expected.append(logits.numpy())
            np.testing.assert_allclose(whole, np.concatenate(expected, 1), atol=1e-6)
            with self.assertRaisesRegex(ValueError, "continue"):
                session.run(ids[:, :1], position_ids=np.full((2, 1), 1), state=state)
            with self.assertRaisesRegex(ValueError, "contiguous"):
                session.run(ids[:, :2], position_ids=np.array([[0, 2], [0, 2]]))
            with self.assertRaisesRegex(ValueError, "token_types"):
                session.run(ids, position_ids=positions, token_types=np.zeros_like(ids))
            with self.assertRaisesRegex(ValueError, "vocabulary"):
                session.run(np.array([[999]]), position_ids=np.array([[0]]))
            with self.assertRaisesRegex(ValueError, "no vision"):
                session.encode_media(np.zeros((4, 3, 2, 2)), [1, 2, 2])
            with self.assertRaisesRegex(ValueError, "batch_size"):
                session.initial_state(0)
            with self.assertRaisesRegex(ValueError, "partition"):
                values = config.to_dict()
                values["text_config"]["linear_attn_config"]["kda_layers"] = [1]
                KimiK31Config(**values)

    def export(self, directory, precision="fp32", vision=True, provider="cpu", cli=False, qdq=False):
        from modelbuilder.builder import create_model
        from modelbuilder.builders.kimi_k31 import KimiK31Model
        from modelbuilder.helpers.kimi_k31 import is_fp32_parameter

        config, module = make_tiny_kimi(vision)
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16, "int4": torch.float32}[precision]
        for name, parameter in module.named_parameters():
            parameter.data = parameter.data.to(torch.float32 if is_fp32_parameter(name) else dtype)
        output = os.path.join(directory, "output")
        cache = os.path.join(directory, "cache")
        os.makedirs(output)
        os.makedirs(cache)
        if cli:
            from safetensors.torch import save_file

            source = os.path.join(directory, "source")
            os.makedirs(source)
            config.save_pretrained(source)
            self.make_word_level_tokenizer().save_pretrained(source)
            save_file(module.state_dict(), os.path.join(source, "model.safetensors"))
            create_model("KimiK31ForConditionalGeneration", source, output, precision, provider, cache)
        else:
            io = {torch.float32: TensorProto.FLOAT, torch.float16: TensorProto.FLOAT16, torch.bfloat16: TensorProto.BFLOAT16}[dtype]
            builder = KimiK31Model(config, io, TensorProto.INT4 if precision == "int4" else io, provider, cache, {"use_qdq": qdq})
            builder.export_module(module)
            builder.save_model(output)
            builder.make_genai_config("", {}, output)
        return config, module, output

    @torch.no_grad()
    def compare_runtime(self, precision="fp32", provider="cpu", cli=False, qdq=False):
        from modelbuilder.helpers.kimi_k31_runtime import KimiK31Session

        with tempfile.TemporaryDirectory() as directory:
            config, module, output = self.export(directory, precision, provider=provider, cli=cli, qdq=qdq)
            if precision == "bf16":
                backends = ("onnxruntime",)
            elif precision == "int4" and not qdq:
                # Native onnx-light has no com.microsoft MatMulNBits kernel;
                # the QDQ test below exercises its standard-ONNX alternative.
                backends = ("onnxruntime", "onnx-light-cpu")
            else:
                backends = ("onnxruntime", "onnx-light", "onnx-light-cpu")
            tolerance = {"fp32": 3e-5, "fp16": 3e-3, "bf16": 3e-2, "int4": 0.08}[precision]
            providers = ["CUDAExecutionProvider"] if provider == "cuda" else ["CPUExecutionProvider"]
            quantized_reference = KimiK31Session(output) if precision == "int4" else None
            if precision == "int4":
                from onnx_light.onnx import helper, load_model

                for component in ("decoder", "vision"):
                    graph = load_model(os.path.join(output, quantized_reference.config[component]), load_external_data=False).graph
                    nodes = [node for node in graph.node if node.op_type == ("DequantizeLinear" if qdq else "MatMulNBits")]
                    self.assertTrue(nodes, component)
                    initializers = {tensor.name: tensor for tensor in graph.initializer}
                    for node in nodes:
                        if qdq:
                            self.assertEqual(initializers[str(node.input[0])].data_type, TensorProto.INT4)
                        else:
                            attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                            self.assertEqual(attrs["bits"], 4)
            for backend in backends:
                session = KimiK31Session(output, backend=backend, providers=providers)
                for batch in (1, 2):
                    for frames, height, width in ((0, 0, 0), (1, 2, 4), (2, 4, 2)):
                        with self.subTest(backend=backend, batch=batch, frames=frames):
                            ids = np.tile([[4, 5, 6, 7]], (batch, 1))
                            types = np.full_like(ids, -1)
                            embeds = np.zeros((batch, 4, 32), dtype=np.float32)
                            if frames:
                                rng = np.random.default_rng(42)
                                patches = rng.normal(size=(frames * height * width, 3, 2, 2)).astype(np.float32)
                                dtype = module.language_model.model.embed_tokens.weight.dtype
                                features = module.encode_media(torch.tensor(patches).to(dtype), torch.tensor([frames, height, width]))
                                actual_features = session.encode_media(patches, [frames, height, width])
                                if precision == "int4":
                                    self.assertLess(
                                        np.linalg.norm(actual_features.astype(np.float32) - features.float().numpy())
                                        / np.linalg.norm(features.float().numpy()),
                                        0.2,
                                    )
                                else:
                                    np.testing.assert_allclose(
                                        actual_features.astype(np.float32), features.float().numpy(), atol=tolerance, rtol=tolerance
                                    )
                                length = len(actual_features)
                                ids[:, 1 : 1 + length] = 31
                                types[:, 1 : 1 + length] = 1
                                embeds[:, 1 : 1 + length] = actual_features.astype(np.float32)
                            pt_state = module.initial_state(batch)
                            state = None
                            expected_logits = []
                            dtype = module.language_model.model.embed_tokens.weight.dtype
                            # The oracle receives its own PyTorch vision output, never runtime features.
                            reference_embeds = torch.tensor(embeds).to(dtype)
                            if frames:
                                reference_embeds[:, 1 : 1 + len(features)] = features
                            for step in range(4):
                                logits, *pt_state = module(
                                    torch.tensor(ids[:, step : step + 1]),
                                    torch.full((batch, 1), step),
                                    torch.tensor(types[:, step : step + 1]),
                                    reference_embeds[:, step : step + 1],
                                    *pt_state,
                                )
                                expected_logits.append(logits.float().numpy())
                            actual, state = session.run(
                                ids, position_ids=np.tile(np.arange(4), (batch, 1)), token_types=types, inputs_embeds=embeds
                            )
                            np.testing.assert_allclose(
                                actual.astype(np.float32), np.concatenate(expected_logits, axis=1), atol=tolerance, rtol=tolerance
                            )
                            if precision != "int4":
                                for name, expected in zip(module.state_names, pt_state):
                                    np.testing.assert_allclose(
                                        state[name].astype(np.float32), expected.float().numpy(), atol=tolerance, rtol=tolerance
                                    )
                            reference_tokens, actual_tokens = ids.copy(), ids.copy()
                            for step in range(3):
                                next_ref = expected_logits[-1][:, -1].argmax(-1)[:, None]
                                next_actual = actual[:, -1].argmax(-1)[:, None]
                                reference_tokens = np.concatenate((reference_tokens, next_ref), axis=1)
                                actual_tokens = np.concatenate((actual_tokens, next_actual), axis=1)
                                expected, *pt_state = module(
                                    torch.tensor(next_ref),
                                    torch.full((batch, 1), 4 + step),
                                    torch.full((batch, 1), -1),
                                    torch.zeros(batch, 1, 32, dtype=dtype),
                                    *pt_state,
                                )
                                expected_logits.append(expected.float().numpy())
                                actual, state = session.run(next_ref, position_ids=np.full((batch, 1), 4 + step), state=state)
                                np.testing.assert_allclose(
                                    actual.astype(np.float32), expected.float().numpy(), atol=tolerance, rtol=tolerance
                                )
                            if precision == "fp32" or quantized_reference is not None:
                                generated = session.generate(
                                    ids, max_new_tokens=3, token_types=types, inputs_embeds=embeds, eos_token_id=[]
                                )
                                if quantized_reference is None:
                                    np.testing.assert_array_equal(actual_tokens, reference_tokens)
                                    np.testing.assert_array_equal(generated, reference_tokens)
                                else:
                                    # Quantization can change argmax. Compare exact tokens
                                    # with ORT executing the same real quantized graph.
                                    np.testing.assert_array_equal(
                                        generated,
                                        quantized_reference.generate(
                                            ids, max_new_tokens=3, token_types=types, inputs_embeds=embeds, eos_token_id=[]
                                        ),
                                    )
                            else:
                                generated = session.generate(
                                    ids, max_new_tokens=3, token_types=types, inputs_embeds=embeds, eos_token_id=[]
                                )
                                # Replay each actual generated prefix independently:
                                # reduced precision may reorder nearly tied tokens.
                                replay = module.initial_state(batch)
                                for index in range(generated.shape[1] - 1):
                                    typ = torch.tensor(types[:, index : index + 1]) if index < 4 else torch.full((batch, 1), -1)
                                    embedding = (
                                        reference_embeds[:, index : index + 1] if index < 4 else torch.zeros(batch, 1, 32, dtype=dtype)
                                    )
                                    logits, *replay = module(
                                        torch.tensor(generated[:, index : index + 1]),
                                        torch.full((batch, 1), index),
                                        typ,
                                        embedding,
                                        *replay,
                                    )
                                    if index >= 3:
                                        reference = logits[:, -1].float().numpy()
                                        selected = generated[:, index + 1]
                                        regret = reference.max(1) - reference[np.arange(batch), selected]
                                        self.assertLess(float(regret.max()), 2 * tolerance)

    @hide_stdout()
    def test_fp32_all_backends_text_image_video(self):
        self.compare_runtime(cli=True)

    @hide_stdout()
    def test_fp16_ort_text_image_video(self):
        self.compare_runtime("fp16")

    @hide_stdout()
    def test_int4_ort_text_image_video(self):
        self.compare_runtime("int4")

    @hide_stdout()
    def test_int4_qdq_all_backends_text_image_video(self):
        self.compare_runtime("int4", qdq=True)

    @hide_stdout()
    @requires_cuda()
    def test_bf16_cuda_text_image_video(self):
        self.compare_runtime("bf16", "cuda")

    @unittest.skip("K3.1 named recurrent-state rebinding is blocked by #604; no GenAI architecture alias is emitted.")
    def test_genai_text_image_video_generation(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
