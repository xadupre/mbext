# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Offline V4.1 text/vision prefill, cached decode and generation parity.

GenAI's arbitrary recurrent-state updates are blocked by #604. In particular,
using a V3 config or checking only the first generation step is not a valid
substitute for V4.1 cached generation.
"""

import json
import os
import tempfile
import unittest

import numpy as np
import torch
from onnx_light.onnx import TensorProto

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda
from modelbuilder.helpers.onnx_helper import torch_tensor_to_numpy


def make_tiny_v41():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from modelbuilder.models.deepseek_v41 import DeepseekV41Config, DeepseekV41ForCausalLM

    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, "<unk>": 3}
    vocab.update({f"word{i}": i for i in range(4, 32)})
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    config = DeepseekV41Config(
        text_config=dict(
            hidden_size=32,
            intermediate_size=32,
            moe_intermediate_size=32,
            vocab_size=32,
            num_hidden_layers=5,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            qk_rope_head_dim=8,
            q_lora_rank=32,
            o_lora_rank=32,
            o_groups=1,
            n_routed_experts=4,
            n_shared_experts=1,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            routed_scaling_factor=1.5,
            scoring_func="sqrtsoftplus",
            hidden_act="silu",
            swiglu_limit=2.0,
            rms_norm_eps=1e-5,
            max_position_embeddings=128,
            sliding_window=4,
            compress_ratios=[0, 2, 2, 1, 1],
            kv_source_layer_ids=[1, 3],
            index_source_layer_ids=[1, 2, 3, 4],
            index_n_heads=2,
            index_head_dim=32,
            index_topk=2,
            candidate_source_layer_id=3,
            candidate_topk_blocks=1,
            candidate_block_size=2,
            hc_mult=2,
            hc_sinkhorn_iters=3,
            hc_eps=1e-6,
            engram_layer_ids=[0],
            engram_num_embeddings=[36],
            engram_max_ngram_size=3,
            engram_vocab_size=5,
            engram_n_heads=2,
            engram_head_dim=8,
            engram_pad_token_id=2,
            engram_compressed_vocab_size=32,
            rope_theta=10000,
            compress_rope_theta=160000,
            rope_scaling=None,
        ),
        vision_config=dict(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            patch_size=2,
            downsample_ratio=2,
            max_image_tokens=32,
            min_pixels=0,
        ),
        image_token_id=31,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
    )
    torch.manual_seed(42)
    return config, tokenizer, DeepseekV41ForCausalLM(config, tokenizer=tokenizer).eval()


class TestDeepseekV41RandomWeights(ExtTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls._threads)
        super().tearDownClass()

    @hide_stdout()
    @torch.no_grad()
    def test_compressor_empty_group_runtime_shape(self):
        from onnx_light.onnx import save_model
        from onnxruntime import InferenceSession

        from modelbuilder.builders.deepseek_v41 import trace_v41_model
        from modelbuilder.models.deepseek_v41 import Compressor, DeepseekV41TextConfig

        torch.manual_seed(42)
        config = DeepseekV41TextConfig(hidden_size=2, head_dim=2)
        module = Compressor(config, 2).eval()
        hidden = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 4
        positions = torch.arange(3, 6).expand(2, -1)
        partial = torch.zeros(2, 1, 2)
        inputs = ["hidden", "positions", "partial_values", "partial_scores"]
        outputs = ["pooled", "pooled_positions", "present_values", "present_scores"]
        axes = {name: {0: "batch"} for name in inputs + outputs}
        for name in inputs[:2]:
            axes[name][1] = "sequence"
        for name in outputs[:2]:
            axes[name][1] = "groups"
        graph = trace_v41_model(module, (hidden, positions, partial, partial.clone()), inputs, outputs, axes)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "compressor.onnx")
            save_model(graph, path)
            session = InferenceSession(path, providers=["CPUExecutionProvider"])
            for batch in (1, 2):
                pt_state = [torch.zeros(batch, 1, 2), torch.zeros(batch, 1, 2)]
                ort_state = [value.numpy() for value in pt_state]
                start = 0
                for sequence in (1, 1, 3, 1):
                    with self.subTest(batch=batch, start=start, sequence=sequence):
                        x = torch.arange(batch * sequence * 2, dtype=torch.float32).reshape(batch, sequence, 2) / 4
                        pos = torch.arange(start, start + sequence).expand(batch, -1)
                        expected = module(x, pos, *pt_state)
                        actual = session.run(None, dict(zip(inputs, [x.numpy(), pos.numpy(), *ort_state])))
                        for reference, value in zip(expected, actual):
                            self.assertEqual(value.shape, tuple(reference.shape))
                            np.testing.assert_allclose(value, reference.numpy(), atol=1e-6, rtol=1e-5)
                        pt_state, ort_state = expected[2:], actual[2:]
                    start += sequence

    def export(self, directory, precision="fp32", through_cli=False, provider="cpu"):
        from safetensors.torch import save_file

        from modelbuilder.builder import create_model
        from modelbuilder.builders.deepseek_v41 import DeepSeekV41Model

        config, tokenizer, module = make_tiny_v41()
        output = os.path.join(directory, "output")
        cache = os.path.join(directory, "cache")
        os.makedirs(output)
        os.makedirs(cache)
        if precision in {"fp16", "bf16"}:
            # The checkpoint loader regenerates nonpersistent RoPE/LUT buffers
            # in FP32; retain the same buffers in the independent reference.
            buffers = dict(module.named_buffers())
            module.to({"fp16": torch.float16, "bf16": torch.bfloat16}[precision])
            for name, value in buffers.items():
                owner, _, attribute = name.rpartition(".")
                setattr(module.get_submodule(owner), attribute, value)
        if through_cli:
            checkpoint = os.path.join(directory, "checkpoint")
            os.makedirs(checkpoint)
            config.save_pretrained(checkpoint)
            tokenizer.save_pretrained(checkpoint)
            save_file(module.state_dict(), os.path.join(checkpoint, "model.safetensors"))
            create_model("deepseek-ai/DeepSeek-V4.1-Flash", checkpoint, output, precision, provider, cache)
        else:
            io_dtype = {"fp32": TensorProto.FLOAT, "fp16": TensorProto.FLOAT16, "bf16": TensorProto.BFLOAT16, "int4": TensorProto.FLOAT}[
                precision
            ]
            builder = DeepSeekV41Model(config, io_dtype, TensorProto.INT4 if precision == "int4" else io_dtype, provider, cache, {})
            builder.export_module(module)
            builder.save_model(output)
            builder.make_genai_config("", {}, output)
        return module, output

    @torch.no_grad()
    def compare_prefill_decode(
        self, module, session, ids, types=None, embeds=None, atol=1e-4, reference_session=None, generated_tokens=None
    ):
        batch = ids.shape[0]
        if generated_tokens is not None:
            self.assertEqual(generated_tokens.shape, (batch, 6))
        if types is None:
            types = np.full(ids.shape, -1, dtype=np.int64)
        if embeds is None:
            embeds = np.zeros((*ids.shape, module.config.text_config.hidden_size), dtype=np.float32)
        pt_state = module.initial_state(batch)
        state = reference_state = None
        start = 0
        for step in range(6):
            if step:
                ids = np.full((batch, 1), 3 + step, dtype=np.int64) if generated_tokens is None else generated_tokens[:, step - 1 : step]
                types = np.full_like(ids, -1)
                embeds = np.zeros((batch, 1, module.config.text_config.hidden_size), dtype=np.float32)
            positions = np.broadcast_to(np.arange(start, start + ids.shape[1]), ids.shape).copy()
            expected, *pt_state = module(
                torch.from_numpy(ids),
                torch.from_numpy(positions),
                torch.from_numpy(types),
                torch.from_numpy(embeds.astype(np.float32, copy=False)).to(module.embed.weight.dtype),
                *pt_state,
            )
            kwargs = dict(position_ids=positions, token_types=types, inputs_embeds=embeds)
            actual, state = session.run(ids, state=state, **kwargs)
            np.testing.assert_allclose(actual, expected.float().numpy(), atol=atol, rtol=atol)
            if generated_tokens is not None:
                token = generated_tokens[:, step]
                np.testing.assert_array_equal(token, actual[:, -1].argmax(-1))
                reference_logits = expected[:, -1].float().numpy()
                gap = reference_logits.max(-1) - reference_logits[np.arange(batch), token]
                tolerance = 2 * atol * (1 + np.abs(reference_logits).max(-1))
                self.assertTrue(np.all(gap <= tolerance), f"Generated tokens exceed the reference error bound: {gap} > {tolerance}")
            if step == 0:
                with self.assertRaisesRegex(ValueError, "continue the recurrent state"):
                    session.run(ids, position_ids=positions, state=state)
            for name, value in zip(module.state_names, pt_state):
                expected_state = torch_tensor_to_numpy(value)
                self.assertEqual(state[name].shape, expected_state.shape, name)
                self.assertEqual(state[name].dtype, expected_state.dtype, name)
                if expected_state.dtype.kind in "iu":
                    np.testing.assert_array_equal(state[name], expected_state, err_msg=name)
                else:
                    self.assertTrue(np.isfinite(state[name].astype(np.float32)).all(), name)
            if reference_session is not None:
                reference, reference_state = reference_session.run(ids, state=reference_state, **kwargs)
                # FP4 cache bins amplify small accumulation differences, even
                # with identical weights. Generation still requires exact tokens.
                np.testing.assert_allclose(actual, reference, atol=0.05, rtol=0.05)
                self.assertEqual(set(state), set(reference_state))
                for name, value in state.items():
                    if value.dtype.kind in "iu":
                        np.testing.assert_array_equal(value, reference_state[name], err_msg=name)
            start += ids.shape[1]

    @torch.no_grad()
    def reference_tokens(self, module, ids, *, types=None, embeds=None, steps=6):
        batch, length = ids.shape
        dtype = module.embed.weight.dtype
        types = np.full_like(ids, -1) if types is None else types
        if embeds is None:
            embeds = np.zeros((*ids.shape, module.config.text_config.hidden_size), dtype=np.float32)
        logits, *state = module(
            torch.from_numpy(ids),
            torch.arange(length).expand(batch, -1),
            torch.from_numpy(types),
            torch.from_numpy(embeds.astype(np.float32, copy=False)).to(dtype),
            *module.initial_state(batch),
        )
        chunks = [ids]
        for step in range(steps):
            token = logits[:, -1].argmax(-1, keepdim=True)
            chunks.append(token.numpy())
            if step + 1 < steps:
                logits, *state = module(
                    token,
                    torch.full_like(token, length + step),
                    torch.full_like(token, -1),
                    torch.zeros(batch, 1, module.config.text_config.hidden_size, dtype=dtype),
                    *state,
                )
        return np.concatenate(chunks, 1)

    @torch.no_grad()
    def compare_image(self, module, session, atol=1e-4, check_tokens=True, reference_session=None, exact_tokens=True):
        from modelbuilder.helpers.deepseek_v41_image import DeepseekV41ImageProcessor

        image = np.arange(72, dtype=np.uint8).reshape(4, 6, 3) * 3
        types, embeds = session.encode_image(image)
        processor = DeepseekV41ImageProcessor(module.config.vision_config.to_dict())
        patches, grid, _ = processor(image)
        features = module.encode_image(torch.from_numpy(patches).to(module.embed.weight.dtype), int(grid[0]), int(grid[1]))
        np.testing.assert_allclose(embeds[types == 1].astype(np.float32), features.float().numpy(), atol=atol, rtol=atol)
        if reference_session is not None:
            reference_types, reference_embeds = reference_session.encode_image(image)
            np.testing.assert_array_equal(types, reference_types)
            np.testing.assert_allclose(embeds, reference_embeds, atol=1e-4, rtol=1e-4)
        ids = np.full((1, len(types) + 2), 31, dtype=np.int64)
        ids[0, [0, -1]] = [4, 5]
        image_types = np.full_like(ids, -1)
        image_types[0, 1:-1] = types
        image_embeds = np.zeros((*ids.shape, module.config.text_config.hidden_size), dtype=np.float32)
        image_embeds[0, 1:-1] = embeds
        self.compare_prefill_decode(module, session, ids, image_types, image_embeds, atol=atol, reference_session=reference_session)
        if check_tokens:
            generated = session.generate(ids, max_new_tokens=6, token_types=image_types, inputs_embeds=image_embeds, eos_token_id=[])
            if not exact_tokens:
                # BF16 arithmetic and FP4 cache rounding can change argmax.
                # Replay every generated prefix against independent PyTorch
                # logits and check the choices against its numerical bounds.
                self.compare_prefill_decode(
                    module, session, ids, image_types, image_embeds, atol=atol, generated_tokens=generated[:, ids.shape[1] :]
                )
                return
            expected = (
                self.reference_tokens(module, ids, types=image_types, embeds=image_embeds)
                if reference_session is None
                else reference_session.generate(ids, max_new_tokens=6, token_types=image_types, inputs_embeds=image_embeds, eos_token_id=[])
            )
            np.testing.assert_array_equal(generated, expected)

    @torch.no_grad()
    def check_int4_backends(self, backend):
        from onnx_light.onnx import helper, load_model

        from modelbuilder.helpers.deepseek_v41_runtime import DeepseekV41Session

        with tempfile.TemporaryDirectory() as directory:
            module, output = self.export(directory, precision="int4")
            reference = DeepseekV41Session(output)
            for component in ("decoder", "vision"):
                graph = load_model(os.path.join(output, reference.config[component]), load_external_data=False).graph
                weights = {value.name: value for value in graph.initializer}
                quantized = [node for node in graph.node if node.op_type == "MatMulNBits"]
                self.assertTrue(quantized, component)
                for node in quantized:
                    self.assertEqual(node.domain, "com.microsoft")
                    attrs = {value.name: helper.get_attribute_value(value) for value in node.attribute}
                    self.assertEqual(attrs["bits"], 4)
                    weight = weights[str(node.input[1])]
                    self.assertEqual(weight.data_type, TensorProto.UINT8)
                    self.assertEqual(weight.dims[-1], attrs["block_size"] // 2)
            session = DeepseekV41Session(output, backend=backend)
            for sequence, batch in ((1, 1), (3, 2), (12, 1)):
                with self.subTest(sequence=sequence, batch=batch):
                    ids = np.arange(batch * sequence).reshape(batch, sequence) % 25 + 4
                    self.compare_prefill_decode(module, session, ids, atol=0.15, reference_session=reference)
                    # The exact token oracle executes the same INT4 graph in
                    # ORT; FP32 PyTorch logits independently bound quantization
                    # error above, without assuming quantization preserves argmax.
                    np.testing.assert_array_equal(
                        session.generate(ids, max_new_tokens=6, eos_token_id=[]), reference.generate(ids, max_new_tokens=6, eos_token_id=[])
                    )
            self.compare_image(module, session, atol=0.15, reference_session=reference)

    @hide_stdout()
    def test_int4_text_and_vision_cpu_backends(self):
        self.check_int4_backends("onnx-light-cpu")

    @unittest.skip("Native MatMulNBits needs onnx-light-cpu; portable QDQ still needs the empty-cache fix: xadupre/onnx-light#5993.")
    @hide_stdout()
    def test_int4_text_and_vision_native(self):
        self.check_int4_backends("onnx-light")

    @hide_stdout()
    def test_bf16_text_and_vision_export(self):
        from onnx_light.onnx import checker, load_model

        with tempfile.TemporaryDirectory() as directory:
            _, output = self.export(directory, precision="bf16", through_cli=True)
            with open(os.path.join(output, "deepseek_v41_config.json")) as stream:
                config = json.load(stream)
            self.assertEqual(config["io_dtype"], "bfloat16")
            for component, input_name in (("decoder", "inputs_embeds"), ("vision", "patches")):
                with self.subTest(component=component):
                    graph = load_model(os.path.join(output, config[component]), load_external_data=True)
                    checker.check_model(graph, full_check=True)
                    inputs = {value.name: value for value in graph.graph.input}
                    self.assertEqual(inputs[input_name].type.tensor_type.elem_type, TensorProto.BFLOAT16)
                    self.assertTrue(any(value.data_type == TensorProto.BFLOAT16 for value in graph.graph.initializer))
            state = {value["input"]: value for value in config["state"]}
            self.assertEqual(state["engram_history"]["dtype"], "int64")
            self.assertEqual(state["layer_1_compressed"]["dtype"], "bfloat16")
            self.assertEqual(state["layer_1_compressed"]["shape"][1], 0)
            self.assertEqual(state["layer_1_partial_kv"]["dtype"], "float32")

    @torch.no_grad()
    def check_bf16_backend(self, backend="onnxruntime", providers=None):
        from modelbuilder.helpers.deepseek_v41_runtime import DeepseekV41Session

        with tempfile.TemporaryDirectory() as directory:
            module, output = self.export(directory, precision="bf16", through_cli=True, provider="cuda" if providers else "cpu")
            session = DeepseekV41Session(output, backend=backend, providers=providers)
            if providers:
                self.assertEqual(session.decoder.get_providers()[0], "CUDAExecutionProvider")
            for sequence, batch in ((1, 1), (3, 2), (12, 1)):
                with self.subTest(sequence=sequence, batch=batch):
                    ids = np.arange(batch * sequence).reshape(batch, sequence) % 25 + 4
                    self.compare_prefill_decode(module, session, ids, atol=0.1)
                    np.testing.assert_array_equal(
                        session.generate(ids, max_new_tokens=6, eos_token_id=[]), self.reference_tokens(module, ids)
                    )
            self.compare_image(module, session, atol=0.1, exact_tokens=False)
            if providers:
                self.assertEqual(session.vision.get_providers()[0], "CUDAExecutionProvider")

    @requires_cuda()
    @hide_stdout()
    def test_bf16_text_and_vision_cuda(self):
        self.check_bf16_backend(providers=["CUDAExecutionProvider"])

    @unittest.skip("ORT CPU has no BF16 Where/MatMul arithmetic: microsoft/onnxruntime#28768.")
    @hide_stdout()
    def test_bf16_text_and_vision_ort_cpu(self):
        self.check_bf16_backend()

    @unittest.skip("BF16 Where is unsupported: xadupre/onnx-light#5998.")
    @hide_stdout()
    def test_bf16_text_and_vision_light_cpu(self):
        self.check_bf16_backend(backend="onnx-light-cpu")

    @unittest.skip("BF16 Where is unsupported: xadupre/onnx-light#5998.")
    @hide_stdout()
    def test_bf16_text_and_vision_native(self):
        self.check_bf16_backend(backend="onnx-light")

    @hide_stdout()
    @torch.no_grad()
    def test_fp32_text_and_vision_all_runtimes(self):
        from modelbuilder.helpers.deepseek_v41_runtime import DeepseekV41Session

        with tempfile.TemporaryDirectory() as directory:
            module, output = self.export(directory, through_cli=True)
            for backend in ("onnxruntime", "onnx-light", "onnx-light-cpu"):
                with self.subTest(backend=backend):
                    if backend == "onnx-light":
                        self.skipTest("Empty-cache Pow broadcasting: xadupre/onnx-light#5993.")
                    session = DeepseekV41Session(output, backend=backend)
                    for sequence, batch in ((1, 1), (3, 2), (12, 1)):
                        ids = np.tile(np.arange(sequence) % 25 + 4, (batch, 1))
                        self.compare_prefill_decode(module, session, ids)
                    ids = np.array([[4, 5, 6]], dtype=np.int64)
                    np.testing.assert_array_equal(
                        session.generate(ids, max_new_tokens=6, eos_token_id=[]), self.reference_tokens(module, ids)
                    )
                    self.compare_image(module, session)

    @hide_stdout()
    def test_fp16_and_int4_cpu_discrepancies(self):
        from modelbuilder.helpers.deepseek_v41_runtime import DeepseekV41Session

        # Half-precision rounding can cross an FP4 cache bin boundary. The
        # generated-token checks below additionally guard against routing drift.
        for precision, atol in (("fp16", 0.05), ("int4", 0.15)):
            with self.subTest(precision=precision):
                with tempfile.TemporaryDirectory() as directory:
                    module, output = self.export(directory, precision=precision)
                    session = DeepseekV41Session(output)
                    ids = np.array([[4, 5, 6, 7, 8]], dtype=np.int64)
                    self.compare_prefill_decode(module, session, ids, atol=atol)
                    # Low-bit cache rounding and INT4 weights can change argmax.
                    self.compare_image(module, session, atol=atol, check_tokens=False)
                    if precision == "fp16":
                        np.testing.assert_array_equal(
                            session.generate(ids, max_new_tokens=6, eos_token_id=[]), self.reference_tokens(module, ids)
                        )

    @unittest.skip("GenAI recurrent-state updates are blocked: xadupre/mbext#604.")
    def test_genai_cached_text_and_vision_generation(self):
        pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
