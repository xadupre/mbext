# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Offline portability and numerical checks for the canonical any target."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnx_light.onnx import TensorProto, helper, load
from transformers import AutoModelForCausalLM, LlamaConfig, MistralConfig, PhiConfig, Qwen3Config

from modelbuilder.builder import create_model, parse_extra_options
from modelbuilder.builders.base import Model
from modelbuilder.builders.portable import make_portable_attention, validate_portable_model
from modelbuilder.ext_test_case import ExtTestCase, hide_stdout


class TestAnyEP(ExtTestCase):
    def _config(self, cls, **extra):
        return cls(
            architectures=[cls.__name__.replace("Config", "ForCausalLM")],
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=32,
            vocab_size=41,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            **extra,
        )

    def _check_export(self, config, precision="fp32", execute=True):
        torch.manual_seed(42)
        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
        reference = AutoModelForCausalLM.from_config(config).eval().to(torch_dtype)
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "export"
            reference.save_pretrained(source)
            self.make_word_level_tokenizer().save_pretrained(source)
            create_model(
                None, str(source), str(output), precision, "any", str(Path(tmp) / "cache"), hf_remote=False, hf_token="false", onnx_opset=23
            )
            path = output / "model.onnx"
            proto = load(str(path), load_external_data=False)
            self.assertFalse(proto.functions)
            self.assertLessEqual({item.domain for item in proto.opset_import}, {"", "ai.onnx"})
            self.assertTrue(all(node.domain in {"", "ai.onnx"} for node in proto.graph.node))
            self.assertNotIn("GroupQueryAttention", {node.op_type for node in proto.graph.node})
            self.assertEqual(sum(node.op_type == "Attention" for node in proto.graph.node), config.num_hidden_layers)
            self.assertNotIn("Softmax", {node.op_type for node in proto.graph.node})
            self.assertNotIn("Tile", {node.op_type for node in proto.graph.node})
            locations = {entry.value for tensor in proto.graph.initializer for entry in tensor.external_data if entry.key == "location"}
            self.assertTrue(locations)
            self.assertLessEqual(locations, {"model.onnx.data", ".weights/model.safetensors"})
            genai = json.loads((output / "genai_config.json").read_text())
            self.assertEqual(genai["model"]["decoder"]["session_options"]["provider_options"], [])
            self.assertFalse(genai["search"]["past_present_share_buffer"])
            if not execute:
                self.assertEqual(proto.graph.input[-1].type.tensor_type.elem_type, TensorProto.BFLOAT16)
                return
            dtype = np.float16 if precision == "fp16" else np.float32
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            cache = {
                f"past_key_values.{i}.{kind}": np.zeros(
                    (2, config.num_key_value_heads, 0, config.hidden_size // config.num_attention_heads), dtype=dtype
                )
                for i in range(config.num_hidden_layers)
                for kind in ("key", "value")
            }
            ids = np.array([[0, 3, 4, 5], [6, 7, 8, 9]], dtype=np.int64)
            mask = (ids != 0).astype(np.int64)
            for step, length in enumerate((4, 1, 2)):
                if step:
                    ids = np.concatenate([ids, np.full((2, length), 10 + step, dtype=np.int64)], axis=1)
                    mask = np.concatenate([mask, np.ones((2, length), dtype=np.int64)], axis=1)
                positions = np.maximum(np.cumsum(mask, axis=1) - 1, 0)
                feeds = {"input_ids": ids[:, -length:], "attention_mask": mask, "position_ids": positions[:, -length:], **cache}
                result = dict(zip([value.name for value in session.get_outputs()], session.run(None, feeds)))
                with torch.no_grad():
                    expected = reference(
                        input_ids=torch.from_numpy(ids),
                        attention_mask=torch.from_numpy(mask),
                        position_ids=torch.from_numpy(positions),
                        use_cache=True,
                    )
                tolerance = 2e-3 if precision == "fp16" else 2e-5
                self.assertEqualArray(
                    expected.logits[:, -1:].float().numpy(), result["logits"][:, -1:].astype(np.float32), atol=tolerance, rtol=tolerance
                )
                for i in range(config.num_hidden_layers):
                    for kind in ("key", "value"):
                        present = result[f"present.{i}.{kind}"]
                        self.assertEqual(
                            present.shape, (2, config.num_key_value_heads, ids.shape[1], config.hidden_size // config.num_attention_heads)
                        )
                        cache[f"past_key_values.{i}.{kind}"] = present

    @hide_stdout()
    def test_llama_prefill_decode(self):
        self._check_export(self._config(LlamaConfig))

    @hide_stdout()
    def test_llama_fp16_prefill_decode(self):
        self._check_export(self._config(LlamaConfig), "fp16")

    @hide_stdout()
    def test_llama_bf16_standard_export(self):
        self._check_export(self._config(LlamaConfig), "bf16", execute=False)

    @hide_stdout()
    def test_qwen_qk_norm_prefill_decode(self):
        self._check_export(self._config(Qwen3Config))

    @hide_stdout()
    def test_mistral_sliding_window_prefill_decode(self):
        self._check_export(self._config(MistralConfig, sliding_window=3))

    @hide_stdout()
    def test_phi_layernorm_partial_rope_prefill_decode(self):
        self._check_export(self._config(PhiConfig, partial_rotary_factor=0.5))

    def test_reject_integer_precisions_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            for precision in ("int2", "int4", "int8", "int16"):
                with self.subTest(precision=precision), self.assertRaisesRegex(ValueError, "--ep any supports only"):
                    create_model(None, os.path.join(tmp, "missing"), tmp, precision, "any", tmp)

    def test_reject_provider_options(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "provider-specific"):
            create_model(None, os.path.join(tmp, "missing"), tmp, "fp32", "any", tmp, enable_cuda_graph=True)

    @hide_stdout()
    def test_reject_provider_options_during_cli_parsing(self):
        for option in ("enable_cuda_graph", "enable_webgpu_graph", "use_cuda_bf16", "use_webgpu_fp32"):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, "provider-specific"):
                parse_extra_options([f"{option}=true"], "any")

    def test_reject_older_opset(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "opset 23"):
            Model(self._config(LlamaConfig), TensorProto.FLOAT, TensorProto.FLOAT, "any", tmp, {"onnx_opset": 21})

    def test_reject_unknown_contrib_operator(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Model(self._config(LlamaConfig), TensorProto.FLOAT, TensorProto.FLOAT, "any", tmp, {})
            with self.assertRaisesRegex(NotImplementedError, "com.microsoft::MoE"):
                model.make_node("MoE", [], ["out"], name="unsupported", domain="com.microsoft")

    def test_reject_nested_contrib_operator(self):
        body = helper.make_graph(
            [helper.make_node("Gelu", ["x"], ["y"], domain="com.microsoft")],
            "branch",
            [],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        graph = helper.make_graph(
            [helper.make_node("If", ["condition"], ["result"], then_branch=body, else_branch=body)],
            "main",
            [helper.make_tensor_value_info("condition", TensorProto.BOOL, []), helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("result", TensorProto.FLOAT, [1])],
        )
        proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
        with self.assertRaisesRegex(NotImplementedError, "com.microsoft::Gelu"):
            validate_portable_model(proto)

    def test_skip_norm_bias_and_activation_lowerings(self):
        rng = np.random.default_rng(42)
        x, skip = [rng.normal(size=(2, 3, 32)).astype(np.float32) for _ in range(2)]
        gamma, beta, bias = [rng.normal(size=32).astype(np.float32) for _ in range(3)]
        for simple in (False, True):
            with self.subTest(simple=simple), tempfile.TemporaryDirectory() as tmp:
                model = Model(self._config(LlamaConfig), TensorProto.FLOAT, TensorProto.FLOAT, "any", tmp, {"onnx_opset": 23})
                feeds = {"x": x, "skip": skip, "gamma": gamma, "bias": bias}
                inputs = ["x", "skip", "gamma", "bias"] if simple else ["x", "skip", "gamma", "beta", "bias"]
                if not simple:
                    feeds["beta"] = beta
                for key, value in feeds.items():
                    model.make_graph_input(key, TensorProto.FLOAT, value.shape)
                model.make_node(
                    "SkipSimplifiedLayerNormalization" if simple else "SkipLayerNormalization",
                    inputs,
                    ["norm", "", "", "sum"],
                    name="norm",
                    domain="com.microsoft",
                    epsilon=1e-5,
                )
                activation = "QuickGelu" if simple else "FastGelu"
                model.make_node(activation, ["norm"] if simple else ["norm", "bias"], ["output"], name="activation", domain="com.microsoft")
                for output in ("sum", "output"):
                    model.make_graph_output(output, TensorProto.FLOAT, x.shape)
                session = ort.InferenceSession(model.to_model_proto().SerializeToString(), providers=["CPUExecutionProvider"])
                summed, activated = session.run(None, feeds)
                expected_sum = torch.from_numpy(x + skip + bias)
                self.assertEqualArray(expected_sum.numpy(), summed, atol=1e-6)
                if simple:
                    norm = expected_sum * torch.rsqrt(expected_sum.square().mean(dim=-1, keepdim=True) + 1e-5)
                    norm *= torch.from_numpy(gamma)
                    expected = norm * torch.sigmoid(1.702 * norm)
                else:
                    norm = torch.nn.functional.layer_norm(expected_sum, (32,), torch.from_numpy(gamma), torch.from_numpy(beta))
                    expected = torch.nn.functional.gelu(norm + torch.from_numpy(bias), approximate="tanh")
                self.assertEqualArray(expected.numpy(), activated, atol=1e-6, rtol=1e-5)

    def test_attention_softcap_sinks_and_fully_padded_queries(self):
        rng = np.random.default_rng(42)
        q = rng.normal(size=(2, 3, 32)).astype(np.float32)
        k = rng.normal(size=(2, 3, 16)).astype(np.float32)
        v = rng.normal(size=(2, 3, 16)).astype(np.float32)
        past_k = rng.normal(size=(2, 2, 2, 8)).astype(np.float32)
        past_v = rng.normal(size=(2, 2, 2, 8)).astype(np.float32)
        mask = np.array([[0, 1, 1, 1, 1], [0, 0, 0, 0, 0]], dtype=np.int64)
        for use_sinks in (False, True):
            with self.subTest(sinks=use_sinks), tempfile.TemporaryDirectory() as tmp:
                model = Model(self._config(LlamaConfig), TensorProto.FLOAT, TensorProto.FLOAT, "any", tmp, {"onnx_opset": 23})
                model.window_size = 3
                model.attention_attrs["softcap"] = 1.5
                feeds = {"q": q, "k": k, "v": v, "past_k": past_k, "past_v": past_v, "attention_mask": mask}
                if use_sinks:
                    feeds["sinks"] = np.linspace(-1, 1, 4, dtype=np.float32)
                for key, value in feeds.items():
                    model.make_graph_input(key, TensorProto.INT64 if key == "attention_mask" else TensorProto.FLOAT, value.shape)
                make_portable_attention(
                    model,
                    "attention",
                    q_path="q",
                    k_path="k",
                    v_path="v",
                    past_k="past_k",
                    past_v="past_v",
                    present_k="present_k",
                    present_v="present_v",
                    sinks="sinks" if use_sinks else "",
                )
                model.make_graph_output("attention/output_0", TensorProto.FLOAT, [2, 3, 32])
                for label in ("k", "v"):
                    model.make_graph_output(f"present_{label}", TensorProto.FLOAT, [2, 2, 5, 8])
                proto = model.to_model_proto()
                self.assertEqual(sum(node.op_type == "Attention" for node in proto.graph.node), 0 if use_sinks else 1)
                session = ort.InferenceSession(proto.SerializeToString(), providers=["CPUExecutionProvider"])
                actual, actual_key, actual_value = session.run(None, feeds)
                query = torch.from_numpy(q).reshape(2, 3, 4, 8).transpose(1, 2)
                key = torch.cat([torch.from_numpy(past_k), torch.from_numpy(k).reshape(2, 3, 2, 8).transpose(1, 2)], dim=2)
                value = torch.cat([torch.from_numpy(past_v), torch.from_numpy(v).reshape(2, 3, 2, 8).transpose(1, 2)], dim=2)
                self.assertEqualArray(key.numpy(), actual_key)
                self.assertEqualArray(value.numpy(), actual_value)
                key, value = key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1)
                scores = torch.tanh((query @ key.transpose(-1, -2)) / np.sqrt(8) / 1.5) * 1.5
                positions = torch.arange(5)
                queries = torch.arange(2, 5)[:, None]
                allowed = (positions <= queries) & (positions > queries - 3)
                allowed = allowed[None, None] & torch.from_numpy(mask[:, None, None].astype(bool))
                scores = scores.masked_fill(~allowed, float("-inf"))
                if use_sinks:
                    sink = torch.from_numpy(feeds["sinks"])[None, :, None, None].expand(2, 4, 3, 1)
                    scores = torch.cat([scores, sink], dim=-1)
                probabilities = torch.softmax(scores, dim=-1)[..., :5]
                probabilities = torch.where(allowed, probabilities, 0)
                expected = (probabilities @ value).transpose(1, 2).reshape(2, 3, 32)
                self.assertEqualArray(expected.numpy(), actual, atol=1e-6, rtol=1e-5)
                self.assertTrue(np.isfinite(actual).all())

    def test_cli_ep_alias(self):
        result = subprocess.run(
            [sys.executable, "-m", "modelbuilder.builder", "--ep", "any", "-p", "int4", "-i", "missing", "-o", "unused"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--ep any supports only", result.stderr)
        self.assertNotIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
