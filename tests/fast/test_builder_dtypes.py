# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for the dtype/token helpers in :mod:`modelbuilder.builder`.
"""

import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from onnx_light.onnx import load
from onnx_light.onnx.numpy_helper import to_array
from onnx_light.onnx import TensorProto as ir

from modelbuilder.builder import create_model, parse_hf_token, set_io_dtype, set_onnx_dtype
from modelbuilder.builders.base import Model
from modelbuilder.ext_test_case import ExtTestCase, hide_stdout


class TestParseHfToken(ExtTestCase):
    def test_disabled_variants_return_none(self):
        for value in ("false", "False", "0"):
            with self.subTest(value=value):
                self.assertIsNone(parse_hf_token(value))

    def test_enabled_variants_return_true(self):
        for value in ("true", "True", "1"):
            with self.subTest(value=value):
                self.assertIs(parse_hf_token(value), True)

    def test_user_token_returned_as_is(self):
        self.assertEqual(parse_hf_token("hf_secret_token"), "hf_secret_token")


class TestSetIoDtype(ExtTestCase):
    def test_fp32_and_int_cpu_are_float(self):
        self.assertEqual(set_io_dtype("fp32", "cpu", {}), ir.DataType.FLOAT)
        for precision in ("int2", "int4", "int8"):
            with self.subTest(precision=precision):
                self.assertEqual(set_io_dtype(precision, "cpu", {}), ir.DataType.FLOAT)

    def test_int4_cpu_is_float(self):
        self.assertEqual(set_io_dtype("int4", "cpu", {}), ir.DataType.FLOAT)

    def test_webgpu_fp32_option(self):
        self.assertEqual(set_io_dtype("int4", "webgpu", {"use_webgpu_fp32": True}), ir.DataType.FLOAT)
        self.assertEqual(set_io_dtype("int4", "webgpu", {}), ir.DataType.FLOAT16)

    def test_bf16_precision(self):
        self.assertEqual(set_io_dtype("bf16", "cpu", {}), ir.DataType.BFLOAT16)

    def test_int4_cuda_bf16_option(self):
        self.assertEqual(set_io_dtype("int4", "cuda", {"use_cuda_bf16": True}), ir.DataType.BFLOAT16)
        self.assertEqual(set_io_dtype("int4", "trt-rtx", {"use_cuda_bf16": True}), ir.DataType.BFLOAT16)

    def test_default_is_fp16(self):
        self.assertEqual(set_io_dtype("fp16", "cuda", {}), ir.DataType.FLOAT16)
        self.assertEqual(set_io_dtype("int4", "cuda", {}), ir.DataType.FLOAT16)


class TestAttentionDtypes(ExtTestCase):
    def test_cpu_bf16_supports_group_query_attention(self):
        model = Model.__new__(Model)
        model.ep = "cpu"
        model.io_dtype = ir.DataType.BFLOAT16
        self.assertTrue(model.is_gqa_supported())


class TestRotaryConfiguration(ExtTestCase):
    @hide_stdout()
    def test_partial_rotary_cache_dimensions(self):
        from transformers import PhiConfig

        for legacy_factor, nested_factor, rotary_dim in (
            (0.5, None, 8),
            (None, 0.25, 4),
            (0.5, 0.25, 4),
            (None, None, 16),
            (None, 1.0, 16),
        ):
            with self.subTest(legacy_factor=legacy_factor, nested_factor=nested_factor):
                config = PhiConfig(
                    architectures=["PhiForCausalLM"],
                    hidden_size=64,
                    intermediate_size=128,
                    max_position_embeddings=32,
                    num_attention_heads=4,
                    num_hidden_layers=1,
                    vocab_size=32,
                )
                if legacy_factor is not None:
                    config.partial_rotary_factor = legacy_factor
                elif hasattr(config, "partial_rotary_factor"):
                    del config.partial_rotary_factor
                config.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}
                if nested_factor is not None:
                    config.rope_parameters["partial_rotary_factor"] = nested_factor
                with tempfile.TemporaryDirectory() as cache_dir:
                    model = Model(config, ir.FLOAT, ir.FLOAT, "cpu", cache_dir, {})
                    model.rope_attrs["save_caches"] = False
                    cos_cache, sin_cache = model.make_rotary_embedding_caches()
                self.assertEqual(cos_cache.shape, (32, rotary_dim // 2))
                self.assertEqual(sin_cache.shape, cos_cache.shape)
                self.assertEqual(model.rope_attrs["rotary_embedding_dim"], 0 if rotary_dim == 16 else rotary_dim)


class TestSetOnnxDtype(ExtTestCase):
    def test_int4_symmetric_default(self):
        self.assertEqual(set_onnx_dtype("int4", {}), ir.DataType.INT4)

    def test_int_precisions_use_nbits_container(self):
        for precision in ("int2", "int4", "int8"):
            with self.subTest(precision=precision):
                self.assertEqual(set_onnx_dtype(precision, {}), ir.DataType.INT4)
                self.assertEqual(set_onnx_dtype(precision, {"int4_is_symmetric": False}), ir.DataType.UINT4)

    def test_int4_asymmetric(self):
        self.assertEqual(set_onnx_dtype("int4", {"int4_is_symmetric": False}), ir.DataType.UINT4)

    def test_float_precisions(self):
        self.assertEqual(set_onnx_dtype("fp32", {}), ir.DataType.FLOAT)
        self.assertEqual(set_onnx_dtype("fp16", {}), ir.DataType.FLOAT16)
        self.assertEqual(set_onnx_dtype("bf16", {}), ir.DataType.BFLOAT16)

    def test_unknown_precision_raises(self):
        with self.assertRaises(KeyError):
            set_onnx_dtype("int3", {})


class TestSourceWeightRelease(ExtTestCase):
    def test_releases_unique_module_parameters(self):
        holder = Model.__new__(Model)
        holder.weights = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.Linear(4, 2))
        holder._initialize_source_tensor_tracking()

        first = holder.weights[0]
        second = holder.weights[1]
        holder._release_source_module(first)

        self.assertEqual(first.weight.numel(), 0)
        self.assertEqual(first.bias.numel(), 0)
        self.assertGreater(second.weight.numel(), 0)

    def test_keeps_tied_parameter_until_final_alias(self):
        holder = Model.__new__(Model)
        holder.weights = torch.nn.Module()
        holder.weights.embedding = torch.nn.Embedding(8, 4)
        holder.weights.lm_head = torch.nn.Linear(4, 8, bias=False)
        holder.weights.lm_head.weight = holder.weights.embedding.weight
        holder._initialize_source_tensor_tracking()

        tied_weight = holder.weights.embedding.weight
        holder._release_source_module(holder.weights.embedding)
        self.assertEqual(tied_weight.numel(), 32)

        holder._release_source_module(holder.weights.lm_head)
        self.assertEqual(tied_weight.numel(), 0)


class TestReuseDownloadedWeights(ExtTestCase):
    @hide_stdout()
    def test_config_only_does_not_require_or_move_checkpoint_weights(self):
        from transformers import LlamaConfig

        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "source", Path(tmp) / "export"
            config = LlamaConfig(
                architectures=["LlamaForCausalLM"],
                hidden_size=16,
                intermediate_size=24,
                num_attention_heads=2,
                num_hidden_layers=1,
                num_key_value_heads=1,
                vocab_size=32,
            )
            config.save_pretrained(source)
            self.make_word_level_tokenizer().save_pretrained(source)
            for has_weights in (False, True):
                with self.subTest(has_weights=has_weights):
                    if has_weights:
                        (source / "model.safetensors").write_bytes(b"not needed for config-only")
                        (output / ".weights").mkdir()
                        (output / ".weights" / "model.safetensors").write_bytes(b"existing destination")
                    create_model(
                        model_name=None,
                        input_path=str(source),
                        output_dir=str(output),
                        precision="fp32",
                        execution_provider="cpu",
                        cache_dir=os.path.join(tmp, "cache"),
                        reuse_downloaded_weights=True,
                        config_only=True,
                    )
                    self.assertTrue((output / "genai_config.json").exists())
                    self.assertFalse((output / "model.onnx").exists())
                    if has_weights:
                        self.assertEqual((source / "model.safetensors").read_bytes(), b"not needed for config-only")
                        self.assertEqual((output / ".weights" / "model.safetensors").read_bytes(), b"existing destination")

    @hide_stdout()
    def test_symlinked_export_remains_relocatable(self):
        from transformers import AutoModelForCausalLM, LlamaConfig

        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=16,
            intermediate_size=24,
            max_position_embeddings=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=1,
            vocab_size=32,
        )
        with tempfile.TemporaryDirectory() as tmp:
            output, alias = Path(tmp) / "export", Path(tmp) / "export_alias"
            output.mkdir()
            alias.symlink_to(output, target_is_directory=True)
            source = alias / ".weights"
            AutoModelForCausalLM.from_config(config).save_pretrained(source)
            self.make_word_level_tokenizer().save_pretrained(source)
            create_model(
                model_name=None,
                input_path=str(source),
                output_dir=str(alias),
                precision="fp32",
                execution_provider="cpu",
                cache_dir=os.path.join(tmp, "cache"),
                reuse_downloaded_weights=True,
            )
            model = load(output / "model.onnx", load_external_data=False)
            locations = {entry.value for tensor in model.graph.initializer for entry in tensor.external_data if entry.key == "location"}
            self.assertEqual(locations, {os.path.join(".weights", "model.safetensors")})
            alias.unlink()
            relocated = Path(tmp) / "relocated"
            output.rename(relocated)
            self._check_with_ort(str(relocated / "model.onnx"), cpu=True)

    @hide_stdout()
    def test_reuses_safetensors_checkpoint(self):
        from transformers import AutoModelForCausalLM, LlamaConfig

        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=16,
            intermediate_size=24,
            max_position_embeddings=16,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_key_value_heads=1,
            tie_word_embeddings=True,
            vocab_size=32,
        )
        model = AutoModelForCausalLM.from_config(config)
        output_dir, cache_dir = self.get_dirs("test_reuse_downloaded_weights")
        model_dir = os.path.join(output_dir, ".weights")
        os.makedirs(model_dir)
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)

        create_model(
            model_name=None,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp16",
            execution_provider="cpu",
            cache_dir=cache_dir,
            reuse_downloaded_weights=True,
        )

        model_proto = load(os.path.join(output_dir, "model.onnx"), load_external_data=False)
        external_locations = {
            entry.value for initializer in model_proto.graph.initializer for entry in initializer.external_data if entry.key == "location"
        }
        self.assertIn(os.path.join(".weights", "model.safetensors"), external_locations)
        self.assertFalse(os.path.exists(os.path.join(output_dir, "model.safetensors")))
        model_with_weights = load(os.path.join(output_dir, "model.onnx"))
        embedding = next(
            initializer for initializer in model_with_weights.graph.initializer if initializer.name == "model.embed_tokens.weight.source"
        )
        self.assertEqualArray(to_array(embedding, output_dir), model.model.embed_tokens.weight.detach().numpy())
        self.assertIn(
            "model.embed_tokens.weight",
            {output for node in model_with_weights.graph.node if node.op_type == "Cast" for output in node.output},
        )
        self.assertFalse(list(Path(output_dir).glob("*.data*")))
        self._check_with_ort(os.path.join(output_dir, "model.onnx"), cpu=True)
        original_bytes = (Path(model_dir) / "model.safetensors").read_bytes()
        create_model(
            model_name=None,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp16",
            execution_provider="cpu",
            cache_dir=cache_dir,
            reuse_downloaded_weights=True,
        )
        self.assertEqual((Path(model_dir) / "model.safetensors").read_bytes(), original_bytes)
        self.assertFalse(list(Path(output_dir).glob("*.data*")))

    @hide_stdout()
    def test_moves_mixed_quantized_shards_and_relocates_export(self):
        import json

        import onnxruntime as ort
        from transformers import AutoModelForCausalLM, LlamaConfig

        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            hidden_size=64,
            intermediate_size=64,
            max_position_embeddings=16,
            num_attention_heads=4,
            num_hidden_layers=1,
            num_key_value_heads=2,
            tie_word_embeddings=True,
            vocab_size=96,
        )
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config).eval()
        with tempfile.TemporaryDirectory() as tmp:
            source, output, cache = (os.path.join(tmp, name) for name in ("checkpoint", "export", "cache"))
            model.save_pretrained(source, max_shard_size="20KB")
            self.make_word_level_tokenizer().save_pretrained(source)
            shards = {path.name: path.read_bytes() for path in Path(source).glob("*.safetensors")}
            self.assertGreater(len(shards), 1)
            (Path(source) / "unrelated.txt").write_text("keep")
            create_model(
                model_name=None,
                input_path=source,
                output_dir=output,
                precision="int4",
                execution_provider="cpu",
                cache_dir=cache,
                reuse_downloaded_weights=True,
                max_shard_size=4096,
                int4_nodes_to_exclude=["/lm_head/MatMul"],
            )
            self.assertFalse(list(Path(source).glob("*.safetensors")))
            self.assertEqual((Path(source) / "unrelated.txt").read_text(), "keep")
            for name, content in shards.items():
                self.assertEqual((Path(output) / ".weights" / name).read_bytes(), content)
            path = os.path.join(output, "model.onnx")
            proto = load(path, load_external_data=False)
            self.assertIn("MatMulNBits", {node.op_type for node in proto.graph.node})
            original, generated, inline = [], [], []
            for initializer in proto.graph.initializer:
                ref = {entry.key: entry.value for entry in initializer.external_data}
                if not ref:
                    inline.append(initializer)
                elif ref["location"].endswith(".safetensors"):
                    original.append(ref)
                    content = shards[os.path.basename(ref["location"])]
                    header_length = struct.unpack("<Q", content[:8])[0]
                    header = json.loads(content[8 : 8 + header_length])
                    start, length = int(ref["offset"]), int(ref["length"])
                    matching = [
                        value
                        for key, value in header.items()
                        if key != "__metadata__" and 8 + header_length + value["data_offsets"][0] == start
                    ]
                    self.assertTrue(matching)
                    metadata = matching[0]
                    self.assertEqual(length, metadata["data_offsets"][1] - metadata["data_offsets"][0])
                    self.assertEqual(tuple(initializer.dims), tuple(metadata["shape"]))
                    self.assertEqual(initializer.data_type, ir.FLOAT)
                    self.assertEqual(to_array(initializer, output).tobytes(), content[start : start + length])
                else:
                    generated.append(ref)
                    self.assertIn(".onnx.data", ref["location"])
                    self.assertGreaterEqual(int(ref["length"]), 1024)
            self.assertTrue(original)
            self.assertTrue(generated)
            self.assertTrue(inline)
            self.assertLess(len({(r["location"], r["offset"], r["length"]) for r in original}), len(original))
            self.assertTrue(any("_Q4" in t.name and t.data_location == ir.EXTERNAL for t in proto.graph.initializer))
            feeds = {
                "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
                "position_ids": np.arange(3, dtype=np.int64).reshape(1, 3),
                "past_key_values.0.key": np.zeros((1, 2, 0, 16), dtype=np.float32),
                "past_key_values.0.value": np.zeros((1, 2, 0, 16), dtype=np.float32),
            }
            session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            feeds = {value.name: feeds[value.name] for value in session.get_inputs()}
            expected = session.run(None, feeds)
            del session
            with torch.no_grad():
                torch_logits = model(torch.tensor([[1, 2, 3]])).logits.numpy()
            np.testing.assert_allclose(expected[0], torch_logits, atol=0.06, rtol=0.1)
            relocated = os.path.join(tmp, "relocated")
            shutil.move(output, relocated)
            shutil.rmtree(source)
            session = ort.InferenceSession(os.path.join(relocated, "model.onnx"), providers=["CPUExecutionProvider"])
            actual = session.run(None, feeds)
            for a, b in zip(expected, actual):
                self.assertEqualArray(a, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
