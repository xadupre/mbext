# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx_light.onnx as onnx
import onnx_light.onnx.helper as oh
import onnxruntime as ort
import torch
from onnx_light.onnx.numpy_helper import to_array
from safetensors.torch import save_file
from transformers import AutoConfig, LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from modelbuilder.builder import create_model
from modelbuilder.builders.llama import LlamaModel
from modelbuilder.builders.qwen import Qwen3Model, QwenModel
from modelbuilder.ext_test_case import ExtTestCase, hide_stdout
from modelbuilder.helpers.checkpoint import DenseCheckpoint


class TestDenseCheckpoint(ExtTestCase):
    def make_config(self, cls, **kwargs):
        return cls(
            vocab_size=41,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=32,
            **kwargs,
        )

    def export(self, builder_type, checkpoint, output, dtype, **options):
        output.mkdir()
        builder = builder_type(
            AutoConfig.from_pretrained(checkpoint), dtype, dtype, "cpu", str(output / "cache"), {"disable_qkv_fusion": True, **options}
        )
        builder.make_model(str(checkpoint))
        builder.save_model(str(output))
        return onnx.load(str(output / "model.onnx"), load_external_data=False)

    def run_decoder(self, path, config, outputs=None):
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        feeds = {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.ones((1, 3), dtype=np.int64),
            "position_ids": np.arange(3, dtype=np.int64)[None, :],
        }
        for layer in range(config.num_hidden_layers):
            for kind in ("key", "value"):
                feeds[f"past_key_values.{layer}.{kind}"] = np.empty((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32)
        for value in session.get_inputs():
            if value.type == "tensor(float16)":
                feeds[value.name] = feeds[value.name].astype(np.float16)
        results = session.run(outputs, {value.name: feeds[value.name] for value in session.get_inputs()})
        return results if outputs is not None else results[0]

    @hide_stdout()
    def test_dense_exports_match_transformers_path_and_reference(self):
        cases = (
            (LlamaConfig, LlamaForCausalLM, LlamaModel, torch.float32, onnx.TensorProto.FLOAT),
            (Qwen2Config, Qwen2ForCausalLM, QwenModel, torch.float32, onnx.TensorProto.FLOAT),
            (Qwen3Config, Qwen3ForCausalLM, Qwen3Model, torch.bfloat16, onnx.TensorProto.FLOAT16),
        )
        for config_type, model_type, builder_type, source_dtype, target_dtype in cases:
            with self.subTest(architecture=model_type.__name__), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = self.make_config(config_type, tie_word_embeddings=config_type != Qwen2Config, attention_bias=True, mlp_bias=True)
                torch.manual_seed(42)
                reference = model_type(config).eval().to(source_dtype)
                reference.save_pretrained(root / "checkpoint", max_shard_size="4KB")
                loaded = DenseCheckpoint(root / "checkpoint", AutoConfig.from_pretrained(root / "checkpoint"))
                self.assertNotIsInstance(loaded, torch.nn.Module)
                if config.tie_word_embeddings:
                    self.assertIs(loaded.embedding, loaded.lm_head.weight)
                self.assertEqual(loaded.embedding.dtype, source_dtype)
                legacy_type = type("LegacyBuilder", (builder_type,), {})
                direct = self.export(builder_type, root / "checkpoint", root / "direct", target_dtype)
                legacy = self.export(legacy_type, root / "checkpoint", root / "legacy", target_dtype)
                legacy_weights = {value.name: value for value in legacy.graph.initializer}
                external = [value for value in direct.graph.initializer if value.name.startswith("/modelbuilder/checkpoint/")]
                self.assertTrue(external)
                self.assertTrue(all(value.data_location == onnx.TensorProto.EXTERNAL and not value.raw_data for value in external))
                for value in external:
                    source_name = value.name.removeprefix("/modelbuilder/checkpoint/")
                    source = loaded.tensors[source_name].sources[0]
                    self.assertEqual(value.data_type, source.dtype)
                    self.assertEqual(list(value.dims), list(source.shape))
                    reference_data = {entry.key: entry.value for entry in value.external_data}
                    self.assertEqual(int(reference_data["offset"]), source.offset)
                    self.assertEqual(int(reference_data["length"]), source.length)
                    self.assertEqual((root / "direct" / reference_data["location"]).stat().st_nlink, 1)
                    self.assertTrue(Path(source.path).exists())
                evaluated = onnx.ModelProto()
                evaluated.CopyFrom(direct)
                for name, value in legacy_weights.items():
                    evaluated.graph.output.append(oh.make_tensor_value_info(name, value.data_type, value.dims))
                onnx.save_model(evaluated, str(root / "direct" / "weights.onnx"))
                values = self.run_decoder(root / "direct" / "weights.onnx", config, list(legacy_weights))
                for (name, value), actual_value in zip(legacy_weights.items(), values):
                    with self.subTest(initializer=name):
                        np.testing.assert_array_equal(actual_value, to_array(value, str(root / "legacy")))
                expected_q = reference.model.layers[0].self_attn.q_proj.weight.detach().float().T.numpy()
                actual_q = dict(zip(legacy_weights, values))["model.layers.0.attn.q_proj.MatMul.weight"]
                np.testing.assert_array_equal(actual_q, expected_q.astype(actual_q.dtype))
                actual = self.run_decoder(root / "direct" / "model.onnx", config)
                previous = self.run_decoder(root / "legacy" / "model.onnx", config)
                np.testing.assert_array_equal(actual, previous)
                with torch.no_grad():
                    expected = reference.float()(torch.tensor([[1, 2, 3]])).logits.numpy()
                np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=2e-3)

    @hide_stdout()
    def test_cli_export_does_not_import_transformers_model_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            revision = "a" * 40
            repository = cache / "models--offline--dense"
            checkpoint = repository / "snapshots" / revision
            model = Qwen3ForCausalLM(self.make_config(Qwen3Config))
            model.save_pretrained(checkpoint)
            self.make_word_level_tokenizer().save_pretrained(checkpoint)
            (repository / "refs").mkdir()
            (repository / "refs" / "main").write_text(revision)
            code = """
import builtins
import os
import struct
import sys
from modelbuilder.builder import create_model
from modelbuilder.helpers import checkpoint
precision = sys.argv[4]
if precision == 'fp32':
    def forbidden(*args, **kwargs):
        raise AssertionError('Checkpoint tensor payload materialized')
    checkpoint.safe_open = forbidden
    original_open = builtins.open
    class HeaderOnlyFile:
        def __init__(self, stream):
            self.stream = stream
            self.limit = 8 + struct.unpack('<Q', stream.read(8))[0]
            stream.seek(0)
        def read(self, size=-1):
            assert 0 <= size <= self.limit - self.stream.tell(), 'Checkpoint payload read'
            return self.stream.read(size)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.stream.close()
        def __getattr__(self, name):
            return getattr(self.stream, name)
    def header_only_open(path, *args, **kwargs):
        stream = original_open(path, *args, **kwargs)
        return HeaderOnlyFile(stream) if os.fspath(path).endswith('.safetensors') else stream
    checkpoint.open = header_only_open
create_model('offline/dense', sys.argv[1], sys.argv[2], precision, 'cpu', sys.argv[3])
if precision == 'fp32':
    create_model('offline/dense', sys.argv[1], sys.argv[2], precision, 'cpu', sys.argv[3])
assert 'transformers.models.qwen3.modeling_qwen3' not in sys.modules
"""
            for mode, source, precision in (("local", str(checkpoint), "fp32"), ("hub", "", "fp32"), ("quant", str(checkpoint), "int4")):
                with self.subTest(mode=mode):
                    output = root / mode
                    result = subprocess.run(
                        [sys.executable, "-c", code, source, str(output), str(cache), precision],
                        capture_output=True,
                        text=True,
                        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertExists(output / "model.onnx")

    @hide_stdout()
    def test_layer_limit_and_excluded_components_preserve_existing_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Qwen3ForCausalLM(self.make_config(Qwen3Config)).save_pretrained(root / "checkpoint")
            options = {"num_hidden_layers": 1, "exclude_embeds": True, "exclude_lm_head": True}
            legacy_type = type("LegacyBuilder", (Qwen3Model,), {})
            direct = self.export(Qwen3Model, root / "checkpoint", root / "direct", onnx.TensorProto.FLOAT, **options)
            legacy = self.export(legacy_type, root / "checkpoint", root / "legacy", onnx.TensorProto.FLOAT, **options)
            self.assertEqual(
                [node.op_type for node in direct.graph.node if not node.name.startswith("/modelbuilder/")],
                [node.op_type for node in legacy.graph.node],
            )
            self.assertEqual([value.name for value in direct.graph.input], [value.name for value in legacy.graph.input])
            self.assertEqual([value.name for value in direct.graph.output], [value.name for value in legacy.graph.output])
            self.assertFalse(any(value.name.startswith("/modelbuilder/checkpoint/model.layers.1.") for value in direct.graph.initializer))

    def test_invalid_checkpoint_tensors_are_reported(self):
        config = self.make_config(Qwen3Config, tie_word_embeddings=False)
        with tempfile.TemporaryDirectory() as directory:
            save_file({"model.embed_tokens.weight": torch.ones(41, 64)}, Path(directory) / "model.safetensors")
            checkpoint = DenseCheckpoint(directory, config)
            with self.assertRaisesRegex(ValueError, "Missing dense checkpoint tensor 'model.layers.0"):
                checkpoint.layer(0)
            save_file({"model.embed_tokens.weight": torch.ones(40, 64)}, Path(directory) / "model.safetensors")
            checkpoint = DenseCheckpoint(directory, config)
            with self.assertRaisesRegex(ValueError, "has shape .* expected"):
                _ = checkpoint.embedding
            save_file({"model.embed_tokens.weight": torch.tensor(1.0)}, Path(directory) / "model.safetensors")
            checkpoint = DenseCheckpoint(directory, config)
            with self.assertRaisesRegex(ValueError, "has shape .* expected"):
                _ = checkpoint.embedding
            save_file({"model.embed_tokens.weight": torch.ones(41, 64, dtype=torch.int64)}, Path(directory) / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "must contain floating-point weights"):
                DenseCheckpoint(directory, config)

    @hide_stdout()
    def test_tied_checkpoint_weight_reuse_preserves_selected_storage(self):
        for embedding_present in (True, False):
            with self.subTest(embedding_present=embedding_present), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "checkpoint"
                source.mkdir()
                model = Qwen3ForCausalLM(self.make_config(Qwen3Config, tie_word_embeddings=True)).to(torch.bfloat16)
                model.config.architectures = ["Qwen3ForCausalLM"]
                model.config.save_pretrained(source)
                self.make_word_level_tokenizer().save_pretrained(source)
                tensors = {name: value.clone() for name, value in model.state_dict().items()}
                tensors["lm_head.weight"].add_(1)
                if not embedding_present:
                    del tensors["model.embed_tokens.weight"]
                save_file(tensors, source / "model.safetensors")
                legacy_type = type("LegacyBuilder", (Qwen3Model,), {})
                self.export(legacy_type, source, root / "legacy", onnx.TensorProto.FLOAT)
                expected = self.run_decoder(root / "legacy" / "model.onnx", model.config)
                self.export(Qwen3Model, source, root / "normal", onnx.TensorProto.FLOAT)
                np.testing.assert_array_equal(self.run_decoder(root / "normal" / "model.onnx", model.config), expected)
                create_model(
                    "offline", str(source), str(root / "direct"), "fp32", "cpu", str(root / "cache"), reuse_downloaded_weights=True
                )
                self.assertFalse((source / "model.safetensors").exists())
                actual = self.run_decoder(root / "direct" / "model.onnx", model.config)
                np.testing.assert_array_equal(actual, expected)

    @hide_stdout()
    def test_bin_checkpoint_keeps_transformers_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Qwen3ForCausalLM(self.make_config(Qwen3Config))
            model.config.architectures = ["Qwen3ForCausalLM"]
            model.config.save_pretrained(directory)
            torch.save(model.state_dict(), Path(directory) / "pytorch_model.bin")
            builder = Qwen3Model(
                AutoConfig.from_pretrained(directory), onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT, "cpu", directory, {}
            )
            self.assertIsInstance(builder.load_weights(directory), torch.nn.Module)
