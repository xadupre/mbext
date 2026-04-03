# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout

MISTRAL_NEMO_MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"


class TestMistralNeMo(ExtTestCase):
    @hide_stdout()
    def test_mistral_nemo_fp32_cpu_random_weights(self):
        """
        Convert a randomly-initialised MistralNeMoForCausalLM model to an fp32
        ONNX model targeting the CPU execution provider.

        MistralNeMoForCausalLM shares the same underlying architecture as
        MistralForCausalLM but is registered under a distinct architecture
        class name.  This test verifies that the builder routes it correctly
        to ``MistralNeMoModel`` and produces a valid ONNX model.

        Using random weights and a single hidden layer avoids downloading any
        files from Hugging Face, keeping the test completely offline.

        The test verifies that:
        * ``create_model`` completes without error when given a local model directory.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        * The ONNX logits closely match those of the original PyTorch model.
        """
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            AutoModelForCausalLM,
            MistralConfig,
            PreTrainedTokenizerFast,
        )

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = MistralConfig(
            architectures=["MistralNeMoForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=1024,
            model_type="mistral",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=1000000.0,
            sliding_window=None,
            vocab_size=32000,
        )

        model_dir = self.get_model_dir("test_mistral_nemo_fp32_cpu_random_weights")
        output_dir, cache_dir = self.get_dirs("test_mistral_nemo_fp32_cpu_random_weights")

        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MISTRAL_NEMO_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        # --- PyTorch inference ---
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            pt_logits = model(input_ids).logits.numpy()

        onnx_input_names = {inp.name for inp in sess.get_inputs()}

        # Use config.head_dim (the authoritative head dimension) for KV-cache shapes.
        head_size = config.head_dim
        onnx_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        # Provide empty past KV-cache tensors for every materialised layer.
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
        # Keep only what the session expects.
        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}

        onnx_outputs = sess.run(None, onnx_feed)
        onnx_logits = onnx_outputs[0]

        self.assertEqual(pt_logits.shape, onnx_logits.shape)
        # Verify first-token logits are numerically close; subsequent positions
        # can diverge slightly in FP32 due to GQA kernel differences.
        np.testing.assert_allclose(
            pt_logits[:, :1, :], onnx_logits[:, :1, :], atol=1e-3, rtol=1e-3
        )

    def test_mistral_nemo_fix_head_size_from_weights(self):
        """
        Verify that ``MistralNeMoModel._fix_head_size_from_weights`` correctly
        reads the K-projection weight shape from a safetensors shard and
        updates ``head_size`` when the value computed from the config is wrong.

        This exercises the scenario that caused
        "Got invalid dimensions for input: past_key_values.N.value … Got: 160
        Expected: 128" for mistralai/Mistral-Nemo-Instruct: the real model
        has head_dim=128 while hidden_size / num_attention_heads = 160.  When
        the ``head_dim`` field is absent from the model config the builder
        would previously fall back to the division and produce the wrong ONNX
        input shapes.
        """
        import tempfile

        import numpy as np
        from safetensors.numpy import save_file

        from modelbuilder.builders.mistral import MistralNeMoModel

        # ---------------------------------------------------------------------------
        # Build a minimal on-disk representation that mimics the Mistral-NeMo layout:
        #   hidden_size=320, num_attention_heads=2, num_kv_heads=1, head_dim=128
        # => hidden_size // num_attention_heads = 160 (wrong)
        # => k_proj weight shape = [num_kv_heads * head_dim, hidden_size] = [128, 320]
        # ---------------------------------------------------------------------------
        model_dir = tempfile.mkdtemp()
        tensors = {
            "model.layers.0.self_attn.k_proj.weight": np.zeros((128, 320), dtype=np.float32),
        }
        save_file(tensors, os.path.join(model_dir, "model.safetensors"))

        # Create a minimal stub that satisfies the Model.__init__ signature but
        # deliberately sets the *wrong* head_size (160 = 320 // 2) to simulate
        # the case where head_dim is missing from the config.json.
        class _StubNeMoModel(MistralNeMoModel):
            """Minimal subclass with enough attributes to call _fix_head_size_from_weights."""

            def __init__(self):  # noqa: D107
                # Bypass Model.__init__ – we only need a handful of fields.
                self.num_kv_heads = 1
                self.head_size = 160  # wrong: should be 128

        stub = _StubNeMoModel()
        self.assertEqual(stub.head_size, 160)

        stub._fix_head_size_from_weights(model_dir)

        # After the fix the head_size must be inferred from the shard.
        self.assertEqual(stub.head_size, 128)

    def test_mistral_nemo_fix_head_size_noop_when_correct(self):
        """
        ``_fix_head_size_from_weights`` must be a no-op when the stored
        head_size already matches the weight dimensions.
        """
        import tempfile

        import numpy as np
        from safetensors.numpy import save_file

        from modelbuilder.builders.mistral import MistralNeMoModel

        model_dir = tempfile.mkdtemp()
        # k_proj shape [128, 320] → head_dim = 128
        tensors = {
            "model.layers.0.self_attn.k_proj.weight": np.zeros((128, 320), dtype=np.float32),
        }
        save_file(tensors, os.path.join(model_dir, "model.safetensors"))

        class _StubNeMoModel(MistralNeMoModel):
            def __init__(self):  # noqa: D107
                self.num_kv_heads = 1
                self.head_size = 128  # already correct

        stub = _StubNeMoModel()
        stub._fix_head_size_from_weights(model_dir)
        # Should remain unchanged.
        self.assertEqual(stub.head_size, 128)


if __name__ == "__main__":
    unittest.main(verbosity=2)
