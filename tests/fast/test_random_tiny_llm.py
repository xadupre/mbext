# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout


class TestRandomTinyLLM(ExtTestCase):
    @hide_stdout()
    def test_tiny_llm_fp32_cpu_random_weights(self):
        """
        Convert a model with the same architecture as arnir0/Tiny-LLM but with
        randomly initialised weights to an fp32 ONNX model targeting the CPU
        execution provider.

        Using random weights avoids downloading the pretrained weights from
        Hugging Face, making this test completely offline and suitable for CI
        environments without internet access.  A single hidden layer is used
        to keep the test fast.

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
            LlamaConfig,
            PreTrainedTokenizerFast,
        )
        from modelbuilder.builder import create_model

        # Config matching the arnir0/Tiny-LLM architecture (LlamaForCausalLM)
        # but with a single hidden layer to keep the test fast.  These values
        # are hardcoded so the test runs completely offline without downloading
        # any files from Hugging Face.
        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        model_dir = self.get_model_dir("test_tiny_llm_fp32_cpu_random_weights")
        output_dir, cache_dir = self.get_dirs("test_tiny_llm_fp32_cpu_random_weights")
        num_hidden_layers = 1

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
            model_name="arnir0/Tiny-LLM",
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

        head_size = config.hidden_size // config.num_attention_heads
        onnx_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(
                batch_size, seq_len
            ),
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

        np.testing.assert_allclose(pt_logits, onnx_logits, atol=1e-3, rtol=1e-3)

    @hide_stdout()
    def test_tiny_llm_fp32_cpu_decoding_step(self):
        """
        Verify the auto-regressive *decoding* step for a randomly initialised
        arnir0/Tiny-LLM (LlamaForCausalLM) converted to fp32 ONNX.

        The test runs two inference passes:

        1. **Prefill** – feed a prompt of ``seq_len`` tokens with an empty
           KV-cache.  The ONNX ``present.*`` tensors are retained as the
           KV-cache for the next step.

        2. **Decode** – feed a single new token together with the KV-cache
           produced in step 1.  The ONNX logits must closely match those
           produced by the original PyTorch model when called with the
           equivalent ``past_key_values`` argument.

        Using random weights keeps the test completely offline and suitable
        for CI environments without internet access.
        """
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            AutoModelForCausalLM,
            LlamaConfig,
            PreTrainedTokenizerFast,
        )
        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        model_dir = self.get_model_dir("test_tiny_llm_fp32_cpu_decoding_step")
        output_dir, cache_dir = self.get_dirs("test_tiny_llm_fp32_cpu_decoding_step")

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
            model_name="arnir0/Tiny-LLM",
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

        onnx_input_names = {inp.name for inp in sess.get_inputs()}
        onnx_output_names = [out.name for out in sess.get_outputs()]

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        # Fix random seed so the input token IDs are deterministic across runs.
        # The seed is local to this scope; it does not persist across tests
        # because each test creates a fresh random model with its own weights.
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # ------------------------------------------------------------------
        # Step 1: prefill with an empty KV-cache
        # ------------------------------------------------------------------
        prefill_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(
                batch_size, seq_len
            ),
        }
        for i in range(num_hidden_layers):
            prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
            prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
        prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

        prefill_outputs = sess.run(None, prefill_feed)
        prefill_results = dict(zip(onnx_output_names, prefill_outputs))

        # The next token is the greedy choice from the prefill logits.  The
        # exact token value is not important here; what matters is that both
        # ONNX and PyTorch process the *same* token in the decode step so
        # their outputs can be compared.
        next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

        # ------------------------------------------------------------------
        # Step 2: single-token decode using the KV-cache from prefill
        # ------------------------------------------------------------------
        decode_feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
            "position_ids": np.array([[seq_len]], dtype=np.int64),
        }
        for i in range(num_hidden_layers):
            decode_feed[f"past_key_values.{i}.key"] = prefill_results[
                f"present.{i}.key"
            ]
            decode_feed[f"past_key_values.{i}.value"] = prefill_results[
                f"present.{i}.value"
            ]
        decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

        decode_outputs = sess.run(None, decode_feed)
        onnx_decode_logits = decode_outputs[0]

        # ------------------------------------------------------------------
        # PyTorch reference: run the same two steps
        # ------------------------------------------------------------------
        with torch.no_grad():
            pt_prefill = model(input_ids)
            pt_past_kv = pt_prefill.past_key_values

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
            pt_decode_logits = pt_decode.logits.numpy()

        np.testing.assert_allclose(
            pt_decode_logits, onnx_decode_logits, atol=1e-3, rtol=1e-3
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
