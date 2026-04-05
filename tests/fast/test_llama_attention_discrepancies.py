# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout

# LlamaForCausalLM architecture matching arnir0/Tiny-LLM but with a single
# hidden layer and smaller dimensions to keep tests fast and completely offline.
_NUM_HIDDEN_LAYERS = 1
_LLAMA_CONFIG_KWARGS = dict(
    architectures=["LlamaForCausalLM"],
    bos_token_id=1,
    eos_token_id=2,
    hidden_act="silu",
    hidden_size=512,
    intermediate_size=1376,
    max_position_embeddings=2048,
    model_type="llama",
    num_attention_heads=8,
    num_hidden_layers=_NUM_HIDDEN_LAYERS,
    num_key_value_heads=4,
    rms_norm_eps=1e-05,
    rope_theta=10000.0,
    vocab_size=32000,
)

_MODEL_ID = "arnir0/Tiny-LLM"


class TestLlamaAttentionDiscrepancies(ExtTestCase):
    """Fast discrepancy tests for LlamaAttention using randomly initialised weights.

    These tests check that the ONNX model produced by ``create_model`` for a
    LlamaForCausalLM architecture (arnir0/Tiny-LLM) numerically agrees with
    the original PyTorch model.  Random weights are used so that no files are
    downloaded from Hugging Face, making every test suitable for offline CI
    environments.

    Discrepancy metrics are recorded via ``log_results`` and hard thresholds
    are enforced with ``assertLess`` to catch any regression in the ONNX
    conversion pipeline.
    """

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_model_and_session(self, test_name):
        """Create a randomly initialised LlamaForCausalLM, convert it to ONNX
        and return the PyTorch model together with an ORT InferenceSession."""
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            AutoModelForCausalLM,
            LlamaConfig,
            PreTrainedTokenizerFast,
        )

        from modelbuilder.builder import create_model

        config = LlamaConfig(**_LLAMA_CONFIG_KWARGS)
        model_dir = self.get_model_dir(test_name)
        output_dir, cache_dir = self.get_dirs(test_name)

        torch.manual_seed(42)
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
            model_name=_MODEL_ID,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=_NUM_HIDDEN_LAYERS,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)
        return config, model, sess

    @staticmethod
    def _empty_kv_cache(config, batch_size, head_size):
        """Return a dict of empty (zero-length sequence) KV-cache arrays."""
        kv = {}
        for i in range(_NUM_HIDDEN_LAYERS):
            kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
            kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
        return kv

    # ------------------------------------------------------------------
    # Test: prefill (forward-pass) discrepancies
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_llama_attention_prefill_discrepancies(self):
        """Check numerical discrepancies between PyTorch and ONNX for the
        LlamaAttention prefill (full-sequence forward) pass.

        A randomly initialised LlamaForCausalLM is converted to an fp32 ONNX
        model.  Both backends are fed the same ``input_ids`` and the resulting
        logits are compared using ``get_numpy_discrepancy``.  Discrepancy
        metrics are logged and a hard threshold on ``max_abs_err`` is enforced.
        """
        import torch

        test_name = "test_llama_attention_prefill_discrepancies"
        config, model, sess = self._build_model_and_session(test_name)

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # PyTorch reference
        with torch.no_grad():
            pt_logits = model(input_ids).logits.numpy()

        # ONNX inference
        onnx_input_names = {inp.name for inp in sess.get_inputs()}
        feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        feed.update(self._empty_kv_cache(config, batch_size, head_size))
        feed = {k: v for k, v in feed.items() if k in onnx_input_names}

        onnx_logits = sess.run(None, feed)[0]

        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        disc.update(
            dict(
                precision="fp32",
                model_id=_MODEL_ID,
                experiment="prefill",
                provider="cpu",
                test=test_name,
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 1e-3)

    # ------------------------------------------------------------------
    # Test: decode-step discrepancies
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_llama_attention_decode_discrepancies(self):
        """Check numerical discrepancies between PyTorch and ONNX for the
        LlamaAttention single-token decode step (with KV-cache).

        Two passes are run:

        1. **Prefill** – a prompt of ``seq_len`` tokens is processed with an
           empty KV-cache.  The ONNX ``present.*`` tensors are kept as the
           KV-cache for the next step.

        2. **Decode** – a single new token is processed using the KV-cache
           produced in step 1.

        The decode-step logits from ONNX and PyTorch are compared with
        ``get_numpy_discrepancy`` and a hard threshold is enforced.
        """
        import torch

        test_name = "test_llama_attention_decode_discrepancies"
        config, model, sess = self._build_model_and_session(test_name)

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads

        onnx_input_names = {inp.name for inp in sess.get_inputs()}
        onnx_output_names = [out.name for out in sess.get_outputs()]

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        # ------------------------------------------------------------------
        # Step 1: prefill
        # ------------------------------------------------------------------
        prefill_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        prefill_feed.update(self._empty_kv_cache(config, batch_size, head_size))
        prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

        prefill_outputs = sess.run(None, prefill_feed)
        prefill_results = dict(zip(onnx_output_names, prefill_outputs))

        next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

        # ------------------------------------------------------------------
        # Step 2: single-token decode using the KV-cache from prefill
        # ------------------------------------------------------------------
        decode_feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
            "position_ids": np.array([[seq_len]], dtype=np.int64),
        }
        for i in range(_NUM_HIDDEN_LAYERS):
            decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
            decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
        decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

        onnx_decode_logits = sess.run(None, decode_feed)[0]

        # ------------------------------------------------------------------
        # PyTorch reference: same two steps
        # ------------------------------------------------------------------
        with torch.no_grad():
            pt_prefill = model(input_ids)
            pt_past_kv = pt_prefill.past_key_values

            next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
            pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
            pt_decode_logits = pt_decode.logits.numpy()

        disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
        disc.update(
            dict(
                precision="fp32",
                model_id=_MODEL_ID,
                experiment="decode",
                provider="cpu",
                test=test_name,
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
