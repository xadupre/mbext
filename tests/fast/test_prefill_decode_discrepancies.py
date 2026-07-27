# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Measures prefill and decode discrepancies on a full Tiny-LLM (Llama) pipeline.

Unlike the attention-only tests in ``test_llama_attention_discrepancies.py``,
this module exercises the entire causal-LM pipeline (embedding → LayerNorm →
attention → MLP → final norm → LM head) and reports discrepancy metrics
separately for the prefill pass and the decode step.
"""

import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

MODEL_NAME = "arnir0/Tiny-LLM"

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
    num_hidden_layers=1,
    num_key_value_heads=4,
    rms_norm_eps=1e-05,
    rope_theta=10000.0,
    vocab_size=32000,
)


@requires_transformers("5.0")
class TestPrefillDecodeDiscrepancies(ExtTestCase):
    """Full-pipeline prefill vs decode discrepancy measurements.

    Each test builds a complete LlamaForCausalLM model with random weights,
    exports it to ONNX, runs prefill and decode passes, and logs
    discrepancy metrics separately for each step.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_and_export(self, basename, precision, provider):
        """Build a random-weight Llama model, export to ONNX, return (model, session)."""
        import torch
        from transformers import AutoModelForCausalLM, LlamaConfig

        from modelbuilder.builder import create_model

        config = LlamaConfig(**_LLAMA_CONFIG_KWARGS)
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)

        tokenizer = self.make_word_level_tokenizer()
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path, provider=provider)
        return config, model, sess

    def _run_prefill(self, config, model, sess, precision, provider, batch_size=1, seq_len=5):
        """Run prefill pass on both PyTorch and ONNX, return discrepancy dict and state."""
        import torch

        vocab_size = config.vocab_size
        num_hidden_layers = config.num_hidden_layers
        num_key_value_heads = config.num_key_value_heads
        head_size = config.hidden_size // config.num_attention_heads

        onnx_input_names = [i.name for i in sess.get_inputs()]
        onnx_output_names = [o.name for o in sess.get_outputs()]

        torch.manual_seed(0)
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(provider)

        # ONNX prefill feed
        np_dtype = self.get_input_np_dtype(precision)
        feed = {
            "input_ids": input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        for i in range(num_hidden_layers):
            feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, num_key_value_heads, 0, head_size), dtype=np_dtype)
            feed[f"past_key_values.{i}.value"] = np.zeros((batch_size, num_key_value_heads, 0, head_size), dtype=np_dtype)
        feed = {k: v for k, v in feed.items() if k in onnx_input_names}

        onnx_results_raw = sess.run(None, feed)
        onnx_results = dict(zip(onnx_output_names, onnx_results_raw))
        onnx_logits = onnx_results["logits"]

        # PyTorch prefill
        with torch.no_grad():
            pt_out = model(input_ids)
        pt_logits = pt_out.logits.detach().cpu().numpy()

        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        return disc, onnx_results, pt_out, input_ids

    def _run_decode(self, config, model, sess, precision, provider, onnx_results, pt_out, batch_size=1, seq_len=5):
        """Run a single decode step using KV cache from prefill, return discrepancy dict."""
        import torch

        num_hidden_layers = config.num_hidden_layers
        onnx_input_names = [i.name for i in sess.get_inputs()]
        onnx_output_names = [o.name for o in sess.get_outputs()]

        # Pick next token from ONNX prefill logits
        next_token = int(np.argmax(onnx_results["logits"][0, -1, :]))
        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)

        # ONNX decode feed
        decode_feed = {
            "input_ids": np.array([[next_token]], dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
            "position_ids": np.array([[seq_len]], dtype=np.int64),
        }
        for i in range(num_hidden_layers):
            decode_feed[f"past_key_values.{i}.key"] = onnx_results[f"present.{i}.key"]
            decode_feed[f"past_key_values.{i}.value"] = onnx_results[f"present.{i}.value"]
        decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

        onnx_decode_raw = sess.run(None, decode_feed)
        onnx_decode_results = dict(zip(onnx_output_names, onnx_decode_raw))
        onnx_decode_logits = onnx_decode_results["logits"]

        # PyTorch decode
        with torch.no_grad():
            pt_decode = model(next_token_tensor, past_key_values=pt_out.past_key_values)
        pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

        disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
        return disc

    def _common_prefill_decode_discrepancies(self, precision, provider):
        """Shared test body: build model, run prefill and decode, log and assert."""
        basename = f"test_prefill_decode_{precision}_{provider}"
        config, model, sess = self._build_and_export(basename, precision, provider)

        atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}

        # --- Prefill ---
        prefill_disc, onnx_results, pt_out, input_ids = self._run_prefill(config, model, sess, precision, provider)
        prefill_disc.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="prefill",
                provider=provider,
                test=basename,
                input_type="text",
                step="prefill",
            )
        )
        self.log_results(prefill_disc)
        self.assertLess(
            prefill_disc["max_abs_err"], atol[precision], f"Prefill max_abs_err={prefill_disc['max_abs_err']:.6e} exceeds threshold"
        )

        # --- Decode ---
        decode_disc = self._run_decode(config, model, sess, precision, provider, onnx_results, pt_out)
        decode_disc.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="decode",
                provider=provider,
                test=basename,
                input_type="text",
                step="decode",
            )
        )
        self.log_results(decode_disc)
        self.assertLess(
            decode_disc["max_abs_err"], atol[precision], f"Decode max_abs_err={decode_disc['max_abs_err']:.6e} exceeds threshold"
        )

    # ------------------------------------------------------------------
    # Test methods
    # ------------------------------------------------------------------

    @hide_stdout()
    def test_prefill_decode_discrepancies_fp32_cpu(self):
        self._common_prefill_decode_discrepancies("fp32", "cpu")

    @hide_stdout()
    def test_prefill_decode_discrepancies_fp16_cpu(self):
        self._common_prefill_decode_discrepancies("fp16", "cpu")

    @unittest.skip("fails due to incorrect model on CUDA fp32")
    @hide_stdout()
    @requires_cuda()
    def test_prefill_decode_discrepancies_fp32_cuda(self):
        self._common_prefill_decode_discrepancies("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_prefill_decode_discrepancies_fp16_cuda(self):
        self._common_prefill_decode_discrepancies("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
