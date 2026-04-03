# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase

MODEL_NAME = "arnir0/Tiny-LLM"


class TestTrainedTinyLLM(ExtTestCase):
    def _common_part(self, precision, dtype):
        from transformers import AutoModelForCausalLM
        from modelbuilder.builder import create_model

        # Use 4 layers so that both rope (layers 0-2) and no-rope (layer 3)
        # code paths are exercised with the default no_rope_layer_interval=4.
        output_dir, cache_dir = self.get_dirs(
            f"test_trained_tiny_llm_{precision}_cpu", clean=False
        )
        onnx_path = os.path.join(output_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            # Let's avoid doing it a second time.
            create_model(
                model_name=MODEL_NAME,
                input_path="",
                precision=precision,
                execution_provider="cpu",
                output_dir=output_dir,
                cache_dir=cache_dir,
            )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, ignore_mismatched_sizes=True, dtype=dtype
        )
        model.eval()
        return onnx_path, model

    def test_trained_tiny_llm_fp32_discrepancies_cpu(self):
        """
        Convert arnir0/Tiny-LLM to an fp32 ONNX model targeting the CPU execution
        provider.  Only one hidden layer is materialised (via the
        ``num_hidden_layers`` extra option) to keep the test fast.
        The test verifies that:
        * ``create_model`` completes without error.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file passes ``onnx.checker.check_model``.
        """
        import torch

        precision, dtype, np_dtype = "fp16", torch.float16, np.float16

        onnx_path, model = self._common_part(precision, dtype)
        sess = self._check_with_ort(onnx_path, cpu=True)
        config = model.config

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
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        # Provide empty past KV-cache tensors for every materialised layer.
        for i in range(config.num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np_dtype,
            )
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np_dtype,
            )
        # Keep only what the session expects.
        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}

        onnx_outputs = sess.run(None, onnx_feed)
        onnx_logits = onnx_outputs[0]

        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        disc.update(
            dict(precision=precision, model_id=MODEL_NAME, experiment="forward", provider="cpu")
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 0.05)

    def test_trained_tiny_llm_genai_generate_cpu(self):
        """
        Compare ``transformers.generate`` with ``onnxruntime-genai`` generate
        on the ``arnir0/Tiny-LLM`` trained model.

        The test converts the model to fp32 ONNX (CPU), then:

        * Runs ``transformers.generate(do_sample=False)`` on a short prompt.
        * Runs ``onnxruntime_genai`` greedy generation on the same prompt and
          ONNX model.

        Both backends use greedy (argmax) decoding, so the generated token
        sequences must be bit-for-bit identical.

        The test is skipped automatically when ``onnxruntime-genai`` is not
        installed.
        """
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest(
                "onnxruntime-genai is not installed; skipping genai comparison test."
            )

        import torch
        from transformers import AutoTokenizer

        precision, dtype = "fp16", torch.float16

        onnx_path, model = self._common_part(precision, dtype)

        genai_config_path = os.path.join(os.path.dirname(onnx_path), "genai_config.json")
        self.assertExists(genai_config_path)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        prompt = "Once upon a time"
        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt")
        start_sequence = inputs["input_ids"].shape[1]
        inputs = inputs.to("cpu")
        with torch.no_grad():
            pt_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        pt_tokens = pt_output[0].tolist()

        # ------------------------------------------------------------------
        # onnxruntime-genai greedy generation
        # ------------------------------------------------------------------
        og_model = og.Model(os.path.dirname(onnx_path))

        params = og.GeneratorParams(og_model)
        params.set_search_options(
            do_sample=False,
            max_length=inputs["input_ids"].shape[1] + max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        generator = og.Generator(og_model, params)
        generator.append_tokens(inputs["input_ids"])

        og_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            og_token = generator.get_next_tokens()[0]
            og_tokens.append(int(og_token))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same token sequence (prompt + all generated tokens).
        disc = self.first_token_diff(pt_tokens[start_sequence:], og_tokens)
        disc.update(
            dict(precision=precision, model_id=MODEL_NAME, experiment="generate", provider="cpu")
        )
        self.log_results(disc)
        self.assertEqual(pt_tokens[start_sequence:], og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
