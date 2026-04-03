# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, long_test

SMOLLM3_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"

# Persistent HF model cache shared across test runs – not cleaned up per test.
_HF_CACHE_DIR = os.path.join("dump_models", "hf_cache")
os.makedirs(_HF_CACHE_DIR, exist_ok=True)


class TestTrainedSmolLM3(ExtTestCase):
    @long_test()
    def test_trained_smollm3_fp32_cpu(self):
        """
        Convert HuggingFaceTB/SmolLM3-3B to an fp32 ONNX model targeting the
        CPU execution provider.  Only a small number of hidden layers is
        materialised (via ``num_hidden_layers``) so the test completes in a
        reasonable time while still exercising both rope and no-rope code
        paths (the default SmolLM3 config places a NoPE layer every 4th layer).

        The test verifies that:
        * ``create_model`` completes without error.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        * The ONNX logits closely match those of the original PyTorch model.
        """
        import torch
        from transformers import AutoModelForCausalLM
        from modelbuilder.builder import create_model

        # Use 4 layers so that both rope (layers 0-2) and no-rope (layer 3)
        # code paths are exercised with the default no_rope_layer_interval=4.
        num_hidden_layers = 4

        output_dir, _ = self.get_dirs("test_smollm3_fp32_cpu")
        create_model(
            model_name=SMOLLM3_MODEL_NAME,
            input_path="",
            precision="fp32",
            execution_provider="cpu",
            output_dir=output_dir,
            cache_dir=_HF_CACHE_DIR,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=True)

        model = AutoModelForCausalLM.from_pretrained(
            SMOLLM3_MODEL_NAME, ignore_mismatched_sizes=True
        )
        model.eval()
        config = model.config

        batch_size = 1
        seq_len = 5
        torch.manual_seed(0)
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

    @long_test()
    def test_trained_smollm3_genai_generate(self):
        """
        Compare ``transformers.generate`` with ``onnxruntime-genai`` generate
        on the ``HuggingFaceTB/SmolLM3-3B`` trained model.

        The test converts the model to fp32 ONNX (CPU) using 4 hidden layers
        (so that both RoPE and NoPE code paths are exercised with the default
        ``no_rope_layer_interval=4``), then:

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
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        from modelbuilder.builder import create_model

        num_hidden_layers = 4

        output_dir, _ = self.get_dirs("test_smollm3_genai_generate")
        create_model(
            model_name=SMOLLM3_MODEL_NAME,
            input_path="",
            precision="fp32",
            execution_provider="cpu",
            output_dir=output_dir,
            cache_dir=_HF_CACHE_DIR,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)

        # Load the PyTorch model with the same reduced layer count so that its
        # outputs match those of the truncated ONNX model exactly.
        config = AutoConfig.from_pretrained(SMOLLM3_MODEL_NAME, cache_dir=_HF_CACHE_DIR)
        config.num_hidden_layers = num_hidden_layers
        model = AutoModelForCausalLM.from_pretrained(
            SMOLLM3_MODEL_NAME,
            config=config,
            ignore_mismatched_sizes=True,
            cache_dir=_HF_CACHE_DIR,
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(SMOLLM3_MODEL_NAME, cache_dir=_HF_CACHE_DIR)

        prompt = "Once upon a time"
        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            pt_output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        # Keep only the newly generated tokens (exclude the prompt).
        pt_tokens = pt_output[0][prompt_len:].tolist()

        # ------------------------------------------------------------------
        # onnxruntime-genai greedy generation
        # ------------------------------------------------------------------
        og_model = og.Model(output_dir)

        params = og.GeneratorParams(og_model)
        params.set_search_options(
            do_sample=False,
            max_length=prompt_len + max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        generator = og.Generator(og_model, params)
        generator.append_tokens(inputs["input_ids"])

        og_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same newly-generated token sequence.
        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
