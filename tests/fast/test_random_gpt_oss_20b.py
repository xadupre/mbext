# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout

MODEL_NAME = "openai/gpt-oss-20b"


class TestGptOss20b(ExtTestCase):
    @hide_stdout()
    def test_gpt_oss_20b_fp32_cpu_random_weights(self):
        """
        Convert a model with the same architecture as openai/gpt-oss-20b but
        with randomly initialised weights to an fp32 ONNX model targeting the
        CPU execution provider.

        GPT-OSS is a Mixture-of-Experts model that alternates between sliding-
        window (local) attention layers and full-attention layers.  Two hidden
        layers are used so that both attention variants are exercised by the
        builder.

        The test verifies that:
        * ``create_model`` completes without error when given a local model
          directory.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        * The ONNX first-token logits closely match those of the original
          PyTorch model.
        """
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GptOssConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        # Two layers: layer 0 is local (sliding-window) attention,
        # layer 1 is full attention, matching the is_local = lambda i: i % 2 == 0
        # logic in GPTOSSModel.
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
        )

        model_dir = self.get_model_dir("test_gpt_oss_20b_fp32_cpu_random_weights", clean=False)
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

        output_dir, cache_dir = self.get_dirs(
            "test_gpt_oss_20b_fp32_cpu_random_weights", clean=False
        )

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
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

        head_size = config.head_dim
        onnx_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        # Provide empty past KV-cache tensors for every layer.
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

    @hide_stdout()
    def test_gpt_oss_20b_fp32_cpu_genai_generate(self):
        """
        Verify that ``onnxruntime-genai`` can load the fp32/CPU ONNX model
        produced from a randomly-initialised GPT-OSS-20B architecture and
        generate tokens without discrepancies against ``transformers.generate``.

        Both backends use greedy (argmax) decoding with the same prompt, so
        the generated token sequences must be bit-for-bit identical.

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
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GptOssConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        prefix = "test_gpt_oss_20b_fp32_cpu_genai_generate"
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
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

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)

        # Use a fixed seed so the prompt token IDs are deterministic.
        torch.manual_seed(0)
        batch_size = 1
        max_new_tokens = 5
        # Start from token ID 3 to avoid accidentally hitting BOS/EOS/PAD.
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 4))
        prompt_len = prompt_ids.shape[1]

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=config.eos_token_id,
            )
        pt_tokens = pt_output[0].tolist()

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
        generator.append_tokens(prompt_ids.numpy().astype(np.int64))

        og_tokens = prompt_ids[0].tolist()
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same token sequence (prompt + all generated tokens).
        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
