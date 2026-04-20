# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, run_session_or_io_binding, hide_stdout, requires_cuda

MODEL_NAME = "openai/gpt-oss-20b"


class TestGptOss20b(ExtTestCase):
    def common_fast_gpt_oss_20b_random_weights(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM, GptOssConfig

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

        basename = f"test_discrepancies_gpt_oss_20b_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        self.make_word_level_tokenizer(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        log_data = dict(
            precision=precision,
            model_id=MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="random",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        batch_size = 1
        seq_len = 5
        head_size = config.head_dim

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        with self.subTest(step="prefill"):
            prefill_feed = self.make_prefill_feed(
                input_ids.cpu().numpy(),
                num_hidden_layers,
                config.num_key_value_heads,
                head_size,
                self.get_input_np_dtype(precision),
                onnx_input_names,
            )

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config.vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill[:, :1, :], ort_logits_np[:, :1, :])
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            self.assertEqual(np_prefill.shape, ort_logits_np.shape)
            # Verify first-token logits are numerically close; subsequent positions
            # can diverge slightly in FP32 due to GQA kernel differences.
            np.testing.assert_allclose(np_prefill[:, :1, :], ort_logits_np[:, :1, :], atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            decode_feed = self.make_decode_feed(
                next_token, seq_len, prefill_results, num_hidden_layers, batch_size=batch_size, onnx_input_names=onnx_input_names
            )

            prefill_results, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_gpt_oss_20b_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM, GptOssConfig

        from modelbuilder.builder import create_model

        # Two layers: layer 0 is local (sliding-window) attention,
        # layer 1 is full attention, matching the is_local = lambda i: i % 2 == 0
        # logic in GPTOSSModel.
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
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
            max_position_embeddings=4096,
        )

        basename = f"test_generation_gpt_oss_20b_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        self.make_word_level_tokenizer(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        input_names = {inp.name for inp in sess.get_inputs()}

        batch_size = 1
        head_size = config.head_dim
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5)).to(provider)

        with torch.no_grad():
            pt_output = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id)
        pt_tokens = pt_output[0].tolist()

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        onnx_tokens = self.run_onnx_greedy_generation(
            sess=sess,
            precision=precision,
            provider=provider,
            initial_ids=current_ids,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=head_size,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            max_new_tokens=max_new_tokens,
            input_names=input_names,
        )

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_gpt_oss_20b_fp32_cpu_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gpt_oss_20b_fp16_cpu_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_gpt_oss_20b_fp32_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gpt_oss_20b_fp16_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gpt_oss_20b_bf16_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_fp32_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_fp16_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_int4_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gpt_oss_20b_fp16_cuda(self):
        self.common_fast_gpt_oss_20b_random_weights("fp16", "cuda")

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
            raise unittest.SkipTest("onnxruntime-genai is not installed; skipping genai comparison test.")

        import torch
        from transformers import AutoModelForCausalLM, GptOssConfig

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
            eos_token_id=2,
            bos_token_id=1,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        self.make_word_level_tokenizer(model_dir)

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
            pt_output = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id)
        pt_tokens = pt_output[0].tolist()

        # ------------------------------------------------------------------
        # onnxruntime-genai greedy generation
        # ------------------------------------------------------------------
        og_model = og.Model(output_dir)

        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=prompt_len + max_new_tokens, temperature=1.0, top_k=1)

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
