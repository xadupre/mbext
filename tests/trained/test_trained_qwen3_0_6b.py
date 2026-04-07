# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, long_test, requires_cuda

QWEN3_MODEL_NAME = "Qwen/Qwen3-0.6B"


class TestTrainedQwen3(ExtTestCase):
    def _common_part(self, precision, dtype, provider="cpu", int4=False):
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        output_dir, cache_dir = self.get_dirs(
            f"test_trained_qwen_qwen3_0_6b_{'int4' if int4 else precision}_{provider}",
            clean=False,
        )
        onnx_path = os.path.join(output_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            create_model(
                model_name=QWEN3_MODEL_NAME,
                input_path="",
                precision="int4" if int4 else precision,
                execution_provider=provider,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        model = AutoModelForCausalLM.from_pretrained(
            QWEN3_MODEL_NAME, ignore_mismatched_sizes=True, torch_dtype=dtype
        )
        model.eval().to(provider).to(dtype)
        return onnx_path, model

    @long_test()
    def test_trained_qwen_qwen3_0_6b_discrepancies_cpu(self):
        """
        Convert Qwen/Qwen3-0.6B to an fp32 ONNX model targeting the CPU
        execution provider.

        The test verifies that:
        * ``create_model`` completes without error.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        * The ONNX logits closely match those of the original PyTorch model.
        """
        import torch

        precision, dtype, np_dtype = "fp32", torch.float32, np.float32

        onnx_path, model = self._common_part(precision, dtype, provider="cpu")
        sess = self._check_with_ort(onnx_path, cpu=True)
        config = model.config

        batch_size = 1
        seq_len = 5

        head_size = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        onnx_feed, torch_feed = self.make_dummy_text_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            np_dtype=np_dtype,
            provider="cpu",
            head_size=head_size,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
        )

        with torch.no_grad():
            pt_logits = model(**torch_feed).logits.numpy()

        onnx_outputs = sess.run(None, onnx_feed)
        onnx_logits = onnx_outputs[0]

        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        disc.update(
            dict(
                precision=precision,
                model_id=QWEN3_MODEL_NAME,
                experiment="forward",
                provider="cpu",
                test="test_trained_qwen_qwen3_0_6b_discrepancies_cpu",
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 0.05)

    @long_test()
    @requires_cuda()
    def test_trained_qwen_qwen3_0_6b_discrepancies_cuda(self):
        """
        Convert Qwen/Qwen3-0.6B to an fp16 ONNX model targeting the CUDA
        execution provider.

        The test verifies that:
        * ``create_model`` completes without error.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        * The ONNX logits closely match those of the original PyTorch model.
        """
        import torch

        precision, dtype, np_dtype = "fp16", torch.float16, np.float16

        onnx_path, model = self._common_part(precision, dtype, provider="cuda")
        sess = self._check_with_ort(onnx_path, cpu=False)
        config = model.config

        batch_size = 1
        seq_len = 5

        head_size = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        onnx_feed, torch_feed = self.make_dummy_text_inputs(
            batch_size=batch_size,
            seq_len=seq_len,
            np_dtype=np_dtype,
            provider="cuda",
            head_size=head_size,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            vocab_size=config.vocab_size,
        )

        with torch.no_grad():
            pt_logits = model(**torch_feed).logits
        pt_logits = pt_logits.detach().cpu().numpy()

        onnx_outputs = sess.run(None, onnx_feed)
        onnx_logits = onnx_outputs[0]

        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        disc.update(
            dict(
                precision=precision,
                model_id=QWEN3_MODEL_NAME,
                experiment="forward",
                provider="cuda",
                test="test_trained_qwen_qwen3_0_6b_discrepancies_cuda",
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 2)

    @long_test()
    def test_trained_qwen_qwen3_0_6b_genai_generate_cpu(self):
        """
        Compare ``transformers.generate`` with ``onnxruntime-genai`` generate
        on the ``Qwen/Qwen3-0.6B`` trained model (fp32, CPU).

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

        precision, dtype = "fp32", torch.float32

        onnx_path, model = self._common_part(precision, dtype, provider="cpu")

        genai_config_path = os.path.join(os.path.dirname(onnx_path), "genai_config.json")
        self.assertExists(genai_config_path)

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_NAME)
        prompt = "Once upon a time"
        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to("cpu")
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
        og_model = og.Model(os.path.dirname(onnx_path))

        params = og.GeneratorParams(og_model)
        params.set_search_options(
            do_sample=False,
            max_length=prompt_len + max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        generator = og.Generator(og_model, params)
        generator.append_tokens(inputs["input_ids"].numpy())

        og_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same newly-generated token sequence.
        disc = self.first_token_diff(pt_tokens, og_tokens)
        disc.update(
            dict(
                precision=precision,
                model_id=QWEN3_MODEL_NAME,
                experiment="generate",
                provider="cpu",
                test="test_trained_qwen_qwen3_0_6b_genai_generate_cpu",
                expected_text=tokenizer.decode(pt_tokens, skip_special_tokens=False),
                genai_text=tokenizer.decode(og_tokens, skip_special_tokens=False),
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertEqual(pt_tokens, og_tokens)

    @long_test()
    def test_trained_qwen_qwen3_0_6b_genai_generate_int4_cpu(self):
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest(
                "onnxruntime-genai is not installed; skipping genai comparison test."
            )

        import torch
        from transformers import AutoTokenizer

        precision, dtype = "fp32", torch.float32

        onnx_path, model = self._common_part(precision, dtype, provider="cpu", int4=True)

        genai_config_path = os.path.join(os.path.dirname(onnx_path), "genai_config.json")
        self.assertExists(genai_config_path)

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_NAME)
        prompt = "Once upon a time"
        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to("cpu")
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
        og_model = og.Model(os.path.dirname(onnx_path))

        params = og.GeneratorParams(og_model)
        params.set_search_options(
            do_sample=False,
            max_length=prompt_len + max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        generator = og.Generator(og_model, params)
        generator.append_tokens(inputs["input_ids"].numpy())

        og_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same newly-generated token sequence.
        disc = self.first_token_diff(pt_tokens, og_tokens)
        disc.update(
            dict(
                precision="int4",
                model_id=QWEN3_MODEL_NAME,
                experiment="generate",
                provider="cpu",
                test="test_trained_qwen_qwen3_0_6b_genai_generate_int4_cpu",
                expected_text=tokenizer.decode(pt_tokens, skip_special_tokens=False),
                genai_text=tokenizer.decode(og_tokens, skip_special_tokens=False),
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertEqual(pt_tokens, og_tokens)

    @long_test()
    @requires_cuda()
    def test_trained_qwen_qwen3_0_6b_genai_generate_cuda(self):
        """
        Compare ``transformers.generate`` with ``onnxruntime-genai`` generate
        on the ``Qwen/Qwen3-0.6B`` trained model (fp16, CUDA).

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

        onnx_path, model = self._common_part(precision, dtype, provider="cuda")

        genai_config_path = os.path.join(os.path.dirname(onnx_path), "genai_config.json")
        self.assertExists(genai_config_path)

        tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_NAME)
        prompt = "Once upon a time"
        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to("cuda")
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
        og_model = og.Model(os.path.dirname(onnx_path))

        params = og.GeneratorParams(og_model)
        params.set_search_options(
            do_sample=False,
            max_length=prompt_len + max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        generator = og.Generator(og_model, params)
        generator.append_tokens(inputs["input_ids"].detach().cpu().numpy())

        og_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same newly-generated token sequence.
        disc = self.first_token_diff(pt_tokens, og_tokens)
        disc.update(
            dict(
                precision=precision,
                model_id=QWEN3_MODEL_NAME,
                experiment="generate",
                provider="cuda",
                test="test_trained_qwen_qwen3_0_6b_genai_generate_cuda",
                expected_text=tokenizer.decode(pt_tokens, skip_special_tokens=False),
                genai_text=tokenizer.decode(og_tokens, skip_special_tokens=False),
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
