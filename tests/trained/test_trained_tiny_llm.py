# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest
from modelbuilder.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    long_test,
    requires_cuda,
)

MODEL_NAME = "arnir0/Tiny-LLM"


class TestTrainedTinyLLM(ExtTestCase):
    def _common_part(self, precision, dtype, provider="cuda", int4=False):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from modelbuilder.builder import create_model

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        text = "What is machine learning?"
        inputs = tokenizer(text, return_tensors="pt")

        output_dir, cache_dir = self.get_dirs(
            f"test_trained_tiny_llm_{'int4' if int4 else precision}_{provider}", clean=False
        )
        onnx_path = os.path.join(output_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            create_model(
                model_name=MODEL_NAME,
                input_path="",
                precision="int4" if int4 else precision,
                execution_provider=provider,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, ignore_mismatched_sizes=True, dtype=dtype
        )
        model.eval().to(provider).to(dtype)

        return (
            onnx_path,
            model,
            dict(
                input_ids=inputs["input_ids"].to(provider),
                attention_mask=inputs["attention_mask"].to(provider),
            ),
            dict(
                input_ids=inputs["input_ids"].detach().cpu().numpy(),
                attention_mask=inputs["attention_mask"].detach().cpu().numpy(),
            ),
            tokenizer,
        )

    def _common_trained_discrepancies(self, precision, provider):
        import torch

        dtype = self.get_input_torch_dtype(precision)

        onnx_path, model, torch_feed, onnx_feed, tokenizer = self._common_part(
            precision, dtype, provider=provider
        )
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")
        self.fill_with_empty_cache(onnx_feed, sess, provider)

        with torch.no_grad():
            pt_logits = model(**torch_feed).logits
        pt_logits = pt_logits.detach().cpu().numpy()

        onnx_outputs = sess.run(None, onnx_feed)
        onnx_logits = onnx_outputs[0]

        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        disc.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="forward",
                provider="cuda",
                test=f"test_trained_tiny_llm_discrepancies_{precision}_{provider}",
                input_type="text",
                kind="prefill",
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 3)

    @long_test()
    @hide_stdout()
    def test_trained_tiny_llm_discrepancies_fp32_cpu(self):
        self._common_trained_discrepancies("fp32", "cpu")

    @long_test()
    @requires_cuda()
    @hide_stdout()
    def test_trained_tiny_llm_discrepancies_fp16_cuda(self):
        self._common_trained_discrepancies("fp16", "cuda")

    def _common_trained_generate(self, precision, provider):
        import torch
        import onnxruntime_genai as og

        dtype = self.get_input_torch_dtype(precision)

        onnx_path, model, torch_feed, onnx_feed, tokenizer = self._common_part(
            precision, dtype, provider=provider
        )

        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        prompt_len = torch_feed["input_ids"].shape[1]
        with torch.no_grad():
            pt_output = model.generate(
                **torch_feed,
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
            max_length=max_new_tokens,
            temperature=1.0,
            top_k=1,
        )

        generator = og.Generator(og_model, params)
        generator.append_tokens(onnx_feed["input_ids"])

        og_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        # Greedy decoding is deterministic: both backends must produce the
        # exact same newly-generated token sequence.
        min_length = min(len(pt_tokens), len(og_tokens))
        pt_tokens = pt_tokens[:min_length]
        og_tokens = og_tokens[:min_length]
        disc = self.first_token_diff(pt_tokens, og_tokens)
        disc.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="generate",
                provider="cuda",
                test=f"test_trained_tiny_llm_genai_generate_{precision}_{provider}",
                expected_text=tokenizer.decode(pt_tokens, skip_special_tokens=False),
                genai_text=tokenizer.decode(og_tokens, skip_special_tokens=False),
                input_type="text",
            )
        )
        self.log_results(disc)
        length = 1 if precision == "int4" else len(pt_tokens)
        self.assertEqual(pt_tokens[:length], og_tokens[:length])

    @long_test()
    @hide_stdout()
    def test_trained_tiny_llm_generate_fp32_cpu(self):
        self._common_trained_generate("fp32", "cpu")

    @long_test()
    @hide_stdout()
    def test_trained_tiny_llm_generate_int4_cpu(self):
        self._common_trained_generate("int4", "cpu")

    @long_test()
    @requires_cuda()
    @hide_stdout()
    def test_trained_tiny_llm_generate_fp16_cuda(self):
        self._common_trained_generate("fp16", "cuda")

    @long_test()
    @requires_cuda()
    @hide_stdout()
    def test_trained_tiny_llm_generate_bf16_cuda(self):
        self._common_trained_generate("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
