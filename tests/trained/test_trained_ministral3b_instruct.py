# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest
from modelbuilder.ext_test_case import ExtTestCase, long_test, requires_cuda

MINISTRAL3B_MODEL_NAME = "mistralai/Ministral-3-3B-Instruct-2512"


class TestTrainedMinistral3BInstruct(ExtTestCase):
    def _common_part(self, precision, dtype, provider="cuda"):
        from transformers import Mistral3ForConditionalGeneration, AutoTokenizer
        from modelbuilder.builder import create_model

        tokenizer = AutoTokenizer.from_pretrained(MINISTRAL3B_MODEL_NAME)
        text = "Explain tokenization in simple terms."
        inputs = tokenizer(text, return_tensors="pt")

        output_dir, cache_dir = self.get_dirs(f"test_trained_ministral3b_instruct_{precision}_{provider}", clean=False)
        onnx_path = os.path.join(output_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            create_model(
                model_name=MINISTRAL3B_MODEL_NAME,
                input_path="",
                precision=precision,
                execution_provider=provider,
                output_dir=output_dir,
                cache_dir=cache_dir,
            )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        # model = AutoModelForCausalLM.from_pretrained(
        #     MINISTRAL3B_MODEL_NAME, ignore_mismatched_sizes=True, dtype=dtype
        model = Mistral3ForConditionalGeneration.from_pretrained(MINISTRAL3B_MODEL_NAME)
        model.eval().to(provider).to(dtype)

        embedding_layer = model.get_input_embeddings()
        inputs_embeds = embedding_layer(inputs["input_ids"].to(provider))
        attention_mask = inputs["attention_mask"].to(provider)

        return (
            onnx_path,
            model,
            dict(inputs_embeds=inputs_embeds, attention_mask=attention_mask),
            dict(inputs_embeds=inputs_embeds.detach().cpu().numpy(), attention_mask=attention_mask.detach().cpu().numpy()),
        )

    def _common_trained_discrepancies(self, precision, provider):
        import torch

        dtype = self.get_input_torch_dtype(precision)

        onnx_path, model, torch_feed, onnx_feed = self._common_part(precision, dtype, provider=provider)
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
                model_id=MINISTRAL3B_MODEL_NAME,
                experiment="forward",
                provider="cuda",
                test=f"test_trained_ministral3b_instruct_discrepancies_{precision}_{provider}",
                input_type="text",
                kind="prefill",
            )
        )
        self.log_results(disc)
        self.assertLess(disc["max_abs_err"], 2)

    @long_test()
    def test_trained_ministral3b_instruct_discrepancies_fp32_cpu(self):
        self._common_trained_discrepancies("fp32", "cpu")

    @long_test()
    def test_trained_ministral3b_instruct_discrepancies_fp16_cuda(self):
        self._common_trained_discrepancies("fp16", "cuda")

    @long_test()
    @requires_cuda()
    def test_trained_ministral3b_instruct_genai_generate_cuda(self):
        """
        Compare ``transformers.generate`` with ``onnxruntime-genai`` generate
        on the ``mistralai/Ministral-3B-Instruct-2512`` trained model (CUDA,
        fp16).

        Both backends use greedy (argmax) decoding, so the generated token
        sequences must be bit-for-bit identical.

        The test is skipped automatically when ``onnxruntime-genai`` is not
        installed.
        """
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest("onnxruntime-genai is not installed; skipping genai comparison test.")

        import torch
        from transformers import AutoTokenizer

        precision, dtype = "fp16", torch.float16

        onnx_path, model = self._common_part(precision, dtype)

        genai_config_path = os.path.join(os.path.dirname(onnx_path), "genai_config.json")
        self.assertExists(genai_config_path)

        tokenizer = AutoTokenizer.from_pretrained(MINISTRAL3B_MODEL_NAME)
        prompt = "Once upon a time"
        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt")
        start_sequence = inputs["input_ids"].shape[1]
        inputs = inputs.to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            pt_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        # Keep only the newly generated tokens (exclude the prompt).
        pt_tokens = pt_output[0][prompt_len:].tolist()

        # ------------------------------------------------------------------
        # onnxruntime-genai greedy generation
        # ------------------------------------------------------------------
        og_model = og.Model(os.path.dirname(onnx_path))

        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=prompt_len + max_new_tokens, temperature=1.0, top_k=1)

        generator = og.Generator(og_model, params)
        generator.append_tokens(inputs["input_ids"])

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
                model_id=MINISTRAL3B_MODEL_NAME,
                experiment="generate",
                provider="cuda",
                test="test_trained_ministral3b_instruct_genai_generate_cuda",
                expected_text=tokenizer.decode(pt_tokens[start_sequence:], skip_special_tokens=False),
                genai_text=tokenizer.decode(og_tokens, skip_special_tokens=False),
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertEqual(pt_tokens, og_tokens)

    @long_test()
    def test_trained_ministral3b_instruct_genai_generate_cpu(self):
        """
        Compare ``transformers.generate`` with ``onnxruntime-genai`` generate
        on the ``mistralai/Ministral-3B-Instruct-2512`` trained model (CPU,
        fp32).

        Both backends use greedy (argmax) decoding, so the generated token
        sequences must be bit-for-bit identical.

        The test is skipped automatically when ``onnxruntime-genai`` is not
        installed.
        """
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest("onnxruntime-genai is not installed; skipping genai comparison test.")

        import torch
        from transformers import AutoTokenizer

        precision, dtype = "fp32", torch.float32

        onnx_path, model = self._common_part(precision, dtype, provider="cpu")

        genai_config_path = os.path.join(os.path.dirname(onnx_path), "genai_config.json")
        self.assertExists(genai_config_path)

        tokenizer = AutoTokenizer.from_pretrained(MINISTRAL3B_MODEL_NAME)
        prompt = "Once upon a time"
        max_new_tokens = 20

        # ------------------------------------------------------------------
        # transformers greedy generation (reference)
        # ------------------------------------------------------------------
        inputs = tokenizer(prompt, return_tensors="pt")
        start_sequence = inputs["input_ids"].shape[1]
        inputs = inputs.to("cuda")
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            pt_output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        # Keep only the newly generated tokens (exclude the prompt).
        pt_tokens = pt_output[0][prompt_len:].tolist()

        # ------------------------------------------------------------------
        # onnxruntime-genai greedy generation
        # ------------------------------------------------------------------
        og_model = og.Model(os.path.dirname(onnx_path))

        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=prompt_len + max_new_tokens, temperature=1.0, top_k=1)

        generator = og.Generator(og_model, params)
        generator.append_tokens(inputs["input_ids"])

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
                model_id=MINISTRAL3B_MODEL_NAME,
                experiment="generate",
                provider="cpu",
                test="test_trained_ministral3b_instruct_genai_generate_cpu",
                expected_text=tokenizer.decode(pt_tokens[start_sequence:], skip_special_tokens=False),
                genai_text=tokenizer.decode(og_tokens, skip_special_tokens=False),
                input_type="text",
            )
        )
        self.log_results(disc)
        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
