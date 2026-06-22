# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Compares ``llama.cpp`` and ``onnxruntime-genai`` on ``arnir0/Tiny-LLM``.

The model architecture and tokenizer come from ``arnir0/Tiny-LLM`` but the
weights are randomly initialised (``from_config``), so the test runs without
relying on any particular trained checkpoint.  Both backends are fed the exact
same prompt token ids and decoded greedily; the reported metric is the first
token difference between the two token sequences.

This test is intentionally excluded from the regular fast/trained CI jobs (it
requires both ``llama-cpp-python`` and the ``llama.cpp`` conversion script) and
is gated behind ``LONGTEST=1`` so it only runs in its dedicated CI job.
"""

import os
import shutil
import subprocess
import sys
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_genai, requires_llama_cpp

MODEL_NAME = "arnir0/Tiny-LLM"


class TestLlamaCppTinyLLM(ExtTestCase):
    def _build_random_tiny_llm(self, model_dir):
        """Saves a random-weight ``arnir0/Tiny-LLM`` and its tokenizer."""
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(MODEL_NAME)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(model_dir)

        # ``convert_hf_to_gguf.py`` runs in an isolated environment that pins a
        # different ``transformers``/``tokenizers`` version (see
        # ``_convert_to_gguf``). Its vocabulary resolution relies on the
        # original SentencePiece file (``tokenizer.model``), but the fast
        # tokenizer's ``save_pretrained`` only emits ``tokenizer.json``. Fetch
        # ``tokenizer.model`` straight from the hub so the converter resolves
        # the vocabulary through the SentencePiece path, independent of any
        # cross-version fast-tokenizer load.
        from huggingface_hub import hf_hub_download

        sp_model = hf_hub_download(repo_id=MODEL_NAME, filename="tokenizer.model")
        shutil.copyfile(sp_model, os.path.join(model_dir, "tokenizer.model"))
        return tokenizer, config

    def _convert_to_gguf(self, convert_script, model_dir, gguf_path):
        """Converts a Hugging Face checkpoint to a float32 GGUF file.

        ``convert_hf_to_gguf.py`` pins ``torch``/``transformers``/``numpy`` to
        versions that conflict with the ones modelbuilder and
        onnxruntime-genai rely on. ``LLAMA_CPP_CONVERT_PYTHON`` lets the caller
        run the conversion with a dedicated interpreter (e.g. an isolated
        virtual environment) so those pins never pollute the test environment.
        """
        python = os.environ.get("LLAMA_CPP_CONVERT_PYTHON", sys.executable)
        subprocess.run([python, convert_script, model_dir, "--outfile", gguf_path, "--outtype", "f32"], check=True)
        self.assertExists(gguf_path)

    def _llama_cpp_tokens(self, gguf_path, prompt_ids, max_new_tokens):
        """Greedy generation with ``llama.cpp`` over explicit prompt token ids."""
        from llama_cpp import Llama

        llm = Llama(model_path=gguf_path, n_ctx=2048, logits_all=False, verbose=False)
        tokens = []
        for token in llm.generate(list(prompt_ids), top_k=1, top_p=1.0, temp=0.0):
            tokens.append(int(token))
            if len(tokens) >= max_new_tokens:
                break
        return tokens

    def _genai_tokens(self, output_dir, prompt_ids, max_new_tokens):
        """Greedy generation with ``onnxruntime-genai`` over explicit prompt token ids."""
        import numpy as np
        import onnxruntime_genai as og

        og_model = og.Model(output_dir)
        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=len(prompt_ids) + max_new_tokens, temperature=1.0, top_k=1)
        generator = og.Generator(og_model, params)
        generator.append_tokens(np.array([list(prompt_ids)], dtype=np.int64))

        tokens = []
        while not generator.is_done() and len(tokens) < max_new_tokens:
            generator.generate_next_token()
            tokens.append(int(generator.get_next_tokens()[0]))
        return tokens

    @requires_genai()
    @requires_llama_cpp()
    @hide_stdout()
    def test_llama_cpp_vs_genai_tiny_llm_fp32_cpu(self):
        from modelbuilder.builder import create_model

        convert_script = self._find_convert_script()
        if convert_script is None:
            self.skipTest("convert_hf_to_gguf.py unavailable (lookup/download failed)")

        basename = "test_llama_cpp_vs_genai_tiny_llm_fp32_cpu"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        tokenizer, _ = self._build_random_tiny_llm(model_dir)

        # Build the ONNX model consumed by onnxruntime-genai.
        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )
        self.assertExists(os.path.join(output_dir, "model.onnx"))

        # Build the GGUF model consumed by llama.cpp.
        gguf_path = os.path.join(output_dir, "model.gguf")
        self._convert_to_gguf(convert_script, model_dir, gguf_path)

        # Feed both backends the exact same prompt token ids so the comparison
        # is independent of any tokenizer mismatch.
        prompt_ids = tokenizer("What is machine learning?", return_tensors="np")["input_ids"][0].tolist()
        max_new_tokens = 20

        llama_tokens = self._llama_cpp_tokens(gguf_path, prompt_ids, max_new_tokens)
        genai_tokens = self._genai_tokens(output_dir, prompt_ids, max_new_tokens)

        # The reported metric is the first token difference between the two
        # greedy token sequences.
        disc = self.first_token_diff(llama_tokens, genai_tokens)
        disc.update(
            dict(
                precision="fp32",
                model_id=MODEL_NAME,
                experiment="generate",
                provider="cpu",
                test=basename,
                backend_a="llama.cpp",
                backend_b="onnxruntime-genai",
                llama_cpp_text=tokenizer.decode(llama_tokens, skip_special_tokens=False),
                genai_text=tokenizer.decode(genai_tokens, skip_special_tokens=False),
                input_type="text",
                kind="llama_cpp",
            )
        )
        self.log_results(disc)

        # Greedy decoding is deterministic: both backends must agree on the
        # first generated token.
        self.assertGreater(len(llama_tokens), 0)
        self.assertGreater(len(genai_tokens), 0)
        self.assertEqual(llama_tokens[0], genai_tokens[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
