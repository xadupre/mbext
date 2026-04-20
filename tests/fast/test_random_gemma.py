# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda

MODEL_NAME = "google/gemma-2b"


class TestRandomGemma(ExtTestCase):
    def common_fast_gemma_random_weights(self, precision, provider):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GemmaConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        num_hidden_layers = 1

        # Minimal GemmaConfig matching the GemmaForCausalLM architecture but
        # with small dimensions to keep the test fast and completely offline.
        # head_dim=64 matches hidden_size // num_attention_heads = 512 // 8.
        config = GemmaConfig(
            architectures=["GemmaForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            hidden_act="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32000,
        )

        basename = f"test_discrepancies_gemma_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "</s>": 1, "<bos>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<bos>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
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
        sess = self.check_ort(onnx_path, provider=provider)

        self.run_prefill_and_decode_check(
            model=model,
            sess=sess,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            precision=precision,
            provider=provider,
            log_data=log_data,
        )

    def common_gemma_greedy_generation(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GemmaConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        num_hidden_layers = 1

        config = GemmaConfig(
            architectures=["GemmaForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            hidden_act="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32000,
        )

        basename = f"test_generation_gemma_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "</s>": 1, "<bos>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<bos>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        log_data = dict(
            precision=precision,
            model_id=MODEL_NAME,
            experiment="generate",
            provider=provider,
            test=basename,
            input_type="text",
            kind="fast",
        )
        self.run_greedy_generation_check(
            model=model,
            sess=sess,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            precision=precision,
            provider=provider,
            log_data=log_data,
        )

    @hide_stdout()
    def test_gemma_fp32_cpu_greedy_generation(self):
        self.common_gemma_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma_fp16_cpu_greedy_generation(self):
        self.common_gemma_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_gemma_fp32_cuda_greedy_generation(self):
        self.common_gemma_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma_fp16_cuda_greedy_generation(self):
        self.common_gemma_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma_bf16_cuda_greedy_generation(self):
        self.common_gemma_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_gemma_fp32_cpu(self):
        self.common_fast_gemma_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma_fp16_cpu(self):
        self.common_fast_gemma_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma_int4_cpu(self):
        self.common_fast_gemma_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gemma_fp16_cuda(self):
        self.common_fast_gemma_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_gemma_fp32_cpu_genai_generate(self):
        try:
            import onnxruntime_genai as og
        except ImportError:
            raise unittest.SkipTest("onnxruntime-genai is not installed; skipping genai comparison test.")

        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GemmaConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        prefix = "test_gemma_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = GemmaConfig(
            architectures=["GemmaForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            hidden_act="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=32000,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "</s>": 1, "<bos>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<bos>", eos_token="</s>", unk_token="<unk>"
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
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)

        torch.manual_seed(0)
        batch_size = 1
        max_new_tokens = 5
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 4))
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            pt_output = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id)
        pt_tokens = pt_output[0].tolist()

        og_model = og.Model(output_dir)
        params = og.GeneratorParams(og_model)
        params.set_search_options(do_sample=False, max_length=prompt_len + max_new_tokens, temperature=1.0, top_k=1)

        generator = og.Generator(og_model, params)
        generator.append_tokens(prompt_ids.numpy().astype(np.int64))

        og_tokens = prompt_ids[0].tolist()
        while not generator.is_done():
            generator.generate_next_token()
            og_tokens.append(int(generator.get_next_tokens()[0]))

        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
