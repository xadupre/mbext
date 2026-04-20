# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest


from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

MODEL_NAME = "arnir0/Tiny-LLM"


class TestRandomTinyLLM(ExtTestCase):
    def common_fast_tiny_llm_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM, LlamaConfig

        # Config matching the arnir0/Tiny-LLM architecture (LlamaForCausalLM)
        # but with a single hidden layer to keep the test fast.  These values
        # are hardcoded so the test runs completely offline without downloading
        # any files from Hugging Face.
        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        atol = {"fp16": 3e-2, "bf16": 2e-2, "fp32": 2e-3 if provider == "cuda" else 2e-4, "int4": 0.5}
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_tiny_llm_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            atol=atol,
            rtol={"fp16": 10, "bf16": 10, "fp32": 1e-4, "int4": 10000},
            kind="fast",
        )

    @hide_stdout()
    def test_fast_discrepancy_tiny_llm_fp32_cpu(self):
        self.common_fast_tiny_llm_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_tiny_llm_fp16_cpu(self):
        self.common_fast_tiny_llm_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_tiny_llm_int4_cpu(self):
        self.common_fast_tiny_llm_random_weights("int4", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_tiny_llm_fp32_cuda(self):
        self.common_fast_tiny_llm_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_tiny_llm_fp16_cuda(self):
        self.common_fast_tiny_llm_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_tiny_llm_bf16_cuda(self):
        self.common_fast_tiny_llm_random_weights("bf16", "cuda")

    def common_tiny_llm_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM, LlamaConfig

        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    @hide_stdout()
    def test_tiny_llm_fp32_cpu_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_tiny_llm_fp16_cpu_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_tiny_llm_fp32_cuda_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_tiny_llm_fp16_cuda_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_tiny_llm_bf16_cuda_greedy_generation(self):
        self.common_tiny_llm_greedy_generation("bf16", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_tiny_llm_fp32_cpu_genai_generate(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        prefix = "test_tiny_llm_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
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
        with torch.no_grad():
            pt_output = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id)
        pt_tokens = pt_output[0].tolist()

        og_tokens = self.run_genai_generation(output_dir, prompt_ids, max_new_tokens)
        self.assertEqual(pt_tokens, og_tokens)


if __name__ == "__main__":
    unittest.main(verbosity=2)
