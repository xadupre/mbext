# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest


from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

CODEPARROT_MODEL_NAME = "codeparrot/codeparrot-small"


class TestCodeParrot(ExtTestCase):
    def _make_config(self):
        from transformers import GPT2Config

        return GPT2Config(
            architectures=["GPT2LMHeadModel"],
            n_embd=64,
            n_head=4,
            n_layer=3,
            n_inner=128,
            n_positions=128,
            vocab_size=1024,
            activation_function="gelu_new",
            bos_token_id=1,
            eos_token_id=2,
        )

    def common_fast_codeparrot_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM

        config = self._make_config()
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=CODEPARROT_MODEL_NAME,
            basename=f"test_discrepancies_codeparrot_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=config.n_layer,
            num_key_value_heads=config.n_head,
            head_size=config.n_embd // config.n_head,
            vocab_size=config.vocab_size,
        )

    def common_codeparrot_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_config()
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=CODEPARROT_MODEL_NAME,
            basename=f"test_generation_codeparrot_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=config.n_layer,
            num_key_value_heads=config.n_head,
            head_size=config.n_embd // config.n_head,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
        )

    @hide_stdout()
    def test_fast_discrepancy_codeparrot_fp32_cpu(self):
        self.common_fast_codeparrot_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_codeparrot_fp16_cpu(self):
        self.common_fast_codeparrot_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_codeparrot_int4_cpu(self):
        self.common_fast_codeparrot_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_codeparrot_fp16_cuda(self):
        self.common_fast_codeparrot_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_codeparrot_fp32_cpu_greedy_generation(self):
        self.common_codeparrot_greedy_generation("fp32", "cpu")

    @hide_stdout()
    @requires_genai()
    def test_codeparrot_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        prefix = "test_codeparrot_fp32_cpu_genai_generate"
        config = self._make_config()

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=CODEPARROT_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
