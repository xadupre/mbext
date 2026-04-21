# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest


from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

MODEL_NAME = "nvidia/Minitron-4B-Base"


class TestNemotron(ExtTestCase):
    def common_fast_nemotron_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron import NemotronConfig

        num_hidden_layers = 1
        config = NemotronConfig(
            architectures=["NemotronForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="relu2",
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            norm_eps=1e-05,
            partial_rotary_factor=0.5,
            vocab_size=32000,
        )

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_nemotron_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            atol={"fp16": 3e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5},
            rtol={"fp16": 10, "bf16": 10, "fp32": 1e-3, "int4": 10000},
            kind="fast",
        )

    def common_nemotron_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron import NemotronConfig

        num_hidden_layers = 1
        config = NemotronConfig(
            architectures=["NemotronForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="relu2",
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            norm_eps=1e-05,
            partial_rotary_factor=0.5,
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
            basename=f"test_generation_nemotron_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    @hide_stdout()
    def test_fast_discrepancy_nemotron_fp32_cpu(self):
        self.common_fast_nemotron_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_fp16_cpu(self):
        self.common_fast_nemotron_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_int4_cpu(self):
        self.common_fast_nemotron_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_fp16_cuda(self):
        self.common_fast_nemotron_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_bf16_cuda(self):
        self.common_fast_nemotron_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_nemotron_fp32_cpu_greedy_generation(self):
        self.common_nemotron_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_nemotron_fp16_cpu_greedy_generation(self):
        self.common_nemotron_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_fp16_cuda_greedy_generation(self):
        self.common_nemotron_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_bf16_cuda_greedy_generation(self):
        self.common_nemotron_greedy_generation("bf16", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_nemotron_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron import NemotronConfig

        from modelbuilder.builder import create_model

        prefix = "test_nemotron_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = NemotronConfig(
            architectures=["NemotronForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="relu2",
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            norm_eps=1e-05,
            partial_rotary_factor=0.5,
            vocab_size=32000,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
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

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
