# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest


from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

MODEL_NAME = "microsoft/phi-2"


class TestPhi(ExtTestCase):
    def common_fast_phi_random_weights(self, precision, provider):
        from transformers import PhiConfig, PhiForCausalLM

        num_hidden_layers = 1

        # Minimal PhiForCausalLM config with small dimensions so the test
        # runs fast and completely offline without downloading any weights.
        # head_size = hidden_size // num_attention_heads = 512 // 8 = 64
        config = PhiConfig(
            architectures=["PhiForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="gelu_new",
            hidden_size=512,
            intermediate_size=2048,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            layer_norm_eps=1e-05,
            vocab_size=51200,
        )

        model = PhiForCausalLM(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_phi_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            atol={"fp16": 3e-2, "bf16": 2e-2, "fp32": 1e-2, "int4": 0.5},
            rtol={"fp16": 10, "bf16": 10, "fp32": 1e-2, "int4": 10000},
            kind="fast",
        )

    def common_phi_greedy_generation(self, precision, provider):
        import torch
        from transformers import PhiConfig, PhiForCausalLM

        num_hidden_layers = 1

        config = PhiConfig(
            architectures=["PhiForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="gelu_new",
            hidden_size=512,
            intermediate_size=2048,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            layer_norm_eps=1e-05,
            vocab_size=51200,
        )

        torch.manual_seed(42)
        model = PhiForCausalLM(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_phi_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.hidden_size // config.num_attention_heads,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
        )

    @hide_stdout()
    def test_fast_discrepancy_phi_fp32_cpu(self):
        self.common_fast_phi_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_phi_fp16_cpu(self):
        self.common_fast_phi_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi_fp16_cuda(self):
        self.common_fast_phi_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_phi_bf16_cuda(self):
        self.common_fast_phi_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_phi_fp32_cpu_greedy_generation(self):
        self.common_phi_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_phi_fp16_cpu_greedy_generation(self):
        self.common_phi_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_phi_fp16_cuda_greedy_generation(self):
        self.common_phi_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_phi_bf16_cuda_greedy_generation(self):
        self.common_phi_greedy_generation("bf16", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_phi_fp32_cpu_genai_generate(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PhiConfig, PhiForCausalLM, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        prefix = "test_phi_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = PhiConfig(
            architectures=["PhiForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="gelu_new",
            hidden_size=512,
            intermediate_size=2048,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            layer_norm_eps=1e-05,
            vocab_size=51200,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = PhiForCausalLM(config)
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
