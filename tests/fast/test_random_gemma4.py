# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import contextlib
import io
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, requires_transformers

MODEL_NAME = "google/gemma-4-E4B-it"


@requires_transformers("5.6")
class TestRandomGemma4(ExtTestCase):
    @staticmethod
    def _make_config(num_hidden_layers=2):
        """Return a minimal Gemma4TextConfig for offline testing.

        Uses equal head_dim and global_head_dim (both 64) to keep the
        KV-cache shapes uniform across layer types.  PLE and shared-KV
        features are disabled so the export exercises only the core
        sliding/full-attention and v_norm paths.

        ``layer_types`` is set explicitly so that the last layer is
        ``"full_attention"`` (required by Gemma4) without relying on
        the config's auto-correction that emits a WARNING.
        """
        from transformers import Gemma4TextConfig

        # Build layer_types: (num_hidden_layers - 1) sliding + 1 full.
        layer_types = ["sliding_attention"] * (num_hidden_layers - 1) + ["full_attention"]

        return Gemma4TextConfig(
            architectures=["Gemma4ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            global_head_dim=64,  # keep head_dim uniform across layer types
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            layer_types=layer_types,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
            # Disable optional features not yet exported
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            enable_moe_block=False,
        )

    def common_fast_gemma4_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM

        config = self._make_config()
        num_hidden_layers = config.num_hidden_layers

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_gemma4_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            # Use a relaxed int4 tolerance: 2 layers with random weights can exceed 0.5
            atol={"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 1.5},
        )

    def common_gemma4_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_config()
        num_hidden_layers = config.num_hidden_layers

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_gemma4_{precision}_{provider}",
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
    def test_fast_discrepancy_gemma4_fp32_cpu(self):
        self.common_fast_gemma4_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma4_fp16_cpu(self):
        self.common_fast_gemma4_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma4_int4_cpu(self):
        self.common_fast_gemma4_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gemma4_fp16_cuda(self):
        self.common_fast_gemma4_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_gemma4_fp32_cpu_greedy_generation(self):
        self.common_gemma4_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma4_fp16_cpu_greedy_generation(self):
        self.common_gemma4_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_gemma4_fp16_cuda_greedy_generation(self):
        self.common_gemma4_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma4_bf16_cuda_greedy_generation(self):
        self.common_gemma4_greedy_generation("bf16", "cuda")

    def test_gemma4_num_kv_shared_layers_warning(self):
        """Creating a Gemma4Model with num_kv_shared_layers > 0 should print a warning."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        config = self._make_config(num_hidden_layers=2)
        config.num_kv_shared_layers = 2

        prefix = "test_gemma4_num_kv_shared_layers_warning"
        model_dir = self.get_model_dir(prefix, clean=True)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=True)

        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            create_model(
                model_name=MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision="fp32",
                execution_provider="cpu",
                cache_dir=cache_dir,
                num_hidden_layers=config.num_hidden_layers,
            )

        output = captured.getvalue()
        self.assertIn("num_kv_shared_layers", output)
        self.assertIn("WARNING", output)

    @hide_stdout()
    @requires_genai()
    def test_gemma4_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        prefix = "test_gemma4_fp32_cpu_genai_generate"
        config = self._make_config(num_hidden_layers=2)

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
