# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
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

    @staticmethod
    def _make_shared_kv_config(num_hidden_layers=4, num_kv_shared=2):
        """Return a Gemma4TextConfig with ``num_kv_shared_layers > 0``.

        Layer layout: [sliding, full, sliding, full, ...]
        The last ``num_kv_shared`` layers share K/V with the last non-shared
        layer of the same type.  Both non-shared and shared sections must
        contain at least one sliding and one full-attention layer so that each
        shared layer has a valid donor.

        ``num_hidden_layers`` must be >= 4 so that at least one non-shared
        sliding *and* one non-shared full-attention layer always exist.
        """
        from transformers import Gemma4TextConfig

        layer_types = []
        for i in range(num_hidden_layers):
            layer_types.append("sliding_attention" if i % 2 == 0 else "full_attention")

        return Gemma4TextConfig(
            architectures=["Gemma4ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            global_head_dim=64,
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
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=num_kv_shared,
            enable_moe_block=False,
        )

    def common_fast_gemma4_shared_kv_random_weights(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_shared_kv_config()
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_gemma4_shared_kv_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": config.num_hidden_layers},
            atol={"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 1.5},
        )

    def common_gemma4_shared_kv_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM

        config = self._make_shared_kv_config()
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_gemma4_shared_kv_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=config.num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": config.num_hidden_layers},
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

    @hide_stdout()
    def test_gemma4_num_kv_shared_layers(self):
        """Gemma4 with num_kv_shared_layers > 0 should export without errors.

        Shared-KV layers (the last ``num_kv_shared_layers`` layers) have no
        k_proj/v_proj/k_norm/v_norm weights and reuse the donor layer's K/V.
        This test verifies that ``create_model`` completes successfully for such
        a configuration and produces a valid ONNX model.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        # 4 layers: [sliding, full, sliding, full]
        # Non-shared layers: [0, 1] (sliding + full)
        # Shared layers:    [2, 3] (sliding reuses layer 0 as donor; full reuses layer 1 as donor)
        # first_kv_shared_layer_idx = 4 - 2 = 2
        # Layer 2 (sliding): last non-shared sliding = layer 0 → donor = 0
        # Layer 3 (full):    last non-shared full    = layer 1 → donor = 1
        num_hidden_layers = 4
        num_kv_shared = 2
        config = self._make_config(num_hidden_layers=num_hidden_layers)
        # Override layer_types so that both non-shared and shared sections
        # contain at least one sliding and one full-attention layer.
        config.layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
        config.num_kv_shared_layers = num_kv_shared

        prefix = "test_gemma4_num_kv_shared_layers"
        model_dir = self.get_model_dir(prefix, clean=True)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=True)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

    @hide_stdout()
    def test_fast_discrepancy_gemma4_shared_kv_fp32_cpu(self):
        self.common_fast_gemma4_shared_kv_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma4_shared_kv_fp16_cpu(self):
        self.common_fast_gemma4_shared_kv_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_gemma4_shared_kv_fp32_cpu_greedy_generation(self):
        self.common_gemma4_shared_kv_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma4_shared_kv_fp16_cpu_greedy_generation(self):
        self.common_gemma4_shared_kv_greedy_generation("fp16", "cpu")

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
