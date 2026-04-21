# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

MODEL_NAME = "google/gemma-3-4b-it"


@requires_transformers("5")
class TestRandomGemma3Conditional(ExtTestCase):
    """Fast offline tests for the Gemma3ForConditionalGeneration text decoder.

    The builder exports only the text component of the VLM (with
    ``exclude_embeds=True``), so the resulting ONNX model takes
    ``inputs_embeds`` rather than ``input_ids``.  The tests verify that
    the exported model produces logits numerically close to those from the
    original ``Gemma3ForConditionalGeneration`` HF model, and that greedy
    token generation is identical.
    """

    @staticmethod
    def _make_config():
        """Return a minimal ``Gemma3Config`` for offline testing."""
        from transformers import Gemma3Config, Gemma3TextConfig

        text_config = Gemma3TextConfig(
            head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=1,
            num_key_value_heads=4,
            query_pre_attn_scalar=64,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
        )
        return Gemma3Config(architectures=["Gemma3ForConditionalGeneration"], text_config=text_config)

    def common_fast_gemma3_conditional_random_weights(self, precision, provider):
        from transformers import Gemma3ForConditionalGeneration

        config = self._make_config()
        num_hidden_layers = config.text_config.num_hidden_layers

        model = Gemma3ForConditionalGeneration(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_gemma3_cond_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.text_config.num_key_value_heads,
            head_size=config.text_config.head_dim,
            vocab_size=config.text_config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            embed_fn=model.model.language_model.embed_tokens,
        )

    def common_gemma3_conditional_greedy_generation(self, precision, provider):
        import torch
        from transformers import Gemma3ForConditionalGeneration

        config = self._make_config()
        num_hidden_layers = config.text_config.num_hidden_layers

        torch.manual_seed(42)
        model = Gemma3ForConditionalGeneration(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer(bos_token="<bos>", bos_token_id=2, eos_token="</s>", eos_token_id=1)
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_gemma3_cond_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.text_config.num_key_value_heads,
            head_size=config.text_config.head_dim,
            vocab_size=config.text_config.vocab_size,
            eos_token_id=config.text_config.eos_token_id,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            embed_fn=model.model.language_model.embed_tokens,
        )

    @hide_stdout()
    def test_fast_discrepancy_gemma3_cond_fp32_cpu(self):
        self.common_fast_gemma3_conditional_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma3_cond_fp16_cpu(self):
        self.common_fast_gemma3_conditional_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma3_cond_int4_cpu(self):
        self.common_fast_gemma3_conditional_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gemma3_cond_fp16_cuda(self):
        self.common_fast_gemma3_conditional_random_weights("fp16", "cuda")

    @hide_stdout()
    def test_gemma3_cond_fp32_cpu_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma3_cond_fp16_cpu_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_cond_fp16_cuda_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_cond_bf16_cuda_greedy_generation(self):
        self.common_gemma3_conditional_greedy_generation("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
