# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

QWEN3_VL_MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"


@requires_transformers("5")
class TestRandomQwen3VL(ExtTestCase):
    @staticmethod
    def _make_config():
        from transformers import Qwen3VLConfig
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

        num_hidden_layers = 1

        # Minimal Qwen3-VL text config with small dimensions for fast, offline testing.
        # mrope_section=[12, 10, 10]: sum=32, and sum*2=64 == head_dim.
        text_config = Qwen3VLTextConfig(
            hidden_size=512,
            intermediate_size=1376,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            max_position_embeddings=2048,
            vocab_size=32000,
            rms_norm_eps=1e-6,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [12, 10, 10]},
            pad_token_id=0,
        )
        config = Qwen3VLConfig(text_config=text_config)
        config.architectures = ["Qwen3VLForConditionalGeneration"]
        config.bos_token_id = 1
        config.eos_token_id = 2
        return config

    @hide_stdout()
    def test_qwen3_vl_fp32_cpu_onnx_light_generate(self):
        import torch
        from transformers import Qwen3VLForConditionalGeneration

        config = self._make_config()
        config.text_config.vocab_size = 257
        config.vision_config.depth = 1
        config.vision_config.hidden_size = 32
        config.vision_config.intermediate_size = 64
        config.vision_config.num_heads = 4
        config.vision_config.out_hidden_size = config.text_config.hidden_size
        config.vision_config.deepstack_visual_indexes = []
        torch.manual_seed(42)
        model = Qwen3VLForConditionalGeneration(config).eval()
        self.run_onnx_light_model_generation_test(model, QWEN3_VL_MODEL_NAME, "native_qwen3_vl")

    def common_fast_qwen3_vl_random_weights(self, precision, provider):
        import torch
        from transformers import Qwen3VLForConditionalGeneration

        config = self._make_config()
        text_config = config.text_config
        num_hidden_layers = text_config.num_hidden_layers

        torch.manual_seed(0)
        model = Qwen3VLForConditionalGeneration(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()

        self.run_vl_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=QWEN3_VL_MODEL_NAME,
            basename=f"test_discrepancies_qwen3_vl_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=text_config.num_key_value_heads,
            head_size=text_config.head_dim,
            vocab_size=text_config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
        )

    @hide_stdout()
    def test_fast_discrepancy_qwen3_vl_fp32_cpu(self):
        self.common_fast_qwen3_vl_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen3_vl_fp16_cpu(self):
        self.common_fast_qwen3_vl_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen3_vl_int4_cpu(self):
        self.common_fast_qwen3_vl_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_vl_int4_cuda(self):
        self.common_fast_qwen3_vl_random_weights("int4", "cuda")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_vl_fp32_cuda(self):
        self.common_fast_qwen3_vl_random_weights("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_vl_fp16_cuda(self):
        self.common_fast_qwen3_vl_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_vl_bf16_cuda(self):
        self.common_fast_qwen3_vl_random_weights("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
