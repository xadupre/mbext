# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest


from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda

MODEL_NAME = "HunYuanDenseV1ForCausalLM"


def _make_config(num_hidden_layers=1):
    from transformers import HunYuanDenseV1Config

    return HunYuanDenseV1Config(
        architectures=[MODEL_NAME],
        bos_token_id=1,
        eos_token_id=2,
        hidden_act="silu",
        hidden_size=512,
        intermediate_size=1376,
        max_position_embeddings=2048,
        model_type="hunyuan_v1_dense",
        num_attention_heads=8,
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=4,
        head_dim=64,
        rms_norm_eps=1e-05,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        vocab_size=128,
        tie_word_embeddings=False,
    )


class TestHunyuanDenseV1(ExtTestCase):
    def common_fast_hunyuan_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM

        num_hidden_layers = 1
        config = _make_config(num_hidden_layers)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_hunyuan_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
        )

    def test_hunyuan_builder_instantiation(self):
        """Smoke test: HunyuanDenseV1Model can be instantiated and overrides the
        attention_qk_rope_and_norm hook (RoPE-then-norm ordering)."""
        import onnx_ir as ir

        from modelbuilder.builders.base import Model
        from modelbuilder.builders.hunyuan import HunyuanDenseV1Model

        config = _make_config()
        builder = HunyuanDenseV1Model(
            config, io_dtype=ir.DataType.FLOAT, onnx_dtype=ir.DataType.FLOAT, ep="cpu", cache_dir=None, extra_options={}
        )
        # Hunyuan flips QK-norm placement; ensure it overrides the base hook.
        self.assertIsNot(HunyuanDenseV1Model.make_attention_qk_rope_and_norm, Model.make_attention_qk_rope_and_norm)
        # is_fused_rope_supported must be False so RotaryEmbedding nodes are emitted
        # explicitly (allowing QK norm to be inserted between RoPE and attention).
        self.assertFalse(builder.is_fused_rope_supported())
        # make_attention_init sets q_norm/k_norm before the base packed matmul check.
        self.assertTrue(builder.attention_attrs["q_norm"])
        self.assertTrue(builder.attention_attrs["k_norm"])

    def test_hunyuan_rope_theta_dynamic_ntk_scaling(self):
        """Dynamic NTK-alpha scaling bakes the effective theta into config.rope_theta."""
        import onnx_ir as ir

        from modelbuilder.builders.hunyuan import HunyuanDenseV1Model

        config = _make_config()
        # Inject NTK-alpha scaling on top of the default rope params.
        head_dim = config.head_dim
        config.rope_theta = 10000.0
        config.rope_scaling = {"alpha": 1000.0, "rope_type": "default"}

        HunyuanDenseV1Model(config, io_dtype=ir.DataType.FLOAT, onnx_dtype=ir.DataType.FLOAT, ep="cpu", cache_dir=None, extra_options={})

        expected_theta = 10000.0 * (1000.0 ** (head_dim / (head_dim - 2)))
        self.assertAlmostEqual(expected_theta, config.rope_theta, rtol=1e-6)
        # rope_scaling must be cleared so downstream uses the baked theta.
        self.assertIsNone(config.rope_scaling)

    @hide_stdout()
    def test_fast_discrepancy_hunyuan_fp32_cpu(self):
        self.common_fast_hunyuan_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_hunyuan_fp16_cpu(self):
        self.common_fast_hunyuan_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_hunyuan_fp16_cuda(self):
        self.common_fast_hunyuan_random_weights("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
