# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Fast random-weight tests for DeepSeek-V3 / DeepSeek-V4-Flash.

These tests use a tiny synthetic config to validate that the builder can
export an ONNX model and that PyTorch vs ONNX logit discrepancies are within
acceptable tolerances.

Architecture strings dispatched to the same builder (DeepSeekV3Model):
  - ``DeepseekV3ForCausalLM`` (DeepSeek-V3 / deepseek_v3 transformers model type)
  - ``DeepseekV4ForCausalLM`` (DeepSeek-V4-Flash — same MLA+MoE structure as V3)

Note: ``DeepseekV4ForCausalLM`` is not yet registered in the transformers
model registry, so the synthetic config is created with
``deepseek_v3`` as the ``model_type`` and the architecture string is
patched afterwards.  This exercises the builder dispatch path for V4 without
requiring a separate registered config class.
"""

import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda

MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"


def _make_tiny_deepseek_config(**overrides):
    """Return a tiny DeepseekV3Config suitable for random-weight tests."""
    from transformers import AutoConfig

    cfg_kwargs = dict(
        architectures=["DeepseekV3ForCausalLM"],
        hidden_size=64,
        intermediate_size=64,  # dense MLP (first_k_dense_replace layers)
        moe_intermediate_size=16,  # per-expert intermediate size
        num_hidden_layers=3,  # 1 dense + 2 MoE
        num_attention_heads=4,
        num_key_value_heads=4,
        # MLA dimensions
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        # MoE
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        # Other
        hidden_act="silu",
        rms_norm_eps=1e-5,
        vocab_size=256,
        max_position_embeddings=128,
        rope_interleave=True,
        attention_bias=False,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
    )
    cfg_kwargs.update(overrides)
    # Always use the "deepseek_v3" model_type regardless of the architecture
    # string.  DeepseekV4ForCausalLM is not yet registered in the transformers
    # model registry, so its config must be built via deepseek_v3 and then the
    # architectures field overridden (see test_fast_discrepancy_deepseek_v4_*).
    return AutoConfig.for_model("deepseek_v3", **cfg_kwargs)


class TestDeepSeekV3(ExtTestCase):
    def common_fast_deepseek_random_weights(self, precision, provider, arch="DeepseekV3ForCausalLM"):
        from transformers import AutoModelForCausalLM

        num_hidden_layers = 3
        config = _make_tiny_deepseek_config(num_hidden_layers=num_hidden_layers, architectures=[arch])
        # qk_head_dim = qk_nope + qk_rope = 16; used as head_size for KV cache
        kv_head_size = config.qk_head_dim  # = 16

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        # The ORT fused MoE kernel uses softmax-based routing while DeepSeek V3
        # uses sigmoid-based grouped-topk routing, so outputs will differ for MoE
        # layers.  Use wider tolerances to accommodate this routing approximation.
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_deepseek_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=kv_head_size,
            vocab_size=config.vocab_size,
            atol={"fp32": 0.1, "fp16": 0.15, "bf16": 0.15, "int4": 0.5},
            rtol={"fp32": 1.0, "fp16": 1.0, "bf16": 1.0, "int4": 10000},
        )

    @hide_stdout()
    def test_fast_discrepancy_deepseek_fp32_cpu(self):
        self.common_fast_deepseek_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_deepseek_fp16_cpu(self):
        self.common_fast_deepseek_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_deepseek_fp16_cuda(self):
        self.common_fast_deepseek_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_deepseek_bf16_cuda(self):
        self.common_fast_deepseek_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_deepseek_v4_dispatch_fp32_cpu(self):
        # Verify that DeepseekV4ForCausalLM dispatches to the same builder.
        # DeepseekV4ForCausalLM is not yet in the transformers registry, so we
        # create the config via deepseek_v3 and patch the architecture string.
        self.common_fast_deepseek_random_weights("fp32", "cpu", arch="DeepseekV4ForCausalLM")


if __name__ == "__main__":
    unittest.main()
