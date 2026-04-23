# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

QWEN2_5_OMNI_MODEL_NAME = "Qwen/Qwen2.5-Omni-3B"


def _make_qwen25_omni_thinker_config(num_hidden_layers=1):
    """Return a minimal ``Qwen2_5OmniThinkerConfig`` for offline unit tests.

    The config uses small model dimensions so that the test runs quickly
    without downloading any model weights.  The ``architectures`` attribute
    is set to ``["Qwen2_5OmniForConditionalGeneration"]`` so that
    :func:`modelbuilder.builder.create_model` dispatches to
    :class:`~modelbuilder.builders.qwen.Qwen25OmniThinkerModel`.

    Qwen2.5-Omni uses multimodal RoPE (mRoPE) with ``mrope_section`` stored
    in ``rope_parameters`` rather than ``rope_scaling``.  The minimal audio
    and vision tower configs are kept tiny to avoid OOM in CI.
    """
    from transformers import Qwen2_5OmniAudioEncoderConfig, Qwen2_5OmniThinkerConfig, Qwen2_5OmniVisionEncoderConfig

    # Minimal audio encoder: single layer, very small hidden size.
    audio_config = Qwen2_5OmniAudioEncoderConfig(
        encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, d_model=32, output_dim=256  # must equal text hidden_size
    )

    # Minimal vision encoder: single layer, very small hidden size.
    vision_config = Qwen2_5OmniVisionEncoderConfig(
        depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256  # must equal text hidden_size
    )

    config = Qwen2_5OmniThinkerConfig(audio_config=audio_config, vision_config=vision_config)

    # Override text config with small dimensions.
    # head_size = hidden_size // num_attention_heads = 256 // 4 = 64
    # sum(mrope_section) * 2 must equal head_size:
    #   sum([8, 12, 12]) = 32, 32 * 2 = 64 == head_size ✓
    config.text_config.hidden_size = 256
    config.text_config.intermediate_size = 512
    config.text_config.num_hidden_layers = num_hidden_layers
    config.text_config.layer_types = ["full_attention"] * num_hidden_layers
    config.text_config.num_attention_heads = 4
    config.text_config.num_key_value_heads = 2
    config.text_config.vocab_size = 32000
    config.text_config.bos_token_id = 1
    config.text_config.eos_token_id = 2
    config.text_config.rms_norm_eps = 1e-6
    config.text_config.max_position_embeddings = 2048
    config.text_config.hidden_act = "silu"
    # mRoPE parameters: sections must satisfy sum(mrope_section) * 2 == head_size
    config.text_config.rope_parameters = {"rope_theta": 10000.0, "rope_type": "default", "mrope_section": [8, 12, 12]}

    # Set architecture to trigger the Qwen2_5OmniForConditionalGeneration
    # dispatch path in builder.py.
    config.architectures = ["Qwen2_5OmniForConditionalGeneration"]
    return config


@requires_transformers("5")
class TestRandomQwen25Omni(ExtTestCase):
    def common_fast_qwen25omni_random_weights(self, precision, provider):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        num_hidden_layers = 1
        config = _make_qwen25_omni_thinker_config(num_hidden_layers)

        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()

        head_size = config.text_config.hidden_size // config.text_config.num_attention_heads
        self.run_vl_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=QWEN2_5_OMNI_MODEL_NAME,
            basename=f"test_discrepancies_qwen25omni_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.text_config.num_key_value_heads,
            head_size=head_size,
            vocab_size=config.text_config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers},
            pt_mode="inputs_embeds",
        )

    @hide_stdout()
    def test_fast_discrepancy_qwen25omni_fp32_cpu(self):
        self.common_fast_qwen25omni_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25omni_fp16_cpu(self):
        self.common_fast_qwen25omni_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_qwen25omni_int4_cpu(self):
        self.common_fast_qwen25omni_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25omni_fp16_cuda(self):
        self.common_fast_qwen25omni_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen25omni_bf16_cuda(self):
        self.common_fast_qwen25omni_random_weights("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
