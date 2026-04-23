# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, requires_transformers

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
    # output_dim must equal text hidden_size.
    audio_config = Qwen2_5OmniAudioEncoderConfig(
        encoder_layers=1, encoder_attention_heads=2, encoder_ffn_dim=32, d_model=32, output_dim=256
    )

    # Minimal vision encoder: single layer, very small hidden size.
    # out_hidden_size must equal text hidden_size.
    vision_config = Qwen2_5OmniVisionEncoderConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=256)

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

    @unittest.skip(
        "RuntimeError: No execution providers were provided or selected – "
        "ORT-GenAI 0.13.x initialises Qwen2_5_VL as a full multimodal pipeline "
        "and cannot load the text-only thinker ONNX on CPU. "
        "Un-skip when ORT-GenAI supports text-only mRoPE decoder inference."
    )
    @hide_stdout()
    @requires_genai()
    def test_qwen25omni_fp32_cpu_genai_generate(self):
        """Verify that ``onnxruntime-genai`` can load and run a Qwen2.5-Omni thinker model.

        This test builds a minimal random-weight Qwen2.5-Omni thinker ONNX model
        (fp32 / CPU) and verifies that ORT-genai can load and generate from it.
        PyTorch parity is not checked here because the mRoPE position-id computation
        differs between PyTorch (internal ``get_rope_index``) and ORT-genai; what we
        validate instead is that the genai pipeline executes without error.
        """
        import torch

        from modelbuilder.builder import create_model

        prefix = "test_qwen25omni_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = _make_qwen25_omni_thinker_config(num_hidden_layers)

        from transformers import Qwen2_5OmniThinkerForConditionalGeneration

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = Qwen2_5OmniThinkerForConditionalGeneration(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=QWEN2_5_OMNI_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        # Pass model=None to skip PyTorch comparison: mRoPE position-id handling
        # differs between PyTorch (computed internally by get_rope_index) and
        # ORT-genai (computed from the genai pipeline).  The test still verifies
        # that model.onnx and genai_config.json are valid and that ORT-genai can
        # execute a generation loop without error.
        self.run_genai_generation_test(
            output_dir, model=None, vocab_size=config.text_config.vocab_size, eos_token_id=config.text_config.eos_token_id
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
