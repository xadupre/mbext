# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, requires_transformers

QWEN3_5_4B_MODEL_NAME = "Qwen/Qwen3.5-4B"


def _make_qwen3_5_4b_config(layer_types=None, num_hidden_layers=None):
    """Return a minimal ``Qwen3_5TextConfig`` suitable for offline unit tests.

    Qwen3.5-4B uses ``Qwen3_5ForCausalLM`` with a flat ``Qwen3_5TextConfig``
    (no ``text_config`` sub-config) and all ``full_attention`` layers.

    Parameters
    ----------
    layer_types:
        List of layer type strings.  Defaults to
        ``["full_attention", "full_attention"]``.
    num_hidden_layers:
        Explicit override; defaults to ``len(layer_types)``.
    """
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    if layer_types is None:
        layer_types = ["full_attention", "full_attention"]
    if num_hidden_layers is None:
        num_hidden_layers = len(layer_types)

    # partial_rotary_factor=0.25, head_dim=64 → rdim=16, rdim_half=8.
    # mrope_section=[2, 3, 3]: sum = 8 = rdim_half. ✓
    rope_cfg = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3], "rope_theta": 10000.0, "partial_rotary_factor": 0.25}

    config = Qwen3_5TextConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=256,
        vocab_size=32000,
        rms_norm_eps=1e-6,
        layer_types=layer_types,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        bos_token_id=1,
        eos_token_id=2,
    )
    config.rope_parameters = rope_cfg
    config.architectures = ["Qwen3_5ForCausalLM"]
    return config


class TestRandomQwen3_5_4B(ExtTestCase):
    # ------------------------------------------------------------------ #
    # Discrepancy tests: full_attention only (standard ORT)               #
    # ------------------------------------------------------------------ #

    def common_fast_qwen3_5_4b_random_weights(self, precision, provider):
        """Export a random-weight Qwen3.5-4B CausalLM and compare PyTorch vs ONNX.

        Uses :meth:`run_vl_random_weights_test` with ``pt_mode="inputs_embeds"``
        since the ONNX decoder is built with ``exclude_embeds=True`` and takes
        ``inputs_embeds`` + 3-D mRoPE ``position_ids``.
        """
        import torch
        from transformers import AutoModelForCausalLM

        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        num_hidden_layers = len(config.layer_types)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()

        self.run_vl_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=QWEN3_5_4B_MODEL_NAME,
            basename=f"test_discrepancies_qwen3_5_4b_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            pt_mode="inputs_embeds",
        )

    @requires_transformers("5")
    @hide_stdout()
    def test_fast_discrepancy_qwen3_5_4b_fp32_cpu(self):
        """Build and run a Qwen3.5-4B (CausalLM) decoder with full_attention layers.

        Qwen3.5-4B uses ``Qwen3_5ForCausalLM`` (text-only) rather than the
        ``Qwen3_5ForConditionalGeneration`` VL model.  The flat
        ``Qwen3_5TextConfig`` is used directly (no ``text_config`` nesting).

        All layers are ``full_attention``, so no custom ORT ops are needed and
        the test can run with the standard ``onnxruntime`` Python binding.
        Exercises mRoPE embedding, QK-norm, OffsetRMSNorm (1+weight), 3-D
        position_ids, and the ``inputs_embeds``-based interface.
        """
        self.common_fast_qwen3_5_4b_random_weights("fp32", "cpu")

    @requires_transformers("5")
    @hide_stdout()
    def test_fast_discrepancy_qwen3_5_4b_fp16_cpu(self):
        """fp16 variant of :meth:`test_fast_discrepancy_qwen3_5_4b_fp32_cpu`."""
        self.common_fast_qwen3_5_4b_random_weights("fp16", "cpu")

    @requires_transformers("5")
    @hide_stdout()
    def test_fast_discrepancy_qwen3_5_4b_int4_cpu(self):
        """int4 variant of :meth:`test_fast_discrepancy_qwen3_5_4b_fp32_cpu`."""
        self.common_fast_qwen3_5_4b_random_weights("int4", "cpu")

    @requires_transformers("5")
    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_qwen3_5_4b_fp16_cuda(self):
        """fp16 / CUDA variant of :meth:`test_fast_discrepancy_qwen3_5_4b_fp32_cpu`."""
        self.common_fast_qwen3_5_4b_random_weights("fp16", "cuda")

    # ------------------------------------------------------------------ #
    # Hybrid architecture build verification                              #
    # linear_attention layers use com.microsoft:CausalConvWithState and  #
    # com.microsoft:LinearAttention custom ops; only model.onnx validity  #
    # is checked here (inference requires ORT-GenAI runtime).            #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_fp32_cpu_hybrid_build(self):
        """Verify that ``create_model`` builds a hybrid Qwen3.5-4B CausalLM model.

        Tests that a ``Qwen3_5ForCausalLM`` model with mixed layer types
        (``full_attention`` + ``linear_attention``) is exported correctly.
        Only verifies that ``model.onnx`` is produced and is a valid ONNX file.
        """
        import torch
        import onnx
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        config = _make_qwen3_5_4b_config(["full_attention", "linear_attention"])
        basename = "test_qwen3_5_4b_fp32_cpu_hybrid_build"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)

        create_model(
            model_name=QWEN3_5_4B_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)
        self.assertIn("LinearAttention", op_types)

    # ------------------------------------------------------------------ #
    # GenAI generation test                                               #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    @requires_genai()
    def test_qwen3_5_4b_fp32_cpu_genai_generate(self):
        """Export a Qwen3.5-4B CausalLM model and verify ORT-GenAI generation.

        Builds a random-weight Qwen3.5-4B model with all ``full_attention``
        layers, exports it to ONNX, then exercises ORT-GenAI greedy generation
        via :meth:`run_genai_generation_test`.  The ``pt_tokens`` reference
        is omitted (``model=None``) because the ONNX decoder uses
        ``inputs_embeds`` while ORT-GenAI dispatches generation through its
        own embedding interface; the test therefore validates that ORT-GenAI
        can successfully load and run the model without raising an exception.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        prefix = "test_qwen3_5_4b_fp32_cpu_genai_generate"
        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=QWEN3_5_4B_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )

        # Pass model=None: skip PyTorch token comparison since the ONNX model
        # uses the inputs_embeds interface; only verify ORT-GenAI can run.
        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
