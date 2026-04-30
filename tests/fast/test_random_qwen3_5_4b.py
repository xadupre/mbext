# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

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
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_and_save_model(self, config, precision, provider):
        """Create a random-weight HF model and build its ONNX export.

        Returns the (model, output_dir) tuple so callers can run inference.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        basename = f"test_qwen3_5_4b_{precision}_{provider}_{'_'.join(config.layer_types)}"
        model_dir_full = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir_full)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir_full)

        create_model(
            model_name=QWEN3_5_4B_MODEL_NAME,
            input_path=model_dir_full,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )
        return model, output_dir

    def _run_text_decoder(self, model, output_dir, config, precision, layer_types, cpu=True):
        """Load the ONNX text decoder and run a single prefill step.

        Returns the ONNX output list.
        """
        import torch

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        text_sess = self._check_with_ort(text_onnx_path, cpu=cpu)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}

        batch_size = 1
        seq_len = 5

        # Compute inputs_embeds from the saved model's embedding layer.
        # Qwen3_5ForCausalLM uses model.model.embed_tokens (flat structure).
        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = model.model.embed_tokens(input_ids).numpy().astype(self.get_input_np_dtype(precision))

        # 3D position_ids [3, batch_size, seq_len] for mRoPE.
        pos = np.arange(seq_len, dtype=np.int64)
        position_ids_3d = np.stack([pos, pos, pos], axis=0)  # [3, seq_len]
        position_ids_3d = np.stack([position_ids_3d] * batch_size, axis=1)  # [3, B, S]

        np_dtype = self.get_input_np_dtype(precision)
        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": position_ids_3d,
        }

        for i, lt in enumerate(layer_types):
            if lt == "full_attention":
                onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, config.head_dim), dtype=np_dtype
                )
                onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, config.head_dim), dtype=np_dtype
                )
            else:
                linear_conv_dim = (
                    config.linear_num_key_heads * config.linear_key_head_dim * 2
                    + config.linear_num_value_heads * config.linear_value_head_dim
                )
                conv_kernel_minus1 = config.linear_conv_kernel_dim - 1
                onnx_feed[f"past_key_values.{i}.conv_state"] = np.zeros((batch_size, linear_conv_dim, conv_kernel_minus1), dtype=np_dtype)
                onnx_feed[f"past_key_values.{i}.recurrent_state"] = np.zeros(
                    (batch_size, config.linear_num_value_heads, config.linear_key_head_dim, config.linear_value_head_dim), dtype=np_dtype
                )

        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}
        outputs = text_sess.run(None, onnx_feed)
        return outputs

    # ------------------------------------------------------------------ #
    # Tests: full_attention only (standard ORT, no custom ops needed)     #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_fp32_cpu_full_attention(self):
        """Build and run a Qwen3.5-4B (CausalLM) decoder with full_attention layers.

        Qwen3.5-4B uses ``Qwen3_5ForCausalLM`` (text-only) rather than the
        ``Qwen3_5ForConditionalGeneration`` VL model.  The flat
        ``Qwen3_5TextConfig`` is used directly (no ``text_config`` nesting).

        All layers are ``full_attention``, so no custom ORT ops are needed and
        the test can run with the standard ``onnxruntime`` Python binding.

        This exercises the mRoPE embedding, the QK-norm in full attention, the
        OffsetRMSNorm (1+weight) layernorm, the 3-D position_ids input, and the
        ``inputs_embeds``-based interface.
        """
        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        model, output_dir = self._build_and_save_model(config, "fp32", "cpu")

        outputs = self._run_text_decoder(model, output_dir, config, "fp32", ["full_attention", "full_attention"])
        self.assertIsNotNone(outputs[0])
        # logits: [batch_size, seq_len, vocab_size]
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_fp16_cpu_full_attention(self):
        """fp16 variant of :meth:`test_qwen3_5_4b_fp32_cpu_full_attention`."""
        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        model, output_dir = self._build_and_save_model(config, "fp16", "cpu")

        outputs = self._run_text_decoder(model, output_dir, config, "fp16", ["full_attention", "full_attention"])
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    @requires_cuda()
    def test_qwen3_5_4b_fp16_cuda_full_attention(self):
        """fp16 / CUDA variant of :meth:`test_qwen3_5_4b_fp32_cpu_full_attention`."""
        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        model, output_dir = self._build_and_save_model(config, "fp16", "cuda")

        outputs = self._run_text_decoder(model, output_dir, config, "fp16", ["full_attention", "full_attention"], cpu=False)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    # ------------------------------------------------------------------ #
    # Tests: hybrid architecture build verification                       #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_fp32_cpu_hybrid_build(self):
        """Verify that ``create_model`` builds a hybrid Qwen3.5 CausalLM model.

        Tests that a ``Qwen3_5ForCausalLM`` model with mixed layer types
        (``full_attention`` + ``linear_attention``) is exported correctly.
        The ``linear_attention`` layers use the ``com.microsoft:CausalConvWithState``
        and ``com.microsoft:LinearAttention`` custom ops, so this test only
        verifies that ``model.onnx`` is produced and is a valid ONNX file.
        """
        import onnx

        config = _make_qwen3_5_4b_config(["full_attention", "linear_attention"])
        _, output_dir = self._build_and_save_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)
        self.assertIn("LinearAttention", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
