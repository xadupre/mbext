# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import json
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

# Fara1.5-27B (https://huggingface.co/microsoft/Fara1.5-27B) is a multimodal
# computer-use agent supervised fine-tuned from Qwen3.5-27B. Its config.json
# reports architectures=["Qwen3_5ForConditionalGeneration"] with a
# Qwen3_5TextConfig-shaped text_config (hybrid full_attention/linear_attention
# layers, full_attention_interval=4, partial_rotary_factor=0.25,
# mrope_section=[11, 11, 10], rope_theta=10000000, head_dim=256), so it is
# already handled by the existing Qwen3.5 builder (``Qwen35TextModel``). This
# test exercises that builder against Fara1.5-27B's specific hyperparameter
# shape at a small scale.
FARA1_5_MODEL_NAME = "microsoft/Fara1.5-27B"


def _make_fara1_5_config(num_hidden_layers=8, full_attention_interval=4):
    """Return a minimal ``Qwen3_5Config`` matching Fara1.5-27B's hyperparameter shape."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig

    # Fara1.5-27B: layer_types alternates 3x linear_attention, then 1x
    # full_attention (full_attention_interval=4), head_dim=256,
    # partial_rotary_factor=0.25 -> rdim=64, rdim_half=32, matching
    # mrope_section=[11, 11, 10] (sums to 32).
    layer_types = ["full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention" for i in range(num_hidden_layers)]

    rope_cfg = {
        "type": "mrope",
        "rope_type": "default",
        "mrope_section": [11, 11, 10],
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
    }

    text_config = Qwen3_5TextConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=256,
        vocab_size=32000,
        rms_norm_eps=1e-06,
        layer_types=layer_types,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
        linear_conv_kernel_dim=4,
    )
    text_config.rope_scaling = rope_cfg
    text_config.rope_parameters = rope_cfg

    config = Qwen3_5Config(text_config=text_config, bos_token_id=1, eos_token_id=2)
    config.architectures = ["Qwen3_5ForConditionalGeneration"]
    return config


class TestRandomFara1_5(ExtTestCase):
    def _build_and_save_model(self, config, precision, provider):
        import torch
        from transformers import AutoModelForImageTextToText

        from modelbuilder.builder import create_model

        basename = f"test_fara1_5_{precision}_{provider}"
        model_dir_full = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForImageTextToText.from_config(config)
        model.eval()
        model.save_pretrained(model_dir_full)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir_full)

        create_model(
            model_name=FARA1_5_MODEL_NAME,
            input_path=model_dir_full,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )
        return model, output_dir

    def _run_text_decoder(self, model, output_dir, config, precision, layer_types, cpu=True):
        import torch

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        text_sess = self._check_with_ort(text_onnx_path, cpu=cpu)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}

        batch_size = 1
        seq_len = 5
        text_cfg = config.text_config

        torch.manual_seed(0)
        input_ids = torch.randint(0, text_cfg.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = model.model.language_model.embed_tokens(input_ids).numpy().astype(self.get_input_np_dtype(precision))

        pos = np.arange(seq_len, dtype=np.int64)
        position_ids_3d = np.stack([pos, pos, pos], axis=0)
        position_ids_3d = np.stack([position_ids_3d] * batch_size, axis=1)

        linear_conv_dim = (
            text_cfg.linear_num_key_heads * text_cfg.linear_key_head_dim * 2
            + text_cfg.linear_num_value_heads * text_cfg.linear_value_head_dim
        )
        conv_kernel_minus1 = text_cfg.linear_conv_kernel_dim - 1

        np_dtype = self.get_input_np_dtype(precision)
        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": position_ids_3d,
        }

        for i, lt in enumerate(layer_types):
            if lt == "full_attention":
                onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim), dtype=np_dtype
                )
                onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim), dtype=np_dtype
                )
            else:
                onnx_feed[f"past_key_values.{i}.conv_state"] = np.zeros((batch_size, linear_conv_dim, conv_kernel_minus1), dtype=np_dtype)
                onnx_feed[f"past_key_values.{i}.recurrent_state"] = np.zeros(
                    (batch_size, text_cfg.linear_num_value_heads, text_cfg.linear_key_head_dim, text_cfg.linear_value_head_dim),
                    dtype=np_dtype,
                )

        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}
        outputs = text_sess.run(None, onnx_feed)
        return outputs

    @requires_transformers("5")
    @hide_stdout()
    def test_fara1_5_fp32_cpu_hybrid_build(self):
        """Build a Fara1.5-27B-shaped hybrid decoder (full_attention_interval=4).

        Verifies that the Qwen3.5 builder handles Fara1.5-27B's specific
        rope configuration (partial_rotary_factor=0.25, mrope_section=[11, 11, 10],
        rope_theta=10000000, head_dim=256) and its hybrid layer pattern.
        """
        import onnx

        config = _make_fara1_5_config()
        model, output_dir = self._build_and_save_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)
        self.assertIn("LinearAttention", op_types)

        with open(os.path.join(output_dir, "genai_config.json")) as f:
            genai_config = json.load(f)
        self.assertTrue(genai_config["search"]["past_present_share_buffer"])

        outputs = self._run_text_decoder(model, output_dir, config, "fp32", config.text_config.layer_types)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    def test_fara1_5_fp16_cpu_hybrid_build(self):
        """fp16 variant of :meth:`test_fara1_5_fp32_cpu_hybrid_build`."""
        config = _make_fara1_5_config()
        model, output_dir = self._build_and_save_model(config, "fp16", "cpu")

        outputs = self._run_text_decoder(model, output_dir, config, "fp16", config.text_config.layer_types)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    @requires_cuda()
    def test_fara1_5_fp16_cuda_hybrid_build(self):
        """fp16 / CUDA variant of :meth:`test_fara1_5_fp32_cpu_hybrid_build`."""
        config = _make_fara1_5_config()
        model, output_dir = self._build_and_save_model(config, "fp16", "cuda")

        outputs = self._run_text_decoder(model, output_dir, config, "fp16", config.text_config.layer_types, cpu=False)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
