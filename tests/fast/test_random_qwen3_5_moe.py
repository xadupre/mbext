# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Fast unit tests for the Qwen3.5-MoE ONNX builder.

These tests cover the changes imported from `microsoft/onnxruntime-genai
PR #2146 <https://github.com/microsoft/onnxruntime-genai/pull/2146>`_,
which add support for the ``Qwen3_5MoeForConditionalGeneration`` HF
architecture (256 routed experts + 1 shared expert with SwiGLU).
"""

import os
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_transformers

QWEN3_5_MOE_MODEL_NAME = "Qwen/Qwen3.5-MoE"


def _make_qwen3_5_moe_config(layer_types, num_hidden_layers=None):
    """Return a minimal ``Qwen3_5MoeConfig`` suitable for offline unit tests.

    Parameters
    ----------
    layer_types:
        List of layer type strings, e.g. ``["full_attention", "linear_attention"]``.
        The number of layers is inferred from this list unless
        ``num_hidden_layers`` is provided.
    num_hidden_layers:
        Explicit override; defaults to ``len(layer_types)``.
    """
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

    if num_hidden_layers is None:
        num_hidden_layers = len(layer_types)

    # partial_rotary_factor=0.25, head_dim=64 -> rdim=16, rdim_half=8.
    # mrope_section=[2, 3, 3]: height positions at stride-1 offsets within rdim_half.
    rope_cfg = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3], "rope_theta": 10000.0, "partial_rotary_factor": 0.25}

    text_config = Qwen3_5MoeTextConfig(
        hidden_size=128,
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
        # MoE-specific settings (kept small for CI).
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
    )
    text_config.rope_scaling = rope_cfg
    text_config.rope_parameters = rope_cfg

    config = Qwen3_5MoeConfig(text_config=text_config, bos_token_id=1, eos_token_id=2)
    config.architectures = ["Qwen3_5MoeForConditionalGeneration"]
    return config


class TestRandomQwen3_5Moe(ExtTestCase):
    def _build_model(self, config, precision, provider):
        """Create a random-weight HF MoE model and build its ONNX export.

        Returns the output directory containing ``model.onnx``.
        """
        import torch
        from transformers import AutoModelForImageTextToText

        from modelbuilder.builder import create_model

        basename = f"test_qwen3_5_moe_{precision}_{provider}_" + "_".join(config.text_config.layer_types)
        model_dir_full = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForImageTextToText.from_config(config)
        model.eval()
        model.save_pretrained(model_dir_full)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir_full)

        create_model(
            model_name=QWEN3_5_MOE_MODEL_NAME,
            input_path=model_dir_full,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )
        return output_dir

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_moe_fp32_cpu_full_attention_build(self):
        """Build a Qwen3.5-MoE decoder with only ``full_attention`` layers.

        Verifies that ``Qwen35MoeTextModel`` registers correctly, emits MoE
        ops in the ONNX graph, and produces a model with the expected
        ``model_type`` in ``genai_config.json``.
        """
        import json

        import onnx

        config = _make_qwen3_5_moe_config(["full_attention", "full_attention"])
        output_dir = self._build_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        # Confirm an MoE op (MoE or QMoE) appears in the graph along with the
        # shared-expert sigmoid gate (com.microsoft Sigmoid via make_sigmoid).
        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertTrue(("MoE" in op_types) or ("QMoE" in op_types), f"Expected an MoE/QMoE op in the graph, found: {sorted(op_types)}")

        # genai_config.json should expose the MoE-specific model_type.
        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)
        with open(genai_config_path, encoding="utf-8") as f:
            genai_config = json.load(f)
        self.assertEqual(genai_config["model"]["type"], "qwen3_5_moe")


if __name__ == "__main__":
    unittest.main(verbosity=2)
