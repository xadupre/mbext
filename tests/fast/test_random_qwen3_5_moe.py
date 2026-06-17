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

import numpy as np

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


def _make_qwen3_5_moe_text_config(layer_types, num_hidden_layers=None):
    """Return a minimal flat ``Qwen3_5MoeTextConfig`` for offline unit tests.

    This is the text-only ``Qwen3_5MoeForCausalLM`` config (no ``text_config``
    sub-config), analogous to the dense ``Qwen3_5ForCausalLM`` flat config used
    by Qwen3.5-4B.
    """
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

    if num_hidden_layers is None:
        num_hidden_layers = len(layer_types)

    rope_cfg = {"type": "mrope", "rope_type": "default", "mrope_section": [2, 3, 3], "rope_theta": 10000.0, "partial_rotary_factor": 0.25}

    config = Qwen3_5MoeTextConfig(
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
        bos_token_id=1,
        eos_token_id=2,
    )
    config.rope_scaling = rope_cfg
    config.rope_parameters = rope_cfg
    config.architectures = ["Qwen3_5MoeForCausalLM"]
    return config


class TestRandomQwen3_5Moe(ExtTestCase):
    def _build_model(self, config, precision, provider, **extra_options):
        """Create a random-weight HF MoE model and build its ONNX export.

        Returns the output directory containing ``model.onnx``.
        """
        import torch
        from transformers import AutoModelForImageTextToText

        from modelbuilder.builder import create_model

        basename = f"test_qwen3_5_moe_{precision}_{provider}_" + "_".join(config.text_config.layer_types)
        if extra_options:
            basename += "_" + "_".join(f"{k}{v}" for k, v in sorted(extra_options.items()))
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
            **extra_options,
        )
        return output_dir

    def _build_text_model(self, config, precision, provider, **extra_options):
        """Create a random-weight flat ``Qwen3_5MoeForCausalLM`` and build it.

        Returns the output directory containing ``model.onnx``.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        basename = f"test_qwen3_5_moe_text_{precision}_{provider}_" + "_".join(config.layer_types)
        if extra_options:
            basename += "_" + "_".join(f"{k}{v}" for k, v in sorted(extra_options.items()))
        model_dir_full = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
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
            **extra_options,
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

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_moe_fp32_cpu_text_only_model_type(self):
        """Verify the text-only Qwen3.5-MoE genai-config ``model_type``.

        When built as a standalone LLM (``exclude_embeds=False`` →
        ``is_text_only=True``) the genai-config ``model.type`` must flip from
        ``qwen3_5_moe`` to ``qwen3_5_moe_text``, matching the
        ``qwen3_5_moe_text`` entry added to ``model_type.h`` in upstream
        `onnxruntime-genai#2186
        <https://github.com/microsoft/onnxruntime-genai/pull/2186>`_.
        """
        import json

        config = _make_qwen3_5_moe_config(["full_attention", "full_attention"])
        output_dir = self._build_model(config, "fp32", "cpu", exclude_embeds=False)

        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)
        with open(genai_config_path, encoding="utf-8") as f:
            genai_config = json.load(f)
        self.assertEqual(genai_config["model"]["type"], "qwen3_5_moe_text")

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_moe_causallm_fp32_cpu_full_attention_build(self):
        """Build a flat ``Qwen3_5MoeForCausalLM`` text-only MoE decoder.

        ``Qwen35MoeCausalLMModel`` handles the flat ``Qwen3_5MoeTextConfig``
        (no ``text_config`` sub-config), mirroring how ``Qwen35CausalLMModel``
        handles the dense ``Qwen3_5ForCausalLM``.  The ONNX graph must include
        the embedding layer (``input_ids`` input), emit an MoE/QMoE op, and
        expose the ``qwen3_5_moe_text`` genai-config ``model_type``.
        """
        import json

        import onnx

        config = _make_qwen3_5_moe_text_config(["full_attention", "full_attention"])
        output_dir = self._build_text_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        # The text-only CausalLM graph includes the embedding, so it takes
        # input_ids directly rather than inputs_embeds.
        input_names = {inp.name for inp in onnx_model.graph.input}
        self.assertIn("input_ids", input_names)

        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertTrue(("MoE" in op_types) or ("QMoE" in op_types), f"Expected an MoE/QMoE op in the graph, found: {sorted(op_types)}")

        genai_config_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_config_path)
        with open(genai_config_path, encoding="utf-8") as f:
            genai_config = json.load(f)
        self.assertEqual(genai_config["model"]["type"], "qwen3_5_moe_text")

    # ------------------------------------------------------------------ #
    # Discrepancy: HF PyTorch vs ONNX Runtime CPU                         #
    # ------------------------------------------------------------------ #

    def _prefill_feed(self, model, config, precision, batch_size=1, seq_len=5):
        """Return ``(inputs_embeds_pt, position_ids_3d, onnx_feed)`` for a
        Qwen3.5-MoE prefill step with all-``full_attention`` layers.
        """
        import torch

        text_cfg = config.text_config
        np_dtype = self.get_input_np_dtype(precision)

        torch.manual_seed(0)
        input_ids = torch.randint(3, text_cfg.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds_pt = model.model.language_model.embed_tokens(input_ids)

        pos = np.arange(seq_len, dtype=np.int64)
        position_ids_3d = np.stack([pos] * 3, axis=0)[:, None, :]  # [3, 1, S]
        position_ids_3d = np.broadcast_to(position_ids_3d, (3, batch_size, seq_len)).copy()

        feed = {
            "inputs_embeds": inputs_embeds_pt.numpy().astype(np_dtype),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": position_ids_3d,
        }
        for i, lt in enumerate(text_cfg.layer_types):
            if lt == "full_attention":
                feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim), dtype=np_dtype
                )
                feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim), dtype=np_dtype
                )
        return inputs_embeds_pt, position_ids_3d, feed

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_moe_fp32_cpu_discrepancy_full_attention(self):
        """Compare ONNX Runtime prefill logits with HF PyTorch forward.

        Builds a tiny random-weight ``Qwen3_5MoeForConditionalGeneration``
        decoder, runs both ``onnxruntime`` CPU and the HF model's
        ``language_model`` forward (with ``inputs_embeds`` and the 3-D
        mRoPE ``position_ids``), and asserts:

        - The per-element maximum absolute difference of the logits is
          below ``1e-3`` (fp32, dominated by the ``com.microsoft:MoE``
          kernel and ``GroupQueryAttention``).
        - The greedy first-token prediction (``argmax`` of the last-row
          logits) agrees between PyTorch and ONNX Runtime — this is the
          "first token difference" that ORT-GenAI greedy generation
          relies on.
        """
        import torch
        from transformers import AutoModelForImageTextToText

        from modelbuilder.builder import create_model

        precision, provider = "fp32", "cpu"
        config = _make_qwen3_5_moe_config(["full_attention", "full_attention"])

        basename = f"test_qwen3_5_moe_disc_{precision}_{provider}"
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

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        sess = self._check_with_ort(onnx_path, cpu=True)
        onnx_input_names = {i.name for i in sess.get_inputs()}

        inputs_embeds_pt, position_ids_3d, feed = self._prefill_feed(model, config, precision)
        feed = {k: v for k, v in feed.items() if k in onnx_input_names}
        ort_outputs = sess.run(None, feed)
        ort_logits = ort_outputs[0]
        self.assertEqual(ort_logits.shape, (1, 5, config.text_config.vocab_size))

        # HF forward using the same inputs_embeds + 3-D position_ids.
        with torch.no_grad():
            pt_out = model.model.language_model(
                inputs_embeds=inputs_embeds_pt,
                position_ids=torch.from_numpy(position_ids_3d),
                attention_mask=torch.ones(1, inputs_embeds_pt.shape[1], dtype=torch.long),
                use_cache=False,
            )
            pt_logits = model.lm_head(pt_out.last_hidden_state).numpy()

        # Discrepancy (uses the same helper as other random-weight tests).
        disc = self.get_numpy_discrepancy(pt_logits, ort_logits)
        self.log_results(
            {
                "step": "prefill",
                "precision": precision,
                "model_id": QWEN3_5_MOE_MODEL_NAME,
                "experiment": "forward",
                "provider": provider,
                "test": basename,
                "input_type": "text",
                "kind": "fast",
                **disc,
            }
        )
        np.testing.assert_allclose(pt_logits, ort_logits, atol=1e-3, rtol=1e-3)

        # First-token agreement: the greedy next token (argmax of the last
        # row of logits) is what ORT-GenAI's greedy generation step consumes.
        pt_first = int(np.argmax(pt_logits[0, -1, :]))
        ort_first = int(np.argmax(ort_logits[0, -1, :]))
        self.assertEqual(pt_first, ort_first)


if __name__ == "__main__":
    unittest.main(verbosity=2)
