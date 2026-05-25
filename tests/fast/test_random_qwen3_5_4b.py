# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

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
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _build_and_save_model(self, config, precision, provider):
        """Create a random-weight ``Qwen3_5ForCausalLM`` and build its ONNX export.

        Returns the output directory path.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        basename = f"test_qwen3_5_4b_{precision}_{provider}_{'_'.join(config.layer_types)}"
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
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )
        return output_dir

    def _run_text_decoder(self, output_dir, config, precision, layer_types, cpu=True):
        """Load the ONNX text decoder and run a single prefill step.

        Returns the ONNX output list.

        ``Qwen35CausalLMModel`` includes the embedding layer in the ONNX graph
        so the model takes ``input_ids`` directly (no ``inputs_embeds``).
        """
        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        text_sess = self._check_with_ort(text_onnx_path, cpu=cpu)
        onnx_input_names = {inp.name for inp in text_sess.get_inputs()}

        batch_size = 1
        seq_len = 5

        np.random.seed(0)
        input_ids = np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.int64)

        # 2D position_ids [batch_size, seq_len].  In text-only mode the
        # graph itself expands these to [3, B, S] for mRoPE via Unsqueeze+Tile.
        pos = np.arange(seq_len, dtype=np.int64)
        position_ids = np.broadcast_to(pos, (batch_size, seq_len)).copy()

        np_dtype = self.get_input_np_dtype(precision)
        onnx_feed = {"input_ids": input_ids, "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64), "position_ids": position_ids}

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
        return text_sess.run(None, onnx_feed)

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
        Exercises mRoPE embedding, QK-norm, OffsetRMSNorm (1+weight), 3-D
        position_ids, and the ``input_ids``-based interface.
        """
        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        output_dir = self._build_and_save_model(config, "fp32", "cpu")

        outputs = self._run_text_decoder(output_dir, config, "fp32", ["full_attention", "full_attention"])
        self.assertIsNotNone(outputs[0])
        # logits: [batch_size, seq_len, vocab_size]
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_fp16_cpu_full_attention(self):
        """fp16 variant of :meth:`test_qwen3_5_4b_fp32_cpu_full_attention`."""
        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        output_dir = self._build_and_save_model(config, "fp16", "cpu")

        outputs = self._run_text_decoder(output_dir, config, "fp16", ["full_attention", "full_attention"])
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    @requires_cuda()
    def test_qwen3_5_4b_fp16_cuda_full_attention(self):
        """fp16 / CUDA variant of :meth:`test_qwen3_5_4b_fp32_cpu_full_attention`."""
        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        output_dir = self._build_and_save_model(config, "fp16", "cuda")

        outputs = self._run_text_decoder(output_dir, config, "fp16", ["full_attention", "full_attention"], cpu=False)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    # ------------------------------------------------------------------ #
    # Tests: hybrid architecture build verification                       #
    # The linear_attention layers use com.microsoft:CausalConvWithState   #
    # and com.microsoft:LinearAttention custom ops.  When ORT < 1.26,    #
    # both ops are provided as ONNX local-function fallbacks so inference #
    # runs on the standard onnxruntime CPU EP as well.                    #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_fp32_cpu_hybrid_build(self):
        """Verify that ``create_model`` builds a hybrid Qwen3.5-4B CausalLM model.

        The hybrid architecture mixes one ``full_attention`` layer and one
        ``linear_attention`` layer.  Both ``com.microsoft:CausalConvWithState``
        and ``com.microsoft:LinearAttention`` custom ops are expected in the
        graph.  When ORT < 1.26, the builder embeds ONNX local-function
        fallbacks for both ops so inference also runs on standard
        ``onnxruntime``.
        """
        import onnx

        config = _make_qwen3_5_4b_config(["full_attention", "linear_attention"])
        output_dir = self._build_and_save_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)
        self.assertIn("LinearAttention", op_types)

        # Run a prefill step to confirm the local-function fallbacks work
        # with the installed onnxruntime (whether stable or nightly).
        outputs = self._run_text_decoder(output_dir, config, "fp32", config.layer_types)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_fp16_cpu_hybrid_build(self):
        """fp16 variant of :meth:`test_qwen3_5_4b_fp32_cpu_hybrid_build`."""
        import onnx

        config = _make_qwen3_5_4b_config(["full_attention", "linear_attention"])
        output_dir = self._build_and_save_model(config, "fp16", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)
        self.assertIsNotNone(onnx_model)

        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)
        self.assertIn("LinearAttention", op_types)

        outputs = self._run_text_decoder(output_dir, config, "fp16", config.layer_types)
        self.assertIsNotNone(outputs[0])
        self.assertEqual(outputs[0].shape, (1, 5, 32000))

    # ------------------------------------------------------------------ #
    # GenAI generation test                                               #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    @requires_genai("0.14")
    def test_qwen3_5_4b_fp32_cpu_genai_generate(self):
        """Export a Qwen3.5-4B CausalLM model and verify ORT-GenAI generation.

        Builds a random-weight Qwen3.5-4B model with all ``full_attention``
        layers, exports it to ONNX, then exercises ORT-GenAI greedy generation
        via :meth:`run_genai_generation_test`.  ``model=None`` is passed to
        skip the PyTorch token comparison because calling ``Qwen3_5ForCausalLM``
        with the ``inputs_embeds`` + 3-D mRoPE position_ids interface triggers a
        ``has_previous_state`` error in the transformers HybridCache; ORT-GenAI
        handles embedding lookup internally so the test validates that the model
        loads and generates tokens successfully.
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

        # Pass model=None: skip PyTorch token comparison (see docstring).
        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id)

    # ------------------------------------------------------------------ #
    # PR #2157 contract: text-only 2D position_ids + in-graph expansion   #
    # ------------------------------------------------------------------ #

    @requires_transformers("5")
    @hide_stdout()
    def test_qwen3_5_4b_text_only_position_ids_2d(self):
        """Verify the text-only Qwen3.5 graph contract from onnxruntime-genai#2157.

        In text-only mode (``Qwen3_5ForCausalLM`` → ``Qwen35CausalLMModel``,
        ``is_text_only=True``) the ONNX graph must:

        * declare ``position_ids`` as 2D ``[batch_size, sequence_length]``
          (instead of 3D ``[3, B, S]`` used by the VL pipeline), and
        * embed an ``Unsqueeze`` + ``Tile`` pair that expands the 2D input
          to ``[3, B, S]`` so the unchanged mRoPE subgraph can consume it.

        The genai-config ``model_type`` must also flip from
        ``qwen3_5`` (VL) to ``qwen3_5_text`` (LLM), matching the
        ``qwen3_5_text`` entry added to ``model_type.h`` in upstream
        onnxruntime-genai#2157.
        """
        import json

        import onnx

        config = _make_qwen3_5_4b_config(["full_attention", "full_attention"])
        output_dir = self._build_and_save_model(config, "fp32", "cpu")

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        onnx_model = onnx.load(text_onnx_path)

        # 1. position_ids graph input is 2D [batch_size, sequence_length].
        pos_input = next(i for i in onnx_model.graph.input if i.name == "position_ids")
        dims = pos_input.type.tensor_type.shape.dim
        self.assertEqual(len(dims), 2)
        self.assertEqual(dims[0].dim_param, "batch_size")
        self.assertEqual(dims[1].dim_param, "sequence_length")

        # 2. The Unsqueeze + Tile expansion nodes added by
        #    ``make_position_ids_reformatting`` are present in the graph.
        node_names = {node.name for node in onnx_model.graph.node}
        self.assertIn("/model/position_ids_expand/Unsqueeze", node_names)
        self.assertIn("/model/position_ids_expand/Tile", node_names)

        # 3. genai-config model type is the text-only variant.
        genai_cfg_path = os.path.join(output_dir, "genai_config.json")
        self.assertExists(genai_cfg_path)
        with open(genai_cfg_path) as f:
            genai_cfg = json.load(f)
        self.assertEqual(genai_cfg["model"]["type"], "qwen3_5_text")


if __name__ == "__main__":
    unittest.main(verbosity=2)
