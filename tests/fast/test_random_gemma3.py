# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import (
    ExtTestCase,
    hide_stdout,
    requires_cuda,
    requires_transformers,
    run_session_or_io_binding,
)

GEMMA3_MODEL_NAME = "google/gemma-3-4b-it"

# Minimal vocabulary used for the test tokenizer.  The exact contents are not
# important; we only need a valid tokenizer artifact so that ``create_model``
# can save the processing files alongside the ONNX model.
_VOCAB = {"<unk>": 0, "<s>": 1, "</s>": 2}


def _make_gemma3_config():
    """Return a minimal ``Gemma3Config`` for ``Gemma3ForConditionalGeneration``.

    Uses 6 hidden layers so that the default ``layer_types`` produced by
    ``Gemma3TextConfig`` includes exactly one global (full_attention) layer
    (layer 5) and five local sliding-attention layers (layers 0-4).  This
    matches the ``is_local()`` pattern in ``Gemma3Model`` which returns
    ``True`` when ``(layer_id + 1) % 6 != 0``.
    """
    from transformers import Gemma3Config, Gemma3TextConfig

    text_config = Gemma3TextConfig(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=64,
        vocab_size=32000,
        query_pre_attn_scalar=64,
        sliding_window=256,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
    )
    config = Gemma3Config(text_config=text_config)
    config.architectures = ["Gemma3ForConditionalGeneration"]
    return config


@requires_transformers("5")
class TestRandomGemma3(ExtTestCase):
    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _build_and_save_model(self, config, precision, provider):
        """Create a random-weight HF model and build its ONNX export.

        Returns the ``(model, output_dir)`` pair so callers can run ONNX
        inference and compare against the PyTorch reference output.
        """
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForImageTextToText, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        basename = f"test_gemma3_conditional_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForImageTextToText.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = _VOCAB
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=GEMMA3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )
        return model, output_dir

    def _run_text_decoder(self, model, output_dir, config, precision, cpu=True):
        """Load the exported text decoder ONNX model and run a prefill step.

        Because ``create_model`` sets ``exclude_embeds=True`` for
        ``Gemma3ForConditionalGeneration``, the decoder accepts
        ``inputs_embeds`` rather than ``input_ids``.

        Returns the ONNX output list (logits first, then KV-cache tensors).
        """
        import torch

        text_cfg = config.text_config
        num_hidden_layers = text_cfg.num_hidden_layers
        batch_size = 1
        seq_len = 5
        np_dtype = self.get_input_np_dtype(precision)

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)

        sess = self._check_with_ort(text_onnx_path, cpu=cpu)
        onnx_input_names = {inp.name for inp in sess.get_inputs()}

        # Derive inputs_embeds from the saved model's embedding table so we
        # can also run the PyTorch reference and compare logits.
        # Seed 0 governs the random input token IDs; seed 42 in
        # _build_and_save_model governs the model weights.  Keeping them
        # separate makes each concern independently reproducible.
        torch.manual_seed(0)
        input_ids = torch.randint(0, text_cfg.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            inputs_embeds = (
                model.model.language_model.embed_tokens(input_ids).numpy().astype(np_dtype)
            )

        onnx_feed = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
        }
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim),
                dtype=np_dtype,
            )
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim),
                dtype=np_dtype,
            )

        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}
        outputs = sess.run(None, onnx_feed)
        return outputs, inputs_embeds

    # ------------------------------------------------------------------ #
    # Common test helpers                                                  #
    # ------------------------------------------------------------------ #

    def common_gemma3_random_weights(self, precision, provider):
        """Build and run a single prefill step; check ONNX vs PyTorch logits."""
        import torch

        config = _make_gemma3_config()
        text_cfg = config.text_config

        model, output_dir = self._build_and_save_model(config, precision, provider)
        outputs, inputs_embeds = self._run_text_decoder(
            model, output_dir, config, precision, cpu=(provider == "cpu")
        )

        # Verify output shape: [batch, seq_len, vocab_size].
        self.assertEqual(outputs[0].shape, (1, 5, text_cfg.vocab_size))

        # Compare against PyTorch reference forward pass.
        with torch.no_grad():
            embeds_tensor = torch.from_numpy(inputs_embeds.astype(np.float32))
            hidden = model.model.language_model(inputs_embeds=embeds_tensor)
            pt_logits = model.lm_head(hidden.last_hidden_state).numpy()

        atol = {"fp32": 1e-3, "fp16": 1e-2, "bf16": 1e-2, "int4": 0.5}
        np.testing.assert_allclose(
            pt_logits, outputs[0].astype(np.float32), atol=atol[precision], rtol=1e-3
        )

    def common_gemma3_prefill_and_decode(self, precision, provider):
        """Two-step (prefill + single-token decode) run-through."""
        config = _make_gemma3_config()
        text_cfg = config.text_config
        num_hidden_layers = text_cfg.num_hidden_layers
        batch_size = 1
        seq_len = 5
        np_dtype = self.get_input_np_dtype(precision)

        _, output_dir = self._build_and_save_model(config, precision, provider)

        text_onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(text_onnx_path)
        sess = self._check_with_ort(text_onnx_path, cpu=(provider == "cpu"))
        onnx_input_names = {inp.name for inp in sess.get_inputs()}

        inputs_embeds_prefill = np.random.randn(batch_size, seq_len, text_cfg.hidden_size).astype(
            np_dtype
        )

        prefill_feed = {
            "inputs_embeds": inputs_embeds_prefill,
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
        }
        for i in range(num_hidden_layers):
            prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim),
                dtype=np_dtype,
            )
            prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, text_cfg.num_key_value_heads, 0, text_cfg.head_dim),
                dtype=np_dtype,
            )
        prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

        with self.subTest(step="prefill"):
            prefill_results, _ = run_session_or_io_binding(
                use_iobinding=(precision == "bf16"),
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=text_cfg.vocab_size,
            )
            self.assertEqual(
                prefill_results["logits"].shape, (batch_size, seq_len, text_cfg.vocab_size)
            )

        with self.subTest(step="decode"):
            inputs_embeds_decode = np.random.randn(batch_size, 1, text_cfg.hidden_size).astype(
                np_dtype
            )
            decode_feed = {
                "inputs_embeds": inputs_embeds_decode,
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            decode_results, _ = run_session_or_io_binding(
                use_iobinding=(precision == "bf16"),
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=text_cfg.vocab_size,
                results=prefill_results,
            )
            self.assertEqual(decode_results["logits"].shape, (batch_size, 1, text_cfg.vocab_size))

    # ------------------------------------------------------------------ #
    # fp32 / CPU tests                                                     #
    # ------------------------------------------------------------------ #

    @hide_stdout()
    def test_gemma3_fp32_cpu_random_weights(self):
        """Build and run Gemma3ForConditionalGeneration with fp32/CPU random weights."""
        self.common_gemma3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_gemma3_fp32_cpu_prefill_and_decode(self):
        """Prefill + single-decode step for Gemma3ForConditionalGeneration, fp32/CPU."""
        self.common_gemma3_prefill_and_decode("fp32", "cpu")

    # ------------------------------------------------------------------ #
    # fp16 / CPU tests                                                     #
    # ------------------------------------------------------------------ #

    @hide_stdout()
    def test_gemma3_fp16_cpu_random_weights(self):
        """Build and run Gemma3ForConditionalGeneration with fp16/CPU random weights."""
        self.common_gemma3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_gemma3_fp16_cpu_prefill_and_decode(self):
        """Prefill + single-decode step for Gemma3ForConditionalGeneration, fp16/CPU."""
        self.common_gemma3_prefill_and_decode("fp16", "cpu")

    # ------------------------------------------------------------------ #
    # CUDA tests                                                           #
    # ------------------------------------------------------------------ #

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_fp16_cuda_random_weights(self):
        """Build and run Gemma3ForConditionalGeneration with fp16/CUDA random weights."""
        self.common_gemma3_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_fp16_cuda_prefill_and_decode(self):
        """Prefill + single-decode step for Gemma3ForConditionalGeneration, fp16/CUDA."""
        self.common_gemma3_prefill_and_decode("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
