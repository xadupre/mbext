# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Fast random-weight tests for Mixtral (``MixtralForCausalLM``).

These tests use a tiny synthetic ``MixtralConfig`` to validate that the
``MixtralModel`` builder can export an ONNX model with the block-sparse
Mixture-of-Experts (QMoE) subgraph.  They run completely offline without
downloading ``mistralai/Mixtral-8x7B-v0.1``.

The QMoE operator is only supported on CUDA in ONNX Runtime, so only the
build (weight extraction + ONNX IR construction) is exercised on CPU; the
prefill/decode inference test is gated behind ``@requires_cuda()``.
"""

import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, run_session_or_io_binding

_MODEL_NAME = "mistralai/Mixtral-8x7B-v0.1"

# Dimensions chosen to keep the test fast while exercising the full pipeline.
# head_size = hidden_size // num_attention_heads = 64 // 4 = 16
_HIDDEN_SIZE = 64
_NUM_ATTN_HEADS = 4
_NUM_KV_HEADS = 2
_HEAD_SIZE = _HIDDEN_SIZE // _NUM_ATTN_HEADS
_INTERMEDIATE_SIZE = 32  # per-expert hidden dimension
_NUM_EXPERTS = 4
_NUM_EXPERTS_PER_TOK = 2
_MAX_POS = 128
_VOCAB_SIZE = 1000
_NUM_HIDDEN_LAYERS = 2


def _make_mixtral_config():
    """Create a tiny ``MixtralConfig`` matching the ``MixtralForCausalLM`` arch."""
    from transformers import MixtralConfig

    return MixtralConfig(
        architectures=["MixtralForCausalLM"],
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=_INTERMEDIATE_SIZE,
        num_hidden_layers=_NUM_HIDDEN_LAYERS,
        num_attention_heads=_NUM_ATTN_HEADS,
        num_key_value_heads=_NUM_KV_HEADS,
        num_local_experts=_NUM_EXPERTS,
        num_experts_per_tok=_NUM_EXPERTS_PER_TOK,
        max_position_embeddings=_MAX_POS,
        rms_norm_eps=1e-5,
        vocab_size=_VOCAB_SIZE,
        hidden_act="silu",
        bos_token_id=1,
        eos_token_id=2,
    )


def _make_mixtral_builder(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options):
    """Return a ``MixtralModel`` subclass whose ``load_weights`` returns a
    freshly constructed random-weight model instead of loading from disk.

    This keeps the test completely offline and avoids downloading the real
    ``mistralai/Mixtral-8x7B-v0.1`` checkpoint.
    """
    import torch
    from transformers import AutoModelForCausalLM

    from modelbuilder.builders.mistral import MixtralModel

    class _MixtralBuilderWithSyntheticWeights(MixtralModel):
        """Subclass that replaces disk-based weight loading with an in-memory
        synthetic PyTorch model created at test time.  Everything else (ONNX
        graph construction, rotary caches, QMoE quantisation) runs as in
        production."""

        def load_weights(self, input_path):  # noqa: ARG002
            torch.manual_seed(0)
            model = AutoModelForCausalLM.from_config(config)
            model.eval()
            return model

    return _MixtralBuilderWithSyntheticWeights(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)


class TestRandomMixtral(ExtTestCase):
    def _build_mixtral_onnx(self, output_dir, cache_dir, ep):
        """Build the Mixtral ONNX model with synthetic random weights.

        The MoE op is only supported with INT4 (QMoE) weights on CUDA, so
        ``onnx_dtype`` is forced to INT4.  Building (weight extraction and ONNX
        IR construction) runs on CPU using block-wise quantization.
        """
        from modelbuilder.builder import set_io_dtype, set_onnx_dtype

        config = _make_mixtral_config()
        extra_options = {}
        io_dtype = set_io_dtype("int4", ep, extra_options)
        onnx_dtype = set_onnx_dtype("int4", extra_options)

        builder = _make_mixtral_builder(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        builder.make_model(cache_dir)
        builder.save_model(output_dir)
        return config

    @hide_stdout()
    def test_mixtral_onnx_build_cpu(self):
        """Build the Mixtral ONNX model with synthetic random weights (CPU safe).

        The ONNX graph construction (weight extraction and ONNX IR building)
        runs entirely on CPU using block-wise QMoE quantization.  Only ORT
        inference of the QMoE operator requires CUDA, so this test can run in
        any environment.
        """
        basename = "test_mixtral_build_cpu"
        output_dir, cache_dir = self.get_dirs(basename)

        config = self._build_mixtral_onnx(output_dir, cache_dir, ep="cpu")

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        # Verify that the ONNX model has the expected inputs and outputs by
        # loading it with the onnx library (no ORT / CUDA required).
        import onnx

        model_proto = onnx.load(onnx_path)
        input_names = {inp.name for inp in model_proto.graph.input}
        output_names = {out.name for out in model_proto.graph.output}
        op_types = {node.op_type for node in model_proto.graph.node}

        # Standard inputs
        self.assertIn("input_ids", input_names)
        self.assertIn("attention_mask", input_names)

        # The block-sparse MoE layers are exported as QMoE ops.
        self.assertIn("QMoE", op_types)

        # KV-cache inputs and outputs
        num_hidden_layers = config.num_hidden_layers
        for i in range(num_hidden_layers):
            self.assertIn(f"past_key_values.{i}.key", input_names)
            self.assertIn(f"past_key_values.{i}.value", input_names)
            self.assertIn(f"present.{i}.key", output_names)
            self.assertIn(f"present.{i}.value", output_names)

        # Logits output
        self.assertIn("logits", output_names)

    @hide_stdout()
    @requires_cuda()
    def test_mixtral_int4_cuda_build_and_run(self):
        """Build the Mixtral ONNX model and run prefill + decode on CUDA.

        The QMoE operator is only supported on CUDA, so this test is gated
        by ``@requires_cuda()``.  We check that:
        * The ONNX file is created without errors.
        * An ORT inference session can be created on CUDA.
        * Prefill produces logits of the correct shape.
        * Decode (using KV-cache from prefill) also succeeds.
        """
        basename = "test_mixtral_int4_cuda"
        output_dir, cache_dir = self.get_dirs(basename)

        config = self._build_mixtral_onnx(output_dir, cache_dir, ep="cuda")

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        sess = self._check_with_ort(onnx_path, cpu=False)
        onnx_input_names = {i.name for i in sess.get_inputs()}

        batch_size = 1
        seq_len = 5
        head_size = _HEAD_SIZE
        num_hidden_layers = config.num_hidden_layers
        precision = "fp16"

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, _NUM_KV_HEADS, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, _NUM_KV_HEADS, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, prefill_logits = run_session_or_io_binding(
                use_iobinding=False, precision=precision, provider="cuda", feed=prefill_feed, sess=sess, vocab_size=config.vocab_size
            )
            self.assertEqual(prefill_logits.shape, (batch_size, seq_len, config.vocab_size))

        with self.subTest(step="decode"):
            if prefill_results is None:
                raise unittest.SkipTest("prefill failed")
            next_token = int(np.argmax(prefill_results["logits"][0, -1, :]))

            decode_feed = {
                "input_ids": np.array([[next_token]], dtype=np.int64),
                "attention_mask": np.ones((batch_size, seq_len + 1), dtype=np.int64),
                "position_ids": np.array([[seq_len]], dtype=np.int64),
            }
            for i in range(num_hidden_layers):
                decode_feed[f"past_key_values.{i}.key"] = prefill_results[f"present.{i}.key"]
                decode_feed[f"past_key_values.{i}.value"] = prefill_results[f"present.{i}.value"]
            decode_feed = {k: v for k, v in decode_feed.items() if k in onnx_input_names}

            _decode_results, decode_logits = run_session_or_io_binding(
                use_iobinding=False,
                precision=precision,
                provider="cuda",
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )
            self.assertEqual(decode_logits.shape, (batch_size, 1, config.vocab_size))

    @hide_stdout()
    @requires_cuda()
    @requires_genai()
    def test_mixtral_int4_cuda_genai_generate(self):
        """Generate tokens with ``onnxruntime-genai`` end to end on CUDA.

        Mixtral exports as a CUDA-only QMoE (INT4) model, so the genai
        greedy-generation loop is gated by ``@requires_cuda()`` (the QMoE
        kernel) and ``@requires_genai()`` (the runtime).  The synthetic
        random-weight checkpoint and a tiny word-level tokenizer are written
        to disk so ``create_model`` produces ``model.onnx`` together with the
        ``genai_config.json`` / tokenizer files that ``og.Model`` needs.

        Because the weights are quantized to INT4 (lossy), the generated
        tokens are not compared against a PyTorch reference; the test only
        asserts that genai runs the QMoE graph and emits the requested number
        of new tokens without error.
        """
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        basename = "test_mixtral_int4_cuda_genai"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        config = _make_mixtral_config()

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        # MixtralForCausalLM forces CUDA + INT4 (QMoE) inside create_model.
        create_model(
            model_name=_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="int4",
            execution_provider="cuda",
            cache_dir=cache_dir,
        )

        self.assertExists(os.path.join(output_dir, "model.onnx"))
        self.assertExists(os.path.join(output_dir, "genai_config.json"))

        max_new_tokens = 5
        prompt_ids = torch.tensor([[config.bos_token_id, 3, 4, 5]], dtype=torch.int64)
        og_tokens = self.run_genai_generation(output_dir, prompt_ids, max_new_tokens)

        self.assertEqual(og_tokens[: prompt_ids.shape[1]], prompt_ids[0].tolist())
        self.assertLessEqual(len(og_tokens) - prompt_ids.shape[1], max_new_tokens)
        self.assertGreater(len(og_tokens), prompt_ids.shape[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
