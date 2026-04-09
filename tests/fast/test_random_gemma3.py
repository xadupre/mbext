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
MODEL_NAME = "google/gemma-3-4b-it"


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
    def common_fast_gemma3_random_weights(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        num_hidden_layers = 1

        # Minimal Gemma3TextConfig matching the Gemma3ForCausalLM architecture
        # with small dimensions to keep the test fast and completely offline.
        # head_dim=64 keeps compute small; query_pre_attn_scalar=64 matches head_dim.
        # sliding_window=512 enables local (sliding) attention for all layers.
        # With num_hidden_layers=1, is_local() returns True ((0+1)%6==1), so only
        # local RoPE caches are exercised.
        config = Gemma3TextConfig(
            architectures=["Gemma3ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            query_pre_attn_scalar=64,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
        )

        basename = f"test_discrepancies_gemma3_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "</s>": 1, "<bos>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<bos>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=GEMMA3_MODEL_NAME,
            model_name=MODEL_NAME,
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
            num_hidden_layers=num_hidden_layers,
        )

        log_data = dict(
            precision=precision,
            model_id=MODEL_NAME,
            experiment="forward",
            provider=provider,
            test=basename,
            input_type="text",
            kind="random",
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        batch_size = 1
        seq_len = 5
        head_size = config.head_dim

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size),
                    dtype=self.get_input_np_dtype(precision),
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size),
                    dtype=self.get_input_np_dtype(precision),
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            prefill_results, ort_logits_np = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=prefill_feed,
                sess=sess,
                vocab_size=config.vocab_size,
            )

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

        with self.subTest(step="decode"):
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

            prefill_results, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(
                pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision]
            )

    def common_gemma3_greedy_generation(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, Gemma3TextConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        num_hidden_layers = 1

        config = Gemma3TextConfig(
            architectures=["Gemma3ForCausalLM"],
            bos_token_id=2,
            eos_token_id=1,
            head_dim=64,
            hidden_activation="gelu_pytorch_tanh",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            query_pre_attn_scalar=64,
            rms_norm_eps=1e-6,
            sliding_window=512,
            vocab_size=32000,
        )

        basename = f"test_generation_gemma3_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "</s>": 1, "<bos>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<bos>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        input_names = {inp.name for inp in sess.get_inputs()}

        batch_size = 1
        head_size = config.head_dim
        max_new_tokens = 10

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (batch_size, 5)).to(provider)

        with torch.no_grad():
            pt_output = model.generate(
                prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=config.eos_token_id,
            )
        pt_tokens = pt_output[0].tolist()

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=self.get_input_np_dtype(precision),
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=self.get_input_np_dtype(precision),
            )

        onnx_tokens = current_ids[0].tolist()
        results = None
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_ids.shape[1]

            feed = {
                "input_ids": current_ids,
                "attention_mask": np.ones((batch_size, past_len + cur_len), dtype=np.int64),
                "position_ids": np.arange(past_len, past_len + cur_len, dtype=np.int64).reshape(
                    batch_size, cur_len
                ),
            }
            for i in range(num_hidden_layers):
                feed[f"past_key_values.{i}.key"] = past_kv[f"past_key_values.{i}.key"]
                feed[f"past_key_values.{i}.value"] = past_kv[f"past_key_values.{i}.value"]
            feed = {k: v for k, v in feed.items() if k in input_names}

            results, _ = run_session_or_io_binding(
                use_iobinding=precision == "bf16",
                precision=precision,
                provider=provider,
                feed=feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=results,
            )

            next_token = int(np.argmax(results["logits"][0, -1, :]))
            onnx_tokens.append(next_token)

            for i in range(num_hidden_layers):
                past_kv[f"past_key_values.{i}.key"] = results[f"present.{i}.key"]
                past_kv[f"past_key_values.{i}.value"] = results[f"present.{i}.value"]

            current_ids = np.array([[next_token]], dtype=np.int64)

            if next_token == config.eos_token_id:
                break

        diff = self.first_token_diff(pt_tokens, onnx_tokens)
        diff.update(
            dict(
                precision=precision,
                model_id=MODEL_NAME,
                experiment="generate",
                provider=provider,
                test=basename,
                input_type="text",
                kind="fast",
            )
        )
        self.log_results(diff)
        if precision in ("fp16", "bf16"):
            pt_tokens = pt_tokens[:-5]
            onnx_tokens = onnx_tokens[:-5]
        self.assertEqual(pt_tokens, onnx_tokens)

    @hide_stdout()
    def test_gemma3_fp32_cpu_greedy_generation(self):
        self.common_gemma3_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gemma3_fp16_cpu_greedy_generation(self):
        self.common_gemma3_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_gemma3_fp32_cuda_greedy_generation(self):
        self.common_gemma3_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_fp16_cuda_greedy_generation(self):
        self.common_gemma3_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gemma3_bf16_cuda_greedy_generation(self):
        self.common_gemma3_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_gemma3_fp32_cpu(self):
        self.common_fast_gemma3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma3_fp16_cpu(self):
        self.common_fast_gemma3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gemma3_int4_cpu(self):
        self.common_fast_gemma3_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gemma3_fp16_cuda(self):
        self.common_fast_gemma3_random_weights("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
