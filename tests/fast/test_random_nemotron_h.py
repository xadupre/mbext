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
    has_transformers,
    hide_stdout,
    requires_cuda,
    requires_genai,
    requires_transformers,
    run_session_or_io_binding,
)

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"

# Small config shared across mixed-block tests
_MAMBA_KWARGS = dict(mamba_num_heads=4, mamba_head_dim=16, ssm_state_size=8, n_groups=1, conv_kernel=4, use_mamba_kernels=False)
_MOE_KWARGS = dict(
    n_routed_experts=4,
    n_shared_experts=1,
    moe_intermediate_size=64,
    moe_shared_expert_intermediate_size=64,
    num_experts_per_tok=2,
    n_group=1,
    topk_group=1,
)


def _make_nemotron_h_config(layers_block_type, **extra):
    from transformers.models.nemotron_h import NemotronHConfig

    return NemotronHConfig(
        architectures=["NemotronHForCausalLM"],
        bos_token_id=1,
        eos_token_id=2,
        hidden_size=64,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=256,
        model_type="nemotron_h",
        num_attention_heads=4,
        num_hidden_layers=len(layers_block_type),
        num_key_value_heads=2,
        layer_norm_epsilon=1e-5,
        vocab_size=256,
        layers_block_type=layers_block_type,
        **_MAMBA_KWARGS,
        **_MOE_KWARGS,
        **extra,
    )


@requires_transformers("5")
class TestNemotronH(ExtTestCase):
    def common_fast_nemotron_h_random_weights(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        basename = f"test_discrepancies_nemotron_h_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
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

        log_data = dict(
            precision=precision, model_id=MODEL_NAME, experiment="forward", provider=provider, test=basename, input_type="text", kind="fast"
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        batch_size = 1
        seq_len = 5
        head_size = config.head_dim

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
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
                # use_cache=False avoids has_previous_state error when NemotronH
                # creates a DynamicCache internally (attention-only config).
                pt_prefill = model(input_ids, use_cache=False)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 3e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

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
                # NemotronH's _update_mamba_mask raises ValueError when
                # past_key_values is a DynamicCache (attention-only).
                # Run from scratch without cache to get the same logits.
                all_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to(provider)], dim=1)
                pt_decode = model(all_ids, use_cache=False)
                pt_decode_logits = pt_decode.logits[:, -1:, :].detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 10, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_nemotron_h_greedy_generation(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        basename = f"test_generation_nemotron_h_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
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
            # use_cache=False avoids the has_previous_state error: NemotronH's
            # _update_mamba_mask raises ValueError when past_key_values is a
            # DynamicCache containing only attention layers (no Mamba layers).
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id, use_cache=False
            )
        pt_tokens = pt_output[0].tolist()

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )

        onnx_tokens = current_ids[0].tolist()
        results = None
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_ids.shape[1]

            feed = {
                "input_ids": current_ids,
                "attention_mask": np.ones((batch_size, past_len + cur_len), dtype=np.int64),
                "position_ids": np.arange(past_len, past_len + cur_len, dtype=np.int64).reshape(batch_size, cur_len),
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

    # ------------------------------------------------------------------
    # Mixed-block tests: mamba and moe block types
    # ------------------------------------------------------------------

    def _make_mamba_feed(self, sess, config, batch_size, seq_len, precision):
        """Build an ORT feed dict for a mamba-only model (single-token decode)."""
        mamba_layers = [i for i, t in enumerate(config.layers_block_type) if t == "mamba"]
        mamba_num_heads = config.mamba_num_heads
        mamba_head_dim = config.mamba_head_dim
        ssm_state_size = config.ssm_state_size
        conv_dim = mamba_num_heads * mamba_head_dim + 2 * config.n_groups * ssm_state_size
        conv_kernel = config.conv_kernel

        input_names = {inp.name for inp in sess.get_inputs()}
        feed = {
            "input_ids": np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        for i in mamba_layers:
            feed[f"past_mamba.{i}.conv_state"] = np.zeros((batch_size, conv_dim, conv_kernel), dtype=np.float32)
            feed[f"past_mamba.{i}.recurrent_state"] = np.zeros(
                (batch_size, mamba_num_heads, mamba_head_dim, ssm_state_size), dtype=np.float32
            )
        return {k: v for k, v in feed.items() if k in input_names}

    def _make_moe_feed(self, sess, config, batch_size, seq_len, precision):
        """Build an ORT feed dict for a moe-only model (no state needed)."""
        attn_layers = [i for i, t in enumerate(config.layers_block_type) if t == "attention"]
        dtype = self.get_input_np_dtype(precision)
        input_names = {inp.name for inp in sess.get_inputs()}
        head_size = config.head_dim
        feed = {
            "input_ids": np.random.randint(0, config.vocab_size, (batch_size, seq_len), dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        for i in attn_layers:
            feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, config.num_key_value_heads, 0, head_size), dtype=dtype)
            feed[f"past_key_values.{i}.value"] = np.zeros((batch_size, config.num_key_value_heads, 0, head_size), dtype=dtype)
        return {k: v for k, v in feed.items() if k in input_names}

    def common_nemotron_h_mamba_block_fp32_cpu(self):
        """Export a mamba-only NemotronH and verify single-token inference."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        layers_block_type = ["mamba"]
        config = _make_nemotron_h_config(layers_block_type)

        basename = "test_nemotron_h_mamba_fp32_cpu"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=len(layers_block_type),
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=True)

        # Run single-token inference – the mamba ONNX model processes 1 token at a time
        batch_size, seq_len = 1, 1
        torch.manual_seed(42)
        feed = self._make_mamba_feed(sess, config, batch_size, seq_len, "fp32")

        ort_results = sess.run(None, feed)
        output_names = [o.name for o in sess.get_outputs()]
        result_dict = dict(zip(output_names, ort_results))

        # logits shape: (batch, 1, vocab_size)
        self.assertEqual(result_dict["logits"].shape, (batch_size, seq_len, config.vocab_size))

        # present_mamba states must have correct shapes
        mamba_layers = [i for i, t in enumerate(config.layers_block_type) if t == "mamba"]
        conv_dim = config.mamba_num_heads * config.mamba_head_dim + 2 * config.n_groups * config.ssm_state_size
        for i in mamba_layers:
            self.assertIn(f"present_mamba.{i}.conv_state", result_dict)
            self.assertIn(f"present_mamba.{i}.recurrent_state", result_dict)
            self.assertEqual(result_dict[f"present_mamba.{i}.conv_state"].shape, (batch_size, conv_dim, config.conv_kernel))
            self.assertEqual(
                result_dict[f"present_mamba.{i}.recurrent_state"].shape,
                (batch_size, config.mamba_num_heads, config.mamba_head_dim, config.ssm_state_size),
            )

        # Compare against PyTorch for a single token with use_cache=False
        input_ids = torch.tensor(feed["input_ids"])
        with torch.no_grad():
            pt_out = model(input_ids, use_cache=False)
        np.testing.assert_allclose(pt_out.logits.numpy(), result_dict["logits"], atol=1e-4, rtol=1e-3)

    def common_nemotron_h_moe_block_fp32_cpu(self):
        """Export a moe-only NemotronH and verify single-token inference."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        layers_block_type = ["moe"]
        config = _make_nemotron_h_config(layers_block_type)

        basename = "test_nemotron_h_moe_fp32_cpu"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=len(layers_block_type),
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=True)

        batch_size, seq_len = 1, 3
        torch.manual_seed(42)
        feed = self._make_moe_feed(sess, config, batch_size, seq_len, "fp32")

        ort_results = sess.run(None, feed)
        output_names = [o.name for o in sess.get_outputs()]
        result_dict = dict(zip(output_names, ort_results))

        # logits shape: (batch, seq, vocab_size)
        self.assertEqual(result_dict["logits"].shape, (batch_size, seq_len, config.vocab_size))

        # Compare against PyTorch
        input_ids = torch.tensor(feed["input_ids"])
        with torch.no_grad():
            pt_out = model(input_ids, use_cache=False)
        np.testing.assert_allclose(pt_out.logits.numpy(), result_dict["logits"], atol=1e-4, rtol=1e-3)

    def common_nemotron_h_mixed_block_fp32_cpu(self):
        """Export a mixed attention+mamba+moe NemotronH and verify ONNX is created."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        layers_block_type = ["attention", "mamba", "moe"]
        config = _make_nemotron_h_config(layers_block_type)

        basename = "test_nemotron_h_mixed_fp32_cpu"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=len(layers_block_type),
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=True)

        input_names = {inp.name for inp in sess.get_inputs()}
        output_names_list = [o.name for o in sess.get_outputs()]

        # Check expected inputs are present: KV for attention, mamba states for mamba
        self.assertIn("past_key_values.0.key", input_names)
        self.assertIn("past_key_values.0.value", input_names)
        self.assertIn("past_mamba.1.conv_state", input_names)
        self.assertIn("past_mamba.1.recurrent_state", input_names)

        # Check expected outputs are present
        self.assertIn("present.0.key", output_names_list)
        self.assertIn("present.0.value", output_names_list)
        self.assertIn("present_mamba.1.conv_state", output_names_list)
        self.assertIn("present_mamba.1.recurrent_state", output_names_list)

        # Run a single-token forward pass and verify output shape
        batch_size, seq_len = 1, 1
        head_size = config.head_dim
        mamba_num_heads = config.mamba_num_heads
        mamba_head_dim = config.mamba_head_dim
        ssm_state_size = config.ssm_state_size
        conv_dim = mamba_num_heads * mamba_head_dim + 2 * config.n_groups * ssm_state_size
        conv_kernel = config.conv_kernel

        feed = {
            "input_ids": np.array([[5]], dtype=np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.zeros((batch_size, seq_len), dtype=np.int64),
            "past_key_values.0.key": np.zeros((batch_size, config.num_key_value_heads, 0, head_size), dtype=np.float32),
            "past_key_values.0.value": np.zeros((batch_size, config.num_key_value_heads, 0, head_size), dtype=np.float32),
            "past_mamba.1.conv_state": np.zeros((batch_size, conv_dim, conv_kernel), dtype=np.float32),
            "past_mamba.1.recurrent_state": np.zeros((batch_size, mamba_num_heads, mamba_head_dim, ssm_state_size), dtype=np.float32),
        }
        feed = {k: v for k, v in feed.items() if k in input_names}

        ort_results = sess.run(None, feed)
        result_dict = dict(zip(output_names_list, ort_results))
        self.assertEqual(result_dict["logits"].shape, (batch_size, seq_len, config.vocab_size))

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_fp32_cpu(self):
        self.common_fast_nemotron_h_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_fp16_cpu(self):
        self.common_fast_nemotron_h_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_int4_cpu(self):
        self.common_fast_nemotron_h_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_fp16_cuda(self):
        self.common_fast_nemotron_h_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_bf16_cuda(self):
        self.common_fast_nemotron_h_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_nemotron_h_fp32_cpu_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_nemotron_h_fp16_cpu_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_h_fp16_cuda_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_h_bf16_cuda_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_nemotron_h_mamba_block_fp32_cpu(self):
        self.common_nemotron_h_mamba_block_fp32_cpu()

    @hide_stdout()
    def test_nemotron_h_moe_block_fp32_cpu(self):
        self.common_nemotron_h_moe_block_fp32_cpu()

    @hide_stdout()
    def test_nemotron_h_mixed_block_fp32_cpu(self):
        self.common_nemotron_h_mixed_block_fp32_cpu()

    @hide_stdout()
    @requires_genai()
    def test_nemotron_h_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        prefix = "test_nemotron_h_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (1, 4))

        pt_tokens = None
        if not has_transformers("5.5"):
            # The code is broken in transformers 5.5 for this model.
            # ValueError: `has_previous_state` can only be called on LinearAttention layers
            with torch.no_grad():
                pt_output = model.generate(prompt_ids, max_new_tokens=5, do_sample=False, pad_token_id=config.eos_token_id)
            pt_tokens = pt_output[0].tolist()

        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id, pt_tokens=pt_tokens, prompt_ids=prompt_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)

    def common_fast_nemotron_h_random_weights(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        basename = f"test_discrepancies_nemotron_h_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
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

        log_data = dict(
            precision=precision, model_id=MODEL_NAME, experiment="forward", provider=provider, test=basename, input_type="text", kind="fast"
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        batch_size = 1
        seq_len = 5
        head_size = config.head_dim

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_results = None
        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in range(num_hidden_layers):
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
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
                # use_cache=False avoids has_previous_state error when NemotronH
                # creates a DynamicCache internally (attention-only config).
                pt_prefill = model(input_ids, use_cache=False)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 3e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

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
                # NemotronH's _update_mamba_mask raises ValueError when
                # past_key_values is a DynamicCache (attention-only).
                # Run from scratch without cache to get the same logits.
                all_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to(provider)], dim=1)
                pt_decode = model(all_ids, use_cache=False)
                pt_decode_logits = pt_decode.logits[:, -1:, :].detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 10, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    def common_nemotron_h_greedy_generation(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        basename = f"test_generation_nemotron_h_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
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
            # use_cache=False avoids the has_previous_state error: NemotronH's
            # _update_mamba_mask raises ValueError when past_key_values is a
            # DynamicCache containing only attention layers (no Mamba layers).
            pt_output = model.generate(
                prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id, use_cache=False
            )
        pt_tokens = pt_output[0].tolist()

        current_ids = prompt_ids.detach().cpu().numpy().astype(np.int64)

        past_kv = {}
        for i in range(num_hidden_layers):
            past_kv[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )
            past_kv[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
            )

        onnx_tokens = current_ids[0].tolist()
        results = None
        for _ in range(max_new_tokens):
            past_len = past_kv["past_key_values.0.key"].shape[2]
            cur_len = current_ids.shape[1]

            feed = {
                "input_ids": current_ids,
                "attention_mask": np.ones((batch_size, past_len + cur_len), dtype=np.int64),
                "position_ids": np.arange(past_len, past_len + cur_len, dtype=np.int64).reshape(batch_size, cur_len),
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
    def test_fast_discrepancy_nemotron_h_fp32_cpu(self):
        self.common_fast_nemotron_h_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_fp16_cpu(self):
        self.common_fast_nemotron_h_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_int4_cpu(self):
        self.common_fast_nemotron_h_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_fp16_cuda(self):
        self.common_fast_nemotron_h_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_bf16_cuda(self):
        self.common_fast_nemotron_h_random_weights("bf16", "cuda")

    @hide_stdout()
    def test_nemotron_h_fp32_cpu_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_nemotron_h_fp16_cpu_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_h_fp16_cuda_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_nemotron_h_bf16_cuda_greedy_generation(self):
        self.common_nemotron_h_greedy_generation("bf16", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_nemotron_h_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        prefix = "test_nemotron_h_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention"] * num_hidden_layers,
            use_mamba_kernels=False,
        )

        model_dir = self.get_model_dir(prefix, clean=False)
        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        tokenizer = self.make_word_level_tokenizer()
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
        )

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (1, 4))

        pt_tokens = None
        if not has_transformers("5.5"):
            # The code is broken in transformers 5.5 for this model.
            # ValueError: `has_previous_state` can only be called on LinearAttention layers
            with torch.no_grad():
                pt_output = model.generate(prompt_ids, max_new_tokens=5, do_sample=False, pad_token_id=config.eos_token_id)
            pt_tokens = pt_output[0].tolist()

        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id, pt_tokens=pt_tokens, prompt_ids=prompt_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
