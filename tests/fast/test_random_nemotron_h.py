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

    def common_fast_nemotron_h_moe_random_weights(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        # Two-layer model: attention (layer 0) + moe (layer 1).
        # Only the attention layer produces KV cache outputs.
        num_hidden_layers = 2
        attn_layer_ids = [0]  # layer 0 is the only attention layer
        layers_block_type = ["attention", "moe"]
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
            layers_block_type=layers_block_type,
            use_mamba_kernels=False,
            # MoE-specific parameters (use small sizes for fast tests)
            n_routed_experts=4,
            moe_intermediate_size=64,
            moe_shared_expert_intermediate_size=64,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            moe_latent_size=None,
            routed_scaling_factor=1.0,
        )

        basename = f"test_discrepancies_nemotron_h_moe_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
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
            # Only attention layers have KV cache entries
            for layer_idx in attn_layer_ids:
                prefill_feed[f"past_key_values.{layer_idx}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{layer_idx}.value"] = np.zeros(
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
            # Only attention layers have KV cache entries
            for layer_idx in attn_layer_ids:
                decode_feed[f"past_key_values.{layer_idx}.key"] = prefill_results[f"present.{layer_idx}.key"]
                decode_feed[f"past_key_values.{layer_idx}.value"] = prefill_results[f"present.{layer_idx}.value"]
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
                all_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to(provider)], dim=1)
                pt_decode = model(all_ids, use_cache=False)
                pt_decode_logits = pt_decode.logits[:, -1:, :].detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, onnx_decode_logits)
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 10, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision])

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_moe_fp32_cpu(self):
        self.common_fast_nemotron_h_moe_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_nemotron_h_moe_fp16_cpu(self):
        self.common_fast_nemotron_h_moe_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_moe_fp16_cuda(self):
        self.common_fast_nemotron_h_moe_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_nemotron_h_moe_bf16_cuda(self):
        self.common_fast_nemotron_h_moe_random_weights("bf16", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_nemotron_h_moe_fp32_cpu_genai_generate(self):
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        prefix = "test_nemotron_h_moe_fp32_cpu_genai_generate"
        num_hidden_layers = 2
        layers_block_type = ["attention", "moe"]
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
            layers_block_type=layers_block_type,
            use_mamba_kernels=False,
            # MoE-specific parameters (use small sizes for fast tests)
            n_routed_experts=4,
            moe_intermediate_size=64,
            moe_shared_expert_intermediate_size=64,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            moe_latent_size=None,
            routed_scaling_factor=1.0,
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

        # NemotronH with MoE layers does not support use_cache=True for mixed
        # attention+moe configs in transformers 5.x, so we skip the PyTorch
        # reference comparison and only verify that genai generation completes
        # without errors.
        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id, prompt_ids=prompt_ids)

    # ------------------------------------------------------------------ #
    # Tests: NemotronH Mamba blocks                                       #
    # The mamba layers use com.microsoft:CausalConvWithState which is NOT #
    # part of standard onnxruntime.  Build-only tests verify that the     #
    # ONNX model is produced and contains the expected custom ops.        #
    # ------------------------------------------------------------------ #

    def _make_nemotronh_mamba_config(self, layers_block_type=None):
        """Return a small NemotronHConfig with mamba layers for fast tests."""
        from transformers.models.nemotron_h import NemotronHConfig

        if layers_block_type is None:
            layers_block_type = ["mamba"]
        num_hidden_layers = len(layers_block_type)
        return NemotronHConfig(
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
            layers_block_type=layers_block_type,
            use_mamba_kernels=False,
            # Mamba-specific parameters (small for fast tests)
            mamba_num_heads=4,
            mamba_head_dim=8,
            ssm_state_size=8,
            conv_kernel=4,
            n_groups=1,
        )

    def _build_mamba_model(self, config, precision, provider, prefix):
        """Build a mamba ONNX model and return (model_dir, output_dir)."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        model_dir = self.get_model_dir(prefix, clean=False)
        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )
        return model, model_dir, output_dir

    def common_nemotron_h_mamba_build(self, precision, provider, layers_block_type=None):
        """Verify that create_model builds a mamba model and emits CausalConvWithState."""
        import onnx

        config = self._make_nemotronh_mamba_config(layers_block_type)
        prefix = f"test_nemotron_h_mamba_build_{precision}_{provider}_{'_'.join(config.layers_block_type)}"
        _, _, output_dir = self._build_mamba_model(config, precision, provider, prefix)

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        onnx_model = onnx.load(onnx_path)
        self.assertIsNotNone(onnx_model)
        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)

    @hide_stdout()
    def test_nemotron_h_mamba_fp32_cpu_build(self):
        """Build a single-layer mamba model (fp32/CPU) and check for CausalConvWithState."""
        self.common_nemotron_h_mamba_build("fp32", "cpu")

    @hide_stdout()
    def test_nemotron_h_mamba_fp16_cpu_build(self):
        """Build a single-layer mamba model (fp16/CPU) and check for CausalConvWithState."""
        self.common_nemotron_h_mamba_build("fp16", "cpu")

    @hide_stdout()
    def test_nemotron_h_mamba_hybrid_fp32_cpu_build(self):
        """Build a hybrid attention+mamba model (fp32/CPU) and check for CausalConvWithState."""
        self.common_nemotron_h_mamba_build("fp32", "cpu", layers_block_type=["attention", "mamba"])

    @hide_stdout()
    @requires_genai()
    def test_nemotron_h_mamba_fp32_cpu_genai_generate(self):
        """Verify genai generation completes for a mamba-only NemotronH model (fp32/CPU)."""
        config = self._make_nemotronh_mamba_config(["mamba"])
        prefix = "test_nemotron_h_mamba_fp32_cpu_genai_generate"
        _, _, output_dir = self._build_mamba_model(config, "fp32", "cpu", prefix)

        import torch

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (1, 4))
        # NemotronH mamba layers use stateful conv/SSM states that differ from
        # the standard KV cache, so we skip the PyTorch reference comparison and
        # only verify that genai generation completes without errors.
        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id, prompt_ids=prompt_ids)

    @hide_stdout()
    @requires_genai()
    def test_nemotron_h_mamba_hybrid_fp32_cpu_genai_generate(self):
        """Verify genai generation completes for a hybrid attention+mamba NemotronH model (fp32/CPU)."""
        config = self._make_nemotronh_mamba_config(["attention", "mamba"])
        prefix = "test_nemotron_h_mamba_hybrid_fp32_cpu_genai_generate"
        _, _, output_dir = self._build_mamba_model(config, "fp32", "cpu", prefix)

        import torch

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (1, 4))
        # Skip PyTorch reference comparison for the same reason as the mamba-only test.
        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id, prompt_ids=prompt_ids)

    def _make_nemotronh_full_hybrid_config(self):
        """Return a small NemotronHConfig with attention, mamba, and moe layers for fast tests."""
        from transformers.models.nemotron_h import NemotronHConfig

        return NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=64,
            intermediate_size=512,
            max_position_embeddings=2048,
            model_type="nemotron_h",
            num_attention_heads=4,
            num_hidden_layers=3,
            num_key_value_heads=2,
            layer_norm_epsilon=1e-05,
            vocab_size=32000,
            layers_block_type=["attention", "mamba", "moe"],
            use_mamba_kernels=False,
            # Mamba-specific parameters (small for fast tests)
            mamba_num_heads=4,
            mamba_head_dim=8,
            ssm_state_size=8,
            conv_kernel=4,
            n_groups=1,
            # MoE-specific parameters (small for fast tests)
            n_routed_experts=4,
            moe_intermediate_size=64,
            moe_shared_expert_intermediate_size=64,
            num_experts_per_tok=2,
            norm_topk_prob=True,
            n_group=1,
            topk_group=1,
            moe_latent_size=None,
            routed_scaling_factor=1.0,
        )

    @hide_stdout()
    def test_nemotron_h_full_hybrid_fp32_cpu_build(self):
        """Build a full hybrid attention+mamba+moe model (fp32/CPU) and check for CausalConvWithState."""
        import onnx
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        config = self._make_nemotronh_full_hybrid_config()
        prefix = "test_nemotron_h_full_hybrid_fp32_cpu_build"
        model_dir = self.get_model_dir(prefix, clean=False)
        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        onnx_model = onnx.load(onnx_path)
        self.assertIsNotNone(onnx_model)
        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertIn("CausalConvWithState", op_types)

    @hide_stdout()
    @requires_genai()
    def test_nemotron_h_full_hybrid_fp32_cpu_genai_generate(self):
        """Verify genai generation completes for a full hybrid attention+mamba+moe NemotronH model (fp32/CPU)."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        config = self._make_nemotronh_full_hybrid_config()
        prefix = "test_nemotron_h_full_hybrid_fp32_cpu_genai_generate"
        model_dir = self.get_model_dir(prefix, clean=False)
        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)
        self.make_word_level_tokenizer().save_pretrained(model_dir)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=config.num_hidden_layers,
        )

        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (1, 4))
        # Skip PyTorch reference comparison: mamba states and mixed block types are
        # not supported by standard ORT/transformers reference generation.
        self.run_genai_generation_test(output_dir, None, config.vocab_size, config.eos_token_id, prompt_ids=prompt_ids)

    # ------------------------------------------------------------------ #
    # Tests: NemotronH full_attention blocks                              #
    # "full_attention" is an alias for standard NoPE multi-head attention #
    # (identical to "attention"). The transformers library does not yet   #
    # expose "full_attention" in MIXER_TYPES, so these tests patch it in  #
    # temporarily to validate the ONNX builder support.                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _full_attention_patches():
        """Context manager that enables "full_attention" support in transformers.

        Two patches are applied simultaneously:

        1. ``MIXER_TYPES`` gains a ``"full_attention"`` entry that maps to the
           standard ``NemotronHAttention`` class.
        2. The ``validate_layer_type`` class-validator on ``NemotronHConfig`` is
           replaced with a relaxed version that also accepts ``"full_attention"``.

        Both patches are restored automatically when the context exits.
        """
        import contextlib

        import transformers.models.nemotron_h.modeling_nemotron_h as modeling
        from transformers.models.nemotron_h import NemotronHConfig

        _VALID_WITH_FULL = {"mamba", "attention", "moe", "mlp", "full_attention"}

        def _relaxed_validator(self):
            if not isinstance(self.layer_types, list):
                raise ValueError(f"`layers_block_type` must be a list. Got: {type(self.layer_types)}")
            invalid = set(self.layer_types) - _VALID_WITH_FULL
            if invalid:
                raise ValueError(f"`layers_block_type` contains invalid types: {invalid}")

        @contextlib.contextmanager
        def _apply():
            # Patch 1: add "full_attention" to the MIXER_TYPES registry so that
            # NemotronHBlock.__init__ can instantiate the mixer.
            original_mixer = dict(modeling.MIXER_TYPES)
            modeling.MIXER_TYPES["full_attention"] = modeling.NemotronHAttention
            # Patch 2: replace the strict layer-type validator with a relaxed one.
            validators = NemotronHConfig.__class_validators__
            original_validators = list(validators)
            for i, v in enumerate(validators):
                if v.__name__ == "validate_layer_type":
                    validators[i] = _relaxed_validator
            try:
                yield
            finally:
                modeling.MIXER_TYPES.clear()
                modeling.MIXER_TYPES.update(original_mixer)
                validators[:] = original_validators

        return _apply()

    def _make_nemotronh_full_attention_config(self, layers_block_type=None):
        """Return a small NemotronHConfig with full_attention layers for fast tests."""
        from transformers.models.nemotron_h import NemotronHConfig

        if layers_block_type is None:
            layers_block_type = ["full_attention"]
        num_hidden_layers = len(layers_block_type)
        with self._full_attention_patches():
            return NemotronHConfig(
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
                layers_block_type=layers_block_type,
                use_mamba_kernels=False,
            )

    def _build_full_attention_model(self, config, precision, provider, prefix):
        """Build a full_attention ONNX model (patching MIXER_TYPES) and return dirs."""
        import torch
        from transformers import AutoModelForCausalLM

        from modelbuilder.builder import create_model

        model_dir = self.get_model_dir(prefix, clean=False)
        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        # "full_attention" is not yet in transformers MIXER_TYPES; apply patches
        # while creating the model, saving it, and running the ONNX export.
        with self._full_attention_patches():
            torch.manual_seed(42)
            model = AutoModelForCausalLM.from_config(config)
            model.eval()
            model.save_pretrained(model_dir)
            self.make_word_level_tokenizer().save_pretrained(model_dir)

            create_model(
                model_name=MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision=precision,
                execution_provider=provider,
                cache_dir=cache_dir,
                num_hidden_layers=config.num_hidden_layers,
            )
        return model, model_dir, output_dir

    @hide_stdout()
    def test_nemotron_h_full_attention_fp32_cpu_build(self):
        """Build a NemotronH model with full_attention layers (fp32/CPU).

        Verifies that the ONNX builder correctly handles the ``full_attention``
        block type, treating it identically to the standard ``attention`` block.
        """
        import onnx

        config = self._make_nemotronh_full_attention_config(["full_attention"])
        prefix = "test_nemotron_h_full_attention_fp32_cpu_build"
        _, _, output_dir = self._build_full_attention_model(config, "fp32", "cpu", prefix)

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        onnx_model = onnx.load(onnx_path)
        self.assertIsNotNone(onnx_model)
        # The "full_attention" layer must produce an attention op. Because the
        # config has num_key_value_heads=2 < num_attention_heads=4 (GQA), the
        # builder emits GroupQueryAttention rather than MultiHeadAttention.
        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertTrue(op_types & {"MultiHeadAttention", "GroupQueryAttention"}, f"Expected an attention op in {op_types}")

    @hide_stdout()
    def test_nemotron_h_full_attention_fp32_cpu_inference(self):
        """Build and run a NemotronH full_attention model (fp32/CPU).

        Verifies that the exported ONNX model for ``full_attention`` layers
        produces output numerically identical to the equivalent ``attention``
        layer ONNX model (same weights, same computation).
        """
        import torch
        from transformers import AutoModelForCausalLM
        from transformers.models.nemotron_h import NemotronHConfig

        from modelbuilder.builder import create_model

        num_hidden_layers = 1
        head_size = 64
        batch_size = 1
        seq_len = 5

        # --- Build the "full_attention" ONNX model ---
        config_fa = self._make_nemotronh_full_attention_config(["full_attention"] * num_hidden_layers)
        prefix_fa = "test_nemotron_h_full_attention_fp32_cpu_inference_fa"
        _, _, output_dir_fa = self._build_full_attention_model(config_fa, "fp32", "cpu", prefix_fa)

        # --- Build an equivalent "attention" ONNX model with the same weights ---
        # Use "attention" block type so no patching is needed for model creation.
        config_attn = NemotronHConfig(
            architectures=["NemotronHForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=256,
            head_dim=head_size,
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
        prefix_attn = "test_nemotron_h_full_attention_fp32_cpu_inference_attn"
        model_dir_attn = self.get_model_dir(prefix_attn, clean=False)
        output_dir_attn, cache_dir_attn = self.get_dirs(prefix_attn, clean=False)

        # Load the full_attention model weights and re-save under the "attention"
        # config so that both ONNX models share identical weights.
        with self._full_attention_patches():
            model_fa_dir = self.get_model_dir(prefix_fa, clean=False)
            model_fa = AutoModelForCausalLM.from_pretrained(model_fa_dir)

        model_attn = AutoModelForCausalLM.from_config(config_attn)
        # Copy weights (both configs are identical except block_type).
        model_attn.load_state_dict(model_fa.state_dict())
        model_attn.eval()
        model_attn.save_pretrained(model_dir_attn)
        self.make_word_level_tokenizer().save_pretrained(model_dir_attn)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir_attn,
            output_dir=output_dir_attn,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir_attn,
            num_hidden_layers=num_hidden_layers,
        )

        # --- Run both ONNX models with the same input and compare outputs ---
        onnx_path_fa = os.path.join(output_dir_fa, "model.onnx")
        onnx_path_attn = os.path.join(output_dir_attn, "model.onnx")
        self.assertExists(onnx_path_fa)
        self.assertExists(onnx_path_attn)

        sess_fa = self._check_with_ort(onnx_path_fa, cpu=True)
        sess_attn = self._check_with_ort(onnx_path_attn, cpu=True)

        torch.manual_seed(0)
        input_ids = torch.randint(0, config_fa.vocab_size, (batch_size, seq_len))

        def _make_feed(sess, attn_layer_ids):
            feed = {
                "input_ids": input_ids.numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            for i in attn_layer_ids:
                feed[f"past_key_values.{i}.key"] = np.zeros((batch_size, config_fa.num_key_value_heads, 0, head_size), dtype=np.float32)
                feed[f"past_key_values.{i}.value"] = np.zeros((batch_size, config_fa.num_key_value_heads, 0, head_size), dtype=np.float32)
            input_names = {inp.name for inp in sess.get_inputs()}
            return {k: v for k, v in feed.items() if k in input_names}

        attn_layer_ids = list(range(num_hidden_layers))
        results_fa = sess_fa.run(None, _make_feed(sess_fa, attn_layer_ids))
        results_attn = sess_attn.run(None, _make_feed(sess_attn, attn_layer_ids))

        # Logits from both models must match (same weights, same computation).
        np.testing.assert_allclose(results_fa[0], results_attn[0], atol=1e-5, rtol=1e-5)

    @hide_stdout()
    def test_nemotron_h_full_attention_hybrid_fp32_cpu_build(self):
        """Build a hybrid full_attention+mamba NemotronH model (fp32/CPU).

        Verifies that the ONNX builder correctly handles a mixed model containing
        both ``full_attention`` and ``mamba`` block types.
        """
        import onnx

        with self._full_attention_patches():
            config = self._make_nemotronh_mamba_config(["full_attention", "mamba"])
        prefix = "test_nemotron_h_full_attention_hybrid_fp32_cpu_build"
        _, _, output_dir = self._build_full_attention_model(config, "fp32", "cpu", prefix)

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        onnx_model = onnx.load(onnx_path)
        self.assertIsNotNone(onnx_model)
        op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertTrue(op_types & {"MultiHeadAttention", "GroupQueryAttention"}, f"Expected an attention op in {op_types}")
        self.assertIn("CausalConvWithState", op_types)


if __name__ == "__main__":
    unittest.main(verbosity=2)
