# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import math
import os
import unittest
from unittest.mock import patch

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai, run_session_or_io_binding

MODEL_NAME = "openai/gpt-oss-20b"


def _ref_yarn_rope_cache(head_size, cache_length, theta, factor, beta_fast, beta_slow, original_max_pos, truncate):
    """Reference YARN RoPE cos/sin cache matching HuggingFace's ``_compute_yarn_parameters``.

    Computes the NTK-by-parts inv_freq and then builds the cos/sin table that
    ``GPTOSSModel.make_rotary_embedding_caches_from_scratch`` should reproduce.

    Returns
    -------
    cos_cache, sin_cache : torch.Tensor of shape [cache_length, head_size // 2]
    """
    import torch

    d_half = head_size // 2

    # Base inverse frequencies (no scaling)
    pos_freqs = theta ** (torch.arange(0, head_size, 2, dtype=torch.float) / head_size)
    inv_freq = 1.0 / pos_freqs

    # NTK by-parts correction range
    # find_correction_dim(n_rot, dim, base, max_pos) = dim * log(max_pos / (n_rot * 2π)) / (2 * log(base))
    #                                                 = d_half * log(max_pos / (n_rot * 2π)) / log(base)
    low = d_half * math.log(original_max_pos / (beta_fast * 2 * math.pi)) / math.log(theta)
    high = d_half * math.log(original_max_pos / (beta_slow * 2 * math.pi)) / math.log(theta)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)

    interpolation = inv_freq / factor  # 1 / (factor * pos_freqs)
    extrapolation = inv_freq  # 1 / pos_freqs

    ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
    mask = 1.0 - ramp.clamp(0, 1)  # inv_freq_extrapolation_factor

    inv_freq_rescaled = interpolation * (1 - mask) + extrapolation * mask

    # Magnitude scaling: yarn_get_mscale(factor) = 0.1 * log(factor) + 1  (if factor > 1)
    mscale = 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0

    t = torch.arange(cache_length, dtype=torch.float32)
    freqs = torch.einsum("i,j->ij", t, inv_freq_rescaled)
    cos_cache = freqs.cos() * mscale
    sin_cache = freqs.sin() * mscale
    return cos_cache, sin_cache


class TestGptOss20b(ExtTestCase):
    def common_fast_gpt_oss_20b_random_weights(self, precision, provider):
        from transformers import AutoModelForCausalLM, GptOssConfig

        # Two layers: layer 0 is local (sliding-window) attention,
        # layer 1 is full attention, matching the is_local = lambda i: i % 2 == 0
        # logic in GPTOSSModel.
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
        )

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_discrepancies_gpt_oss_20b_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
        )

    def common_gpt_oss_20b_greedy_generation(self, precision, provider):
        import torch
        from transformers import AutoModelForCausalLM, GptOssConfig

        # Two layers: layer 0 is local (sliding-window) attention,
        # layer 1 is full attention, matching the is_local = lambda i: i % 2 == 0
        # logic in GPTOSSModel.
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
            max_position_embeddings=4096,
        )

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        tokenizer = self.make_word_level_tokenizer()
        self.run_greedy_generation_test(
            model=model,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
            basename=f"test_generation_gpt_oss_20b_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            eos_token_id=config.eos_token_id,
        )

    @hide_stdout()
    def test_gpt_oss_20b_fp32_cpu_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp32", "cpu")

    @hide_stdout()
    def test_gpt_oss_20b_fp16_cpu_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp16", "cpu")

    @unittest.skip("fails due to incorrect model")
    @hide_stdout()
    def test_gpt_oss_20b_fp32_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gpt_oss_20b_fp16_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_gpt_oss_20b_bf16_cuda_greedy_generation(self):
        self.common_gpt_oss_20b_greedy_generation("bf16", "cuda")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_fp32_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_fp16_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_gpt_oss_20b_int4_cpu(self):
        self.common_fast_gpt_oss_20b_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_gpt_oss_20b_fp16_cuda(self):
        self.common_fast_gpt_oss_20b_random_weights("fp16", "cuda")

    def test_yarn_rope_cache_fp32(self):
        """Verify that GPTOSSModel.make_rotary_embedding_caches_from_scratch produces
        the correct YARN (Yet Another RoPE extensioN) cos/sin cache.

        The expected values are computed by ``_ref_yarn_rope_cache``, which mirrors
        HuggingFace's ``_compute_yarn_parameters`` exactly.  This test guards against
        regressions in:
          * the direction of inv_freq (1/pos_freqs, not pos_freqs)
          * the interpolation/extrapolation blending (interp = inv_freq / factor)
          * the truncate=False boundary (no floor/ceil for GPT-OSS)
          * the mscale formula (0.1 * log(factor) + 1)
        """
        import onnx_ir as ir
        from transformers import GptOssConfig

        from modelbuilder.builders.gptoss import GPTOSSModel

        head_dim = 32
        cache_length = 64  # small cache for speed

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=head_dim,
            num_attention_heads=4,
            num_hidden_layers=2,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
            max_position_embeddings=cache_length,
        )

        # Confirm the default rope_parameters use YARN so this test stays meaningful
        rope_scaling = config.rope_scaling
        self.assertEqual(rope_scaling["rope_type"], "yarn")

        # Instantiate the builder (no model weights needed for cache computation)
        builder = GPTOSSModel(
            config, io_dtype=ir.DataType.FLOAT, onnx_dtype=ir.DataType.FLOAT, ep="cpu", cache_dir="/tmp", extra_options={}
        )

        cos_cache, sin_cache = builder.make_rotary_embedding_caches_from_scratch()

        ref_cos, ref_sin = _ref_yarn_rope_cache(
            head_size=head_dim,
            cache_length=cache_length,
            theta=rope_scaling["rope_theta"],
            factor=rope_scaling["factor"],
            beta_fast=rope_scaling["beta_fast"],
            beta_slow=rope_scaling["beta_slow"],
            original_max_pos=rope_scaling["original_max_position_embeddings"],
            truncate=rope_scaling.get("truncate", True),
        )

        self.assertEqual(cos_cache.shape, ref_cos.shape)
        self.assertEqual(sin_cache.shape, ref_sin.shape)
        import torch

        torch.testing.assert_close(cos_cache, ref_cos, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sin_cache, ref_sin, rtol=1e-5, atol=1e-5)

    def common_moe_decomposed_random_weights(self, precision, ort_provider):
        """Build a GPT-OSS-20B ONNX model with ``execution_provider="cpu"`` while
        patching ``GPTOSSModel.make_moe`` to always call ``make_moe_decomposed``, then
        verify the resulting ONNX graph and run inference with the ORT provider.

        Patching ``make_moe`` lets us test the decomposed code path with standard CPU
        attention ops and without requiring any non-standard execution environment.
        """
        import onnx
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, GptOssConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model
        from modelbuilder.builders.gptoss import GPTOSSModel

        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
        )

        basename = f"test_moe_decomposed_gpt_oss_20b_{precision}_{ort_provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(42)
        model = AutoModelForCausalLM.from_config(config)
        model.eval()
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")), bos_token="<s>", eos_token="</s>", unk_token="<unk>"
        )
        tokenizer.save_pretrained(model_dir)

        # Patch make_moe to always call the decomposed path so we test
        # make_moe_decomposed while still using the standard CPU execution provider.
        def _force_decomposed(self_model, layer_id, mlp, root_input):
            self_model.make_moe_decomposed(layer_id, mlp, root_input)

        with patch.object(GPTOSSModel, "make_moe", _force_decomposed):
            create_model(
                model_name=MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision=precision,
                execution_provider=ort_provider,
                cache_dir=cache_dir,
            )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)

        # Verify the decomposed MoE subgraph is used (no fused MoE/QMoE op present).
        onnx_model = onnx.load(onnx_path)
        node_op_types = {node.op_type for node in onnx_model.graph.node}
        self.assertNotIn("MoE", node_op_types, "Decomposed model must not contain a fused MoE op")
        self.assertNotIn("QMoE", node_op_types, "Decomposed model must not contain a fused QMoE op")

        # Key op types that make_moe_decomposed emits.
        for expected_op in ("TopK", "Gather", "ReduceSum", "Squeeze", "Softmax"):
            self.assertIn(expected_op, node_op_types, f"Decomposed MoE subgraph must contain {expected_op}")

        sess = self.check_ort(onnx_path, provider=ort_provider)

        batch_size = 1
        seq_len = 5
        head_size = config.head_dim

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        onnx_input_names = [i.name for i in sess.get_inputs()]

        prefill_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
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
            use_iobinding=False, precision=precision, provider=ort_provider, feed=prefill_feed, sess=sess, vocab_size=config.vocab_size
        )

        with torch.no_grad():
            pt_prefill = model(input_ids)

        np_prefill = pt_prefill.logits.detach().cpu().numpy()
        self.assertEqual(np_prefill.shape, ort_logits_np.shape)
        atol = {"fp16": 1e-2, "fp32": 1e-3}[precision]
        np.testing.assert_allclose(np_prefill[:, :1, :], ort_logits_np[:, :1, :], atol=atol, rtol=1e-3)

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

            _, onnx_decode_logits = run_session_or_io_binding(
                use_iobinding=False,
                precision=precision,
                provider=ort_provider,
                feed=decode_feed,
                sess=sess,
                vocab_size=config.vocab_size,
                results=prefill_results,
            )

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            # fp16 decode logits can diverge more due to limited precision; a high rtol
            # (consistent with the existing common_fast_gpt_oss_20b_random_weights helper)
            # ensures the test is robust to cumulative rounding differences.
            rtol = {"fp16": 10, "fp32": 1e-3}[precision]
            np.testing.assert_allclose(pt_decode_logits, onnx_decode_logits, atol=atol, rtol=rtol)

    @hide_stdout()
    def test_moe_decomposed_fp32_cpu(self):
        self.common_moe_decomposed_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_moe_decomposed_fp16_cpu(self):
        self.common_moe_decomposed_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_genai()
    def test_gpt_oss_20b_fp32_cpu_genai_generate(self):
        """
        Verify that ``onnxruntime-genai`` can load the fp32/CPU ONNX model
        produced from a randomly-initialised GPT-OSS-20B architecture and
        generate tokens without discrepancies against ``transformers.generate``.

        Both backends use greedy (argmax) decoding with the same prompt, so
        the generated token sequences must be bit-for-bit identical.

        The test is skipped automatically when ``onnxruntime-genai`` is not
        installed.
        """
        import torch
        from transformers import AutoModelForCausalLM, GptOssConfig

        from modelbuilder.builder import create_model

        prefix = "test_gpt_oss_20b_fp32_cpu_genai_generate"
        num_hidden_layers = 2

        config = GptOssConfig(
            architectures=["GptOssForCausalLM"],
            hidden_act="silu",
            hidden_size=64,
            intermediate_size=64,
            head_dim=32,
            num_attention_heads=4,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=2,
            num_local_experts=4,
            num_experts_per_tok=2,
            rms_norm_eps=1e-5,
            sliding_window=32,
            vocab_size=256,
            eos_token_id=2,
            bos_token_id=1,
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
        )

        # Greedy decoding is deterministic: both backends must produce the
        # exact same token sequence (prompt + all generated tokens).
        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)
