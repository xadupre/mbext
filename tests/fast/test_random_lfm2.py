# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_transformers

MODEL_NAME = "Lfm2ForCausalLM"


@requires_transformers("4.55")
class TestLFM2(ExtTestCase):
    def _make_config(self, num_hidden_layers=4):
        from transformers import Lfm2Config

        return Lfm2Config(
            architectures=["Lfm2ForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=128,
            max_position_embeddings=256,
            num_hidden_layers=num_hidden_layers,
            vocab_size=512,
            norm_eps=1e-05,
            # Hybrid layer pattern: alternate conv and full_attention layers
            layer_types=["conv", "full_attention", "conv", "full_attention"][:num_hidden_layers],
            conv_L_cache=3,
            # Small multiple to keep intermediate_size tiny.
            block_multiple_of=32,
        )

    def common_fast_lfm2_random_weights(self, precision, provider):
        import torch
        from transformers import Lfm2ForCausalLM

        from modelbuilder.builder import create_model

        num_hidden_layers = 4
        config = self._make_config(num_hidden_layers=num_hidden_layers)

        basename = f"test_discrepancies_lfm2_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        torch.manual_seed(0)
        model = Lfm2ForCausalLM(config)
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
        )

        log_data = dict(
            precision=precision, model_id=MODEL_NAME, experiment="forward", provider=provider, test=basename, input_type="text", kind="fast"
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        batch_size = 1
        seq_len = 5
        head_size = config.hidden_size // config.num_attention_heads
        attn_indices = [i for i, t in enumerate(config.layer_types) if t == "full_attention"]
        conv_indices = [i for i, t in enumerate(config.layer_types) if t == "conv"]

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]

        with self.subTest(step="prefill"):
            prefill_feed = {
                "input_ids": input_ids.cpu().numpy().astype(np.int64),
                "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
                "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
            }
            # KV cache inputs only for attention layers
            for i in attn_indices:
                prefill_feed[f"past_key_values.{i}.key"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
                prefill_feed[f"past_key_values.{i}.value"] = np.zeros(
                    (batch_size, config.num_key_value_heads, 0, head_size), dtype=self.get_input_np_dtype(precision)
                )
            # Conv cache inputs only for conv layers
            for i in conv_indices:
                prefill_feed[f"past_conv.{i}"] = np.zeros(
                    (batch_size, config.hidden_size, config.conv_L_cache), dtype=self.get_input_np_dtype(precision)
                )
            prefill_feed = {k: v for k, v in prefill_feed.items() if k in onnx_input_names}

            # Run ORT directly: the standard run_session_or_io_binding helper
            # assumes layer 0 has KV cache, which is not the case for LFM2
            # (layer 0 is a conv layer).
            onnx_output_names = [o.name for o in sess.get_outputs()]
            ort_outputs = sess.run(None, prefill_feed)
            ort_logits_np = ort_outputs[onnx_output_names.index("logits")]

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, ort_logits_np)
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 3e-2, "bf16": 2e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(np_prefill, ort_logits_np, atol=atol[precision], rtol=1e-3)

    @hide_stdout()
    def test_fast_discrepancy_lfm2_fp32_cpu(self):
        self.common_fast_lfm2_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_lfm2_fp16_cpu(self):
        self.common_fast_lfm2_random_weights("fp16", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_lfm2_fp16_cuda(self):
        self.common_fast_lfm2_random_weights("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_lfm2_bf16_cuda(self):
        self.common_fast_lfm2_random_weights("bf16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
