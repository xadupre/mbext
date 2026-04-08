# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda

SMOLLM3_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"


class TestSmolLM3(ExtTestCase):
    def common_fast_smollm3_random_weights(self, precision, provider):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        from modelbuilder.builder import create_model

        # num_hidden_layers=4 is required so that both rope and no-rope
        # layers are exercised (no_rope_layers=[1,1,1,0] by default).
        num_hidden_layers = 4
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        basename = f"test_discrepancies_smollm3_{precision}_{provider}"
        model_dir = self.get_model_dir(basename)
        output_dir, cache_dir = self.get_dirs(basename)

        model = AutoModelForCausalLM.from_config(config)
        model.eval().to(provider)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        create_model(
            model_name=SMOLLM3_MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision=precision,
            execution_provider=provider,
            cache_dir=cache_dir,
        )

        log_data = dict(
            precision=precision,
            model_id=SMOLLM3_MODEL_NAME,
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
        head_size = config.hidden_size // config.num_attention_heads

        torch.manual_seed(0)
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(provider)
        onnx_input_names = [i.name for i in sess.get_inputs()]
        onnx_output_names = [i.name for i in sess.get_outputs()]

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
            prefill_outputs = sess.run(None, prefill_feed)
            prefill_results = dict(zip(onnx_output_names, prefill_outputs))

            with torch.no_grad():
                pt_prefill = model(input_ids)

            np_prefill = pt_prefill.logits.detach().cpu().numpy()
            disc = self.get_numpy_discrepancy(np_prefill, prefill_outputs[0])
            self.log_results({"step": "prefill", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            np.testing.assert_allclose(
                np_prefill, prefill_outputs[0], atol=atol[precision], rtol=1e-3
            )

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

            decode_outputs = sess.run(None, decode_feed)
            onnx_decode_logits = decode_outputs[0]

            with torch.no_grad():
                pt_past_kv = pt_prefill.past_key_values
                next_token_tensor = torch.tensor([[next_token]], dtype=torch.long).to(provider)
                pt_decode = model(next_token_tensor, past_key_values=pt_past_kv)
                pt_decode_logits = pt_decode.logits.detach().cpu().numpy()

            disc = self.get_numpy_discrepancy(pt_decode_logits, decode_outputs[0])
            self.log_results({"step": "decode", **disc, **log_data})
            atol = {"fp16": 1e-2, "bf16": 1e-2, "fp32": 1e-3, "int4": 0.5}
            rtol = {"fp16": 10, "bf16": 1e-2, "fp32": 1e-3, "int4": 10000}
            np.testing.assert_allclose(
                pt_decode_logits, onnx_decode_logits, atol=atol[precision], rtol=rtol[precision]
            )

    @hide_stdout()
    def test_fast_discrepancy_smollm3_fp32_cpu(self):
        self.common_fast_smollm3_random_weights("fp32", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_fp16_cpu(self):
        self.common_fast_smollm3_random_weights("fp16", "cpu")

    @hide_stdout()
    def test_fast_discrepancy_smollm3_int4_cpu(self):
        self.common_fast_smollm3_random_weights("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_fast_discrepancy_smollm3_fp16_cuda(self):
        self.common_fast_smollm3_random_weights("fp16", "cuda")


if __name__ == "__main__":
    unittest.main(verbosity=2)
