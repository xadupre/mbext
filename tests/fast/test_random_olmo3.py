# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout

MODEL_NAME = "Olmo3ForCausalLM"


class TestOLMo3(ExtTestCase):
    @hide_stdout()
    def test_olmo3_fp32_cpu_random_weights(self):
        """
        Convert a model with the same architecture as allenai/Olmo-3-7B-Instruct
        but with randomly initialised weights to an fp32 ONNX model targeting the
        CPU execution provider.

        The test verifies that:
        * ``create_model`` completes without error when given a local model directory.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        """
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.olmo3 import Olmo3Config

        from modelbuilder.builder import create_model

        num_hidden_layers = 1

        config = Olmo3Config(
            architectures=["Olmo3ForCausalLM"],
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="olmo3",
            num_attention_heads=8,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            sliding_window=256,
            vocab_size=50304,
        )

        model_dir = self.get_model_dir("test_olmo3_fp32_cpu_random_weights", clean=False)
        model = AutoModelForCausalLM.from_config(config)
        model.save_pretrained(model_dir)

        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs("test_olmo3_fp32_cpu_random_weights", clean=False)

        create_model(
            model_name=MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
        )

        onnx_path = os.path.join(output_dir, "model.onnx")
        self.assertExists(onnx_path)
        sess = self.check_ort(onnx_path)

        # --- PyTorch inference ---
        batch_size = 1
        seq_len = 5
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        with torch.no_grad():
            pt_logits = model(input_ids).logits.numpy()

        onnx_input_names = {inp.name for inp in sess.get_inputs()}

        head_size = config.hidden_size // config.num_attention_heads
        onnx_feed = {
            "input_ids": input_ids.numpy().astype(np.int64),
            "attention_mask": np.ones((batch_size, seq_len), dtype=np.int64),
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(batch_size, seq_len),
        }
        # Provide empty past KV-cache tensors for every materialised layer.
        for i in range(num_hidden_layers):
            onnx_feed[f"past_key_values.{i}.key"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
            onnx_feed[f"past_key_values.{i}.value"] = np.zeros(
                (batch_size, config.num_key_value_heads, 0, head_size),
                dtype=np.float32,
            )
        # Keep only what the session expects.
        onnx_feed = {k: v for k, v in onnx_feed.items() if k in onnx_input_names}

        onnx_outputs = sess.run(None, onnx_feed)
        onnx_logits = onnx_outputs[0]

        np.testing.assert_allclose(pt_logits, onnx_logits, atol=1e-3, rtol=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
