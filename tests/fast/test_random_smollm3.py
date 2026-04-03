# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import unittest

import numpy as np

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout

SMOLLM3_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"


class TestSmolLM3(ExtTestCase):
    @hide_stdout()
    def test_smollm3_fp32_cpu_random_weights(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        from modelbuilder.builder import create_model
        import torch

        num_hidden_layers = 2

        # Minimal SmolLM3 config with 4 layers so that both rope and no-rope
        # layers are exercised (no_rope_layers=[1,1,1,0] by default).
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        model_dir = self.get_model_dir("test_smollm3_fp32_cpu_random_weights")
        model = AutoModelForCausalLM.from_config(config)
        model.save_pretrained(model_dir)

        # Create and save a minimal tokenizer so that save_processing()
        # inside create_model() can load and copy it to the output folder.
        vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>")),
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
        )
        tokenizer.save_pretrained(model_dir)

        output_dir = os.path.join(model_dir, "onnx_output")
        cache_dir = os.path.join(model_dir, "cache")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        output_dir, cache_dir = self.get_dirs("test_smollm3_fp32_cpu_random_weights")
        create_model(
            model_name=SMOLLM3_MODEL_NAME,
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
            "position_ids": np.arange(seq_len, dtype=np.int64).reshape(
                batch_size, seq_len
            ),
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
