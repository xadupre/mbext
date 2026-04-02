# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Unit test checking that the model builder works with an OLMo2-style model.
The test uses randomly initialised weights to avoid downloading the pretrained
weights from Hugging Face, making it completely offline and suitable for CI
environments without internet access.
"""

import os
import tempfile
import unittest


class TestOLMo2(unittest.TestCase):

    def test_olmo2_fp32_cpu_random_weights(self):
        """
        Convert a model with the same architecture as allenai/OLMo-2-7B but
        with randomly initialised weights to an fp32 ONNX model targeting the
        CPU execution provider.

        The test verifies that:
        * ``create_model`` completes without error when given a local model directory.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.olmo2 import Olmo2Config

        from modelbuilder.builder import create_model

        config = Olmo2Config(
            architectures=["Olmo2ForCausalLM"],
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="olmo2",
            num_attention_heads=8,
            num_hidden_layers=2,
            num_key_value_heads=4,
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            vocab_size=50304,
        )

        MODEL_NAME = "allenai/Olmo-3-7B-Instruct"

        with tempfile.TemporaryDirectory() as model_dir:
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

            output_dir = os.path.join(model_dir, "onnx_output")
            cache_dir = os.path.join(model_dir, "cache")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)

            create_model(
                model_name=MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision="fp32",
                execution_provider="cpu",
                cache_dir=cache_dir,
                num_hidden_layers=1,
            )

            onnx_path = os.path.join(output_dir, "model.onnx")
            assert os.path.exists(
                onnx_path
            ), f"Expected ONNX model not found at {onnx_path}"

            import onnxruntime

            onnxruntime.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
