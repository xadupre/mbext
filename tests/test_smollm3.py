# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Unit test checking that the model builder works with HuggingFaceTB/SmolLM3-3B.

The test creates a minimal SmolLM3-style model with random weights to verify
that the conversion to ONNX works correctly, including the per-layer RoPE
control (no_rope_layers) and sliding window attention (layer_types) features.
No internet access is required.
"""

import os
import tempfile
import unittest

SMOLLM3_MODEL_NAME = "HuggingFaceTB/SmolLM3-3B"


class TestSmolLM3(unittest.TestCase):

    def test_smollm3_fp32_cpu_random_weights(self):
        """
        Convert a model with the SmolLM3 architecture but randomly initialised
        weights to an fp32 ONNX model targeting the CPU execution provider.

        Using random weights avoids downloading the pretrained weights from
        Hugging Face, making this test completely offline and suitable for CI
        environments without internet access.  Only 4 hidden layers are used
        to keep the test fast, covering both rope and no-rope layer types
        (no_rope_layers defaults to [1, 1, 1, 0] for 4 layers, meaning the
        last layer has no RoPE).

        The test verifies that:
        * ``create_model`` completes without error when given a local model directory.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
        from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

        from modelbuilder.builder import create_model

        # Minimal SmolLM3 config with 4 layers so that both rope and no-rope
        # layers are exercised (no_rope_layers=[1,1,1,0] by default).
        config = SmolLM3Config(
            architectures=["SmolLM3ForCausalLM"],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10000.0,
            rms_norm_eps=1e-6,
            vocab_size=128256,
            pad_token_id=None,
        )

        with tempfile.TemporaryDirectory() as model_dir:
            # Create a model with random weights from the config and save it.
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

            create_model(
                model_name=SMOLLM3_MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision="fp32",
                execution_provider="cpu",
                cache_dir=cache_dir,
            )

            onnx_path = os.path.join(output_dir, "model.onnx")
            assert os.path.exists(
                onnx_path
            ), f"Expected ONNX model not found at {onnx_path}"

            # Validate that the ONNX model can be loaded by the runtime.
            import onnxruntime

            onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
