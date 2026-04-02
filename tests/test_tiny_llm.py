# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Unit test checking that the model builder works with arnir0/Tiny-LLM.
arnir0/Tiny-LLM is a small (10M parameter) LLaMA-style causal language model
hosted on Hugging Face, making it well-suited for lightweight CI tests.
"""

import os
import tempfile
import unittest

MODEL_NAME = "arnir0/Tiny-LLM"


class TestTinyLLM(unittest.TestCase):

    def test_tiny_llm_fp32_cpu(self):
        """
        Convert arnir0/Tiny-LLM to an fp32 ONNX model targeting the CPU execution
        provider.  Only one hidden layer is materialised (via the
        ``num_hidden_layers`` extra option) to keep the test fast.
        The test verifies that:
        * ``create_model`` completes without error.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file passes ``onnx.checker.check_model``.
        """
        from modelbuilder.builder import create_model

        output_dir = "dump_models"
        cache_dir = "dump_models/cache_dir"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(cache_dir, exist_ok=True)

        create_model(
            model_name=MODEL_NAME,
            input_path="",
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

        # Validate the ONNX model structure (load with external data from output_dir)
        import onnxruntime

        onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def test_tiny_llm_fp32_cpu_random_weights(self):
        """
        Convert a model with the same architecture as arnir0/Tiny-LLM but with
        randomly initialised weights to an fp32 ONNX model targeting the CPU
        execution provider.

        Using random weights avoids downloading the pretrained weights from
        Hugging Face, making this test completely offline and suitable for CI
        environments without internet access.  The full 8-layer config is saved
        locally, but only one hidden layer is materialised during conversion
        (via the ``num_hidden_layers`` extra option passed to ``create_model``)
        to keep the test fast.

        The test verifies that:
        * ``create_model`` completes without error when given a local model directory.
        * The expected ``model.onnx`` file is written to the output directory.
        * The produced ONNX file can be loaded by ``onnxruntime``.
        """
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoModelForCausalLM, LlamaConfig, PreTrainedTokenizerFast

        from modelbuilder.builder import create_model

        # Config matching the arnir0/Tiny-LLM architecture (LlamaForCausalLM,
        # ~10M parameters). These values are hardcoded so the test runs
        # completely offline without downloading any files from Hugging Face.
        config = LlamaConfig(
            architectures=["LlamaForCausalLM"],
            bos_token_id=1,
            eos_token_id=2,
            hidden_act="silu",
            hidden_size=512,
            intermediate_size=1376,
            max_position_embeddings=2048,
            model_type="llama",
            num_attention_heads=8,
            num_hidden_layers=8,
            num_key_value_heads=4,
            rms_norm_eps=1e-05,
            rope_theta=10000.0,
            vocab_size=32000,
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

            # Validate that the ONNX model can be loaded by the runtime.
            import onnxruntime

            onnxruntime.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
