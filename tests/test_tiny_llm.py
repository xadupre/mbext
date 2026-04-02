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

import onnx
import pytest

from builder import create_model


MODEL_NAME = "arnir0/Tiny-LLM"


def test_tiny_llm_fp32_cpu():
    """
    Convert arnir0/Tiny-LLM to an fp32 ONNX model targeting the CPU execution
    provider.  Only one hidden layer is materialised (via the
    ``num_hidden_layers`` extra option) to keep the test fast.
    The test verifies that:
    * ``create_model`` completes without error.
    * The expected ``model.onnx`` file is written to the output directory.
    * The produced ONNX file passes ``onnx.checker.check_model``.
    """
    with tempfile.TemporaryDirectory() as output_dir:
        with tempfile.TemporaryDirectory() as cache_dir:
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
            assert os.path.exists(onnx_path), f"Expected ONNX model not found at {onnx_path}"

            # Validate the ONNX model structure (load with external data from output_dir)
            model_proto = onnx.load(onnx_path, load_external_data=False)
            onnx.checker.check_model(model_proto)
