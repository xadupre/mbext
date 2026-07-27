# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Fast test file for the ``private_model`` example.

This is the ``fast-test-file`` of the ``--private`` option. It validates the
custom builder end to end:

* a discrepancy test comparing the PyTorch reference against the exported ONNX
  model (``fp32`` and ``fp16``), and
* a genai generation test comparing ``model.generate`` against
  ``onnxruntime-genai`` (skipped when ``onnxruntime-genai`` is not installed).

It can be run in three equivalent ways::

    # 1. directly as a script
    python examples/private_model/test.py

    # 2. through the --private option (no model id => run the fast tests)
    python -m modelbuilder.builder --private \
        "examples/private_model/modeling.py;examples/private_model/convert.py;examples/private_model/test.py"

    # 3. with pytest
    pytest examples/private_model/test.py

The conversion itself is exercised through the ``--private`` option: the test
passes ``private=<modeling>;<convert>;<test>`` to
:func:`modelbuilder.builder.create_model` so the custom builder defined in
``convert.py`` is used.
"""

import importlib.util
import os
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_genai

HERE = os.path.dirname(os.path.abspath(__file__))
MODELING_FILE = os.path.join(HERE, "modeling.py")
CONVERT_FILE = os.path.join(HERE, "convert.py")
TEST_FILE = os.path.join(HERE, "test.py")

# Value passed to the --private option: 'modeling-file;convert-file;fast-test-file'.
PRIVATE_OPTION = f"{MODELING_FILE};{CONVERT_FILE};{TEST_FILE}"


def _load_modeling():
    """Load the sibling ``modeling.py`` helpers by path.

    The test file is executed as a standalone script (``python test.py`` or
    through ``--private``), so it cannot rely on package-relative imports.
    """
    spec = importlib.util.spec_from_file_location("private_model_modeling", MODELING_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modeling = _load_modeling()


class TestPrivateTinyModel(ExtTestCase):
    """End-to-end tests exercising the --private option on the example."""

    def _common_discrepancy(self, precision, provider):
        num_hidden_layers = 1
        config = modeling.make_config(num_hidden_layers=num_hidden_layers)
        model = modeling.make_model(config)
        model.to(provider)
        tokenizer = modeling.make_tokenizer()

        self.run_random_weights_test(
            model=model,
            tokenizer=tokenizer,
            model_name=modeling.MODEL_NAME,
            basename=f"test_discrepancies_private_tiny_{precision}_{provider}",
            precision=precision,
            provider=provider,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=config.num_key_value_heads,
            head_size=config.head_dim,
            vocab_size=config.vocab_size,
            create_model_kwargs={"num_hidden_layers": num_hidden_layers, "private": PRIVATE_OPTION},
        )

    @hide_stdout()
    def test_discrepancy_private_tiny_fp32_cpu(self):
        self._common_discrepancy("fp32", "cpu")

    @hide_stdout()
    def test_discrepancy_private_tiny_fp16_cpu(self):
        self._common_discrepancy("fp16", "cpu")

    @hide_stdout()
    @requires_genai()
    def test_private_tiny_fp32_cpu_genai_generate(self):
        import torch

        from modelbuilder.builder import create_model

        prefix = "test_private_tiny_fp32_cpu_genai_generate"
        num_hidden_layers = 1
        config = modeling.make_config(num_hidden_layers=num_hidden_layers)

        torch.manual_seed(42)
        model = modeling.make_model(config)

        model_dir = self.get_model_dir(prefix, clean=False)
        model.save_pretrained(model_dir)
        modeling.make_tokenizer().save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)

        create_model(
            model_name=modeling.MODEL_NAME,
            input_path=model_dir,
            output_dir=output_dir,
            precision="fp32",
            execution_provider="cpu",
            cache_dir=cache_dir,
            num_hidden_layers=num_hidden_layers,
            private=PRIVATE_OPTION,
        )

        self.run_genai_generation_test(output_dir, model, config.vocab_size, config.eos_token_id)


if __name__ == "__main__":
    # Reset any leftover argv (e.g. when run through the --private option) so
    # unittest does not try to parse the builder's own options.
    import sys

    sys.argv = [sys.argv[0]]
    unittest.main(verbosity=2)
