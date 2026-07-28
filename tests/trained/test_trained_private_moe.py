# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Trained test for the ``private_model`` example.

The fast test (``examples/private_model/test.py``) exercises the ``--private``
option with *random* weights. This module instead checks the **whole trained
model**: it trains the private Mixture-of-Experts model for a few steps (see
:func:`make_trained_model`), saves it as a local checkpoint and converts it with
the ``--private`` option **without truncating the number of layers**, then
compares the PyTorch reference against the exported ONNX model (discrepancy) and
against ``onnxruntime-genai`` (generation).

It follows the same structure as the other ``tests/trained`` examples and is
gated by :func:`~modelbuilder.ext_test_case.long_test` (set ``LONGTEST=1`` to
run it).
"""

import importlib.util
import os
import unittest

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, long_test, requires_genai

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "examples", "private_model"))
MODELING_FILE = os.path.join(EXAMPLE_DIR, "modeling.py")
CONVERT_FILE = os.path.join(EXAMPLE_DIR, "convert.py")
TEST_FILE = os.path.join(EXAMPLE_DIR, "test.py")

# Value passed to the --private option: 'modeling-file;convert-file;fast-test-file'.
PRIVATE_OPTION = f"{MODELING_FILE};{CONVERT_FILE};{TEST_FILE}"


def _load_modeling():
    """Load the example ``modeling.py`` helpers by path."""
    spec = importlib.util.spec_from_file_location("private_model_modeling", MODELING_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


modeling = _load_modeling()


class TestTrainedPrivateMoE(ExtTestCase):
    """End-to-end tests exercising the --private option on the *whole trained* model."""

    def _train_and_convert(self, precision, provider):
        """Train the whole model, save it and convert it with the --private option.

        Returns the output directory, the trained PyTorch model, the config and
        the tokenizer.
        """
        from modelbuilder.builder import create_model

        prefix = f"test_trained_private_moe_{precision}_{provider}"

        config = modeling.make_config()
        model = modeling.make_trained_model(config)
        tokenizer = modeling.make_tokenizer()

        model_dir = self.get_model_dir(prefix, clean=False)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)

        output_dir, cache_dir = self.get_dirs(prefix, clean=False)
        onnx_path = os.path.join(output_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            # No num_hidden_layers override: convert the whole trained model.
            create_model(
                model_name=modeling.MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision=precision,
                execution_provider=provider,
                cache_dir=cache_dir,
                private=PRIVATE_OPTION,
            )
        self.assertExists(onnx_path)
        return output_dir, model, config, tokenizer

    def _common_trained_discrepancies(self, precision, provider):
        import numpy as np
        import torch

        output_dir, model, config, _ = self._train_and_convert(precision, provider)
        onnx_path = os.path.join(output_dir, "model.onnx")

        sess = self._check_with_ort(onnx_path, cpu=provider == "cpu")

        torch.manual_seed(0)
        input_ids = torch.randint(3, config.vocab_size, (1, 8))
        attention_mask = torch.ones_like(input_ids)

        onnx_feed = {
            "input_ids": input_ids.detach().cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.detach().cpu().numpy().astype(np.int64),
        }
        self.fill_with_empty_cache(onnx_feed, sess, provider)

        dtype = self.get_input_torch_dtype(precision)
        model.to(provider).to(dtype)
        with torch.no_grad():
            pt_logits = model(input_ids=input_ids.to(provider), attention_mask=attention_mask.to(provider)).logits
        pt_logits = pt_logits.to(torch.float32).detach().cpu().numpy()

        onnx_logits = sess.run(None, onnx_feed)[0]

        disc = self.get_numpy_discrepancy(pt_logits, onnx_logits)
        disc.update(
            dict(
                precision=precision,
                model_id=modeling.MODEL_NAME,
                experiment="forward",
                provider=provider,
                test=f"test_trained_private_moe_discrepancies_{precision}_{provider}",
                input_type="text",
                kind="trained",
                step="prefill",
            )
        )
        self.log_results(disc)
        atol = {"fp32": 2e-3, "int4": 2.0}[precision]
        self.assertLess(disc["max_abs_err"], atol)

    @long_test()
    @hide_stdout()
    def test_trained_private_moe_discrepancies_fp32_cpu(self):
        self._common_trained_discrepancies("fp32", "cpu")

    @long_test()
    @hide_stdout()
    def test_trained_private_moe_discrepancies_int4_cpu(self):
        self._common_trained_discrepancies("int4", "cpu")

    def _common_trained_generate(self, precision, provider):
        import torch

        output_dir, model, config, _ = self._train_and_convert(precision, provider)

        max_new_tokens = 5
        torch.manual_seed(0)
        prompt_ids = torch.randint(3, config.vocab_size, (1, 4))

        with torch.no_grad():
            pt_output = model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=config.eos_token_id)
        pt_tokens = pt_output[0].tolist()

        og_tokens = self.run_genai_generation(output_dir, prompt_ids, max_new_tokens)

        disc = self.first_token_diff(pt_tokens, og_tokens)
        disc.update(
            dict(
                precision=precision,
                model_id=modeling.MODEL_NAME,
                experiment="genai_generate",
                provider=provider,
                test=f"test_trained_private_moe_genai_generate_{precision}_{provider}",
                input_type="text",
                kind="trained",
            )
        )
        self.log_results(disc)

        self.assertEqual(og_tokens[: prompt_ids.shape[1]], prompt_ids[0].tolist())
        self.assertGreater(len(og_tokens), prompt_ids.shape[1])
        # For lossless precision, generation must match the PyTorch reference.
        if precision not in ("int4", "fp16"):
            self.assertEqual(pt_tokens, og_tokens)

    @long_test()
    @hide_stdout()
    @requires_genai()
    def test_trained_private_moe_generate_fp32_cpu(self):
        self._common_trained_generate("fp32", "cpu")

    @long_test()
    @hide_stdout()
    @requires_genai()
    def test_trained_private_moe_generate_int4_cpu(self):
        self._common_trained_generate("int4", "cpu")


if __name__ == "__main__":
    unittest.main(verbosity=2)
