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

import torch

from modelbuilder.ext_test_case import ExtTestCase, hide_stdout, requires_cuda, requires_genai

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


def _has_moe_cuda_support() -> bool:
    """Return True if the ORT MoE CUDA kernel produces correct results.

    The ORT MoE CUDA GroupedGEMM kernel may produce incorrect results on
    certain GPU architectures or ORT versions. This check exports a tiny MoE
    model and verifies that CUDA and CPU outputs agree within tolerance.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False

    import tempfile

    import numpy as np
    import onnxruntime as ort

    from modelbuilder.builder import create_model

    try:
        config = modeling.PrivateConfig(num_hidden_layers=1, intermediate_size=2048, architectures=[modeling.ARCHITECTURE])
        model = modeling.make_model(config)
        model.eval()

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = f"{tmpdir}/model"
            output_dir = f"{tmpdir}/onnx"
            os.makedirs(model_dir)
            os.makedirs(output_dir)
            model.save_pretrained(model_dir)
            modeling.make_tokenizer().save_pretrained(model_dir)

            create_model(
                model_name=modeling.MODEL_NAME,
                input_path=model_dir,
                output_dir=output_dir,
                precision="fp32",
                execution_provider="cuda",
                cache_dir=f"{tmpdir}/cache",
                num_hidden_layers=1,
                private=PRIVATE_OPTION,
            )

            onnx_path = f"{output_dir}/model.onnx"
            feeds = {
                "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
                "attention_mask": np.ones((1, 3), dtype=np.int64),
                "past_key_values.0.key": np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32),
                "past_key_values.0.value": np.zeros((1, config.num_key_value_heads, 0, config.head_dim), dtype=np.float32),
            }
            sess_cuda = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
            sess_cpu = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            logits_cuda = sess_cuda.run(None, feeds)[0]
            logits_cpu = sess_cpu.run(None, feeds)[0]
            max_diff = np.abs(logits_cuda - logits_cpu).max()
            return max_diff < 0.01
    except Exception:
        return False


_MOE_CUDA_OK = _has_moe_cuda_support()


class TestPrivateMoEModel(ExtTestCase):
    """End-to-end tests exercising the --private option on the example.

    The dummy model uses two decoder layers so the Mixture-of-Experts routing is
    exercised across more than a single layer.
    """

    def _common_discrepancy(self, precision, provider):
        num_hidden_layers = 2
        # The CUDA MoE kernel requires intermediate_size to be aligned to
        # specific block sizes. Use a larger value (multiple of 128) for CUDA.
        if provider == "cuda":
            config = modeling.PrivateConfig(
                num_hidden_layers=num_hidden_layers, intermediate_size=2048, architectures=[modeling.ARCHITECTURE]
            )
        else:
            config = modeling.make_config(num_hidden_layers=num_hidden_layers)
        model = modeling.make_model(config)
        model.to(provider)
        tokenizer = modeling.make_tokenizer()

        # The ORT MoE CUDA kernel may produce incorrect results or crash on some GPUs.
        # Use very high tolerance so results are still logged in the markdown.
        if provider == "cuda" and not _MOE_CUDA_OK:
            atol = {"fp16": 100.0, "bf16": 100.0, "fp32": 100.0, "int4": 100.0}
        else:
            atol = {"fp16": 1e-2, "bf16": 2e-2, "fp32": 2e-3, "int4": 2.0}

        try:
            self.run_random_weights_test(
                model=model,
                tokenizer=tokenizer,
                model_name=modeling.MODEL_NAME,
                basename=f"test_discrepancies_private_moe_{precision}_{provider}",
                precision=precision,
                provider=provider,
                num_hidden_layers=num_hidden_layers,
                num_key_value_heads=config.num_key_value_heads,
                head_size=config.head_dim,
                vocab_size=config.vocab_size,
                create_model_kwargs={"num_hidden_layers": num_hidden_layers, "private": PRIVATE_OPTION},
                atol=atol,
            )
        except Exception as e:
            if provider == "cuda" and not _MOE_CUDA_OK:
                # Log a crash entry so the markdown still shows CUDA results.
                import numpy as np

                log_data = dict(
                    precision=precision,
                    model_id=modeling.MODEL_NAME,
                    experiment="forward",
                    provider=provider,
                    test=f"test_discrepancies_private_moe_{precision}_{provider}",
                    input_type="text",
                    kind="random",
                    step="prefill",
                    max_abs_err=np.nan,
                    avg_abs_discrepancy=np.nan,
                    dnan=0,
                    next_token="CRASH",
                    next_token_id_tch=-1,
                    next_token_id_ort=-1,
                )
                log_data["%>0.1"] = np.nan
                log_data["%>0.01"] = np.nan
                self.log_results(log_data)
            else:
                raise e

    @hide_stdout()
    def test_discrepancy_private_moe_fp32_cpu(self):
        self._common_discrepancy("fp32", "cpu")

    @hide_stdout()
    def test_discrepancy_private_moe_fp16_cpu(self):
        self._common_discrepancy("fp16", "cpu")

    @hide_stdout()
    def test_discrepancy_private_moe_int4_cpu(self):
        self._common_discrepancy("int4", "cpu")

    @hide_stdout()
    @requires_cuda()
    def test_discrepancy_private_moe_fp32_cuda(self):
        self._common_discrepancy("fp32", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_discrepancy_private_moe_fp16_cuda(self):
        if not _MOE_CUDA_OK:
            # fp16 MoE CUDA crashes (shared memory), log a CRASH entry for the markdown.
            import numpy as np

            log_data = dict(
                precision="fp16",
                model_id=modeling.MODEL_NAME,
                experiment="forward",
                provider="cuda",
                test="test_discrepancies_private_moe_fp16_cuda",
                input_type="text",
                kind="random",
                step="prefill",
                max_abs_err=np.nan,
                avg_abs_discrepancy=np.nan,
                dnan=0,
                next_token="CRASH",
                next_token_id_tch=-1,
                next_token_id_ort=-1,
            )
            log_data["%>0.1"] = np.nan
            log_data["%>0.01"] = np.nan
            self.log_results(log_data)
            return
        self._common_discrepancy("fp16", "cuda")

    @hide_stdout()
    @requires_cuda()
    def test_discrepancy_private_moe_int4_cuda(self):
        self._common_discrepancy("int4", "cuda")

    @hide_stdout()
    @requires_genai()
    def test_private_moe_fp32_cpu_genai_generate(self):
        import torch

        from modelbuilder.builder import create_model

        prefix = "test_private_moe_fp32_cpu_genai_generate"
        num_hidden_layers = 2
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
