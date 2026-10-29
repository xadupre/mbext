# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import os
import subprocess
import sys
import textwrap
import unittest

from modelbuilder.ext_test_case import ExtTestCase


class TestDelayedImports(ExtTestCase):
    def _run_python(self, code, no_site=False):
        command = [sys.executable]
        if no_site:
            command.append("-S")
        command.extend(["-c", textwrap.dedent(code)])
        result = subprocess.run(
            command, cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))), capture_output=True, text=True, timeout=60
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_lightweight_modules(self):
        self._run_python(
            """
            import sys
            from modelbuilder import builder, builders, pipeline

            assert builder.parse_private_option(";convert.py;test.py") == (
                "", "convert.py", "test.py")
            assert builder.parse_shard_size("2MiB") == 2 * 1024**2
            assert builder.parse_hf_token("false") is None
            options = {"use_qdq": "true"}
            builder.check_extra_options(options, "cpu")
            assert options == {"use_qdq": True}
            assert "Model" in builders.__all__
            assert "Model" in dir(builders)
            try:
                builders.missing_attribute
            except AttributeError:
                pass
            else:
                raise AssertionError("Unknown attributes must raise AttributeError")
            for name in ("torch", "transformers", "numpy", "onnx_light", "scipy",
                         "modelbuilder.builders.base"):
                assert name not in sys.modules, name
            """,
            no_site=True,
        )

    def test_cli_help(self):
        self._run_python(
            """
            import runpy
            import sys

            sys.argv = ["modelbuilder.builder", "--help"]
            try:
                runpy.run_module("modelbuilder.builder", run_name="__main__")
            except SystemExit as exc:
                assert exc.code == 0, exc.code
            else:
                raise AssertionError("--help must exit")
            for name in ("torch", "transformers", "numpy", "onnx_light"):
                assert name not in sys.modules, name
            """,
            no_site=True,
        )

    def test_private_helpers_without_conversion_dependencies(self):
        self._run_python(
            """
            import os
            import tempfile
            from modelbuilder.builder import load_private_model_builder, run_private_tests

            with tempfile.TemporaryDirectory() as tmp:
                converter = os.path.join(tmp, "convert.py")
                with open(converter, "w") as stream:
                    stream.write("MODEL_BUILDER = object()")
                assert load_private_model_builder(f";{converter};") is not None
                test = os.path.join(tmp, "test.py")
                with open(test, "w") as stream:
                    stream.write("import sys; assert sys.argv == [__file__]")
                run_private_tests(f";;{test}")
            """,
            no_site=True,
        )

    def test_single_model_pipeline_without_onnx(self):
        self._run_python(
            """
            import json
            import os
            import tempfile
            from modelbuilder.pipeline import save_pipeline

            with tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "genai_config.json"), "w") as stream:
                    json.dump({"model": {"decoder": {"filename": "model.onnx"}}}, stream)
                assert save_pipeline(tmp) is None
            """,
            no_site=True,
        )

    def test_dtype_helpers_without_torch(self):
        self._run_python("""
            import sys
            from modelbuilder.builder import set_io_dtype, set_onnx_dtype
            from onnx_light.onnx import TensorProto

            assert set_io_dtype("fp32", "cpu", {}) == TensorProto.FLOAT
            assert set_onnx_dtype("int4", {}) == TensorProto.INT4
            for name in ("torch", "transformers", "modelbuilder.builders.base"):
                assert name not in sys.modules, name
            """)

    def test_lazy_model_export(self):
        self._run_python("""
            import sys
            import modelbuilder.builders as builders

            assert "modelbuilder.builders.base" not in sys.modules
            from modelbuilder.builders import Model
            from modelbuilder.builders.base import Model as BaseModel

            assert Model is BaseModel
            assert builders.Model is Model
            assert builders.__dict__["Model"] is Model
            assert "transformers" not in sys.modules
            assert "modelbuilder.helpers.quantization" not in sys.modules
            """)

    def test_no_grad_is_preserved(self):
        import torch

        from modelbuilder.builder import _torch_no_grad

        @_torch_no_grad
        def multiply(value):
            self.assertFalse(torch.is_grad_enabled())
            return value * 2

        @_torch_no_grad
        def fail():
            self.assertFalse(torch.is_grad_enabled())
            raise ValueError("conversion failed")

        with torch.enable_grad():
            value = torch.tensor([1.0], requires_grad=True)
            self.assertFalse(multiply(value).requires_grad)
            self.assertTrue(torch.is_grad_enabled())
            with self.assertRaisesRegex(ValueError, "conversion failed"):
                fail()
            self.assertTrue(torch.is_grad_enabled())
        self.assertEqual(multiply.__name__, "multiply")

    def test_statistics_delay_scipy(self):
        self._run_python("""
            import sys
            import numpy as np
            from modelbuilder.quant_stats import _tensor_statistics

            assert "scipy" not in sys.modules
            constant = _tensor_statistics("constant", np.ones(8))
            assert constant["normal_distance"] == 1.0
            assert "scipy" not in sys.modules
            varied = _tensor_statistics("varied", np.arange(8, dtype=np.float32))
            assert 0 <= varied["normal_distance"] <= 1
            assert "scipy.stats" in sys.modules
            """)


if __name__ == "__main__":
    unittest.main(verbosity=2)
