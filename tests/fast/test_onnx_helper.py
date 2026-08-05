# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
Tests for :mod:`modelbuilder.helpers.onnx_helper`.
"""

import importlib
import unittest
from unittest import mock

from modelbuilder.ext_test_case import ExtTestCase


class TestOnnxHelper(ExtTestCase):
    def test_use_onnx_light_default(self):
        from modelbuilder.helpers import onnx_helper

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(onnx_helper.use_onnx_light())

    def test_use_onnx_light_truthy_values(self):
        from modelbuilder.helpers import onnx_helper

        for value in ("1", "True", "true"):
            with mock.patch.dict("os.environ", {"USE_ONNX_LIGHT": value}):
                self.assertTrue(onnx_helper.use_onnx_light())

    def test_use_onnx_light_falsy_values(self):
        from modelbuilder.helpers import onnx_helper

        for value in ("", "0", "false", "no"):
            with mock.patch.dict("os.environ", {"USE_ONNX_LIGHT": value}):
                self.assertFalse(onnx_helper.use_onnx_light())

    def test_default_onnx_is_standard_onnx(self):
        import onnx

        from modelbuilder.helpers import onnx_helper

        self.assertIs(onnx_helper.onnx, onnx)

    def test_get_default_onnx_opset_returns_positive_int(self):
        from modelbuilder.helpers import onnx_helper

        opset = onnx_helper.get_default_onnx_opset()
        self.assertIsInstance(opset, int)
        self.assertGreater(opset, 0)

    def test_get_default_onnx_opset_fallback_without_onnxruntime(self):
        from modelbuilder.helpers import onnx_helper

        real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("onnxruntime"):
                raise ImportError("onnxruntime is not available")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            self.assertEqual(onnx_helper.get_default_onnx_opset(), onnx_helper.DEFAULT_ONNX_OPSET)

    def test_env_selects_onnx_light(self):
        # When USE_ONNX_LIGHT is set, re-importing the helper must pull in
        # onnx_light.onnx instead of onnx.
        from modelbuilder.helpers import onnx_helper

        try:
            with mock.patch.dict("os.environ", {"USE_ONNX_LIGHT": "1"}):
                try:
                    import onnx_light.onnx as onnx_light_onnx
                except ImportError:
                    # onnx-light is not installed: the helper must fail loudly
                    # when asked to use it.
                    with self.assertRaises(ImportError):
                        importlib.reload(onnx_helper)
                else:
                    importlib.reload(onnx_helper)
                    self.assertIs(onnx_helper.onnx, onnx_light_onnx)
        finally:
            # Restore the module to its default (onnx-backed) state.
            importlib.reload(onnx_helper)


if __name__ == "__main__":
    unittest.main(verbosity=2)
