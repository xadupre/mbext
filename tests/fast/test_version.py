# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import re
import tomllib
import unittest
from pathlib import Path

import modelbuilder


class TestVersion(unittest.TestCase):
    def test_version_defined(self):
        self.assertTrue(hasattr(modelbuilder, "__version__"))
        self.assertIsInstance(modelbuilder.__version__, str)
        self.assertRegex(modelbuilder.__version__, re.compile(r"^\d+\.\d+\.\d+$"))

    def test_version_matches_package_metadata(self):
        pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
        with pyproject.open("rb") as f:
            version = tomllib.load(f)["project"]["version"]
        self.assertEqual(version, "0.2.1")
        self.assertEqual(modelbuilder.__version__, version)


if __name__ == "__main__":
    unittest.main(verbosity=2)
