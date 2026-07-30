# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import re
import unittest

import modelbuilder


class TestVersion(unittest.TestCase):
    def test_version_defined(self):
        self.assertTrue(hasattr(modelbuilder, "__version__"))
        self.assertIsInstance(modelbuilder.__version__, str)
        self.assertRegex(modelbuilder.__version__, re.compile(r"^\d+\.\d+\.\d+$"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
