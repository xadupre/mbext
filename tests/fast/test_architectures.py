# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from modelbuilder.architectures import SupportedArchitecture, list_supported_architectures
from modelbuilder.ext_test_case import ExtTestCase


class TestArchitectures(ExtTestCase):
    def test_list_supported_architectures_not_empty(self):
        archs = list_supported_architectures()
        self.assertNotEmpty(archs)
        for arch in archs:
            self.assertIsInstance(arch, SupportedArchitecture)

    def test_list_supported_architectures_sorted_and_unique(self):
        archs = list_supported_architectures()
        names = [a.name for a in archs]
        self.assertEqual(names, sorted(names, key=str.lower))
        self.assertEqual(len(names), len(set(names)))

    def test_known_architectures_present(self):
        names = {a.name for a in list_supported_architectures()}
        for expected in ("LlamaForCausalLM", "MixtralForCausalLM", "Qwen3ForCausalLM", "GemmaForCausalLM"):
            self.assertIn(expected, names)

    def test_builder_module_resolved(self):
        # Every dispatch branch should resolve to a builder module/class so the
        # generated documentation table has no empty cells.
        for arch in list_supported_architectures():
            self.assertNotEmpty(arch.builder_module)
            self.assertNotEmpty(arch.builder_class)
            self.assertStartsWith("modelbuilder.builders", arch.builder_module)


if __name__ == "__main__":
    unittest.main(verbosity=2)
