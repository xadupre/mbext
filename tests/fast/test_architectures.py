# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import unittest

from modelbuilder.architectures import SupportedArchitecture, list_supported_architectures, list_supported_model_ids
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

    def test_representative_model_ids(self):
        archs = {arch.name: arch for arch in list_supported_architectures()}
        for name, model_id in (
            ("GemmaForCausalLM", "google/gemma-2b"),
            ("LlamaForCausalLM", "meta-llama/Llama-2-7b-hf"),
            ("Phi3ForCausalLM", "microsoft/Phi-3-mini-4k-instruct"),
            ("WhisperForConditionalGeneration", "openai/whisper-tiny"),
        ):
            self.assertIn(model_id, archs[name].model_ids)
        for arch in archs.values():
            self.assertIsInstance(arch.model_ids, tuple)
            for model_id in arch.model_ids:
                self.assertEqual(len(model_id.split("/")), 2)
        self.assertEqual(SupportedArchitecture("NewArchitecture").model_ids, ())

    def test_list_supported_model_ids(self):
        model_ids = list_supported_model_ids()
        self.assertNotEmpty(model_ids)
        self.assertEqual(model_ids, sorted(set(model_ids)))
        self.assertEqual(set(model_ids), {model_id for arch in list_supported_architectures() for model_id in arch.model_ids})
        self.assertEqual(model_ids.count("Qwen/Qwen2.5-Omni-3B"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
