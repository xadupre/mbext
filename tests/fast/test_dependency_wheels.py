# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Check native dependency requirements and pinned wheel releases."""

import tomllib
from pathlib import Path

from modelbuilder.ext_test_case import ExtTestCase

ROOT = Path(__file__).resolve().parents[2]


class TestDependencyWheels(ExtTestCase):
    def test_onnx_light_requirement(self):
        with (ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)["project"]

        requirements = [req for req in project["dependencies"] if req.startswith("onnx-light")]
        self.assertEqual(requirements, ["onnx-light>=0.2.0"])

    def test_onnx_light_wheels(self):
        with (ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)["project"]

        for requirements, package, version in (
            (project["optional-dependencies"]["dev"], "onnx-light-cpu", "0.2.0"),
        ):
            wheels = [requirement for requirement in requirements if requirement.startswith(f"{package} @ ")]
            self.assertTrue(wheels)
            for wheel in wheels:
                self.assertIn(f"/releases/download/{version}/", wheel)
                self.assertIn(f"/{package.replace('-', '_')}-{version}-", wheel)

    def test_ci_transformers_matrix(self):
        with (ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)["project"]
        self.assertIn("transformers<=5.20.2,>=5.19", project["optional-dependencies"]["dev"])

        for name in ("fast_tests.yml", "trained_private_model_tests.yml", "trained_tiny_llm_tests.yml"):
            with self.subTest(workflow=name):
                workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertIn('transformers-version: ["5.19.0", "5.20.2"]', workflow)
                self.assertIn('- version: "2.14.1"', workflow)
                for version in ("5.19.0", "5.20.2"):
                    self.assertIn(f'- transformers-version: "{version}"', workflow)

    def test_ci_onnx_light_wheel_matrix(self):
        workflow = (ROOT / ".github/workflows/fast_tests.yml").read_text(encoding="utf-8")
        wheel_job = workflow.split("\n  onnx-light-wheels:", 1)[1].split("\n  tests:", 1)[0]
        self.assertIn('python-version: ["3.14"]', wheel_job)
        self.assertIn("os: [ubuntu-latest, macos-latest, windows-latest]", wheel_job)
