# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Check native dependency requirements and minimum wheel releases."""

import runpy
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory

from modelbuilder.ext_test_case import ExtTestCase

ROOT = Path(__file__).resolve().parents[2]
_release = runpy.run_path(str(ROOT / "tools" / "onnx_light_release.py"))
release_version = _release["release_version"]
wheel_url = _release["wheel_url"]


class TestDependencyWheels(ExtTestCase):
    def test_onnx_light_requirement(self):
        with (ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)["project"]

        requirements = [req for req in project["dependencies"] if req.startswith("onnx-light")]
        self.assertEqual(requirements, ["onnx-light>=0.2.7"])
        self.assertEqual(release_version(), requirements[0].split(">=")[1])

    def test_release_version_from_pyproject(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pyproject.toml"
            for dependencies, expected in (
                ('"numpy>=2", "onnx-light==1.2.3.4"', "1.2.3.4"),
                ('"onnx-light>=0.2.5"', "0.2.5"),
                ('"onnx-light-cpu==0.2.5"', None),
                ('"onnx-light>0.2.5"', None),
                ('"onnx-light>=0.2.5,<0.3"', None),
                ('"onnx-light==0.2.5", "onnx-light==0.2.6"', None),
                ('"onnx-light>=0.2.5", "onnx-light==0.2.6"', None),
            ):
                with self.subTest(dependencies=dependencies):
                    path.write_text(f"[project]\ndependencies = [{dependencies}]\n", encoding="utf-8")
                    if expected is None:
                        with self.assertRaisesRegex(ValueError, "exactly one onnx-light"):
                            release_version(path)
                    else:
                        self.assertEqual(release_version(path), expected)

    def test_release_wheel_urls(self):
        version = release_version()
        for system, machine, tag in (
            ("Linux", "x86_64", "manylinux_2_27_x86_64.manylinux_2_28_x86_64"),
            ("Linux", "aarch64", "manylinux_2_27_aarch64.manylinux_2_28_aarch64"),
            ("Darwin", "x86_64", "macosx_13_0_universal2"),
            ("Darwin", "arm64", "macosx_13_0_universal2"),
            ("Windows", "AMD64", "win_amd64"),
            ("Windows", "ARM64", "win_arm64"),
        ):
            for package in ("onnx-light", "onnx-light-cpu"):
                if system == "Windows" and package == "onnx-light-cpu":
                    with self.assertRaisesRegex(ValueError, "Unsupported onnx-light"):
                        wheel_url(system, machine, (3, 12), version, package=package)
                    continue
                for minor in (12, 13, 14):
                    with self.subTest(system=system, machine=machine, minor=minor, package=package):
                        self.assertEqual(
                            wheel_url(system, machine, (3, minor), version, package=package),
                            f"https://github.com/xadupre/{package}/releases/download/{version}/"
                            f"{package.replace('-', '_')}-{version}-cp3{minor}-cp3{minor}-{tag}.whl",
                        )

    def test_unsupported_release_wheels(self):
        for system, machine, python_version in (
            ("Linux", "ppc64le", (3, 12)),
            ("Windows", "x86", (3, 12)),
            ("Darwin", "ppc", (3, 12)),
            ("FreeBSD", "x86_64", (3, 12)),
            ("Linux", "x86_64", (3, 11)),
            ("Linux", "x86_64", (3, 15)),
        ):
            with self.subTest(system=system, machine=machine, python=python_version):
                with self.assertRaisesRegex(ValueError, "Unsupported onnx-light"):
                    wheel_url(system, machine, python_version, release_version())

    def test_onnx_light_cpu_requirement(self):
        with (ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)["project"]

        requirements = [req for req in project["optional-dependencies"]["dev"] if req.startswith("onnx-light-cpu")]
        self.assertEqual(requirements, ["onnx-light-cpu>=0.2.7; sys_platform == 'linux' or sys_platform == 'darwin'"])
        self.assertEqual(release_version(package="onnx-light-cpu"), "0.2.7")

    def test_cpu_release_version_from_pyproject(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "pyproject.toml"
            path.write_text(
                '[project]\ndependencies = ["onnx-light>=0.2.7"]\n'
                "[project.optional-dependencies]\ndev = [\"onnx-light-cpu>=0.2.8; sys_platform == 'linux'\"]\n",
                encoding="utf-8",
            )
            self.assertEqual(release_version(path, package="onnx-light-cpu"), "0.2.8")

    def test_ci_uses_github_wheels(self):
        installations = 0
        for path in (ROOT / ".github/workflows").glob("*.yml"):
            lines = path.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if "run: pip install" in line and ('".[dev' in line or line.rstrip().endswith(" .")):
                    with self.subTest(workflow=path.name, command=line):
                        previous_step = "\n".join(lines[:index]).rsplit("- name:", 2)[-2]
                        command = "run: python tools/onnx_light_release.py"
                        if '".[dev' in line:
                            command += " --cpu"
                        self.assertIn(command, previous_step)
                    installations += 1
        self.assertEqual(installations, 8)

    def test_ci_transformers_matrix(self):
        with (ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)["project"]
        self.assertIn("transformers<=5.22.1,>=5.19", project["optional-dependencies"]["dev"])

        for name in ("fast_tests.yml", "trained_private_model_tests.yml", "trained_tiny_llm_tests.yml"):
            with self.subTest(workflow=name):
                workflow = (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
                self.assertIn('transformers-version: ["5.19.0", "5.22.1"]', workflow)
                self.assertIn('- version: "2.14.1"', workflow)
                for version in ("5.19.0", "5.22.1"):
                    self.assertIn(f'- transformers-version: "{version}"', workflow)

    def test_ci_transformers_nightly(self):
        workflow = (ROOT / ".github/workflows/fast_tests_ort_nightly.yml").read_text(encoding="utf-8")
        self.assertIn('"transformers==5.22.1"', workflow)

    def test_ci_onnx_light_wheel_matrix(self):
        workflow = (ROOT / ".github/workflows/fast_tests.yml").read_text(encoding="utf-8")
        wheel_job = workflow.split("\n  onnx-light-wheels:", 1)[1].split("\n  tests:", 1)[0]
        self.assertIn('python-version: ["3.14"]', wheel_job)
        self.assertIn("os: [ubuntu-latest, macos-latest, windows-latest]", wheel_job)
