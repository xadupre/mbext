# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Install GitHub wheels for the onnx-light pinned or minimum versions."""

import argparse
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def release_version(pyproject=Path(__file__).resolve().parents[1] / "pyproject.toml", package="onnx-light"):
    with pyproject.open("rb") as stream:
        project = tomllib.load(stream)["project"]
    dependencies = project["dependencies"] if package == "onnx-light" else project["optional-dependencies"]["dev"]
    versions = [
        match[1]
        for dependency in dependencies
        if (match := re.fullmatch(rf"{re.escape(package)}(?:==|>=)(\d+(?:\.\d+)+)", dependency.split(";")[0].strip()))
    ]
    if len(versions) != 1:
        raise ValueError(f"Expected exactly one {package} pinned or minimum version.")
    return versions[0]


def wheel_url(system, machine, python_version, version, package="onnx-light"):
    if package not in {"onnx-light", "onnx-light-cpu"}:
        raise ValueError(f"Unsupported release package: {package}")
    if python_version not in ((3, 12), (3, 13), (3, 14)):
        raise ValueError(f"Unsupported onnx-light Python version: {python_version}")
    machine = machine.lower()
    if system == "Linux" and machine in {"x86_64", "aarch64"}:
        platform_tag = f"manylinux_2_27_{machine}.manylinux_2_28_{machine}"
    elif system == "Darwin" and machine in {"x86_64", "arm64"}:
        platform_tag = "macosx_13_0_universal2"
    elif system == "Windows" and machine in {"amd64", "arm64"} and package == "onnx-light":
        platform_tag = f"win_{machine}"
    else:
        raise ValueError(f"Unsupported onnx-light release platform: {system} {machine}")
    python_tag = f"cp{python_version[0]}{python_version[1]}"
    name = f"{package.replace('-', '_')}-{version}-{python_tag}-{python_tag}-{platform_tag}.whl"
    return f"https://github.com/xadupre/{package}/releases/download/{version}/{name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", action="store_true", help="Also install onnx-light-cpu.")
    args = parser.parse_args()
    packages = ["onnx-light", "onnx-light-cpu"] if args.cpu else ["onnx-light"]
    urls = [
        wheel_url(platform.system(), platform.machine(), sys.version_info[:2], release_version(package=package), package=package)
        for package in packages
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *urls])
