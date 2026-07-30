# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Sphinx configuration for the mbext documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

project = "mbext"
copyright = "Microsoft Corporation"
author = "Microsoft Corporation"

try:
    from modelbuilder import __version__ as release
except Exception:  # pragma: no cover - fallback when the package is unavailable
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_gallery.gen_gallery",
    "supported_architectures",
]

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]
html_title = "mbext"
html_favicon = "_static/logo.svg"
html_theme_options = {
    "light_logo": "logo.svg",
    "dark_logo": "logo-dark.svg",
}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None), "onnx": ("https://onnx.ai/onnx/", None)}

# -- sphinx-gallery ----------------------------------------------------------
# The gallery examples build tiny random-weight models so they run in seconds,
# matching the "short CI / fast tests" philosophy of the project. The matplotlib
# scraper is disabled because the examples produce text output, not figures.
sphinx_gallery_conf = {
    "examples_dirs": ["examples"],
    "gallery_dirs": ["auto_examples"],
    "filename_pattern": r".*\.py",
    "image_scrapers": (),
    "remove_config_comments": True,
    "download_all_examples": False,
    "abort_on_example_error": True,
}
