# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Checkpoint ownership and shared storage for a single export."""

import errno
import json
import os
import shutil
import tempfile

from onnx_light.onnx import load_model


def checkpoint_paths(directory):
    """Selects checkpoint shards, not unrelated safetensors files."""
    index = os.path.join(directory, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as stream:
            names = sorted(set(json.load(stream)["weight_map"].values()))
    elif os.path.isfile(os.path.join(directory, "model.safetensors")):
        names = ["model.safetensors"]
    else:
        raise ValueError(f"No safetensors checkpoint was found in {os.fspath(directory)!r}.")
    for name in names:
        if os.path.basename(name) != name or not name.endswith(".safetensors"):
            raise ValueError(f"Invalid checkpoint shard name {name!r}.")
    return [os.path.abspath(os.path.join(directory, name)) for name in names]


class WeightReuse:
    """Share one relocation plan and generated-weight save across components."""

    def __init__(self, source_dir, output_dir):
        self.output_dir = os.path.realpath(output_dir)
        self.paths = checkpoint_paths(source_dir)
        self.destinations = {}
        self.exports = []
        self.generated_payloads = {}
        for source in self.paths:
            if not os.path.isfile(source) or os.path.islink(source):
                raise ValueError(f"--reuse-weights requires regular safetensors files: {source!r}.")
            if os.path.commonpath((os.path.realpath(source), self.output_dir)) == self.output_dir:
                destination = source
            else:
                destination = os.path.join(self.output_dir, ".weights", os.path.basename(source))
            if os.path.commonpath((os.path.realpath(destination), self.output_dir)) != self.output_dir:
                raise ValueError(f"Weight destination escapes the export directory: {destination!r}.")
            if source != destination and os.path.lexists(destination):
                raise FileExistsError(f"Refusing to overwrite checkpoint weight {destination!r}.")
            self.destinations[source] = destination

    def __deepcopy__(self, memo):
        # Composite builders copy their options, but storage belongs to the export.
        return self

    def collect_exports(self):
        """Read final component protos, including private-exporter graph splits."""
        paths = [path for path, _, _ in self.exports]
        config_path = os.path.join(self.output_dir, "genai_config.json")
        if os.path.isfile(config_path):
            with open(config_path) as stream:
                config = json.load(stream)
            paths.extend(
                os.path.join(self.output_dir, section["filename"])
                for section in config["model"].values()
                if isinstance(section, dict) and "filename" in section
            )

        # Preserve per-component external data settings when available.
        export_settings = {os.path.abspath(path): (data_path, limit) for path, data_path, limit in self.exports}
        default_data_path, default_limit = self.exports[0][1:]

        exports = []
        for path in dict.fromkeys(os.path.abspath(path) for path in paths):
            if os.path.dirname(os.path.realpath(path)) != self.output_dir:
                raise ValueError("Reused-weight component models must be saved beside the export configuration.")
            data_path, limit = export_settings.get(path, (default_data_path, default_limit))
            exports.append((load_model(path, load_external_data=False), path, data_path, limit))
        return exports

    def relocate(self):
        """Publish a complete file before removing its source, even across devices."""
        for source, destination in self.destinations.items():
            if source == destination:
                continue
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            if os.path.commonpath((os.path.realpath(destination), self.output_dir)) != self.output_dir:
                raise ValueError(f"Weight destination escapes the export directory: {destination!r}.")
            if os.path.islink(source):
                raise ValueError(f"Source weight {source!r} is a symbolic link.")
            print(f"Moving checkpoint weight {source} to {destination} (source ownership transferred)")
            try:
                os.link(source, destination)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                with tempfile.NamedTemporaryFile(dir=os.path.dirname(destination), delete=False) as temporary:
                    temporary_path = temporary.name
                try:
                    shutil.copyfile(source, temporary_path)
                    # Hard-link publication never replaces an existing destination.
                    os.link(temporary_path, destination)
                finally:
                    os.unlink(temporary_path)
            os.unlink(source)
