# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""K3.1 sequential prefill and cached generation on the three ONNX backends."""

import json
import os

import numpy as np

from .deepseek_v41_runtime import DeepseekV41Session


class KimiK31Session(DeepseekV41Session):
    """Share ONNX/BF16 bindings and greedy generation, not a GenAI cache schema."""

    def __init__(self, directory, *, backend="onnxruntime", providers=None):
        self.directory = os.fspath(directory)
        with open(os.path.join(directory, "kimi_k31_config.json")) as stream:
            self.config = json.load(stream)
        if self.config.get("format") != "kimi_k31" or self.config.get("version") != 1:
            raise ValueError("Unsupported Kimi K3.1 export manifest.")
        if backend not in {"onnxruntime", "onnx-light", "onnx-light-cpu"}:
            raise ValueError(f"Unknown Kimi runtime backend {backend!r}.")
        self.backend, self.providers = backend, providers
        self.decoder = self._session(self.config["decoder"])
        self.vision = None

    def initial_state(self, batch_size):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        return {
            spec["input"]: np.full([batch_size, *spec["shape"][1:]], spec["fill"], dtype=self._dtype(spec["dtype"]))
            for spec in self.config["state"]
        }

    def run(self, input_ids, *, position_ids, token_types=None, inputs_embeds=None, state=None):
        """Prefill arbitrary prompt lengths by advancing the one-token ONNX graph."""
        ids = np.asarray(input_ids, dtype=np.int64)
        positions = np.asarray(position_ids, dtype=np.int64)
        if ids.ndim != 2 or not all(ids.shape) or positions.shape != ids.shape:
            raise ValueError("input_ids and position_ids need matching nonempty [batch, sequence] shapes.")
        start = int(positions[0, 0])
        if start < 0 or not np.all(positions == np.arange(start, start + ids.shape[1])):
            raise ValueError("Kimi positions must be synchronized and contiguous; padding is not supported.")
        if start + ids.shape[1] > self.config["context_length"]:
            raise ValueError("position_ids exceed the model context length.")
        if token_types is not None:
            token_types = np.asarray(token_types, dtype=np.int64)
            if token_types.shape != ids.shape or np.any((token_types != -1) & (token_types != 1)):
                raise ValueError("Kimi token_types must match input_ids and be -1 (text) or 1 (media).")
        if inputs_embeds is not None:
            inputs_embeds = np.asarray(inputs_embeds, dtype=self._dtype(self.config["io_dtype"]))
            if inputs_embeds.shape != (*ids.shape, self.config["hidden_size"]):
                raise ValueError("inputs_embeds has an incompatible shape.")
        logits = []
        for index in range(ids.shape[1]):
            step = slice(index, index + 1)
            output, state = super().run(
                ids[:, step],
                position_ids=positions[:, step],
                token_types=None if token_types is None else token_types[:, step],
                inputs_embeds=None if inputs_embeds is None else inputs_embeds[:, step],
                state=state,
            )
            logits.append(output)
        return np.concatenate(logits, axis=1), state

    def encode_media(self, patches, grid_shape):
        """Encode one already-preprocessed image or temporally pooled frame group.

        ``patches`` are normalized RGB [T*H*W, 3, patch, patch], in temporal,
        then row-major order. Grid dimensions must match the processor output.
        Multiple images/frame groups are encoded independently.
        """
        if "vision" not in self.config:
            raise ValueError("This Kimi export has no vision component.")
        grid = np.asarray(grid_shape, dtype=np.int64)
        c = self.config["vision_config"]
        if grid.shape != (3,) or np.any(grid <= 0):
            raise ValueError("grid_shape must contain positive [time, height, width].")
        if grid[0] > c["init_pos_emb_time"] or max(grid[1:]) > 512:
            raise ValueError("Media grid exceeds MoonViT's temporal or spatial position limits.")
        if any(size % merge for size, merge in zip(grid[1:], c["merge_kernel_size"])):
            raise ValueError("Media grid must be divisible by the spatial merge dimensions.")
        patches = np.asarray(patches, dtype=self._dtype(self.config["io_dtype"]))
        expected = (int(np.prod(grid)), 3, c["patch_size"], c["patch_size"])
        if patches.shape != expected:
            raise ValueError(f"Expected media patches of shape {expected}, got {patches.shape}.")
        if self.vision is None:
            self.vision = self._session(self.config["vision"])
        return self._run(self.vision, ["image_features"], {"patches": patches, "grid_shape": grid})[0]

    def encode_image(self, image):
        raise NotImplementedError("Use the checkpoint's Kimi image processor, then encode_media(patches, grid_shape).")
