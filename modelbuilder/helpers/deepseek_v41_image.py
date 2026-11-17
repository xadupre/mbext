# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Local image preprocessing for DeepSeek-V4.1-Flash."""

import math

import numpy as np


class DeepseekV41ImageProcessor:
    """Create ViT patches and the model's start/row/end image-token layout."""

    def __init__(self, config, *, dtype=np.float32):
        self.patch_size = config["patch_size"]
        self.downsample_ratio = config["downsample_ratio"]
        self.max_image_tokens = config["max_image_tokens"]
        self.min_pixels = config["min_pixels"]
        self.max_wh_ratio = config.get("max_wh_ratio")
        self.dtype = dtype
        if (
            self.patch_size < 1
            or self.downsample_ratio < 1
            or self.max_image_tokens < 4
            or self.min_pixels < 0
            or (self.max_wh_ratio is not None and self.max_wh_ratio <= 0)
        ):
            raise ValueError("Invalid DeepSeek-V4.1 image preprocessing configuration.")

    def plan(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("Image dimensions must be positive.")
        if self.max_wh_ratio is not None:
            width = min(width, height * self.max_wh_ratio)
        if width * height < self.min_pixels:
            factor = math.sqrt(self.min_pixels / (width * height))
            width, height = int(width * factor), int(height * factor)
        patch = self.patch_size
        out_width = math.ceil(width / patch) * patch
        out_height = math.ceil(height / patch) * patch
        cell = patch * self.downsample_ratio
        rows = math.ceil(out_height / cell)
        columns = math.ceil(out_width / cell)
        if rows * (columns + 1) + 2 > self.max_image_tokens:
            ratio = height / width
            max_width = math.sqrt((self.max_image_tokens - 2) / ratio + 0.25) - 0.5
            max_height = max_width * ratio
            if max_width < 1:
                out_height = ((self.max_image_tokens - 2) // 2) * cell
                out_width = cell
            elif max_height < 1:
                out_height = cell
                out_width = (self.max_image_tokens - 3) * cell
            else:
                factor = min(math.floor(max_width) * cell / width, math.floor(max_height) * cell / height)
                out_height = math.floor(height * factor / patch) * patch
                out_width = math.floor(width * factor / patch) * patch
        return out_height // patch, out_width // patch

    def __call__(self, image):
        from PIL import Image, ImageOps

        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        height, width = self.plan(image.width, image.height)
        target = (width * self.patch_size, height * self.patch_size)
        if self.max_wh_ratio is not None and image.width >= self.max_wh_ratio * image.height:
            image = image.resize(target)
        else:
            image = ImageOps.pad(image, target, color=(127, 127, 127))
        pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
        pixels = (pixels / 255.0 - 0.5) / 0.5
        patch = self.patch_size
        patches = (
            pixels.reshape(3, height, patch, width, patch)
            .transpose(1, 3, 0, 2, 4)
            .reshape(height * width, 3, patch, patch)
            .astype(self.dtype)
        )
        rows = math.ceil(height / self.downsample_ratio)
        columns = math.ceil(width / self.downsample_ratio)
        types = np.asarray([0] + ([1] * columns + [2]) * rows + [3], dtype=np.int64)
        return patches, np.asarray([height, width], dtype=np.int64), types
