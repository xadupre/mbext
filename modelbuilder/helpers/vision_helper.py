# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import numpy as np


def prepare_qwen25_omni_vision_inputs(pixel_values, grid_thw, vision_config):
    """Prepare ONNX vision feeds for processor-produced Qwen2.5-Omni patches.

    ``grid_thw`` contains one ``(temporal, height, width)`` patch grid per
    image/video. Patches must be in the HF processor's spatial-merge order.
    Temporal sizes count groups of ``temporal_patch_size`` frames, not raw frames.
    Outputs remain in processor order, ready for the embedding model.
    """
    grid = np.asarray(grid_thw)
    merge = vision_config.spatial_merge_size
    window = vision_config.window_size // vision_config.patch_size // merge
    if (
        grid.ndim != 2
        or grid.shape[1] != 3
        or not np.issubdtype(grid.dtype, np.integer)
        or np.any(grid <= 0)
        or np.any(grid[:, 1:] % merge)
        or window < 1
        or len(grid) == 0
    ):
        raise ValueError("grid_thw must contain positive integer (t, h, w) grids with h and w divisible by spatial_merge_size")
    pixels = np.asarray(pixel_values, dtype=np.float32)
    in_dim = vision_config.in_channels * vision_config.temporal_patch_size * vision_config.patch_size**2
    if pixels.shape != (sum(int(t) * int(h) * int(w) for t, h, w in grid), in_dim):
        raise ValueError("pixel_values shape does not match grid_thw and the vision patch dimensions")

    positions, frame_ids, window_ids = [], [], []
    frame_offset = window_offset = 0
    for t, h, w in grid:
        rows, cols = np.indices((h, w))
        rows = rows.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        cols = cols.reshape(h // merge, merge, w // merge, merge).transpose(0, 2, 1, 3).reshape(-1)
        spatial = np.stack((rows, cols), axis=-1)
        windows_w = (w // merge + window - 1) // window
        windows_h = (h // merge + window - 1) // window
        spatial_windows = (rows // (merge * window)) * windows_w + cols // (merge * window)
        windows_per_frame = windows_h * windows_w
        positions.append(np.tile(spatial, (t, 1)))
        frame_ids.append(np.repeat(np.arange(t, dtype=np.int64) + frame_offset, h * w))
        window_ids.append((spatial_windows[None, :] + np.arange(t)[:, None] * windows_per_frame + window_offset).reshape(-1))
        frame_offset += t
        window_offset += t * windows_per_frame

    rotary_dim = vision_config.hidden_size // vision_config.num_heads // 2
    inv_freq = 1.0 / (10000 ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))
    rotary = (np.concatenate(positions).astype(np.float32)[..., None] * inv_freq).reshape(len(pixels), -1)
    return {
        "pixel_values": pixels,
        "rotary_pos_emb": rotary,
        "frame_ids": np.concatenate(frame_ids),
        "window_ids": np.concatenate(window_ids).astype(np.int64),
    }
