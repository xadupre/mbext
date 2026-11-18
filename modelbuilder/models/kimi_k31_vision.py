# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""K3.1 MoonViT: dynamic image/video grids and PatchMergerV2."""

import torch
import torch.nn.functional as F
from torch import nn

from .kimi_k31 import RMSNorm


class VisionBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads = c.vt_num_attention_heads
        self.dim = c.qkv_hidden_size // self.heads
        self.norm0 = RMSNorm(c.vt_hidden_size, eps=None)
        self.norm1 = RMSNorm(c.vt_hidden_size, eps=None)
        self.wqkv = nn.Linear(c.vt_hidden_size, 3 * c.qkv_hidden_size, bias=c.attn_bias)
        self.wo = nn.Linear(c.qkv_hidden_size, c.vt_hidden_size, bias=c.attn_bias)
        self.mlp = nn.Module()
        self.mlp.fc0 = nn.Linear(c.vt_hidden_size, c.vt_intermediate_size, bias=c.linear_bias)
        self.mlp.fc1 = nn.Linear(c.vt_intermediate_size, c.vt_hidden_size, bias=c.linear_bias)

    def forward(self, x, cos, sin):
        q, k, v = self.wqkv(self.norm0(x)).reshape(-1, 3, self.heads, self.dim).unbind(1)

        def rotate(value):
            dtype = value.dtype
            pairs = value.float().reshape(-1, self.heads, self.dim // 2, 2)
            real = pairs[:, :, :, 0] * cos - pairs[:, :, :, 1] * sin
            imag = pairs[:, :, :, 0] * sin + pairs[:, :, :, 1] * cos
            return torch.stack((real, imag), dim=3).reshape(-1, self.heads, self.dim).to(dtype)

        q, k = rotate(q).transpose(0, 1), rotate(k).transpose(0, 1)
        v = v.transpose(0, 1)
        scores = torch.matmul(q.float(), k.float().transpose(1, 2)) * self.dim**-0.5
        output = torch.matmul(scores.softmax(2).to(v.dtype), v).transpose(0, 1).reshape(-1, self.heads * self.dim)
        x = x + self.wo(output)
        return x + self.mlp.fc1(F.gelu(self.mlp.fc0(self.norm1(x)), approximate="tanh"))


class VisionTower(nn.Module):
    def __init__(self, c):
        super().__init__()
        if (c.norm_type, c.mlp_type, c.merge_type, c.mm_projector_type, c.pos_emb_type) != (
            "rmsnorm",
            "mlp2",
            "sd2_tpool",
            "patchmergerv2",
            "divided_fixed",
        ):
            raise ValueError("Unsupported K3.1 vision architecture.")
        if c.qkv_hidden_size % c.vt_num_attention_heads or (c.qkv_hidden_size // c.vt_num_attention_heads) % 4:
            raise ValueError("MoonViT rotary head dimensions must be divisible by four.")
        self.c = c
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, c.vt_hidden_size, c.patch_size, stride=c.patch_size, bias=c.patch_embed_proj_bias)
        self.patch_embed.pos_emb = nn.Module()
        self.patch_embed.pos_emb.weight = nn.Parameter(torch.empty(c.init_pos_emb_height, c.init_pos_emb_width, c.vt_hidden_size))
        nn.init.normal_(self.patch_embed.pos_emb.weight, std=0.02)
        self.encoder = nn.Module()
        self.encoder.blocks = nn.ModuleList([VisionBlock(c) for _ in range(c.vt_num_hidden_layers)])
        self.encoder.final_layernorm = RMSNorm(c.vt_hidden_size, eps=None)

    def forward(self, patches, grid_shape):
        c = self.c
        t, h, w = grid_shape.unbind(0)
        x = self.patch_embed.proj(patches).flatten(1)
        spatial = self.patch_embed.pos_emb.weight.permute(2, 0, 1).unsqueeze(0)
        spatial = F.interpolate(spatial.float(), size=(h, w), mode=c.pos_emb_interpolation_mode, align_corners=False)
        spatial = spatial.squeeze(0).permute(1, 2, 0).reshape(1, h * w, c.vt_hidden_size)
        time = torch.arange(t, device=x.device).float().unsqueeze(1)
        freq = 10000.0 ** (-torch.arange(c.vt_hidden_size // 2, device=x.device).float() / (c.vt_hidden_size / 2))
        phase = time * freq
        temporal = torch.cat((phase.sin(), phase.cos()), dim=1).unsqueeze(1)
        pos = spatial + torch.where(t > 1, temporal, torch.zeros_like(temporal))
        x = x + pos.reshape(-1, c.vt_hidden_size).to(x.dtype)
        head_dim = c.qkv_hidden_size // c.vt_num_attention_heads
        freq = 10000.0 ** (-torch.arange(0, head_dim, 4, device=x.device).float() / head_dim)
        rows = torch.arange(h, device=x.device).unsqueeze(1).expand(h, w).reshape(-1)
        cols = torch.arange(w, device=x.device).unsqueeze(0).expand(h, w).reshape(-1)
        angles = torch.stack((cols.float().unsqueeze(1) * freq, rows.float().unsqueeze(1) * freq), dim=2)
        angles = angles.reshape(h * w, head_dim // 2).repeat(t, 1).unsqueeze(1)
        for block in self.encoder.blocks:
            x = block(x, angles.cos(), angles.sin())
        x = self.encoder.final_layernorm(x)
        kh, kw = c.merge_kernel_size
        x = x.reshape(t, h // kh, kh, w // kw, kw, c.vt_hidden_size)
        return x.permute(0, 1, 3, 2, 4, 5).mean(0).reshape(-1, kh * kw * c.vt_hidden_size)


class VisionProjector(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        width = c.mm_hidden_size * c.merge_kernel_size[0] * c.merge_kernel_size[1]
        self.proj = nn.Sequential(nn.Linear(width, width, bias=False), nn.GELU(), nn.Linear(width, hidden, bias=False))
        self.post_norm = RMSNorm(hidden, c.projector_ln_eps)

    def forward(self, features):
        return self.post_norm(self.proj(features))
