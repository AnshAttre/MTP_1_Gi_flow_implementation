"""Coordinate-aware channel-window token flow model."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .giflow import EMA


def sinusoidal_embedding(values: torch.Tensor, dim: int) -> torch.Tensor:
    if dim % 2:
        raise ValueError("embedding dimension must be even")
    positions = values[..., None]
    scales = torch.exp(
        torch.arange(0, dim, 2, device=values.device, dtype=values.dtype)
        * (-math.log(10000.0) / dim)
    )
    angles = positions * scales
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ChannelTokenFlow(nn.Module):
    """Rectified flow over channel-window tokens ordered as (time, channels)."""

    def __init__(
        self,
        patch_size: int,
        hidden: int = 128,
        layers: int = 4,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        self.patch_size = int(patch_size)
        self.hidden = hidden
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_size, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.position_proj = nn.Linear(64 + 16, hidden)
        self.mask_embed = nn.Embedding(3, hidden)
        self.time_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.decoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.out_proj = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, patch_size))

    def _patches(self, x: torch.Tensor):
        b, n, r = x.shape
        if r % self.patch_size:
            raise ValueError("window length must be divisible by patch_size")
        patches = r // self.patch_size
        # (B, patches, channels, patch_size) -> raster (B, patches*channels, patch_size)
        x = x.reshape(b, n, patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(b, patches * n, self.patch_size)
        return x, patches

    def _patch_mask(self, mask: torch.Tensor):
        b, n, r = mask.shape
        patches = r // self.patch_size
        return mask.reshape(b, n, patches, self.patch_size).any(dim=-1).permute(0, 2, 1).reshape(b, patches * n)

    def _token_positions(self, positions: torch.Tensor | None, n: int, patches: int, device, dtype):
        b = positions.shape[0] if positions is not None else 1
        if positions is None:
            positions = torch.zeros((b, n, 3), device=device, dtype=dtype)
        spatial = torch.cat((
            sinusoidal_embedding(positions[..., 0], 22),
            sinusoidal_embedding(positions[..., 1], 22),
            sinusoidal_embedding(positions[..., 2], 20),
        ), dim=-1)
        spatial = spatial[:, None, :, :].expand(b, patches, n, 64).reshape(b, patches * n, 64)
        time_index = torch.arange(patches, device=device, dtype=dtype).view(1, patches, 1)
        time = sinusoidal_embedding(time_index.expand(b, patches, 1), 16).reshape(b, patches, 16)
        time = time[:, :, None, :].expand(b, patches, n, 16).reshape(b, patches * n, 16)
        return torch.cat((spatial, time), dim=-1)

    def forward(self, x_t, x_cond, cond_mask, target_mask, flow_time, positions=None, node_mask=None):
        x_tokens, patches = self._patches(x_t)
        cond_tokens, _ = self._patches(x_cond)
        visible = self._patch_mask(cond_mask).long()
        target = self._patch_mask(target_mask).long()
        mask_state = torch.clamp(visible + target, max=2)
        n = x_t.shape[1]
        position = self._token_positions(positions, n, patches, x_t.device, x_t.dtype)
        if position.shape[0] == 1 and x_t.shape[0] > 1:
            position = position.expand(x_t.shape[0], -1, -1)
        h = self.patch_embed(x_tokens) + self.position_proj(position) + self.mask_embed(mask_state)
        if node_mask is not None:
            node = node_mask[:, None, :, None].expand(-1, patches, -1, 1).reshape(-1, patches * n, 1)
            key_padding_mask = node[:, :, 0].eq(0)
        else:
            key_padding_mask = None
        h = h + self.time_embed(flow_time[:, None, None].to(x_t.dtype)).expand(-1, h.shape[1], -1)
        latent = self.encoder(h, src_key_padding_mask=key_padding_mask)
        decoded = self.decoder(latent, src_key_padding_mask=key_padding_mask)
        velocity = self.out_proj(decoded)
        return self._unpatch(velocity, x_t.shape[1])

    def _unpatch(self, tokens, channels):
        b, length, patch_size = tokens.shape
        patches = length // channels
        tokens = tokens.reshape(b, patches, channels, patch_size)
        return tokens.permute(0, 2, 1, 3).reshape(b, channels, patches * patch_size)

    def loss(self, x_true, cond_mask, target_mask, timestamps=None, node_mask=None, positions=None, return_components=False):
        b = x_true.shape[0]
        x_cond = x_true * cond_mask
        x0 = torch.randn_like(x_true)
        flow_time = torch.rand(b, device=x_true.device)
        t = flow_time[:, None, None]
        x_t = (1.0 - t) * x0 + t * x_true
        target_velocity = x_true - x0
        predicted_velocity = self.forward(
            x_t, x_cond, cond_mask, target_mask, flow_time, positions, node_mask
        )
        target = target_mask.to(dtype=x_true.dtype)
        denom = target.sum().clamp_min(1.0)
        flow_loss = (((predicted_velocity - target_velocity) * target) ** 2).sum() / denom

        endpoint = x_t + (1.0 - t) * predicted_velocity
        cond_denom = cond_mask.sum().clamp_min(1.0)
        preservation_loss = (((endpoint - x_true) * cond_mask) ** 2).sum() / cond_denom
        total = flow_loss + 0.1 * preservation_loss
        components = {
            "flow_loss": flow_loss,
            "preservation_loss": preservation_loss,
            "residual_loss": x_true.new_zeros(()),
            "smm_regularization": x_true.new_zeros(()),
        }
        return (total, components) if return_components else total

    @torch.no_grad()
    def impute(self, x_true, cond_mask, n_steps=20, timestamps=None, node_mask=None, positions=None):
        x_cond = x_true * cond_mask
        target_mask = 1.0 - cond_mask
        if node_mask is not None:
            target_mask = target_mask * node_mask[:, :, None]
        x = torch.randn_like(x_true)
        for step in range(n_steps):
            flow_time = torch.full((x.shape[0],), step / max(n_steps - 1, 1), device=x.device)
            velocity = self.forward(x, x_cond, cond_mask, target_mask, flow_time, positions, node_mask)
            x = x + velocity * (1.0 / n_steps)
        result = torch.where(cond_mask > 0, x_cond, x)
        if node_mask is not None:
            result = result * node_mask[:, :, None]
        return result
