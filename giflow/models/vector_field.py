"""Hybrid vector field model v_t(X_t; theta, M, L) -- Section 3.3.

Three components, concatenated with the lifted features and the flow-step
embedding, then projected and refined by spatiotemporal message passing:
  spatial attention  (Eq. 11)
  temporal attention (Eq. 13)
  spatiotemporal propagation (Eq. 14)

The `use_*` flags reproduce the Table 5 ablations.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .layers import (
    FlowStepEmbedding,
    SpatialAttention,
    SpatioTemporalPropagation,
    TemporalAttention,
)


class GiFlowVectorField(nn.Module):
    def __init__(
        self,
        adj_s: np.ndarray,
        adj_s_norm: np.ndarray,
        adj_t_norm: np.ndarray,
        hidden: int = 64,
        emb_dim: int = 32,
        n_mp_layers: int = 4,
        k_hops: int = 2,
        dropout: float = 0.0,
        max_len: int = 512,
        n_timestamp_classes: tuple[int, ...] = (),
        use_spatial_attention: bool = True,
        use_temporal_attention: bool = True,
        use_propagation: bool = True,
    ):
        super().__init__()
        self.use_spatial_attention = use_spatial_attention
        self.use_temporal_attention = use_temporal_attention
        self.use_propagation = use_propagation
        self.hidden = hidden

        # lift [x_t, x_cond, mask] to the hidden dimension
        self.input_proj = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )
        self.step_emb = FlowStepEmbedding(hidden)

        if use_spatial_attention:
            self.spatial_attn = SpatialAttention(hidden, adj_s, emb_dim, dropout)
        if use_temporal_attention:
            self.temporal_attn = TemporalAttention(
                hidden, max_len=max_len, n_timestamp_classes=n_timestamp_classes, dropout=dropout
            )

        # concat: features + step embedding + whichever attention branches are on
        n_parts = 2 + int(use_spatial_attention) + int(use_temporal_attention)
        self.fuse = nn.Linear(n_parts * hidden, hidden)

        if use_propagation:
            self.prop = SpatioTemporalPropagation(
                hidden, adj_s_norm, adj_t_norm, n_mp_layers, k_hops, dropout
            )

        self.out_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        x_t: torch.Tensor,
        x_cond: torch.Tensor,
        mask: torch.Tensor,
        t: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_t, x_cond, mask: (B, N, R); t: (B,). Returns the vector field (B, N, R)."""
        b, n, r = x_t.shape
        feat = self.input_proj(torch.stack([x_t, x_cond, mask], dim=-1))  # (B,N,R,H)

        parts = [feat]
        if self.use_spatial_attention:
            parts.append(self.spatial_attn(feat))
        if self.use_temporal_attention:
            parts.append(self.temporal_attn(feat, timestamps))
        parts.append(self.step_emb(t).view(b, 1, 1, self.hidden).expand(b, n, r, self.hidden))

        h = self.fuse(torch.cat(parts, dim=-1))
        if self.use_propagation:
            h = self.prop(h)
        return self.out_mlp(h).squeeze(-1)
