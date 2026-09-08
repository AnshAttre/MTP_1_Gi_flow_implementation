"""Building blocks for the GiFlow vector field model (Section 3.3).

Dense-adjacency implementations of the two GNNs the paper cites, so no
torch-geometric dependency is needed (the graphs here are small):

  * GraphConv (Morris et al., 2019) -- x'_i = W1 x_i + W2 sum_{j in N(i)} x_j.
    Used to propagate the static node embeddings before spatial attention.
  * SGC (Wu et al., 2019) -- X' = S^K X W with S = D^-1/2 (A + I) D^-1/2.
    Used for the spatiotemporal message-passing layers.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """Morris et al. (2019) graph convolution on a dense adjacency."""

    def __init__(self, in_dim: int, out_dim: int, adj: np.ndarray, bias: bool = True):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_neigh = nn.Linear(in_dim, out_dim, bias=False)
        self.register_buffer("adj", torch.as_tensor(np.asarray(adj), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., N, F) -> (..., N, out_dim)."""
        neigh = torch.einsum("ij,...jf->...if", self.adj, x)
        return self.lin_self(x) + self.lin_neigh(neigh)


class SGConv(nn.Module):
    """Wu et al. (2019) simplified graph convolution: S^K X W, precomputed S^K."""

    def __init__(self, in_dim: int, out_dim: int, adj_norm: np.ndarray, k: int = 2):
        super().__init__()
        s = np.asarray(adj_norm, dtype=np.float64)
        sk = np.linalg.matrix_power(s, k) if k > 1 else s
        self.register_buffer("prop", torch.as_tensor(sk, dtype=torch.float32))
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., N, F) -> (..., N, out_dim)."""
        return self.lin(torch.einsum("ij,...jf->...if", self.prop, x))


class SpatialAttention(nn.Module):
    """Eq. (11). Keys/queries come from propagated static node embeddings, so the
    N x N attention map is shared across the batch and across timesteps."""

    def __init__(self, hidden: int, adj_s: np.ndarray, emb_dim: int = 32, dropout: float = 0.0):
        super().__init__()
        n = len(adj_s)
        self.node_emb = nn.Parameter(torch.randn(n, emb_dim) * 0.02)
        self.emb_gnn = GraphConv(emb_dim, hidden, adj_s)
        self.w_q = nn.Linear(hidden, hidden, bias=False)
        self.w_k = nn.Linear(hidden, hidden, bias=False)
        self.w_v = nn.Linear(hidden, hidden, bias=False)
        self.value_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.out_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, N, R, H) -> (B, N, R, H)."""
        x_n = self.emb_gnn(self.node_emb)              # (N, H)
        q = self.w_q(x_n)
        k = self.w_k(x_n)
        attn = torch.softmax(q @ k.T * self.scale, dim=-1)   # (N, N)
        attn = self.drop(attn)
        v = self.w_v(self.value_mlp(feat))             # (B, N, R, H)
        agg = torch.einsum("mn,bnrh->bmrh", attn, v)
        return self.out_mlp(agg)


class TemporalAttention(nn.Module):
    """Eq. (12)-(13). Sinusoidal positional encoding plus an optional learnable
    timestamp embedding; self-attention over the R axis, per node."""

    def __init__(
        self,
        hidden: int,
        max_len: int = 512,
        n_timestamp_classes: tuple[int, ...] = (),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.w_q = nn.Linear(hidden, hidden, bias=False)
        self.w_k = nn.Linear(hidden, hidden, bias=False)
        self.w_v = nn.Linear(hidden, hidden, bias=False)
        self.out_mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.drop = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(hidden)
        self.register_buffer("pos_enc", self._sinusoidal(max_len, hidden), persistent=False)
        self.stamp_embs = nn.ModuleList(
            [nn.Embedding(c, hidden) for c in n_timestamp_classes]
        )

    @staticmethod
    def _sinusoidal(max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, feat: torch.Tensor, timestamps: torch.Tensor | None = None) -> torch.Tensor:
        """feat: (B, N, R, H); timestamps: (B, R, n_fields) integer codes."""
        b, n, r, h = feat.shape
        x = self.in_mlp(feat) + self.pos_enc[:r].view(1, 1, r, h)
        if timestamps is not None and self.stamp_embs:
            te = 0.0
            for i, emb in enumerate(self.stamp_embs):
                te = te + emb(timestamps[..., i])       # (B, R, H)
            x = x + te.unsqueeze(1)
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        attn = torch.softmax(torch.einsum("bnrh,bnsh->bnrs", q, k) * self.scale, dim=-1)
        attn = self.drop(attn)
        agg = torch.einsum("bnrs,bnsh->bnrh", attn, v)
        return self.out_mlp(agg)


class SpatioTemporalPropagation(nn.Module):
    """Eq. (14): L_MP layers, each doing spatial message passing per timestep and
    temporal message passing per node."""

    def __init__(
        self,
        hidden: int,
        adj_s_norm: np.ndarray,
        adj_t_norm: np.ndarray,
        n_layers: int = 4,
        k_hops: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.spatial = nn.ModuleList(
            [SGConv(hidden, hidden, adj_s_norm, k=k_hops) for _ in range(n_layers)]
        )
        self.temporal = nn.ModuleList(
            [SGConv(hidden, hidden, adj_t_norm, k=k_hops) for _ in range(n_layers)]
        )
        self.mix = nn.ModuleList([nn.Linear(2 * hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, N, R, H) -> (B, N, R, H)."""
        for gs, gt, mix, norm in zip(self.spatial, self.temporal, self.mix, self.norms):
            # spatial: propagate over N, independently for each timestep
            hs = gs(h.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            # temporal: propagate over R, independently for each node
            ht = gt(h)
            upd = mix(torch.cat([hs, ht], dim=-1))
            h = norm(h + self.drop(F.relu(upd)))
        return h


class FlowStepEmbedding(nn.Module):
    """Sinusoidal embedding of the flow step t in [0, 1], projected by an MLP."""

    def __init__(self, dim: int, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.mlp = nn.Sequential(nn.Linear(2 * n_freq, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) -> (B, dim)."""
        freqs = torch.exp(
            torch.linspace(0.0, math.log(1000.0), self.n_freq, device=t.device)
        )
        ang = t.view(-1, 1) * freqs.view(1, -1)
        return self.mlp(torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1))
