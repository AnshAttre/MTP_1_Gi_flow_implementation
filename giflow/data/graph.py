"""Graph construction: spatial graph from node features, temporal line graph.

Follows Section 4.2 of the paper: pairwise distances -> Gaussian kernel with decay
rate omega = std(Psi) -> threshold into a binary adjacency matrix.
"""
from __future__ import annotations

import numpy as np


def gaussian_kernel_adjacency(
    positions: np.ndarray,
    threshold: float = 0.1,
    omega: float | None = None,
    binary: bool = True,
) -> np.ndarray:
    """Spatial adjacency from node coordinates/features.

    Psi_ij = ||p_i - p_j||, W_ij = exp(-(Psi_ij / omega)^2) with omega = std(Psi),
    then thresholded into a binary adjacency (paper Sec. 4.2).
    """
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim == 1:
        p = p[:, None]
    diff = p[:, None, :] - p[None, :, :]
    psi = np.sqrt((diff ** 2).sum(-1))
    if omega is None:
        # std over off-diagonal entries
        off = psi[~np.eye(len(psi), dtype=bool)]
        omega = float(off.std())
    omega = max(omega, 1e-12)
    w = np.exp(-((psi / omega) ** 2))
    np.fill_diagonal(w, 0.0)
    w[w < threshold] = 0.0
    if binary:
        w = (w > 0).astype(np.float64)
    return w


def distance_matrix_adjacency(
    dist: np.ndarray, threshold: float = 0.1, omega: float | None = None, binary: bool = True
) -> np.ndarray:
    """Same as above but starting from a precomputed pairwise distance matrix."""
    psi = np.asarray(dist, dtype=np.float64)
    if omega is None:
        off = psi[~np.eye(len(psi), dtype=bool)]
        omega = float(off.std())
    omega = max(omega, 1e-12)
    w = np.exp(-((psi / omega) ** 2))
    np.fill_diagonal(w, 0.0)
    w[w < threshold] = 0.0
    if binary:
        w = (w > 0).astype(np.float64)
    return w


def knn_adjacency(positions: np.ndarray, k: int = 5, symmetric: bool = True) -> np.ndarray:
    """k-nearest-neighbour graph on node positions (used for the synthetic dataset)."""
    p = np.asarray(positions, dtype=np.float64)
    diff = p[:, None, :] - p[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    n = len(p)
    a = np.zeros((n, n))
    idx = np.argsort(d, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    a[rows, idx.reshape(-1)] = 1.0
    if symmetric:
        a = np.maximum(a, a.T)
    return a


def line_graph_adjacency(length: int) -> np.ndarray:
    """Temporal line graph: r <-> r+1. Models the auto-regressive nature (Sec. 4.2)."""
    a = np.zeros((length, length))
    i = np.arange(length - 1)
    a[i, i + 1] = 1.0
    a[i + 1, i] = 1.0
    return a


def laplacian(adj: np.ndarray, normalized: bool = False) -> np.ndarray:
    """L = D - A (unnormalized, as in the paper) or the symmetric normalized version."""
    a = np.asarray(adj, dtype=np.float64)
    deg = a.sum(1)
    if not normalized:
        return np.diag(deg) - a
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    return np.eye(len(a)) - (dinv[:, None] * a * dinv[None, :])


def gcn_normalize(adj: np.ndarray) -> np.ndarray:
    """D^-1/2 (A + I) D^-1/2, the propagation operator used by SGC (Wu et al., 2019)."""
    a = np.asarray(adj, dtype=np.float64) + np.eye(len(adj))
    deg = a.sum(1)
    dinv = np.where(deg > 0, deg ** -0.5, 0.0)
    return dinv[:, None] * a * dinv[None, :]
