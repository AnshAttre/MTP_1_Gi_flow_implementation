"""Synthetic smooth spatiotemporal graph signal (Section 4.1).

50 nodes drawn uniformly in a 50x50 square, KNN graph with k=5. The signal is a
low-frequency graph signal that evolves as x_r = x_{r-1} + L^{-1/2} f_r, with
L^{-1/2} = U Lambda^{-1/2} U^T built from the non-trivial spectrum, then
corrupted by i.i.d. Gaussian noise of std sigma.
"""
from __future__ import annotations

import numpy as np

from .graph import knn_adjacency, laplacian


def generate_synthetic(
    n_nodes: int = 50,
    n_steps: int = 3000,
    domain: float = 50.0,
    k: int = 5,
    sigma: float = 0.1,
    n_low_freq: int | None = None,
    seed: int = 0,
):
    """Returns (X, adj, positions) with X of shape (n_nodes, n_steps)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.0, domain, size=(n_nodes, 2))
    adj = knn_adjacency(pos, k=k)
    lap = laplacian(adj)

    evals, evecs = np.linalg.eigh(lap)
    evals = np.clip(evals, 0.0, None)
    # Lambda^{-1/2} with the first (trivial) eigenvalue zeroed out
    inv_sqrt = np.zeros_like(evals)
    inv_sqrt[1:] = evals[1:] ** -0.5
    l_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T

    # initial signal: low-frequency in the spectral domain
    if n_low_freq is None:
        n_low_freq = max(2, n_nodes // 10)
    coeff = np.zeros(n_nodes)
    coeff[:n_low_freq] = rng.normal(size=n_low_freq)
    x0 = evecs @ coeff

    x = np.empty((n_nodes, n_steps))
    x[:, 0] = x0
    for r in range(1, n_steps):
        f = rng.normal(size=n_nodes)
        x[:, r] = x[:, r - 1] + l_inv_sqrt @ f

    x_noisy = x + rng.normal(scale=sigma, size=x.shape)
    return x_noisy, adj, pos
