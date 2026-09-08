"""Missing-data injection strategies (Section 4 of the paper).

(1) Point missing : randomly mask a fraction rho of the available data.
(2) Block missing : repeatedly pick a random (node, start timestep) and mask a
    contiguous segment, until a fraction rho of the available data is masked.
"""
from __future__ import annotations

import numpy as np


def point_missing(mask: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Return an *eval* mask (1 = entry was removed and must be imputed)."""
    avail = np.argwhere(mask > 0)
    n_drop = int(round(rho * len(avail)))
    if n_drop == 0:
        return np.zeros_like(mask)
    sel = rng.choice(len(avail), size=n_drop, replace=False)
    out = np.zeros_like(mask)
    out[avail[sel, 0], avail[sel, 1]] = 1.0
    return out


def block_missing(
    mask: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    min_len: int = 12,
    max_len: int = 36,
) -> np.ndarray:
    """Contiguous per-node blocks until the target fraction is reached."""
    n, r = mask.shape
    n_avail = int(mask.sum())
    target = int(round(rho * n_avail))
    out = np.zeros_like(mask)
    if target == 0:
        return out
    guard = 0
    max_guard = 100 * (target // max(min_len, 1) + 1) + 1000
    while out.sum() < target and guard < max_guard:
        guard += 1
        node = int(rng.integers(0, n))
        blk = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(r - blk, 1)))
        seg = slice(start, min(start + blk, r))
        out[node, seg] = np.maximum(out[node, seg], mask[node, seg])
    # trim any overshoot so the realised rate matches rho
    over = int(out.sum()) - target
    if over > 0:
        idx = np.argwhere(out > 0)
        sel = rng.choice(len(idx), size=over, replace=False)
        out[idx[sel, 0], idx[sel, 1]] = 0.0
    return out


def make_eval_mask(
    observed_mask: np.ndarray, strategy: str, rho: float, seed: int = 0, **kw
) -> np.ndarray:
    """Dispatch. `observed_mask` marks genuinely available data (1 = present)."""
    rng = np.random.default_rng(seed)
    if strategy == "point":
        return point_missing(observed_mask, rho, rng)
    if strategy == "block":
        return block_missing(observed_mask, rho, rng, **kw)
    raise ValueError(f"unknown missing strategy: {strategy!r}")


def random_subset_mask(mask, keep_ratio_range=(0.3, 0.9), generator=None):
    """Sample a conditioning submask of `mask` (torch). Used during training to
    hide extra points so the model must actually predict rather than copy."""
    import torch

    p = torch.empty(mask.shape[0], 1, 1, device=mask.device).uniform_(*keep_ratio_range)
    keep = (torch.rand(mask.shape, device=mask.device, generator=generator) < p).float()
    return mask * keep
