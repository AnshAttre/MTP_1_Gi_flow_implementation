"""Electrode coordinates and spherical-spline interpolation (Perrin et al., 1989).

Spherical splines are the standard interpolator for missing EEG channels and
the usual reference in the EEG super-resolution literature. Here they serve
as a baseline and as an extra source in the Anchored Flow mixture.
"""
from __future__ import annotations

import numpy as np
import torch

# cerebellar electrodes of the SEED NeuroScan cap have no 10-05 name; I1/I2 sit
# in the same place below O1/O2 and keep every position distinct
_ALIASES = {"CB1": "I1", "CB2": "I2", "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def spline_positions(channel_names) -> np.ndarray:
    """Unit-sphere coordinates (N, 3) for 10-05 channel names, via MNE.

    The montage is fit with a least-squares sphere, centred, then projected
    onto the unit sphere, so spline distances are angles on the head.
    """
    import mne

    try:
        montage = mne.channels.make_standard_montage("colin27_1005")
    except ValueError:
        montage = mne.channels.make_standard_montage("standard_1005")
    table = {k.upper(): v for k, v in montage.get_positions()["ch_pos"].items()}
    names = [_ALIASES.get(str(n).strip().upper(), str(n).strip().upper()) for n in channel_names]
    missing = [n for n in names if n not in table]
    if missing:
        raise ValueError("no 10-05 position for channels %s" % missing)
    pos = np.asarray([table[n] for n in names], dtype=np.float64)
    # least-squares sphere: |p|^2 = 2 p.c + (r^2 - |c|^2)
    a = np.c_[2 * pos, np.ones(len(pos))]
    sol, *_ = np.linalg.lstsq(a, (pos ** 2).sum(1), rcond=None)
    centred = pos - sol[:3]
    return centred / np.linalg.norm(centred, axis=1, keepdims=True)


def spherical_spline_g(positions: np.ndarray, stiffness: int = 4,
                       n_terms: int = 50) -> np.ndarray:
    """Perrin's G matrix: g(cos) = 1/(4 pi) sum (2n+1) / (n(n+1))^m P_n(cos)."""
    from scipy.special import eval_legendre

    cos = np.clip(positions @ positions.T, -1.0, 1.0)
    n = np.arange(1, n_terms + 1)
    coef = (2 * n + 1) / (n * (n + 1)) ** stiffness / (4 * np.pi)
    return sum(c * eval_legendre(k, cos) for k, c in zip(n, coef))


def spline_interpolate(g: torch.Tensor, x_cond: torch.Tensor, cond_mask: torch.Tensor,
                       reg: float = 1e-5) -> torch.Tensor:
    """Spherical-spline estimate of every electrode from the visible ones.

    g: (N, N); x_cond, cond_mask: (B, N, R). An electrode counts as visible in
    a window when any of its samples is visible (exact for whole-electrode
    masks). Hidden electrodes become identity rows with zero right-hand side,
    so one batched solve handles a different visible set per window.
    Visible entries are returned as observed.
    """
    b, n, r = x_cond.shape
    vis = (cond_mask.sum(-1) > 0).to(g.dtype)                       # (B, N)
    outer = vis[:, :, None] * vis[:, None, :]
    eye = torch.eye(n, device=g.device, dtype=g.dtype)
    system = torch.zeros(b, n + 1, n + 1, device=g.device, dtype=g.dtype)
    system[:, :n, :n] = g * outer + eye * (1.0 - vis)[:, None, :] + reg * eye * vis[:, None, :]
    system[:, :n, n] = vis
    system[:, n, :n] = vis
    rhs = torch.zeros(b, n + 1, r, device=g.device, dtype=g.dtype)
    rhs[:, :n] = x_cond.to(g.dtype) * vis[:, :, None]
    coef = torch.linalg.solve(system, rhs)
    est = torch.einsum("ij,bjr->bir", g, coef[:, :n]) + coef[:, n:]
    est = est.to(x_cond.dtype)
    return cond_mask * x_cond + (1.0 - cond_mask) * est


def covariance_interpolate(cov: torch.Tensor, x_cond: torch.Tensor, cond_mask: torch.Tensor,
                           reg: float = 1e-2) -> torch.Tensor:
    """Linear least-squares (kriging) estimate of hidden electrodes from visible ones.

    cov: (N, N) electrode covariance from training data. Hidden electrodes get
    cov[h, v] cov[v, v]^-1 x_v, the best linear predictor under that covariance;
    unlike the spline it uses the recorded spatial structure, not head geometry.
    Same batched solve as `spline_interpolate`, without the constant term.
    """
    b, n, r = x_cond.shape
    vis = (cond_mask.sum(-1) > 0).to(cov.dtype)
    outer = vis[:, :, None] * vis[:, None, :]
    eye = torch.eye(n, device=cov.device, dtype=cov.dtype)
    ridge = reg * torch.diagonal(cov).mean()
    system = cov * outer + eye * (1.0 - vis)[:, None, :] + ridge * eye * vis[:, None, :]
    rhs = x_cond.to(cov.dtype) * vis[:, :, None]
    coef = torch.linalg.solve(system, rhs)
    est = torch.einsum("ij,bjr->bir", cov, coef).to(x_cond.dtype)
    return cond_mask * x_cond + (1.0 - cond_mask) * est
