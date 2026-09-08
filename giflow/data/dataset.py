"""Windowing, splitting and normalisation (Appendix C.3).

Windows of length 24; 70% / 10% / 20% of the windows go to train / val / test.
Standardisation uses training-split statistics over observed entries only.

Masks carried per window:
  observed_mask : 1 where the raw data genuinely exists
  eval_mask     : 1 where we artificially removed data (this is what gets scored)
  cond_mask     : observed_mask * (1 - eval_mask), i.e. what the model may see
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .masking import make_eval_mask


class StandardScaler:
    def __init__(self, mean: float, std: float):
        self.mean = float(mean)
        self.std = float(max(std, 1e-8))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean


class SpatioTemporalWindows(Dataset):
    """Sliding windows over a (N, R_total) signal."""

    def __init__(
        self,
        x: np.ndarray,
        observed_mask: np.ndarray,
        eval_mask: np.ndarray,
        indices: np.ndarray,
        window: int,
        scaler: StandardScaler,
        timestamps: np.ndarray | None = None,
    ):
        self.x = np.asarray(x, dtype=np.float32)
        self.observed_mask = np.asarray(observed_mask, dtype=np.float32)
        self.eval_mask = np.asarray(eval_mask, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = int(window)
        self.scaler = scaler
        self.timestamps = None if timestamps is None else np.asarray(timestamps, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        s = int(self.indices[i])
        sl = slice(s, s + self.window)
        obs = self.observed_mask[:, sl]
        ev = self.eval_mask[:, sl] * obs
        cond = obs * (1.0 - ev)
        # normalise, then zero out anything not genuinely observed
        x = self.scaler.transform(self.x[:, sl]) * obs
        item = {
            "x_true": torch.from_numpy(x.copy()),
            "observed_mask": torch.from_numpy(obs.copy()),
            "eval_mask": torch.from_numpy(ev.copy()),
            "cond_mask": torch.from_numpy(cond.copy()),
            # for the tau optimisation loader interface
            "x_obs": torch.from_numpy((x * cond).copy()),
            "mask": torch.from_numpy(cond.copy()),
        }
        if self.timestamps is not None:
            item["timestamps"] = torch.from_numpy(self.timestamps[sl].copy())
        return item


def build_splits(
    x: np.ndarray,
    observed_mask: np.ndarray | None = None,
    window: int = 24,
    stride: int = 1,
    missing_strategy: str = "point",
    rho: float = 0.2,
    seed: int = 0,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    timestamps: np.ndarray | None = None,
    max_windows: int | None = None,
):
    """Returns (train_ds, val_ds, test_ds, scaler, info).

    Windows are assigned to splits in contiguous blocks so train/val/test do not
    overlap in time, then the train indices are shuffled by the sampler.
    """
    x = np.asarray(x, dtype=np.float64)
    n, total = x.shape
    if observed_mask is None:
        observed_mask = (~np.isnan(x)).astype(np.float64)
    observed_mask = np.asarray(observed_mask, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0)

    eval_mask = make_eval_mask(observed_mask, missing_strategy, rho, seed=seed)

    starts = np.arange(0, total - window + 1, stride)
    if max_windows is not None and len(starts) > max_windows:
        # thin out uniformly; keeps temporal coverage while shrinking cost
        keep = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = starts[np.unique(keep)]
    n_tr = int(round(ratios[0] * len(starts)))
    n_va = int(round(ratios[1] * len(starts)))
    tr_idx = starts[:n_tr]
    va_idx = starts[n_tr : n_tr + n_va]
    te_idx = starts[n_tr + n_va :]

    # scaler from training-window observed entries only
    tr_end = int(tr_idx[-1]) + window if len(tr_idx) else total
    cond_train = observed_mask[:, :tr_end] * (1.0 - eval_mask[:, :tr_end])
    vals = x[:, :tr_end][cond_train > 0]
    scaler = StandardScaler(vals.mean(), vals.std())

    common = dict(window=window, scaler=scaler, timestamps=timestamps)
    train_ds = SpatioTemporalWindows(x, observed_mask, eval_mask, tr_idx, **common)
    val_ds = SpatioTemporalWindows(x, observed_mask, eval_mask, va_idx, **common)
    test_ds = SpatioTemporalWindows(x, observed_mask, eval_mask, te_idx, **common)
    info = {
        "n_nodes": n,
        "n_timesteps": total,
        "n_windows": len(starts),
        "split_sizes": (len(tr_idx), len(va_idx), len(te_idx)),
        "observed_rate": float(observed_mask.mean()),
        "eval_rate": float(eval_mask.mean()),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
    }
    return train_ds, val_ds, test_ds, scaler, info


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 32, num_workers: int = 0):
    kw = dict(num_workers=num_workers, pin_memory=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **kw),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
    )
