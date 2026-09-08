"""MAE / RMSE / MAPE evaluated only on the artificially removed entries."""
from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def imputation_metrics(pred, target, mask, mape_eps: float = 1e-2) -> dict:
    """`mask` = 1 on the entries that were removed and must be scored.

    MAPE is reported in percent and skips targets whose magnitude is below
    `mape_eps`, which otherwise blow the ratio up.
    """
    pred, target, mask = _to_numpy(pred), _to_numpy(target), _to_numpy(mask)
    sel = mask > 0
    if sel.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0}
    p, t = pred[sel], target[sel]
    err = p - t
    mae = float(np.abs(err).mean())
    rmse = float(np.sqrt((err ** 2).mean()))
    ok = np.abs(t) > mape_eps
    mape = float((np.abs(err[ok] / t[ok])).mean() * 100.0) if ok.sum() else float("nan")
    return {"mae": mae, "rmse": rmse, "mape": mape, "n": int(sel.sum())}


class MetricAccumulator:
    """Streams batch-wise sums so metrics match a single pass over the full test set."""

    def __init__(self, mape_eps: float = 1e-2):
        self.mape_eps = mape_eps
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.pct_sum = 0.0
        self.n = 0
        self.n_pct = 0

    def update(self, pred, target, mask):
        pred, target, mask = _to_numpy(pred), _to_numpy(target), _to_numpy(mask)
        sel = mask > 0
        if sel.sum() == 0:
            return
        p, t = pred[sel], target[sel]
        err = p - t
        self.abs_sum += float(np.abs(err).sum())
        self.sq_sum += float((err ** 2).sum())
        self.n += int(sel.sum())
        ok = np.abs(t) > self.mape_eps
        if ok.sum():
            self.pct_sum += float(np.abs(err[ok] / t[ok]).sum())
            self.n_pct += int(ok.sum())

    def compute(self) -> dict:
        if self.n == 0:
            return {"mae": float("nan"), "rmse": float("nan"), "mape": float("nan"), "n": 0}
        return {
            "mae": self.abs_sum / self.n,
            "rmse": float(np.sqrt(self.sq_sum / self.n)),
            "mape": (self.pct_sum / self.n_pct * 100.0) if self.n_pct else float("nan"),
            "n": self.n,
        }


def aggregate_trials(results: list[dict]) -> dict:
    """mean +/- std over independent seeds, as reported in the paper's tables."""
    out = {}
    for key in ("mae", "rmse", "mape"):
        vals = np.array([r[key] for r in results], dtype=float)
        out[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out
