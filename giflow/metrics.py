"""Masked EEG reconstruction metrics, evaluated on artificially removed entries."""
from __future__ import annotations

import numpy as np
import torch


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def imputation_metrics(pred, target, mask, mape_eps: float = 1e-2) -> dict:
    """Score hidden entries; PCC is pooled and PSNR uses their target range."""
    acc = MetricAccumulator(mape_eps=mape_eps)
    acc.update(pred, target, mask)
    return acc.compute()


class MetricAccumulator:
    """Streams batch-wise sums so metrics match a single pass over the full test set."""

    def __init__(self, mape_eps: float = 1e-2):
        self.mape_eps = mape_eps
        self.abs_sum = 0.0
        self.sq_sum = 0.0
        self.pred_sq_sum = 0.0
        self.target_sq_sum = 0.0
        self.pct_sum = 0.0
        self.pred_sum = 0.0
        self.target_sum = 0.0
        self.pred_target_sum = 0.0
        self.target_min = float("inf")
        self.target_max = float("-inf")
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
        self.pred_sq_sum += float((p ** 2).sum())
        self.target_sq_sum += float((t ** 2).sum())
        self.pred_sum += float(p.sum())
        self.target_sum += float(t.sum())
        self.pred_target_sum += float((p * t).sum())
        self.target_min = min(self.target_min, float(t.min()))
        self.target_max = max(self.target_max, float(t.max()))
        self.n += int(sel.sum())
        ok = np.abs(t) > self.mape_eps
        if ok.sum():
            self.pct_sum += float(np.abs(err[ok] / t[ok]).sum())
            self.n_pct += int(ok.sum())

    def compute(self) -> dict:
        if self.n == 0:
            return {
                "pcc": float("nan"), "nmse": float("nan"), "psnr": float("nan"),
                "snr": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "mape": float("nan"), "n": 0,
            }
        mse = self.sq_sum / self.n
        pred_var = self.pred_sq_sum - self.pred_sum ** 2 / self.n
        target_var = self.target_sq_sum - self.target_sum ** 2 / self.n
        covariance = self.pred_target_sum - self.pred_sum * self.target_sum / self.n
        pcc_denom = np.sqrt(max(pred_var, 0.0) * max(target_var, 0.0))
        pcc = covariance / pcc_denom if pcc_denom > 0 else float("nan")
        nmse = self.sq_sum / self.target_sq_sum if self.target_sq_sum > 0 else float("nan")
        data_range = self.target_max - self.target_min
        psnr = (
            float("inf") if mse == 0 and data_range > 0
            else 20.0 * np.log10(data_range / np.sqrt(mse))
            if mse > 0 and data_range > 0 else float("nan")
        )
        snr = (
            float("inf") if self.sq_sum == 0 and self.target_sq_sum > 0
            else 10.0 * np.log10(self.target_sq_sum / self.sq_sum)
            if self.sq_sum > 0 and self.target_sq_sum > 0 else float("nan")
        )
        return {
            "pcc": float(pcc),
            "nmse": float(nmse),
            "psnr": float(psnr),
            "snr": float(snr),
            "mae": self.abs_sum / self.n,
            "rmse": float(np.sqrt(mse)),
            "mape": (self.pct_sum / self.n_pct * 100.0) if self.n_pct else float("nan"),
            "n": self.n,
        }


def aggregate_trials(results: list[dict]) -> dict:
    """mean +/- std over independent seeds, as reported in the paper's tables."""
    out = {}
    for key in ("pcc", "nmse", "psnr", "snr", "mae", "rmse", "mape"):
        vals = np.array([r[key] for r in results], dtype=float)
        out[key] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return out
