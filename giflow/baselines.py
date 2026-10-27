"""Non-parametric baselines (Appendix C.2) -- cheap sanity checks for the model.

Mean-S, Mean-T, Linear interpolation, KNN (neighbour average) and FP (feature
propagation, Rossi et al. 2022). All operate on one window at a time and only
see the conditioning mask.
"""
from __future__ import annotations

import numpy as np

from .metrics import MetricAccumulator


def mean_spatial(x, cond):
    num = (x * cond).sum(0, keepdims=True)
    den = cond.sum(0, keepdims=True)
    fill = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.broadcast_to(fill, x.shape).copy()


def mean_temporal(x, cond):
    num = (x * cond).sum(1, keepdims=True)
    den = cond.sum(1, keepdims=True)
    fill = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return np.broadcast_to(fill, x.shape).copy()


def linear_interp(x, cond):
    out = x.copy()
    r = x.shape[1]
    grid = np.arange(r)
    for i in range(x.shape[0]):
        known = cond[i] > 0
        if known.sum() == 0:
            out[i] = 0.0
        elif known.sum() == 1:
            out[i] = x[i, known][0]
        else:
            out[i] = np.interp(grid, grid[known], x[i, known])
    return out


def knn_spatial(x, cond, adj):
    w = adj
    num = w @ (x * cond)
    den = w @ cond
    out = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    # nodes with no informative neighbour fall back to the spatial mean
    fallback = mean_spatial(x, cond)
    return np.where(den > 0, out, fallback)


def feature_propagation(x, cond, adj_norm, n_iter: int = 40):
    """Rossi et al. (2022): iterate X <- S X, resetting the known entries."""
    out = x * cond
    for _ in range(n_iter):
        out = adj_norm @ out
        out = cond * x + (1.0 - cond) * out
    return out


ALL = ("mean_s", "mean_t", "linear", "knn", "fp")


def run_baselines(loader, scaler, adj, adj_norm, which=ALL, dataset_names=None) -> dict:
    joint = isinstance(scaler, (list, tuple))
    n_datasets = len(scaler) if joint else 1
    accs = {k: [MetricAccumulator() for _ in range(n_datasets)] for k in which}
    for batch in loader:
        xb = batch["x_true"].numpy()
        cb = batch["cond_mask"].numpy()
        eb = batch["eval_mask"].numpy()
        for b in range(xb.shape[0]):
            x, cond, ev = xb[b], cb[b], eb[b]
            dataset_id = int(batch["dataset_id"][b]) if joint else 0
            preds = {}
            if "mean_s" in which:
                preds["mean_s"] = mean_spatial(x, cond)
            if "mean_t" in which:
                preds["mean_t"] = mean_temporal(x, cond)
            if "linear" in which:
                preds["linear"] = linear_interp(x, cond)
            if "knn" in which:
                preds["knn"] = knn_spatial(x, cond, adj)
            if "fp" in which:
                preds["fp"] = feature_propagation(x, cond, adj_norm)
            for k, p in preds.items():
                dataset_scaler = scaler[dataset_id] if joint else scaler
                accs[k][dataset_id].update(
                    dataset_scaler.inverse_transform(p),
                    dataset_scaler.inverse_transform(x),
                    ev,
                )
    results = {
        name: [acc.compute() for acc in accumulators]
        for name, accumulators in accs.items()
    }
    if not joint:
        return {name: values[0] for name, values in results.items()}
    keys = ("pcc", "nmse", "psnr", "snr", "mae", "rmse", "mape")
    labels = dataset_names or tuple(str(i) for i in range(n_datasets))
    output = {}
    for name, per_dataset in results.items():
        aggregate = {
            key: float(np.nanmean([value[key] for value in per_dataset]))
            for key in keys
        }
        aggregate["n"] = sum(value["n"] for value in per_dataset)
        aggregate["per_dataset"] = dict(zip(labels, per_dataset))
        output[name] = aggregate
    return output
