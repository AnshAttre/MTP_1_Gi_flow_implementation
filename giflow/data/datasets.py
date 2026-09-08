"""Dataset loading + graph construction, returning everything `fit` needs.

Synthetic works out of the box. The three real datasets (Appendix C.1) need their
files placed under `data/`; each loader below lists the filenames it accepts.

  Air-36  : 36 PM2.5 stations in Beijing, 8760 hourly steps
  AQI     : 437 PM2.5 stations across 43 Chinese cities, 8760 hourly steps
  PeMS08  : 170 traffic sensors in California, 17856 steps at 5-minute intervals
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .graph import (
    distance_matrix_adjacency,
    gaussian_kernel_adjacency,
    gcn_normalize,
    knn_adjacency,
    laplacian,
    line_graph_adjacency,
)
from .synthetic import generate_synthetic

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class DatasetBundle:
    """Container for the signal, masks, graphs and metadata of one dataset."""

    def __init__(self, name, x, observed_mask, adj_s, timestamps=None, ts_classes=()):
        self.name = name
        self.x = np.asarray(x, dtype=np.float64)          # (N, R_total)
        self.observed_mask = np.asarray(observed_mask, dtype=np.float64)
        self.adj_s = np.asarray(adj_s, dtype=np.float64)  # (N, N)
        self.timestamps = timestamps                      # (R_total, n_fields) int codes
        self.ts_classes = tuple(ts_classes)

    def graphs(self, window: int, normalized_laplacian: bool = False):
        """Spatial/temporal Laplacians and GCN-normalised adjacencies."""
        adj_t = line_graph_adjacency(window)
        return {
            "adj_s": self.adj_s,
            "lap_s": laplacian(self.adj_s, normalized_laplacian),
            "lap_t": laplacian(adj_t, normalized_laplacian),
            "adj_s_norm": gcn_normalize(self.adj_s),
            "adj_t_norm": gcn_normalize(adj_t),
        }

    def summary(self) -> str:
        deg = self.adj_s.sum(1)
        return (
            "%s: N=%d, R=%d, observed=%.1f%%, avg degree=%.2f"
            % (
                self.name,
                self.x.shape[0],
                self.x.shape[1],
                100 * self.observed_mask.mean(),
                deg.mean(),
            )
        )


def _first_existing(names, root: Path):
    for n in names:
        p = root / n
        if p.exists():
            return p
    return None


def _hour_dow_codes(n_steps: int, start_hour: int = 0):
    """Fallback timestamp codes (hour-of-day, day-of-week) for hourly data."""
    idx = np.arange(n_steps) + start_hour
    return np.stack([idx % 24, (idx // 24) % 7], axis=1), (24, 7)


def load_synthetic(n_nodes=50, n_steps=3000, sigma=0.1, k=5, seed=0) -> DatasetBundle:
    x, adj, _ = generate_synthetic(
        n_nodes=n_nodes, n_steps=n_steps, k=k, sigma=sigma, seed=seed
    )
    return DatasetBundle(
        "synthetic(N=%d,sigma=%.1f)" % (n_nodes, sigma), x, np.ones_like(x), adj
    )


def load_air(variant="air36", threshold=0.1, root: Path | None = None) -> DatasetBundle:
    """Air-36 / AQI. Accepts an .h5 with a DataFrame, or an .npz with x/mask/dist.

    Expected shapes: values (R, N); optional `dist` (N, N) real station distances,
    otherwise the graph falls back to a kernel over the value-correlation profile.
    """
    root = root or DATA_DIR
    if variant == "air36":
        cands = ["small36.h5", "small_36.h5", "air36.h5", "air36.npz", "AirQuality36.h5"]
        name = "Air-36"
    else:
        cands = ["full437.h5", "air437.h5", "aqi.h5", "aqi.npz", "AirQuality437.h5"]
        name = "AQI"
    path = _first_existing(cands, root)
    if path is None:
        raise FileNotFoundError(
            "%s not found. Place one of %s under %s.\n"
            "The air-quality data comes from Yi et al. (2016) / Zheng et al. (2015); "
            "the GRIN release (github.com/Graph-Machine-Learning-Group/grin) ships "
            "small36.h5 and full437.h5." % (name, cands, root)
        )

    dist = None
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        vals = z["x"] if "x" in z else z["data"]
        mask = z["mask"] if "mask" in z else (~np.isnan(vals)).astype(float)
        dist = z["dist"] if "dist" in z else None
        stamps = None
    else:
        import pandas as pd

        df = pd.read_hdf(path)
        vals = df.to_numpy(dtype=np.float64)
        mask = (~np.isnan(vals)).astype(np.float64)
        stamps = None
        try:
            hours = df.index.hour.to_numpy()
            dows = df.index.dayofweek.to_numpy()
            stamps = np.stack([hours, dows], axis=1)
        except Exception:
            pass
        dp = _first_existing(["%s_dist.npy" % variant, "dist_%s.npy" % variant], root)
        if dp is not None:
            dist = np.load(dp)

    x = np.nan_to_num(vals, nan=0.0).T          # (N, R)
    mask = mask.T
    if dist is not None:
        adj = distance_matrix_adjacency(dist, threshold=threshold)
    else:
        # no coordinates available: build the kernel on the observed value profile
        prof = np.where(mask > 0, x, np.nan)
        prof = np.nan_to_num(prof - np.nanmean(prof, axis=1, keepdims=True), nan=0.0)
        adj = gaussian_kernel_adjacency(prof, threshold=threshold)
    if stamps is None:
        stamps, ts_classes = _hour_dow_codes(x.shape[1])
    else:
        ts_classes = (24, 7)
    return DatasetBundle(name, x, mask, adj, stamps, ts_classes)


def load_pems08(threshold=0.1, root: Path | None = None, feature=0) -> DatasetBundle:
    """PeMS08. Accepts PEMS08.npz with key `data` of shape (T, N, F)."""
    root = root or DATA_DIR
    path = _first_existing(["PEMS08.npz", "pems08.npz", "PeMS08.npz"], root)
    if path is None:
        raise FileNotFoundError(
            "PeMS08 not found. Place PEMS08.npz under %s (ASTGNN/STSGCN releases, "
            "e.g. github.com/guoshnBJTU/ASTGNN, ship it)." % root
        )
    z = np.load(path)
    data = z["data"] if "data" in z else z[list(z.keys())[0]]
    if data.ndim == 3:
        data = data[..., feature]
    x = np.asarray(data, dtype=np.float64).T          # (N, T)
    mask = (x != 0).astype(np.float64)                # zeros are sensor dropouts

    adj = None
    ap = _first_existing(["PEMS08.csv", "distance08.csv", "pems08_adj.npy"], root)
    if ap is not None and ap.suffix == ".npy":
        adj = distance_matrix_adjacency(np.load(ap), threshold=threshold)
    elif ap is not None:
        import pandas as pd

        df = pd.read_csv(ap)
        n = x.shape[0]
        d = np.full((n, n), np.inf)
        np.fill_diagonal(d, 0.0)
        cols = {c.lower(): c for c in df.columns}
        f, t, c = cols.get("from"), cols.get("to"), cols.get("cost", cols.get("distance"))
        for _, row in df.iterrows():
            i, j = int(row[f]), int(row[t])
            d[i, j] = d[j, i] = float(row[c]) if c else 1.0
        finite = d[np.isfinite(d)]
        d[~np.isfinite(d)] = finite.max() * 10 if len(finite) else 1.0
        adj = distance_matrix_adjacency(d, threshold=threshold)
    if adj is None:
        prof = x - x.mean(axis=1, keepdims=True)
        adj = gaussian_kernel_adjacency(prof, threshold=threshold)

    # 5-minute sampling: 288 slots per day
    idx = np.arange(x.shape[1])
    stamps = np.stack([idx % 288, (idx // 288) % 7], axis=1)
    return DatasetBundle("PeMS08", x, mask, adj, stamps, (288, 7))


def load_dataset(name: str, **kw) -> DatasetBundle:
    name = name.lower()
    if name.startswith("syn"):
        return load_synthetic(**kw)
    if name in ("air36", "air-36"):
        return load_air("air36", **kw)
    if name in ("aqi", "air437", "air-437"):
        return load_air("aqi", **kw)
    if name in ("pems08", "pems-08"):
        return load_pems08(**kw)
    raise ValueError("unknown dataset: %r" % name)
