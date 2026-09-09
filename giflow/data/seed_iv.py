"""SEED-IV EEG loader for GiFlow.

SEED-IV is an EEG emotion dataset: 62 electrodes sampled at 250 Hz. The
preprocessed form used here is one array per subject-session,

    X_prc1.npy   (n_windows, 62, 1000)     4-second windows
    labels.npy   (n_windows,)              emotion label 0..3

which maps onto the paper's setting with the 62 electrodes as graph nodes and
the samples within a window as timesteps -- i.e. imputing dropped EEG channels
or corrupted stretches of signal.

Two things differ from the air-quality / traffic datasets:

  * A window is 1000 samples long. Temporal attention is O(R^2) per node, so
    1000 is not usable directly; windows are re-cut into `window` -length pieces
    (`--window 100` is a sane starting point, vs the paper's 24).
  * There is no natural station-distance graph. The spatial graph is built from
    the correlation between electrodes over the training portion, which is the
    standard fallback for EEG. Pass `montage_positions` if you have 10-20
    electrode coordinates and want a distance graph instead.

Layout expected under `root` (either is accepted):

    root/<session>/<subject>/X_prc1.npy        # the raw download
    root/seed_iv_<...>.npz                     # the packed form scripts/pack_seed_iv.py makes
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .datasets import DatasetBundle
from .graph import distance_matrix_adjacency, gaussian_kernel_adjacency

# SEED-IV 62-channel order (ESI NeuroScan), used only for reporting
SEED_IV_CHANNELS = (
    "FP1 FPZ FP2 AF3 AF4 F7 F5 F3 F1 FZ F2 F4 F6 F8 FT7 FC5 FC3 FC1 FCZ FC2 FC4 "
    "FC6 FT8 T7 C5 C3 C1 CZ C2 C4 C6 T8 TP7 CP5 CP3 CP1 CPZ CP2 CP4 CP6 TP8 P7 "
    "P5 P3 P1 PZ P2 P4 P6 P8 PO7 PO5 PO3 POZ PO4 PO6 PO8 CB1 O1 OZ O2 CB2"
).split()


def _correlation_adjacency(x: np.ndarray, threshold: float = 0.1, binary: bool = True):
    """Electrode graph from the absolute correlation between channel time series.

    `x` is (N, T). Correlation is turned into a distance (1 - |corr|) and then
    passed through the same Gaussian-kernel + threshold pipeline the paper uses
    for stations, so the `--threshold` semantics stay consistent.
    """
    xc = x - x.mean(axis=1, keepdims=True)
    sd = xc.std(axis=1)
    sd[sd < 1e-12] = 1e-12
    corr = (xc @ xc.T) / (len(x[0]) * np.outer(sd, sd))
    dist = 1.0 - np.abs(np.clip(corr, -1.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    return distance_matrix_adjacency(dist, threshold=threshold, binary=binary)


def _discover(root: Path):
    """Find every X_prc1.npy under root, returning (session, subject, path)."""
    found = []
    for p in sorted(root.rglob("X_prc1.npy")):
        rel = p.relative_to(root).parts
        subject = rel[-2] if len(rel) >= 2 else p.parent.name
        session = rel[-3] if len(rel) >= 3 else "0"
        found.append((session, subject, p))
    return found


def load_seed_iv(
    root: str | Path = "data/seed4",
    window: int = 100,
    subjects: tuple[str, ...] | None = None,
    sessions: tuple[str, ...] | None = None,
    max_windows_per_subject: int | None = None,
    threshold: float = 0.1,
    montage_positions: np.ndarray | None = None,
    channel_dropout: bool = False,
    verbose: bool = True,
) -> DatasetBundle:
    """Concatenate SEED-IV windows into one (62, T) signal GiFlow can window over.

    Args:
        window: only used to trim T to a multiple of the window length.
        subjects / sessions: restrict which recordings are loaded. A single
            subject is usually the right call first -- concatenating subjects
            splices unrelated recordings end to end.
        max_windows_per_subject: cap for quick runs.
        channel_dropout: if True, mark whole channels missing at random in the
            observed mask, which is the realistic EEG failure mode (a dead
            electrode) rather than scattered point dropout.
    """
    root = Path(root)
    if root.is_file() and root.suffix == ".npz":
        z = np.load(root, allow_pickle=True)
        x = z["x"].astype(np.float64)
        adj = z["adj"].astype(np.float64) if "adj" in z else None
        mask = z["mask"].astype(np.float64) if "mask" in z else np.ones_like(x)
        name = "SEED-IV(packed:%s)" % root.stem
        if adj is None:
            adj = _correlation_adjacency(x, threshold)
        return DatasetBundle(name, x, mask, adj)

    if not root.exists():
        raise FileNotFoundError(
            "SEED-IV not found at %s. Expected <session>/<subject>/X_prc1.npy "
            "underneath it, or a packed .npz from scripts/pack_seed_iv.py." % root
        )

    found = _discover(root)
    if not found:
        raise FileNotFoundError("no X_prc1.npy found under %s" % root)
    if sessions:
        found = [f for f in found if f[0] in sessions]
    if subjects:
        found = [f for f in found if f[1] in subjects]
    if not found:
        raise ValueError("no recordings left after filtering by session/subject")

    chunks = []
    used = []
    for session, subject, path in found:
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0] if max_windows_per_subject is None else min(
            arr.shape[0], max_windows_per_subject
        )
        a = np.asarray(arr[:n], dtype=np.float64)     # (n, 62, 1000)
        # (n, C, T) -> (C, n*T): lay the windows end to end along time
        chunks.append(a.transpose(1, 0, 2).reshape(a.shape[1], -1))
        used.append("%s/%s[%d]" % (session, subject, n))
    x = np.concatenate(chunks, axis=1)

    # trim to a whole number of windows
    total = (x.shape[1] // window) * window
    x = x[:, :total]

    if montage_positions is not None:
        adj = gaussian_kernel_adjacency(montage_positions, threshold=threshold)
    else:
        # build the graph on the first 70% only, so the test split does not
        # leak into the graph the model is given
        adj = _correlation_adjacency(x[:, : int(0.7 * x.shape[1])], threshold)

    mask = np.ones_like(x)
    if channel_dropout:
        rng = np.random.default_rng(0)
        dead = rng.choice(x.shape[0], size=max(1, x.shape[0] // 20), replace=False)
        mask[dead] = 0.0

    if verbose:
        print("[seed-iv] %d recording(s): %s" % (len(used), ", ".join(used[:6])
              + (" ..." if len(used) > 6 else "")))
        print("[seed-iv] signal (%d channels, %d samples), avg degree %.2f"
              % (x.shape[0], x.shape[1], adj.sum(1).mean()))

    return DatasetBundle("SEED-IV", x, mask, adj)
