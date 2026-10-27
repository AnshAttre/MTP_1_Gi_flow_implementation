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

import re
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


def _subject_id(path: Path) -> str | None:
    match = re.match(r"^(\d+)(?:_|$)", path.stem)
    return str(int(match.group(1))) if match else None


def _requested_subject_ids(subjects: tuple[str, ...] | None) -> set[str] | None:
    if subjects is None:
        return None
    normalized = set()
    for subject in subjects:
        match = re.match(r"^(?:s)?0*(\d+)$", str(subject), re.IGNORECASE)
        normalized.add(str(int(match.group(1))) if match else str(subject))
    return normalized


def _channel_names(n_channels: int):
    if n_channels != len(SEED_IV_CHANNELS):
        raise ValueError(
            "SEED-IV recording has %d channels but no channel-name metadata; "
            "cannot safely align it to another dataset" % n_channels
        )
    return tuple(SEED_IV_CHANNELS)


def load_seed_iv(
    root: str | Path = "data/seed4",
    window: int = 100,
    subjects: tuple[str, ...] | None = None,
    sessions: tuple[str, ...] | None = None,
    max_windows_per_subject: int | None = None,
    max_recordings: int | None = None,
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
    if window < 1:
        raise ValueError("window must be positive")
    if root.is_file() and root.suffix == ".npz":
        z = np.load(root, allow_pickle=True)
        x = z["x"].astype(np.float32)
        adj = z["adj"].astype(np.float64) if "adj" in z else None
        mask = z["mask"].astype(np.float64) if "mask" in z else np.ones_like(x)
        name = "SEED-IV(packed:%s)" % root.stem
        if adj is None:
            adj = _correlation_adjacency(x, threshold)
        channel_names = (
            tuple(v.decode() if isinstance(v, bytes) else str(v)
                  for v in z["channel_names"].tolist())
            if "channel_names" in z else _channel_names(x.shape[0])
        )
        segments = None
        if "recording_lengths" in z:
            lengths = z["recording_lengths"].astype(int).tolist()
            boundaries = np.cumsum([0] + lengths)
            segments = tuple(zip(boundaries[:-1], boundaries[1:]))
        return DatasetBundle(name, x, mask, adj, channel_names=channel_names, segments=segments)

    if not root.exists():
        raise FileNotFoundError(
            "SEED-IV not found at %s. Expected <session>/<subject>/X_prc1.npy "
            "underneath it, or a packed .npz from scripts/pack_seed_iv.py." % root
        )

    found = _discover(root)
    if not found:
        mat_paths = sorted(root.rglob("*.mat"))
        if not mat_paths:
            raise FileNotFoundError("no X_prc1.npy or .mat recordings found under %s" % root)
        wanted_subjects = _requested_subject_ids(subjects)
        if wanted_subjects is not None:
            mat_paths = [
                path for path in mat_paths
                if _subject_id(path) in wanted_subjects
            ]
        if sessions:
            wanted_sessions = set(map(str, sessions))
            mat_paths = [
                path for path in mat_paths
                if path.parent.name in wanted_sessions
            ]
        if max_recordings is not None:
            if max_recordings < 1:
                raise ValueError("max_recordings must be positive")
            mat_paths = mat_paths[:max_recordings]
        if not mat_paths:
            raise ValueError("no MATLAB recordings left after filtering by session/subject")

        from scipy.io import loadmat, whosmat

        # Inspect MATLAB headers first, then choose evenly-spaced windows per
        # participant. Only selected trial variables are loaded on the second
        # pass, keeping the joint DEAP/SEED experiment memory-bounded.
        inventory = []
        subject_windows: dict[str, list[tuple[int, str, int]]] = {}
        for path_index, path in enumerate(mat_paths):
            subject_id = _subject_id(path)
            if subject_id is None:
                raise ValueError(
                    "cannot determine a participant ID from MATLAB filename %s"
                    % path.name
                )
            trials = []
            for variable, shape, _ in whosmat(path):
                if not re.fullmatch(r"[a-z0-9]+_eeg\d+", variable, re.IGNORECASE):
                    continue
                if len(shape) != 2 or shape.count(len(SEED_IV_CHANNELS)) != 1:
                    continue
                channel_axis = shape.index(len(SEED_IV_CHANNELS))
                n_samples = shape[1 - channel_axis]
                n_windows = n_samples // window
                if n_windows:
                    trials.append((variable, channel_axis, n_windows))
                    if max_windows_per_subject is not None:
                        subject_windows.setdefault(subject_id, []).extend(
                            (path_index, variable, local_index)
                            for local_index in range(n_windows)
                        )
            if trials:
                inventory.append((path_index, path, subject_id, trials))
        if not inventory:
            raise ValueError(
                "no 62-channel EEG trial variables found in MATLAB files under %s"
                % root
            )
        if max_windows_per_subject is not None:
            if max_windows_per_subject < 1:
                raise ValueError("max_windows_per_subject must be positive")
            selected_by_subject = {}
            for subject_id, candidates in subject_windows.items():
                count = min(max_windows_per_subject, len(candidates))
                selected_indices = np.linspace(
                    0, len(candidates) - 1, count
                ).round().astype(int)
                selected_by_subject[subject_id] = {
                    candidates[index] for index in selected_indices
                }
        else:
            selected_by_subject = None

        chunks, segments, segment_group_ids = [], [], []
        cursor = 0
        for path_index, path, subject_id, trials in inventory:
            selected = (
                None if selected_by_subject is None
                else selected_by_subject[subject_id]
            )
            variables = (
                [variable for variable, _, _ in trials]
                if selected is None
                else sorted({
                    variable for index, variable, _ in selected
                    if index == path_index
                })
            )
            values_by_variable = (
                loadmat(path, variable_names=variables) if variables else {}
            )
            for variable, channel_axis, n_windows in trials:
                chosen_windows = (
                    range(n_windows)
                    if selected is None
                    else [
                        local_index for local_index in range(n_windows)
                        if (path_index, variable, local_index) in selected
                    ]
                )
                if not chosen_windows:
                    continue
                values = values_by_variable[variable]
                if channel_axis == 1:
                    values = values.T
                values = np.asarray(values, dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError("%s:%s contains non-finite EEG samples" % (path, variable))
                for local_index in chosen_windows:
                    chunk = values[
                        :, local_index * window : (local_index + 1) * window
                    ]
                    chunks.append(chunk)
                    segments.append((cursor, cursor + chunk.shape[1]))
                    segment_group_ids.append(subject_id)
                    cursor += chunk.shape[1]
        x = np.concatenate(chunks, axis=1)
        adj = _correlation_adjacency(x[:, : int(0.7 * x.shape[1])], threshold)
        mask = np.ones_like(x)
        if channel_dropout:
            rng = np.random.default_rng(0)
            dead = rng.choice(x.shape[0], size=max(1, x.shape[0] // 20), replace=False)
            mask[dead] = 0.0
        if verbose:
            print("[seed-iv] loaded %d MATLAB recordings (%d channels, %d samples)"
                  % (len(inventory), x.shape[0], x.shape[1]))
        return DatasetBundle(
            "SEED-IV-MAT", x, mask, adj, channel_names=_channel_names(x.shape[0]),
            segments=segments,
            segment_group_ids=(
                None if any(group is None for group in segment_group_ids)
                else segment_group_ids
            ),
        )
    if sessions:
        found = [f for f in found if f[0] in sessions]
    if subjects:
        found = [f for f in found if f[1] in subjects]
    if not found:
        raise ValueError("no recordings left after filtering by session/subject")

    chunks = []
    used = []
    segments = []
    segment_group_ids = []
    cursor = 0
    for session, subject, path in found:
        arr = np.load(path, mmap_mode="r")
        if max_windows_per_subject is None:
            selected = np.arange(arr.shape[0])
        else:
            if max_windows_per_subject < 1:
                raise ValueError("max_windows_per_subject must be positive")
            n = min(arr.shape[0], max_windows_per_subject)
            selected = np.linspace(0, arr.shape[0] - 1, n).round().astype(int)
        a = np.asarray(arr[selected], dtype=np.float32)     # (n, 62, time)
        # (n, C, T) -> (C, n*T): lay the windows end to end along time
        chunk = a.transpose(1, 0, 2).reshape(a.shape[1], -1)
        chunks.append(chunk)
        segments.append((cursor, cursor + chunk.shape[1]))
        subject_id = subject.split("_", 1)[0]
        segment_group_ids.append(subject_id)
        cursor += chunk.shape[1]
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

    return DatasetBundle(
        "SEED-IV", x, mask, adj, channel_names=_channel_names(x.shape[0]),
        segments=segments, segment_group_ids=segment_group_ids,
    )
