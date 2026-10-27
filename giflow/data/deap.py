"""DEAP raw BioSemi BDF loader for graph-based EEG imputation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .datasets import DatasetBundle
from .seed_iv import _correlation_adjacency

DEAP_EEG_CHANNELS = (
    "FP1 AF3 F7 F3 FC1 FC5 T7 C3 CP1 CP5 P7 P3 PZ PO3 O1 OZ "
    "O2 PO4 P4 P8 CP6 CP2 C4 T8 FC6 FC2 F4 F8 AF4 FP2 FZ CZ"
).split()


def load_deap(
    root: str | Path = "data/DEAP-RAW",
    target_sfreq: float = 200.0,
    threshold: float = 0.1,
    subjects: tuple[str, ...] | None = None,
    max_windows_per_subject: int | None = None,
    window: int = 100,
    verbose: bool = True,
) -> DatasetBundle:
    """Load subject BDF files, retaining EEG channels and subject boundaries.

    DEAP BDF recordings are resampled to ``target_sfreq`` before the runner
    creates fixed-length windows. Only the 32 named EEG channels are retained;
    peripheral BioSemi channels and the status trigger channel are excluded.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError("DEAP raw BDF directory not found: %s" % root)
    paths = sorted(root.glob("s*.bdf"))
    if subjects:
        wanted = {str(subject).lower() for subject in subjects}
        paths = [
            path for path in paths
            if path.stem.lower() in wanted or path.stem[1:].lower() in wanted
        ]
    if not paths:
        raise FileNotFoundError("no DEAP s*.bdf recordings found under %s" % root)

    try:
        import mne
    except ImportError as exc:
        raise ImportError(
            "DEAP BDF loading requires MNE; install the project requirements."
        ) from exc

    chunks: list[np.ndarray] = []
    segments: list[tuple[int, int]] = []
    group_ids: list[str] = []
    cursor = 0
    for path in paths:
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        by_canonical = {name.strip().upper(): name for name in raw.ch_names}
        missing = sorted(set(DEAP_EEG_CHANNELS) - set(by_canonical))
        if missing:
            raise ValueError(
                "%s is missing required DEAP EEG channels: %s"
                % (path.name, ", ".join(missing))
            )
        raw.pick([by_canonical[name] for name in DEAP_EEG_CHANNELS])
        if not np.isclose(raw.info["sfreq"], target_sfreq):
            raw.resample(target_sfreq, npad="auto", verbose="ERROR")
        values = np.asarray(raw.get_data(), dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError("%s contains non-finite EEG values after reading" % path.name)
        if max_windows_per_subject is None:
            chunks.append(values)
            segments.append((cursor, cursor + values.shape[1]))
            group_ids.append(path.stem[1:])
            cursor += values.shape[1]
        else:
            if max_windows_per_subject < 1:
                raise ValueError("max_windows_per_subject must be positive")
            candidate_starts = np.arange(
                0, values.shape[1] - window + 1, window, dtype=np.int64
            )
            n_keep = min(max_windows_per_subject, len(candidate_starts))
            selected = candidate_starts[
                np.linspace(0, len(candidate_starts) - 1, n_keep).round().astype(int)
            ]
            for start in np.unique(selected):
                chunk = values[:, start : start + window]
                chunks.append(chunk)
                segments.append((cursor, cursor + chunk.shape[1]))
                group_ids.append(path.stem[1:])
                cursor += chunk.shape[1]
        if verbose:
            print(
                "[deap] %s: %d EEG channels, %d samples at %.1f Hz"
                % (path.name, values.shape[0], values.shape[1], target_sfreq)
            )
        raw.close()

    x = np.concatenate(chunks, axis=1)
    adj = _correlation_adjacency(x[:, : int(0.7 * x.shape[1])], threshold)
    if verbose:
        print(
            "[deap] loaded %d subject recordings (%d channels, %d samples)"
            % (len(chunks), x.shape[0], x.shape[1])
        )
    return DatasetBundle(
        "DEAP-RAW", x, np.ones_like(x), adj, channel_names=DEAP_EEG_CHANNELS,
        segments=segments, segment_group_ids=group_ids,
    )
