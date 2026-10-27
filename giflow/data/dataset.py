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
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

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
        dataset_id: int = 0,
        node_presence: np.ndarray | None = None,
        group_ids: np.ndarray | None = None,
    ):
        self.x = np.asarray(x, dtype=np.float32)
        self.observed_mask = np.asarray(observed_mask, dtype=np.float32)
        self.eval_mask = np.asarray(eval_mask, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.window = int(window)
        self.scaler = scaler
        self.timestamps = None if timestamps is None else np.asarray(timestamps, dtype=np.int64)
        self.dataset_id = int(dataset_id)
        self.node_presence = np.asarray(
            np.ones(self.x.shape[0]) if node_presence is None else node_presence,
            dtype=np.float32,
        )
        self.group_ids = None if group_ids is None else np.asarray(group_ids, dtype=object)
        if self.group_ids is not None and self.group_ids.shape != self.indices.shape:
            raise ValueError("group_ids must contain one identifier per window")
        if self.node_presence.shape != (self.x.shape[0],):
            raise ValueError("node_presence must contain one value per node")

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
            "dataset_id": torch.tensor(self.dataset_id, dtype=torch.int64),
            "node_mask": torch.from_numpy(self.node_presence.copy()),
        }
        if self.timestamps is not None:
            item["timestamps"] = torch.from_numpy(self.timestamps[sl].copy())
        if self.group_ids is not None:
            item["group_id"] = str(self.group_ids[i])
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
    segments: tuple | None = None,
    dataset_id: int = 0,
    node_presence: np.ndarray | None = None,
    segment_group_ids: tuple[str, ...] | None = None,
    channel_ids: tuple[str, ...] | None = None,
    split_mode: str = "window",
    split_seed: int = 0,
    test_groups: tuple[str, ...] | None = None,
):
    """Returns (train_ds, val_ds, test_ds, scaler, info).

    Windows are assigned to splits in contiguous blocks so train/val/test do not
    overlap in time, then the train indices are shuffled by the sampler.
    """
    x = np.asarray(x, dtype=np.float32)
    n, total = x.shape
    if observed_mask is None:
        observed_mask = (~np.isnan(x)).astype(np.float64)
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)

    eval_mask = make_eval_mask(
        observed_mask, missing_strategy, rho, seed=seed, channel_ids=channel_ids
    )

    if split_mode not in ("window", "subject"):
        raise ValueError("split_mode must be 'window' or 'subject'")
    if split_mode == "subject" and (segments is None or segment_group_ids is None):
        raise ValueError(
            "subject split requires recording segments with subject group identifiers"
        )
    if segments is None:
        starts = np.arange(0, total - window + 1, stride)
        start_groups = None
    else:
        valid_starts = []
        valid_groups = []
        for segment_index, (start, end) in enumerate(segments):
            if end - start < window:
                continue
            segment_starts = np.arange(start, end - window + 1, stride)
            valid_starts.append(segment_starts)
            if segment_group_ids is not None:
                group_id = segment_group_ids[segment_index]
            else:
                group_id = None
            if split_mode == "subject":
                if group_id is None or str(group_id).lower() == "none":
                    raise ValueError(
                        "subject split cannot use recordings without subject IDs"
                    )
            if segment_group_ids is not None:
                valid_groups.extend([str(group_id)] * len(segment_starts))
        starts = np.concatenate(valid_starts) if valid_starts else np.empty(0, dtype=np.int64)
        start_groups = (
            np.asarray(valid_groups, dtype=object)
            if segment_group_ids is not None else None
        )
    if len(starts) == 0:
        raise ValueError("no complete windows available for the requested window and segments")
    if max_windows is not None and len(starts) > max_windows:
        if max_windows < 1:
            raise ValueError("max_windows must be positive")
        if start_groups is None:
            keep = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        else:
            # Retain representation from every participant when thinning.
            unique_groups = sorted(set(start_groups.tolist()))
            if max_windows < len(unique_groups):
                raise ValueError(
                    "max_windows must be at least the number of participant groups"
                )
            group_indices = {
                group_id: np.flatnonzero(start_groups == group_id)
                for group_id in unique_groups
            }
            sizes = np.asarray([len(group_indices[group]) for group in unique_groups])
            quotas = np.ones(len(unique_groups), dtype=int)
            remaining_budget = max_windows - len(unique_groups)
            if remaining_budget:
                proportional = remaining_budget * sizes / sizes.sum()
                additions = np.floor(proportional).astype(int)
                quotas += additions
                left = max_windows - int(quotas.sum())
                remainders = proportional - additions
                for index in np.argsort(-remainders)[:left]:
                    quotas[index] += 1
            keep_indices = []
            for group_index, group_id in enumerate(unique_groups):
                group_members = group_indices[group_id]
                count = min(quotas[group_index], len(group_members))
                selected = np.linspace(
                    0, len(group_members) - 1, count
                ).round().astype(int)
                keep_indices.extend(group_members[selected])
            keep = np.asarray(sorted(keep_indices), dtype=int)
        keep = np.unique(keep)
        starts = starts[keep]
        if start_groups is not None:
            start_groups = start_groups[keep]
    if split_mode == "window":
        n_tr = int(round(ratios[0] * len(starts)))
        n_va = int(round(ratios[1] * len(starts)))
        tr_idx = starts[:n_tr]
        va_idx = starts[n_tr : n_tr + n_va]
        te_idx = starts[n_tr + n_va :]
        split_groups = None
    else:
        grouped_starts = {
            group_id: starts[start_groups == group_id]
            for group_id in sorted(set(start_groups.tolist()))
        }
        all_groups = list(grouped_starts)
        if test_groups is not None:
            test_group_ids = [str(group_id) for group_id in test_groups]
            unknown = sorted(set(test_group_ids) - set(all_groups))
            if unknown:
                raise ValueError(
                    "requested held-out subject IDs are absent: %s" % ", ".join(unknown)
                )
        else:
            rng = np.random.default_rng(split_seed)
            order = rng.permutation(len(all_groups))
            n_test = max(1, int(round(ratios[2] * len(all_groups))))
            test_group_ids = [all_groups[i] for i in order[:n_test]]
        remaining = [group_id for group_id in all_groups if group_id not in test_group_ids]
        if len(remaining) < 2:
            raise ValueError("subject split needs at least two non-test groups")
        rng = np.random.default_rng(split_seed)
        validation_count = max(1, int(round(ratios[1] * len(all_groups))))
        validation_count = min(validation_count, len(remaining) - 1)
        validation_order = rng.permutation(len(remaining))
        validation_group_ids = [remaining[i] for i in validation_order[:validation_count]]
        train_group_ids = [
            group_id for group_id in remaining if group_id not in validation_group_ids
        ]
        tr_idx = np.concatenate([grouped_starts[g] for g in train_group_ids])
        va_idx = np.concatenate([grouped_starts[g] for g in validation_group_ids])
        te_idx = np.concatenate([grouped_starts[g] for g in test_group_ids])
        split_groups = {
            "train": train_group_ids,
            "validation": validation_group_ids,
            "test": test_group_ids,
        }

    # Subject splits can be non-contiguous in the concatenated signal. Fit only
    # on the selected training windows rather than the prefix up to their end.
    if split_mode == "subject":
        value_sum = 0.0
        square_sum = 0.0
        value_count = 0
        for start in tr_idx:
            sl = slice(int(start), int(start) + window)
            train_mask = observed_mask[:, sl] * (1.0 - eval_mask[:, sl])
            vals = x[:, sl][train_mask > 0]
            value_sum += float(vals.sum(dtype=np.float64))
            square_sum += float(np.square(vals, dtype=np.float64).sum())
            value_count += len(vals)
        if value_count == 0:
            raise ValueError("training split has no observed values for normalization")
        mean = value_sum / value_count
        std = max(square_sum / value_count - mean * mean, 0.0) ** 0.5
        scaler = StandardScaler(mean, std)
    else:
        tr_end = int(tr_idx[-1]) + window if len(tr_idx) else total
        cond_train = observed_mask[:, :tr_end] * (1.0 - eval_mask[:, :tr_end])
        vals = x[:, :tr_end][cond_train > 0]
        scaler = StandardScaler(
            vals.mean(dtype=np.float64), vals.std(dtype=np.float64)
        )

    common = dict(
        window=window, scaler=scaler, timestamps=timestamps, dataset_id=dataset_id,
        node_presence=node_presence,
    )
    if start_groups is None:
        train_groups = validation_groups = test_window_groups = None
    else:
        group_by_start = dict(zip(starts.tolist(), start_groups.tolist()))
        train_groups = np.asarray([group_by_start[int(i)] for i in tr_idx], dtype=object)
        validation_groups = np.asarray(
            [group_by_start[int(i)] for i in va_idx], dtype=object
        )
        test_window_groups = np.asarray(
            [group_by_start[int(i)] for i in te_idx], dtype=object
        )
    train_ds = SpatioTemporalWindows(
        x, observed_mask, eval_mask, tr_idx, group_ids=train_groups, **common
    )
    val_ds = SpatioTemporalWindows(
        x, observed_mask, eval_mask, va_idx, group_ids=validation_groups, **common
    )
    test_ds = SpatioTemporalWindows(
        x, observed_mask, eval_mask, te_idx, group_ids=test_window_groups, **common
    )
    info = {
        "n_nodes": n,
        "n_timesteps": total,
        "n_windows": len(starts),
        "split_sizes": (len(tr_idx), len(va_idx), len(te_idx)),
        "split_mode": split_mode,
        "observed_rate": float(observed_mask.mean()),
        "eval_rate": float(eval_mask.mean()),
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
    }
    if split_groups is not None:
        info["subject_groups"] = split_groups
    return train_ds, val_ds, test_ds, scaler, info


def make_loaders(train_ds, val_ds, test_ds, batch_size: int = 32, num_workers: int = 0):
    kw = dict(num_workers=num_workers, pin_memory=False)
    if isinstance(train_ds, ConcatDataset):
        weights = torch.cat([
            torch.full((len(dataset),), 1.0 / max(len(dataset), 1))
            for dataset in train_ds.datasets
        ])
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, drop_last=False, **kw
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=False, **kw
        )
    return (
        train_loader,
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False, **kw),
    )
