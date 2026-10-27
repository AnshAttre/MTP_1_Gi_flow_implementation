"""Missing-data injection strategies (Section 4 of the paper).

(1) Point missing : randomly mask a fraction rho of the available data.
(2) Block missing : repeatedly pick a random (node, start timestep) and mask a
    contiguous segment, until a fraction rho of the available data is masked.
"""
from __future__ import annotations

import numpy as np


def point_missing(mask: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Return an *eval* mask (1 = entry was removed and must be imputed)."""
    avail = np.argwhere(mask > 0)
    n_drop = int(round(rho * len(avail)))
    if n_drop == 0:
        return np.zeros_like(mask)
    sel = rng.choice(len(avail), size=n_drop, replace=False)
    out = np.zeros_like(mask)
    out[avail[sel, 0], avail[sel, 1]] = 1.0
    return out


def block_missing(
    mask: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    min_len: int = 12,
    max_len: int = 36,
) -> np.ndarray:
    """Contiguous per-node blocks until the target fraction is reached."""
    n, r = mask.shape
    n_avail = int(mask.sum())
    target = int(round(rho * n_avail))
    out = np.zeros_like(mask)
    if target == 0:
        return out
    guard = 0
    max_guard = 100 * (target // max(min_len, 1) + 1) + 1000
    while out.sum() < target and guard < max_guard:
        guard += 1
        node = int(rng.integers(0, n))
        blk = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(r - blk, 1)))
        seg = slice(start, min(start + blk, r))
        out[node, seg] = np.maximum(out[node, seg], mask[node, seg])
    # trim any overshoot so the realised rate matches rho
    over = int(out.sum()) - target
    if over > 0:
        idx = np.argwhere(out > 0)
        sel = rng.choice(len(idx), size=over, replace=False)
        out[idx[sel, 0], idx[sel, 1]] = 0.0
    return out


def channel_missing(mask: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Remove a fraction of whole channels for the dead-electrode setting."""
    available_channels = np.flatnonzero(mask.sum(axis=1) > 0)
    n_channels = len(available_channels)
    n_drop = min(n_channels, int(round(rho * n_channels)))
    out = np.zeros_like(mask)
    if n_drop == 0:
        return out
    channels = rng.choice(available_channels, size=n_drop, replace=False)
    out[channels] = mask[channels]
    return out


def stream_missing(
    mask: np.ndarray,
    rho: float,
    rng: np.random.Generator,
    min_len: int = 20,
    max_len: int = 80,
) -> np.ndarray:
    """Remove shared contiguous time intervals across all available channels."""
    n_channels, n_steps = mask.shape
    target = int(round(rho * int(mask.sum())))
    out = np.zeros_like(mask)
    if target == 0:
        return out
    if min_len < 1 or max_len < min_len:
        raise ValueError("stream block lengths must satisfy 1 <= min_len <= max_len")
    attempts = 0
    max_attempts = 100 * (target // min_len + 1) + 1000
    while out.sum() < target and attempts < max_attempts:
        attempts += 1
        length = int(rng.integers(min_len, max_len + 1))
        start = int(rng.integers(0, max(n_steps - length + 1, 1)))
        end = min(start + length, n_steps)
        out[:, start:end] = np.maximum(out[:, start:end], mask[:, start:end])
    excess = int(out.sum()) - target
    if excess > 0:
        fully_hidden_columns = [
            step for step in range(n_steps)
            if out[:, step].sum() > 0 and np.array_equal(out[:, step], mask[:, step])
        ]
        for step in reversed(fully_hidden_columns):
            count = int(out[:, step].sum())
            if count <= excess:
                out[:, step] = 0.0
                excess -= count
            if excess == 0:
                break
    if excess > 0:
        indices = np.argwhere(out > 0)
        trim = rng.choice(len(indices), size=excess, replace=False)
        out[indices[trim, 0], indices[trim, 1]] = 0.0
    return out


def make_eval_mask(
    observed_mask: np.ndarray,
    strategy: str,
    rho: float,
    seed: int = 0,
    channel_ids: tuple[str, ...] | None = None,
    **kw,
) -> np.ndarray:
    """Dispatch with optional channel-name-stable random masks."""
    rng = np.random.default_rng(seed)
    if channel_ids is not None:
        if len(channel_ids) != observed_mask.shape[0]:
            raise ValueError("channel_ids must contain one identifier per channel")
        available = np.flatnonzero(np.asarray(observed_mask).sum(axis=1) > 0)
        ordered = sorted(available, key=lambda index: str(channel_ids[index]).upper())
        out = np.zeros_like(observed_mask)
        if not ordered:
            return out
        out[ordered] = _make_eval_mask(
            np.asarray(observed_mask)[ordered], strategy, rho, rng, kw
        )
        return out
    return _make_eval_mask(observed_mask, strategy, rho, rng, kw)


def _make_eval_mask(observed_mask, strategy, rho, rng, kw):
    if strategy == "point":
        return point_missing(observed_mask, rho, rng)
    if strategy == "block":
        return block_missing(observed_mask, rho, rng, **kw)
    if strategy in ("channel", "electrode"):
        return channel_missing(observed_mask, rho, rng)
    if strategy in ("stream", "outage"):
        return stream_missing(observed_mask, rho, rng, **kw)
    raise ValueError(f"unknown missing strategy: {strategy!r}")


def random_subset_mask(
    mask, keep_ratio_range=(0.3, 0.9), channel_drop_prob=0.0, generator=None
):
    """Sample a conditioning submask of `mask` (torch). Used during training to
    hide extra points so the model must actually predict rather than copy.

    A generator is only passed through when its device matches the mask's --
    torch refuses a CPU generator for a CUDA tensor, so on GPU we draw from the
    global RNG (already seeded by set_seed) rather than crashing.
    """
    import torch

    if generator is not None and generator.device.type != mask.device.type:
        generator = None
    p = torch.empty(mask.shape[0], 1, 1, device=mask.device).uniform_(*keep_ratio_range)
    keep = (torch.rand(mask.shape, device=mask.device, generator=generator) < p).float()
    if channel_drop_prob > 0:
        channel_keep = (
            torch.rand(
                (mask.shape[0], mask.shape[1], 1),
                device=mask.device,
                generator=generator,
            )
            >= channel_drop_prob
        ).float()
        keep = keep * channel_keep
    return mask * keep
