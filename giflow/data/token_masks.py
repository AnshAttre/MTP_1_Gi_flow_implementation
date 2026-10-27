"""Channel-window token metadata and masking curricula."""
from __future__ import annotations

import numpy as np


def channel_positions(channel_names, fallback=None) -> np.ndarray:
    """Return normalized 3D scalp coordinates for standard EEG channel names."""
    try:
        import mne
        montage = mne.channels.make_standard_montage("colin27_1020")
        positions = montage.get_positions()["ch_pos"]
    except Exception:
        positions = {}

    resolved = []
    for index, name in enumerate(channel_names):
        canonical = str(name).strip().upper()
        aliases = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
                   "CB1": "O1", "CB2": "O2", "FPZ": "FPZ", "FZ": "FZ",
                   "CZ": "CZ", "PZ": "PZ", "OZ": "OZ"}
        canonical = aliases.get(canonical, canonical)
        upper_positions = {str(name).upper(): value for name, value in positions.items()}
        position = upper_positions.get(canonical)
        if position is None and fallback is not None and index < len(fallback):
            position = np.asarray(fallback[index], dtype=np.float64)
        if position is None:
            raise ValueError("No 3D montage position available for channel %s" % canonical)
        resolved.append(position)

    positions_array = np.asarray(resolved, dtype=np.float64)
    radius = np.linalg.norm(positions_array, axis=1, keepdims=True)
    safe_radius = np.maximum(radius, 1e-6)
    return positions_array / safe_radius


def raster_tokens(values: np.ndarray, observed_mask: np.ndarray, patch_size: int):
    """Rasterize a (channels, samples) grid as (channels * patches, patch_size)."""
    channels, samples = values.shape
    if patch_size <= 0 or samples % patch_size:
        raise ValueError("samples must be divisible by patch_size")
    patches = samples // patch_size
    tokens = values.reshape(channels, patches, patch_size)
    token_mask = observed_mask.reshape(channels, patches, patch_size).any(axis=-1).astype(np.float32)
    flat_tokens = tokens.transpose(1, 0, 2).reshape(channels * patches, patch_size)
    flat_mask = token_mask.transpose(1, 0).reshape(channels * patches)
    channel_index = np.tile(np.arange(channels), patches)
    time_index = np.repeat(np.arange(patches), channels)
    return flat_tokens, flat_mask, channel_index, time_index


def token_dropout_mask(
    observed_mask: np.ndarray,
    patch_size: int,
    rho: float,
    rng: np.random.Generator,
    channel_dropout_prob: float = 0.0,
) -> np.ndarray:
    """Sample channel-window targets at rates in [0, rho]. Mask is (channels, patches)."""
    if not 0 <= rho <= 1:
        raise ValueError("rho must be between zero and one")
    channels, samples = observed_mask.shape
    if samples % patch_size:
        raise ValueError("samples must be divisible by patch_size")
    patches = samples // patch_size
    obs = observed_mask.reshape(channels, patches, patch_size).any(axis=-1)
    available = np.argwhere(obs > 0)
    if not available.size:
        return np.zeros((channels, patches), dtype=np.float32)

    dropped = set()
    if rng.random() < channel_dropout_prob:
        available_channels = np.flatnonzero(obs.any(axis=1))
        if len(available_channels):
            n_drop = rng.integers(0, len(available_channels) + 1)
            if n_drop:
                channels_to_drop = rng.choice(available_channels, size=n_drop, replace=False)
                for channel in channels_to_drop:
                    for patch in np.flatnonzero(obs[channel]):
                        dropped.add((int(channel), int(patch)))

    candidates = [tuple(index) for index in available if tuple(index) not in dropped]
    target_count = int(round(rho * len(candidates)))
    if target_count:
        selected = rng.choice(len(candidates), size=min(target_count, len(candidates)), replace=False)
        dropped.update(candidates[int(index)] for index in selected)

    out = np.zeros((channels, patches), dtype=np.float32)
    for channel, patch in dropped:
        out[channel, patch] = 1.0
    return out
