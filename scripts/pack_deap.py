"""Pack DEAP raw BDF files into the packed-npz format the SEED-IV loader reads.

Per participant: 32 EEG channels, average reference, 0.5-75 Hz band-pass,
resample to 200 Hz, then 60 evenly spaced 5 s chunks (1000 samples). Each
recording is centred per channel, divided by one scalar (its overall std, so
the topography is kept) and soft-clipped with 5*tanh(x/5), as in the SEED-IV
input. Load the result with
    --dataset seed4 --seed-root <out.npz> --split-mode subject

usage: python scripts/pack_deap.py <dir with s01.bdf..s32.bdf> <out.npz>
"""
import os
import sys
from pathlib import Path

import mne
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from giflow.data.deap import DEAP_EEG_CHANNELS
from giflow.data.seed_iv import _correlation_adjacency

N_CHUNKS, CHUNK = 60, 1000


def main(bdf_dir, out):
    paths = sorted(Path(bdf_dir).glob("s*.bdf"))
    if not paths:
        raise FileNotFoundError("no s*.bdf files under %s" % bdf_dir)
    chunks, records = [], []
    for path in paths:
        raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        by_name = {c.strip().upper(): c for c in raw.ch_names}
        raw.pick([by_name[c] for c in DEAP_EEG_CHANNELS])
        raw.set_eeg_reference("average", verbose="ERROR")
        raw.filter(0.5, 75.0, verbose="ERROR")
        raw.resample(200.0, verbose="ERROR")
        x = raw.get_data().astype(np.float32) * 1e6          # microvolts
        del raw
        starts = np.linspace(0, x.shape[1] - CHUNK, N_CHUNKS).round().astype(int)
        seg = np.concatenate([x[:, s:s + CHUNK] for s in starts], axis=1)
        seg = 5.0 * np.tanh((seg - seg.mean(1, keepdims=True)) / seg.std() / 5.0)
        chunks.append(seg)
        records.append("1/%d_deap:%d" % (int(path.stem[1:3]), N_CHUNKS))
        print(path.name, x.shape, flush=True)
    x = np.concatenate(chunks, axis=1)
    adj = _correlation_adjacency(x[:, :: max(1, x.shape[1] // 200000)], 0.1)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, x=x.astype(np.float16), adj=adj, recordings=np.array(records),
                        channel_names=np.array(DEAP_EEG_CHANNELS))
    print("saved", out, x.shape, len(records), "participants")


if __name__ == "__main__":
    main(*sys.argv[1:3])
