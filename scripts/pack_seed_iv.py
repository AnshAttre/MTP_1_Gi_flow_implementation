"""Pack the SEED-IV download into a slim .npz for Kaggle upload.

The raw folder is ~5.5 GB, most of it inspection PNGs, .mat files and the
float64 duplicates. Only X_prc1.npy + labels.npy matter for imputation, and
float32 halves those. Result: ~1.3 GB for all 17 subject-sessions, or ~80 MB
for one.

  # one subject, quick to upload and enough to get training working
  python scripts/pack_seed_iv.py --root ~/Downloads/seed4 --subjects 12_20150804 \
      --out data/seed_iv_one.npz

  # everything
  python scripts/pack_seed_iv.py --root ~/Downloads/seed4 --out data/seed_iv_all.npz

Upload the .npz as a Kaggle Dataset, then point the notebook at it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giflow.data.seed_iv import _correlation_adjacency, _discover


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="the extracted seed4 folder")
    p.add_argument("--out", default="data/seed_iv_all.npz")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--max-windows-per-subject", type=int, default=None)
    p.add_argument("--threshold", type=float, default=0.1)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--compress", action="store_true",
                   help="savez_compressed; smaller upload, slower to load")
    args = p.parse_args(argv)

    root = Path(args.root).expanduser()
    found = _discover(root)
    if not found:
        raise SystemExit("no X_prc1.npy found under %s" % root)
    if args.sessions:
        found = [f for f in found if f[0] in args.sessions]
    if args.subjects:
        found = [f for f in found if f[1] in args.subjects]
    if not found:
        raise SystemExit("nothing left after filtering; available: %s"
                         % sorted({f[1] for f in _discover(root)}))

    print("packing %d recording(s) at %s" % (len(found), args.dtype))
    chunks, labels, meta = [], [], []
    for session, subject, path in found:
        arr = np.load(path, mmap_mode="r")
        n = arr.shape[0] if args.max_windows_per_subject is None else min(
            arr.shape[0], args.max_windows_per_subject
        )
        a = np.asarray(arr[:n], dtype=args.dtype)
        chunks.append(a.transpose(1, 0, 2).reshape(a.shape[1], -1))
        lp = path.parent / "labels.npy"
        if lp.exists():
            labels.append(np.load(lp)[:n])
        meta.append("%s/%s:%d" % (session, subject, n))
        print("  %s/%-14s %s -> %d samples" % (session, subject, a.shape, chunks[-1].shape[1]))

    x = np.concatenate(chunks, axis=1)
    del chunks
    print("combined signal: (%d channels, %d samples)  %.2f GB in memory"
          % (x.shape[0], x.shape[1], x.nbytes / 1e9))

    # graph from the training portion only, so the test split cannot leak in
    print("building correlation graph on the first 70% ...")
    adj = _correlation_adjacency(
        np.asarray(x[:, : int(0.7 * x.shape[1])], dtype=np.float64), args.threshold
    )
    print("  avg degree %.2f" % adj.sum(1).mean())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # deliberately no "mask" key: the signal is fully observed, and the loader
    # falls back to ones_like(x). Writing a placeholder of the wrong shape here
    # would be picked up as a real mask.
    payload = dict(
        x=x,
        adj=adj.astype(np.float32),
        recordings=np.array(meta),
    )
    if labels:
        payload["labels"] = np.concatenate(labels)
    saver = np.savez_compressed if args.compress else np.savez
    saver(out, **payload)
    print("wrote %s  (%.2f GB on disk)" % (out, out.stat().st_size / 1e9))
    print("\nnext: upload this file as a Kaggle Dataset, then run the notebook with")
    print("  --dataset seed4 --seed-root /kaggle/input/<your-dataset-slug>/%s" % out.name)


if __name__ == "__main__":
    main()
