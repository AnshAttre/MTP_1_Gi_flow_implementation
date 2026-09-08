"""Measure per-epoch cost on this machine and extrapolate full training time.

Times a handful of real training steps plus one validation pass for each dataset
shape in the paper, then projects the total wall-clock for the Appendix C.3
protocol (up to 300 epochs, early stopping patience 10, 5 seeds).

  python scripts/benchmark.py
  python scripts/benchmark.py --configs air36 aqi --threads 12
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giflow.data.dataset import build_splits, make_loaders
from giflow.data.datasets import DatasetBundle, load_synthetic
from giflow.data.graph import gaussian_kernel_adjacency
from giflow.data.masking import random_subset_mask
from giflow.train import TrainConfig, count_params
from giflow.models.giflow import EMA, GiFlow

# (name, n_nodes, n_timesteps) -- the real datasets are stood in for by synthetic
# signals of the same shape, so the timing holds without the data being present.
SHAPES = {
    "synthetic": (50, 3000),
    "air36": (36, 8760),
    "pems08": (170, 17856),
    "aqi": (437, 8760),
}


def peak_rss_gb() -> float:
    try:
        import ctypes
        import ctypes.wintypes as wt

        class PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wt.DWORD),
                ("PageFaultCount", wt.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb
        )
        return pmc.PeakWorkingSetSize / 1e9
    except Exception:
        return float("nan")


def bench_one(name, n_nodes, n_steps, args):
    x, adj, _ = None, None, None
    bundle = load_synthetic(n_nodes=n_nodes, n_steps=n_steps, sigma=0.1, seed=0)
    # rebuild the graph the way the real datasets do (kernel + threshold),
    # so node degree, and hence message-passing cost, is representative
    prof = bundle.x - bundle.x.mean(axis=1, keepdims=True)
    adj = gaussian_kernel_adjacency(prof, threshold=0.1)
    if adj.sum() == 0:
        adj = bundle.adj_s
    bundle.adj_s = adj

    splits = build_splits(
        bundle.x,
        bundle.observed_mask,
        window=args.window,
        stride=args.stride,
        missing_strategy="point",
        rho=0.2,
        seed=0,
        max_windows=args.max_windows,
    )
    train_ds, val_ds, test_ds, scaler, info = splits
    graphs = bundle.graphs(args.window)
    train_loader, val_loader, _ = make_loaders(
        train_ds, val_ds, test_ds, args.batch_size
    )

    t0 = time.time()
    model = GiFlow(
        adj_s=graphs["adj_s"],
        lap_s=graphs["lap_s"],
        lap_t=graphs["lap_t"],
        adj_s_norm=graphs["adj_s_norm"],
        adj_t_norm=graphs["adj_t_norm"],
        hidden=args.hidden,
        n_mp_layers=args.layers,
        dropout=0.1,
    )
    build_s = time.time() - t0

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    ema = EMA(model)
    gen = torch.Generator().manual_seed(0)

    # --- time n_steps_bench training steps
    model.train()
    it = iter(train_loader)
    times = []
    for i in range(args.steps_bench + 2):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader)
            batch = next(it)
        s = time.time()
        cond_full = batch["cond_mask"]
        cond = random_subset_mask(cond_full, (0.3, 0.9), generator=gen)
        target = cond_full - cond
        loss = model.loss(batch["x_true"], cond, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        ema.update(model)
        if i >= 2:  # discard warmup
            times.append(time.time() - s)
    step_s = float(np.median(times))

    # --- time one imputation batch (20 Euler steps)
    model.eval()
    vb = next(iter(val_loader))
    s = time.time()
    with torch.no_grad():
        model.impute(vb["x_true"], vb["cond_mask"], n_steps=args.euler_steps)
    infer_batch_s = time.time() - s

    n_train_batches = len(train_loader)
    n_val_batches = len(val_loader)
    n_test_batches = max(1, int(np.ceil(info["split_sizes"][2] / args.batch_size)))
    epoch_s = step_s * n_train_batches + infer_batch_s * n_val_batches

    return {
        "name": name,
        "n_nodes": n_nodes,
        "n_timesteps": n_steps,
        "avg_degree": float(adj.sum(1).mean()),
        "windows": info["n_windows"],
        "split": info["split_sizes"],
        "params": count_params(model),
        "model_build_s": round(build_s, 2),
        "train_step_s": round(step_s, 4),
        "train_batches": n_train_batches,
        "impute_batch_s": round(infer_batch_s, 3),
        "epoch_s": round(epoch_s, 1),
        "epoch_min": round(epoch_s / 60, 2),
        "test_pass_min": round(infer_batch_s * n_test_batches / 60, 2),
        "proj_50_epochs_h": round(epoch_s * 50 / 3600, 2),
        "proj_300_epochs_h": round(epoch_s * 300 / 3600, 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=list(SHAPES))
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--euler-steps", type=int, default=20)
    p.add_argument("--steps-bench", type=int, default=8)
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", default="runs/benchmark.json")
    args = p.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    print("torch %s | threads %d | device cpu" % (torch.__version__, torch.get_num_threads()))
    print("config: window=%d batch=%d hidden=%d layers=%d euler=%d\n"
          % (args.window, args.batch_size, args.hidden, args.layers, args.euler_steps))

    rows = []
    for name in args.configs:
        if name not in SHAPES:
            print("skip unknown config %r" % name)
            continue
        n, r = SHAPES[name]
        print("[bench] %s  N=%d R=%d ..." % (name, n, r), flush=True)
        try:
            row = bench_one(name, n, r, args)
        except Exception as e:
            print("   FAILED: %r" % (e,))
            continue
        rows.append(row)
        print("   deg=%.1f windows=%d params=%d" % (row["avg_degree"], row["windows"], row["params"]))
        print("   train step %.3fs x %d batches -> epoch %.1fs (%.2f min)"
              % (row["train_step_s"], row["train_batches"], row["epoch_s"], row["epoch_min"]))
        print("   impute batch %.3fs | projected: 50 ep %.2f h, 300 ep %.2f h | peak RSS %.2f GB\n"
              % (row["impute_batch_s"], row["proj_50_epochs_h"], row["proj_300_epochs_h"],
                 row["peak_rss_gb"]))

    print("=" * 92)
    print("%-10s %6s %7s %8s %10s %10s %11s %11s" % (
        "dataset", "N", "windows", "params", "step(s)", "epoch(min)", "50ep(h)", "300ep(h)"))
    print("-" * 92)
    for r in rows:
        print("%-10s %6d %7d %8d %10.3f %10.2f %11.2f %11.2f" % (
            r["name"], r["n_nodes"], r["windows"], r["params"], r["train_step_s"],
            r["epoch_min"], r["proj_50_epochs_h"], r["proj_300_epochs_h"]))
    print("=" * 92)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
    print("saved -> %s" % out)
    return rows


if __name__ == "__main__":
    main()
