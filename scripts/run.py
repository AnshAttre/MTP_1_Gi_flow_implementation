"""Entry point: train + evaluate GiFlow on one dataset / missing pattern.

Examples
--------
  python scripts/run.py --dataset synthetic --sigma 0.1 --missing point --rho 0.2
  python scripts/run.py --dataset air36 --missing block --rho 0.2 --epochs 100
  python scripts/run.py --dataset synthetic --variant fm_gauss   # ablation
  python scripts/run.py --dataset synthetic --seeds 5            # 5 trials
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

from giflow.baselines import run_baselines
from giflow.data.dataset import build_splits, make_loaders
from giflow.data.datasets import load_dataset
from giflow.metrics import aggregate_trials
from giflow.train import TrainConfig, fit

VARIANTS = {
    "giflow": {},
    "fm_gauss": {"gaussian_prior": True},
    "gfm": {"learn_temporal_prior": False},          # spatial-only prior
    "tfm": {"learn_spatial_prior": False},           # temporal-only prior
    "no_spatial_attn": {"use_spatial_attention": False},
    "no_temporal_attn": {"use_temporal_attention": False},
    "no_st_attn": {"use_spatial_attention": False, "use_temporal_attention": False},
    "no_propagation": {"use_propagation": False},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--variant", default="giflow", choices=sorted(VARIANTS))
    p.add_argument("--missing", default="point", choices=["point", "block"])
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--threshold", type=float, default=0.1, help="graph binarisation threshold")
    # synthetic knobs
    p.add_argument("--nodes", type=int, default=50)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--sigma", type=float, default=0.1)
    # SEED-IV knobs
    p.add_argument("--seed-root", default="data/seed4",
                   help="SEED-IV root dir, or a packed .npz")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--max-windows-per-subject", type=int, default=None)
    p.add_argument("--channel-dropout", action="store_true",
                   help="mark whole EEG channels missing (dead-electrode setting)")
    # optimisation
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--alpha-tau", type=float, default=0.01)
    p.add_argument("--tau-epochs", type=int, default=100)
    p.add_argument("--euler-steps", type=int, default=20)
    p.add_argument("--prior-mode", default="exact", choices=["exact", "taylor"])
    p.add_argument("--prior-renormalize", action="store_true")
    p.add_argument("--max-tau", type=float, default=10.0,
                   help="upper bound on the filtering factors; 0 disables the bound")
    p.add_argument("--normalized-laplacian", action="store_true",
                   help="use the symmetric normalised Laplacian instead of D-A")
    # budget controls, important on CPU
    p.add_argument("--max-windows", type=int, default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--seed0", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--skip-baselines", action="store_true")
    return p.parse_args(argv)


def build(args, seed):
    kw = {}
    if args.dataset.lower().startswith("syn"):
        kw = dict(n_nodes=args.nodes, n_steps=args.steps, sigma=args.sigma, seed=seed)
    elif args.dataset.lower() in ("seed4", "seed-iv", "seediv", "seed_iv"):
        kw = dict(
            root=args.seed_root,
            window=args.window,
            subjects=tuple(args.subjects) if args.subjects else None,
            sessions=tuple(args.sessions) if args.sessions else None,
            max_windows_per_subject=args.max_windows_per_subject,
            threshold=args.threshold,
            channel_dropout=args.channel_dropout,
        )
    else:
        kw = dict(threshold=args.threshold)
    bundle = load_dataset(args.dataset, **kw)
    splits = build_splits(
        bundle.x,
        bundle.observed_mask,
        window=args.window,
        stride=args.stride,
        missing_strategy=args.missing,
        rho=args.rho,
        seed=seed,
        timestamps=bundle.timestamps,
        max_windows=args.max_windows,
    )
    train_ds, val_ds, test_ds, scaler, info = splits
    graphs = bundle.graphs(args.window, normalized_laplacian=args.normalized_laplacian)
    return bundle, (train_ds, val_ds, test_ds), scaler, info, graphs


def main(argv=None):
    args = parse_args(argv)
    out_root = Path(args.out or ("runs/%s_%s_%s_rho%g" % (
        args.dataset, args.variant, args.missing, args.rho)))
    out_root.mkdir(parents=True, exist_ok=True)

    trials, histories = [], []
    for i in range(args.seeds):
        seed = args.seed0 + i
        print("\n" + "=" * 72)
        print("dataset=%s variant=%s missing=%s rho=%g seed=%d"
              % (args.dataset, args.variant, args.missing, args.rho, seed))
        print("=" * 72)
        bundle, splits, scaler, info, graphs = build(args, seed)
        print("[data] " + bundle.summary())
        print("[data] windows=%d split=%s eval_rate=%.3f"
              % (info["n_windows"], info["split_sizes"], info["eval_rate"]))

        if i == 0 and not args.skip_baselines:
            _, _, test_loader = make_loaders(*splits, batch_size=args.batch_size)
            t0 = time.time()
            base = run_baselines(
                test_loader, scaler, graphs["adj_s"], graphs["adj_s_norm"]
            )
            print("[baselines] (%.1fs)" % (time.time() - t0))
            for k, v in base.items():
                print("   %-8s MAE %8.4f  RMSE %8.4f  MAPE %7.2f%%"
                      % (k, v["mae"], v["rmse"], v["mape"]))
            (out_root / "baselines.json").write_text(json.dumps(base, indent=2))

        cfg = TrainConfig(
            lr=args.lr,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            n_mp_layers=args.layers,
            hidden=args.hidden,
            alpha_tau=args.alpha_tau,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            patience=args.patience,
            euler_steps=args.euler_steps,
            tau_epochs=args.tau_epochs,
            prior_mode=args.prior_mode,
            prior_renormalize=args.prior_renormalize,
            max_tau=(None if args.max_tau <= 0 else args.max_tau),
            normalized_laplacian=args.normalized_laplacian,
            seed=seed,
            device=args.device,
            threads=args.threads,
            out_dir=str(out_root / ("seed%d" % seed)),
            **VARIANTS[args.variant],
        )
        _, _, hist = fit(
            bundle.x,
            graphs["adj_s"],
            graphs["lap_s"],
            graphs["lap_t"],
            graphs["adj_s_norm"],
            graphs["adj_t_norm"],
            splits,
            scaler,
            cfg,
            n_timestamp_classes=bundle.ts_classes,
        )
        trials.append(hist["test"])
        histories.append(hist)

    agg = aggregate_trials(trials)
    print("\n" + "-" * 72)
    print("FINAL over %d seed(s): MAE %.4f+-%.4f  RMSE %.4f+-%.4f  MAPE %.2f+-%.2f"
          % (len(trials), agg["mae"]["mean"], agg["mae"]["std"],
             agg["rmse"]["mean"], agg["rmse"]["std"],
             agg["mape"]["mean"], agg["mape"]["std"]))
    total = sum(h["train_seconds"] for h in histories)
    print("total training wall-clock: %.1f min" % (total / 60))
    (out_root / "summary.json").write_text(
        json.dumps({"args": vars(args), "aggregate": agg, "trials": trials}, indent=2)
    )
    return agg


if __name__ == "__main__":
    main()
