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
from torch.utils.data import ConcatDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giflow.baselines import run_baselines
from giflow.data.dataset import build_splits, make_loaders
from giflow.data.datasets import align_bundles, load_dataset
from giflow.data.graph import gcn_normalize, laplacian
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
    "srg_flow": {"use_srg_guidance": True},
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="synthetic")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="jointly train on multiple datasets, e.g. seed4 seed4")
    p.add_argument("--variant", default="giflow", choices=sorted(VARIANTS))
    p.add_argument("--missing", default="point", choices=["point", "block", "channel"])
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
    p.add_argument("--seed-roots", nargs="+", default=None,
                   help="one SEED-IV root per item in --datasets")
    p.add_argument("--subjects", nargs="*", default=None)
    p.add_argument("--sessions", nargs="*", default=None)
    p.add_argument("--max-windows-per-subject", type=int, default=None)
    p.add_argument("--max-recordings", type=int, default=None,
                   help="cap MATLAB recordings loaded from a raw SEED-IV tree")
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
    p.add_argument("--preservation-weight", type=float, default=0.1,
                   help="weight of the loss preserving conditioned values")
    p.add_argument("--srg-lambda-res", type=float, default=1.0,
                   help="SRG-flow residual correction and auxiliary-loss weight")
    p.add_argument("--srg-lambda-smm", type=float, default=0.01,
                   help="SRG-flow step-modulation regularization weight")
    p.add_argument("--clamp-observed-each-step", action="store_true",
                   help="restore conditioned values after every Euler update")
    p.add_argument("--train-channel-drop-prob", type=float, default=0.0,
                   help="probability of hiding each electrode during training")
    p.add_argument("--prior-mode", default="exact", choices=["exact", "taylor"])
    p.add_argument("--prior-renormalize", action="store_true")
    p.add_argument("--max-tau", type=float, default=10.0,
                   help="upper bound on the filtering factors; 0 disables the bound")
    p.add_argument("--min-tau-s", type=float, default=0.0,
                   help="lower bound on spatial filtering; use >0 to force smoothing")
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
    names = args.datasets or [args.dataset]
    roots = args.seed_roots
    if roots is not None and len(roots) != len(names):
        raise ValueError("--seed-roots must have one path per --datasets entry")
    bundles = []
    for index, name in enumerate(names):
        kw = {"threshold": args.threshold}
        if name.lower().startswith("syn"):
            kw = dict(n_nodes=args.nodes, n_steps=args.steps, sigma=args.sigma, seed=seed + index)
        elif name.lower() in ("seed4", "seed-iv", "seediv", "seed_iv"):
            kw = dict(
                root=(roots[index] if roots else args.seed_root),
                window=args.window,
                subjects=tuple(args.subjects) if args.subjects else None,
                sessions=tuple(args.sessions) if args.sessions else None,
                max_windows_per_subject=args.max_windows_per_subject,
                max_recordings=args.max_recordings,
                threshold=args.threshold,
                channel_dropout=args.channel_dropout,
            )
        bundles.append(load_dataset(name, **kw))

    aligned, combined_adj, channel_names = align_bundles(bundles)
    timestamp_signatures = {(b.timestamps is not None, b.ts_classes) for b in aligned}
    if len(timestamp_signatures) != 1:
        raise ValueError("joint datasets must use compatible timestamp features")

    split_groups, scalers, infos = [], [], []
    for dataset_id, bundle in enumerate(aligned):
        train_ds, val_ds, test_ds, scaler, info = build_splits(
            bundle.x,
            bundle.observed_mask,
            window=args.window,
            stride=args.stride,
            missing_strategy=args.missing,
            rho=args.rho,
            seed=seed,
            timestamps=bundle.timestamps,
            max_windows=args.max_windows,
            segments=bundle.segments,
            dataset_id=dataset_id,
            node_presence=bundle.node_presence,
        )
        split_groups.append((train_ds, val_ds, test_ds))
        scalers.append(scaler)
        infos.append(info)

    if len(aligned) == 1:
        splits, scaler = split_groups[0], scalers[0]
    else:
        splits = tuple(
            ConcatDataset([group[part] for group in split_groups]) for part in range(3)
        )
        scaler = scalers

    graphs = aligned[0].graphs(
        args.window, normalized_laplacian=args.normalized_laplacian
    )
    graphs["adj_s"] = combined_adj
    graphs["lap_s"] = laplacian(combined_adj, args.normalized_laplacian)
    graphs["adj_s_norm"] = gcn_normalize(combined_adj)
    info = infos[0] if len(infos) == 1 else {
        "datasets": [dict(name=b.name, **item) for b, item in zip(aligned, infos)],
        "channel_names": list(channel_names),
    }
    return (aligned[0] if len(aligned) == 1 else aligned), splits, scaler, info, graphs


def main(argv=None):
    args = parse_args(argv)
    dataset_tag = "_".join(args.datasets or [args.dataset])
    out_root = Path(args.out or ("runs/%s_%s_%s_rho%g" % (
        dataset_tag, args.variant, args.missing, args.rho)))
    out_root.mkdir(parents=True, exist_ok=True)

    trials, histories = [], []
    for i in range(args.seeds):
        seed = args.seed0 + i
        print("\n" + "=" * 72)
        print("dataset=%s variant=%s missing=%s rho=%g seed=%d"
              % (args.datasets or args.dataset, args.variant, args.missing, args.rho, seed))
        print("=" * 72)
        bundle, splits, scaler, info, graphs = build(args, seed)
        bundles = bundle if isinstance(bundle, list) else [bundle]
        dataset_labels = tuple(
            ("%s[%d]" % (item.name, index + 1) if len(bundles) > 1 else item.name)
            for index, item in enumerate(bundles)
        )
        for label, item in zip(dataset_labels, bundles):
            print("[data] %s" % item.summary() if len(bundles) == 1 else "[data] %s: %s" % (label, item.summary()))
        if len(bundles) == 1:
            print("[data] windows=%d split=%s eval_rate=%.3f"
                  % (info["n_windows"], info["split_sizes"], info["eval_rate"]))
        else:
            for item in info["datasets"]:
                print("[data] %s windows=%d split=%s eval_rate=%.3f"
                      % (item["name"], item["n_windows"], item["split_sizes"], item["eval_rate"]))

        if i == 0 and not args.skip_baselines and len(bundles) == 1:
            _, _, test_loader = make_loaders(*splits, batch_size=args.batch_size)
            t0 = time.time()
            base = run_baselines(
                test_loader, scaler, graphs["adj_s"], graphs["adj_s_norm"]
            )
            print("[baselines] (%.1fs)" % (time.time() - t0))
            for k, v in base.items():
                    print("   %-8s PCC %7.4f  NMSE %8.4f  PSNR %7.2f dB  SNR %7.2f dB"
                        % (k, v["pcc"], v["nmse"], v["psnr"], v["snr"]))
            (out_root / "baselines.json").write_text(json.dumps(base, indent=2))
        elif i == 0 and len(bundles) > 1 and not args.skip_baselines:
            print("[baselines] skipped for joint training; baseline runners are single-dataset")

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
            preservation_weight=args.preservation_weight,
            srg_lambda_res=args.srg_lambda_res,
            srg_lambda_smm=args.srg_lambda_smm,
            clamp_observed_each_step=args.clamp_observed_each_step,
            train_channel_drop_prob=args.train_channel_drop_prob,
            tau_epochs=args.tau_epochs,
            prior_mode=args.prior_mode,
            prior_renormalize=(
                args.prior_renormalize
                or any(not item.node_presence.all() for item in bundles)
            ),
            max_tau=(None if args.max_tau <= 0 else args.max_tau),
            min_tau_s=args.min_tau_s,
            normalized_laplacian=args.normalized_laplacian,
            seed=seed,
            device=args.device,
            threads=args.threads,
            out_dir=str(out_root / ("seed%d" % seed)),
            **VARIANTS[args.variant],
        )
        _, _, hist = fit(
            bundles[0].x,
            graphs["adj_s"],
            graphs["lap_s"],
            graphs["lap_t"],
            graphs["adj_s_norm"],
            graphs["adj_t_norm"],
            splits,
            scaler,
            cfg,
            n_timestamp_classes=bundles[0].ts_classes,
            dataset_names=dataset_labels,
        )
        trials.append(hist["test"])
        histories.append(hist)

    agg = aggregate_trials(trials)
    print("\n" + "-" * 72)
    print("FINAL over %d seed(s): PCC %.4f+-%.4f  NMSE %.4f+-%.4f  PSNR %.2f+-%.2f dB  SNR %.2f+-%.2f dB"
          % (len(trials), agg["pcc"]["mean"], agg["pcc"]["std"],
             agg["nmse"]["mean"], agg["nmse"]["std"],
             agg["psnr"]["mean"], agg["psnr"]["std"],
             agg["snr"]["mean"], agg["snr"]["std"]))
    total = sum(h["train_seconds"] for h in histories)
    print("total training wall-clock: %.1f min" % (total / 60))
    (out_root / "summary.json").write_text(
        json.dumps({"args": vars(args), "aggregate": agg, "trials": trials}, indent=2)
    )
    return agg


if __name__ == "__main__":
    main()
