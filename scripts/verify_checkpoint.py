"""Verify a committed checkpoint: rebuild the model from its history.json, load
the weights, re-evaluate on the same split, and compare against the recorded
test metrics.

  python scripts/verify_checkpoint.py --ckpt-dir checkpoints/seed4_point_rho0.2 \
      --seed-root data/seed_iv_one.npz --window 100 --stride 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giflow.data.dataset import build_splits, make_loaders
from giflow.data.datasets import load_dataset
from giflow.metrics import MetricAccumulator
from giflow.models.giflow import GiFlow


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", required=True)
    p.add_argument("--dataset", default="seed4")
    p.add_argument("--seed-root", default="data/seed_iv_one.npz")
    p.add_argument("--window", type=int, default=100)
    p.add_argument("--stride", type=int, default=100)
    p.add_argument("--missing", default="point")
    p.add_argument("--rho", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tol", type=float, default=0.02,
                   help="relative tolerance when comparing to the recorded MAE")
    args = p.parse_args(argv)

    d = Path(args.ckpt_dir)
    hist = json.loads((d / "history.json").read_text())
    cfg = hist["config"]
    recorded = hist["test"]
    print("recorded test: MAE %.5f  RMSE %.5f  MAPE %.2f%%  (n=%d)"
          % (recorded["mae"], recorded["rmse"], recorded["mape"], recorded["n"]))
    print("run config: hidden=%d layers=%d dropout=%s euler=%d seed=%d"
          % (cfg["hidden"], cfg["n_mp_layers"], cfg["dropout"],
             cfg["euler_steps"], cfg["seed"]))
    print("frozen tau: tau_s=%.6f tau_t=%.6f"
          % (hist["tau"]["tau_s"], hist["tau"]["tau_t"]))

    kw = {}
    if args.dataset.lower() in ("seed4", "seed-iv", "seediv", "seed_iv"):
        kw = dict(root=args.seed_root, window=args.window)
    bundle = load_dataset(args.dataset, **kw)
    print("[data] " + bundle.summary())

    tr, va, te, scaler, info = build_splits(
        bundle.x, bundle.observed_mask,
        window=args.window, stride=args.stride,
        missing_strategy=args.missing, rho=args.rho, seed=cfg["seed"],
        timestamps=bundle.timestamps,
    )
    print("[data] windows=%d split=%s" % (info["n_windows"], info["split_sizes"]))
    _, _, test_loader = make_loaders(tr, va, te, args.batch_size)
    graphs = bundle.graphs(args.window, cfg.get("normalized_laplacian", False))

    model = GiFlow(
        adj_s=graphs["adj_s"], lap_s=graphs["lap_s"], lap_t=graphs["lap_t"],
        adj_s_norm=graphs["adj_s_norm"], adj_t_norm=graphs["adj_t_norm"],
        hidden=cfg["hidden"], emb_dim=cfg["emb_dim"],
        n_mp_layers=cfg["n_mp_layers"], k_hops=cfg["k_hops"],
        dropout=cfg["dropout"], prior_mode=cfg["prior_mode"],
        prior_order=cfg["prior_order"], max_tau=cfg["max_tau"],
        gaussian_prior=cfg["gaussian_prior"],
        prior_renormalize=cfg["prior_renormalize"],
        learn_spatial_prior=cfg["learn_spatial_prior"],
        learn_temporal_prior=cfg["learn_temporal_prior"],
        max_len=max(512, args.window),
        n_timestamp_classes=bundle.ts_classes,
        use_spatial_attention=cfg["use_spatial_attention"],
        use_temporal_attention=cfg["use_temporal_attention"],
        use_propagation=cfg["use_propagation"],
    ).to(args.device)

    sd = torch.load(d / "best.pt", map_location=args.device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("[load] %d params in file | missing=%d unexpected=%d"
          % (len(sd), len(missing), len(unexpected)))
    if missing:
        print("   MISSING:", list(missing)[:6])
    if unexpected:
        print("   UNEXPECTED:", list(unexpected)[:6])

    # the checkpoint carries the frozen tau; confirm it round-tripped
    print("[load] tau from checkpoint: tau_s=%.6f tau_t=%.6f"
          % (float(model.prior.tau_s.detach()), float(model.prior.tau_t.detach())))

    model.eval()
    acc = MetricAccumulator()
    with torch.no_grad():
        for batch in test_loader:
            x = batch["x_true"].to(args.device)
            cond = batch["cond_mask"].to(args.device)
            ev = batch["eval_mask"].to(args.device)
            ts = batch.get("timestamps")
            pred = model.impute(x, cond, n_steps=cfg["euler_steps"],
                                timestamps=None if ts is None else ts.to(args.device))
            acc.update(scaler.inverse_transform(pred),
                       scaler.inverse_transform(x), ev)
    got = acc.compute()
    print("\nre-evaluated : MAE %.5f  RMSE %.5f  MAPE %.2f%%  (n=%d)"
          % (got["mae"], got["rmse"], got["mape"], got["n"]))

    rel = abs(got["mae"] - recorded["mae"]) / max(recorded["mae"], 1e-12)
    same_n = got["n"] == recorded["n"]
    ok = rel <= args.tol and same_n
    print("delta MAE %.5f (%.2f%% relative) | n matches: %s" % (
        got["mae"] - recorded["mae"], 100 * rel, same_n))
    print("\n%s" % ("VERIFIED" if ok else "MISMATCH -- investigate"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
