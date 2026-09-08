"""Training / evaluation loop for GiFlow (Appendix C.3 protocol).

Adam, max 300 epochs, early stopping on validation MAE with patience 10,
EMA of the weights with decay 0.9999, Euler solver with 20 steps.
The filtering factors are optimised first (Problem 5) on the training split and
then frozen for training and inference.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .data.dataset import make_loaders
from .data.masking import random_subset_mask
from .metrics import MetricAccumulator
from .models.giflow import EMA, GiFlow
from .models.prior import optimize_filtering_factors


@dataclass
class TrainConfig:
    # optimisation (search spaces from Appendix C.3)
    lr: float = 1e-3                  # {0.005, 0.001, 0.0005}
    weight_decay: float = 5e-5        # {0, 5e-4, 5e-5, 5e-3}
    dropout: float = 0.1              # {0, 0.1, 0.2, 0.3}
    n_mp_layers: int = 4              # {2, 4, 6, 8}
    hidden: int = 64                  # {32, 64, 128}
    emb_dim: int = 32
    k_hops: int = 2
    alpha_tau: float = 0.01           # {0.1, 0.01, 0.001, 0.0001}
    batch_size: int = 32
    max_epochs: int = 300
    patience: int = 10
    ema_decay: float = 0.9999
    euler_steps: int = 20
    grad_clip: float = 1.0
    # prior
    tau_s_init: float = 1.0
    tau_t_init: float = 1.0
    tau_epochs: int = 100
    tau_lr: float = 0.05
    tau_batch_size: int = 64
    prior_mode: str = "exact"
    prior_order: int = 10
    prior_renormalize: bool = False
    max_tau: float | None = 10.0
    normalized_laplacian: bool = False
    # variants
    gaussian_prior: bool = False          # FM-Gauss
    learn_spatial_prior: bool = True      # False -> TFM (temporal-only prior)
    learn_temporal_prior: bool = True     # False -> GFM (spatial-only prior)
    use_spatial_attention: bool = True
    use_temporal_attention: bool = True
    use_propagation: bool = True
    # training-time extra masking so the model predicts instead of copying
    train_keep_range: tuple = (0.3, 0.9)
    # misc
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    log_every: int = 1
    eval_every: int = 1
    out_dir: str = "runs/default"
    threads: int | None = None


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _timestamps(batch, device):
    ts = batch.get("timestamps")
    return None if ts is None else ts.to(device)


@torch.no_grad()
def evaluate(model, loader, scaler, cfg: TrainConfig, device: str) -> dict:
    """Impute the loader's windows and score on the eval mask, in original units."""
    model.eval()
    acc = MetricAccumulator()
    for batch in loader:
        x_true = batch["x_true"].to(device)
        cond = batch["cond_mask"].to(device)
        ev = batch["eval_mask"].to(device)
        pred = model.impute(
            x_true, cond, n_steps=cfg.euler_steps, timestamps=_timestamps(batch, device)
        )
        acc.update(scaler.inverse_transform(pred), scaler.inverse_transform(x_true), ev)
    return acc.compute()


def fit(
    x,
    adj_s,
    lap_s,
    lap_t,
    adj_s_norm,
    adj_t_norm,
    splits,
    scaler,
    cfg: TrainConfig,
    n_timestamp_classes: tuple = (),
    verbose: bool = True,
):
    """Run the full pipeline: tau optimisation, then flow-matching training."""
    if cfg.threads:
        torch.set_num_threads(cfg.threads)
    set_seed(cfg.seed)
    device = cfg.device
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = splits
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers
    )

    model = GiFlow(
        adj_s=adj_s,
        lap_s=lap_s,
        lap_t=lap_t,
        adj_s_norm=adj_s_norm,
        adj_t_norm=adj_t_norm,
        hidden=cfg.hidden,
        emb_dim=cfg.emb_dim,
        n_mp_layers=cfg.n_mp_layers,
        k_hops=cfg.k_hops,
        dropout=cfg.dropout,
        tau_s=cfg.tau_s_init,
        tau_t=cfg.tau_t_init,
        prior_mode=cfg.prior_mode,
        prior_order=cfg.prior_order,
        learn_spatial_prior=cfg.learn_spatial_prior,
        learn_temporal_prior=cfg.learn_temporal_prior,
        prior_renormalize=cfg.prior_renormalize,
        max_tau=cfg.max_tau,
        gaussian_prior=cfg.gaussian_prior,
        max_len=max(512, train_ds.window),
        n_timestamp_classes=n_timestamp_classes,
        use_spatial_attention=cfg.use_spatial_attention,
        use_temporal_attention=cfg.use_temporal_attention,
        use_propagation=cfg.use_propagation,
    ).to(device)

    history = {"tau": None, "epochs": [], "config": asdict(cfg)}

    # ---------------------------------------------------------- Stage 1: tau
    if not cfg.gaussian_prior:
        tau_loader, _, _ = make_loaders(
            train_ds, val_ds, test_ds, cfg.tau_batch_size, cfg.num_workers
        )
        if verbose:
            print("[stage 1] optimising filtering factors (Problem 5)")
        t0 = time.time()
        tau_res = optimize_filtering_factors(
            model.prior,
            tau_loader,
            alpha_tau=cfg.alpha_tau,
            epochs=cfg.tau_epochs,
            lr=cfg.tau_lr,
            device=device,
            verbose=verbose,
        )
        tau_res["seconds"] = time.time() - t0
        history["tau"] = tau_res
        if verbose:
            print(
                "[stage 1] done in %.1fs  tau_s=%.4f tau_t=%.4f"
                % (tau_res["seconds"], tau_res["tau_s"], tau_res["tau_t"])
            )
    # freeze the factors for training and inference
    for p in model.prior.parameters():
        p.requires_grad_(False)

    # ------------------------------------------- transport cost (Theorem 3.2)
    with torch.no_grad():
        tc, ntc = 0.0, 0
        for batch in val_loader:
            xt = batch["x_true"].to(device)
            cm = batch["cond_mask"].to(device)
            x0 = model.source_sample(xt * cm, cm)
            tc += float(model.prior.transport_cost(xt, x0).detach())
            ntc += 1
        history["transport_cost"] = tc / max(ntc, 1)
    if verbose:
        print("[info] transport cost E||X_1 - X_0||^2 = %.2f" % history["transport_cost"])
        print("[info] trainable params: %d" % count_params(model))

    # ------------------------------------------------------- Stage 2: flow FM
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ema = EMA(model, cfg.ema_decay)

    best = {"mae": float("inf"), "epoch": -1}
    bad = 0
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    ckpt = out_dir / "best.pt"
    t_start = time.time()

    for epoch in range(cfg.max_epochs):
        model.train()
        tot, nb = 0.0, 0
        te0 = time.time()
        for batch in train_loader:
            x_true = batch["x_true"].to(device)
            obs = batch["observed_mask"].to(device)
            cond_full = batch["cond_mask"].to(device)
            # hide a random subset of the visible entries; score on what we hid
            cond = random_subset_mask(cond_full, cfg.train_keep_range, generator=gen)
            target = cond_full - cond
            if float(target.sum()) == 0:
                target = cond_full
            loss = model.loss(x_true, cond, target, _timestamps(batch, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            ema.update(model)
            tot += float(loss.detach())
            nb += 1
        train_loss = tot / max(nb, 1)
        epoch_time = time.time() - te0

        rec = {"epoch": epoch, "train_loss": train_loss, "seconds": epoch_time}
        if (epoch + 1) % cfg.eval_every == 0:
            val = evaluate(ema.module(), val_loader, scaler, cfg, device)
            rec["val"] = val
            if val["mae"] < best["mae"] - 1e-6:
                best = {"mae": val["mae"], "epoch": epoch}
                torch.save(ema.module().state_dict(), ckpt)
                bad = 0
            else:
                bad += 1
        history["epochs"].append(rec)
        if verbose and (epoch % cfg.log_every == 0):
            v = rec.get("val", {})
            print(
                "epoch %3d | loss %.4f | val MAE %s | %.1fs | patience %d/%d"
                % (
                    epoch,
                    train_loss,
                    ("%.4f" % v["mae"]) if v else "-",
                    epoch_time,
                    bad,
                    cfg.patience,
                )
            )
        if bad >= cfg.patience:
            if verbose:
                print("[early stop] no val improvement for %d epochs" % cfg.patience)
            break

    history["train_seconds"] = time.time() - t_start
    history["best"] = best

    # --------------------------------------------------------------- test
    eval_model = ema.module()
    if ckpt.exists():
        eval_model.load_state_dict(torch.load(ckpt, map_location=device))
    t0 = time.time()
    test = evaluate(eval_model, test_loader, scaler, cfg, device)
    history["test"] = test
    history["test_seconds"] = time.time() - t0
    if verbose:
        print(
            "[test] MAE %.4f  RMSE %.4f  MAPE %.2f%%  (n=%d, %.1fs)"
            % (test["mae"], test["rmse"], test["mape"], test["n"], history["test_seconds"])
        )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2, default=str))
    return model, eval_model, history
