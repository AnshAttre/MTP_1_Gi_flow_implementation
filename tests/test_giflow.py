"""Correctness checks for the GiFlow implementation.

Run with:  python tests/test_giflow.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from giflow.data.dataset import build_splits, make_loaders
from giflow.data.datasets import load_synthetic
from giflow.data.graph import gcn_normalize, knn_adjacency, laplacian, line_graph_adjacency
from giflow.data.masking import make_eval_mask
from giflow.metrics import imputation_metrics
from giflow.models.giflow import EMA, GiFlow
from giflow.models.prior import GraphInformedPrior, HeatKernel, truncation_bound

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("  %s %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))


def graphs(n=20, r=12, seed=0):
    rng = np.random.default_rng(seed)
    adj_s = knn_adjacency(rng.uniform(0, 50, (n, 2)), 5)
    adj_t = line_graph_adjacency(r)
    return dict(
        adj_s=adj_s,
        lap_s=laplacian(adj_s),
        lap_t=laplacian(adj_t),
        adj_s_norm=gcn_normalize(adj_s),
        adj_t_norm=gcn_normalize(adj_t),
    )


def test_heat_kernel():
    print("\n[heat kernel]")
    from scipy.linalg import expm

    g = graphs()
    hk = HeatKernel(g["lap_s"], mode="exact")
    for tau in (0.05, 0.5, 2.0):
        m = hk.matrix(torch.tensor(tau)).numpy()
        err = np.abs(m - expm(-tau * g["lap_s"])).max()
        check("exact expm matches scipy at tau=%.2f" % tau, err < 1e-5, "maxerr %.2e" % err)
    m = hk.matrix(torch.tensor(0.5)).numpy()
    check("heat kernel rows sum to 1 (L1=0)", abs(m.sum(1) - 1).max() < 1e-5)
    check("heat kernel is symmetric", np.abs(m - m.T).max() < 1e-6)
    check("tau=0 gives the identity",
          np.abs(hk.matrix(torch.tensor(0.0)).numpy() - np.eye(len(m))).max() < 1e-5)

    # Proposition 3.1, in float64 to avoid float32 roundoff dominating
    Ls, Lt = g["lap_s"], g["lap_t"]
    Cs = np.abs(np.linalg.eigvalsh(Ls)).max()
    Ct = np.abs(np.linalg.eigvalsh(Lt)).max()
    X = np.random.default_rng(1).normal(size=(len(Ls), len(Lt)))

    def taylor(L, tau, K):
        acc = np.eye(len(L)); term = acc
        for k in range(1, K + 1):
            term = term @ L * (-tau / k)
            acc = acc + term
        return acc

    def tail(z, K):
        return max(math.exp(z) - sum(z ** k / math.factorial(k) for k in range(K + 1)), 0.0)

    ok, errs = True, []
    for tau in (0.1, 0.3):
        for K in (4, 8, 12):
            actual = np.linalg.norm(
                expm(-tau * Ls) @ X @ expm(-tau * Lt) - taylor(Ls, tau, K) @ X @ taylor(Lt, tau, K)
            )
            bnd = (tail(tau * Cs, K) * math.exp(tau * Ct)
                   + math.exp(tau * Cs) * tail(tau * Ct, K)) * np.linalg.norm(X)
            ok &= actual <= bnd
            errs.append(actual <= bnd)
    check("Proposition 3.1 truncation bound holds", ok, "%d/%d cases" % (sum(errs), len(errs)))

    # smaller tau needs fewer hops for the same error (the Prop 3.1 reading)
    e_small = np.linalg.norm(expm(-0.1 * Ls) @ X - taylor(Ls, 0.1, 4) @ X)
    e_large = np.linalg.norm(expm(-0.6 * Ls) @ X - taylor(Ls, 0.6, 4) @ X)
    check("truncation error grows with tau at fixed K", e_small < e_large,
          "%.2e < %.2e" % (e_small, e_large))


def test_prior():
    print("\n[graph-informed prior]")
    g = graphs()
    p = GraphInformedPrior(g["lap_s"], g["lap_t"], 0.3, 0.3)
    x = torch.randn(4, 20, 12)
    m = (torch.rand(4, 20, 12) > 0.2).float()
    out = p(x * m, m)
    check("prior output shape", tuple(out.shape) == (4, 20, 12))

    # Eq. (4): matrix form must equal the Kronecker-sum form on vec(X)
    fs = p.kernel_s.matrix(p.tau_s).detach().numpy()
    ft = p.kernel_t.matrix(p.tau_t).detach().numpy()
    xm = (x * m)[0].numpy()
    kron = np.kron(ft.T, fs) @ xm.reshape(-1, order="F")
    check("Eq.(4) matrix form == Kronecker form",
          np.abs(kron - (fs @ xm @ ft).reshape(-1, order="F")).max() < 1e-4)

    # tau -> 0 recovers the masked observation. Use the unbounded parameterisation:
    # with max_tau set, _to_raw floors the sigmoid at max_tau * 1e-6, so the residual
    # is the first-order term tau * L * x rather than zero.
    p0 = GraphInformedPrior(g["lap_s"], g["lap_t"], 1e-8, 1e-8, max_tau=None)
    dev = float((p0(x * m, m) - x * m).abs().max())
    check("tau->0 returns the masked signal", dev < 1e-4, "max dev %.2e" % dev)

    # and the residual must shrink linearly with tau
    devs = []
    for tau in (1e-3, 1e-4, 1e-5):
        pt = GraphInformedPrior(g["lap_s"], g["lap_t"], tau, tau, max_tau=None)
        devs.append(float((pt(x * m, m) - x * m).abs().max()))
    check("prior -> masked signal at rate O(tau)",
          devs[0] > devs[1] > devs[2] and devs[0] / devs[1] > 5,
          "devs %s" % ["%.1e" % d for d in devs])

    # Theorem 3.2: the graph-informed prior transports less than a Gaussian one
    b = load_synthetic(n_nodes=30, n_steps=400, seed=0)
    tr, va, te, sc, _ = build_splits(b.x, b.observed_mask, window=12, rho=0.2, seed=0)
    loader, _, _ = make_loaders(tr, va, te, batch_size=32)
    batch = next(iter(loader))
    xb, cb = batch["x_true"], batch["cond_mask"]
    gg = b.graphs(12)
    pri = GraphInformedPrior(gg["lap_s"], gg["lap_t"], 0.05, 0.3)
    with torch.no_grad():
        c_graph = float(pri.transport_cost(xb, pri(xb * cb, cb)))
        c_gauss = float(pri.transport_cost(xb, torch.randn_like(xb)))
    check("Theorem 3.2: cost(graph prior) <= cost(Gaussian)", c_graph <= c_gauss,
          "%.1f vs %.1f" % (c_graph, c_gauss))

    # ablation switches genuinely disable a direction
    gfm = GraphInformedPrior(g["lap_s"], g["lap_t"], 0.3, 0.3, learn_temporal=False)
    tfm = GraphInformedPrior(g["lap_s"], g["lap_t"], 0.3, 0.3, learn_spatial=False)
    check("GFM pins tau_t=0", float(gfm.tau_t) == 0.0)
    check("TFM pins tau_s=0", float(tfm.tau_s) == 0.0)

    # bounded parameterisation respects max_tau
    pb = GraphInformedPrior(g["lap_s"], g["lap_t"], 1.0, 1.0, max_tau=2.0)
    with torch.no_grad():
        pb.raw_tau_s.fill_(50.0)
    check("max_tau caps the filtering factor", float(pb.tau_s.detach()) <= 2.0 + 1e-4,
          "tau_s=%.4f" % float(pb.tau_s.detach()))


def test_vector_field_and_flow():
    print("\n[vector field / flow]")
    g = graphs()
    model = GiFlow(hidden=16, n_mp_layers=2, emb_dim=8, **g)
    x = torch.randn(3, 20, 12)
    m = (torch.rand(3, 20, 12) > 0.2).float()
    t = torch.rand(3)
    v = model.vector_field(x, x * m, m, t)
    check("vector field output shape", tuple(v.shape) == (3, 20, 12))

    loss = model.loss(x, m, m)
    loss.backward()
    n_grad = sum(1 for p in model.vector_field.parameters() if p.grad is not None
                 and p.grad.abs().sum() > 0)
    n_tot = sum(1 for _ in model.vector_field.parameters())
    check("loss is finite", torch.isfinite(loss).item(), "%.4f" % float(loss.detach()))
    check("gradients reach the vector field", n_grad > 0.8 * n_tot,
          "%d/%d tensors" % (n_grad, n_tot))

    # imputation preserves observed entries exactly
    with torch.no_grad():
        pred = model.impute(x, m, n_steps=4)
    check("imputation preserves observed entries",
          torch.allclose(pred * m, x * m, atol=1e-5))
    check("imputation output is finite", bool(torch.isfinite(pred).all()))

    # the Euler solver starts from the prior: 0 steps == the prior itself
    with torch.no_grad():
        p0 = model.source_sample(x * m, m)
        got = model.impute(x, m, n_steps=1)
        expect = p0 + model.vector_field(p0, x * m, m, torch.zeros(3))
    check("1-step Euler == prior + v_0",
          torch.allclose((got * (1 - m)), (expect * (1 - m)), atol=1e-4))

    # ablations build and run
    for kw in ({"use_spatial_attention": False}, {"use_temporal_attention": False},
               {"use_spatial_attention": False, "use_temporal_attention": False},
               {"use_propagation": False}, {"gaussian_prior": True}):
        mm = GiFlow(hidden=16, n_mp_layers=2, emb_dim=8, **g, **kw)
        with torch.no_grad():
            o = mm.impute(x, m, n_steps=2)
        check("ablation runs: %s" % list(kw), bool(torch.isfinite(o).all()))


def test_masking_and_metrics():
    print("\n[masking / metrics]")
    obs = np.ones((20, 200))
    for strat in ("point", "block"):
        ev = make_eval_mask(obs, strat, 0.3, seed=0)
        check("%s missing hits the target rate" % strat, abs(ev.mean() - 0.3) < 1e-6,
              "%.4f" % ev.mean())
        check("%s eval mask is a subset of observed" % strat, bool(((ev <= obs).all())))
    ev = make_eval_mask(obs, "block", 0.3, seed=0)
    runs = [np.diff(np.flatnonzero(ev[i])).tolist() for i in range(20) if ev[i].sum() > 1]
    contig = np.mean([np.mean([d == 1 for d in r]) for r in runs if r])
    check("block missing is mostly contiguous", contig > 0.7, "%.2f contiguous" % contig)

    # partially observed input: eval mask must never exceed availability
    obs2 = (np.random.default_rng(0).random((20, 200)) > 0.3).astype(float)
    ev2 = make_eval_mask(obs2, "point", 0.2, seed=0)
    check("eval mask respects genuine gaps", bool((ev2 * (1 - obs2)).sum() == 0))

    pred = np.array([[1.0, 2.0, 3.0]])
    tgt = np.array([[1.0, 4.0, 3.0]])
    msk = np.array([[0.0, 1.0, 0.0]])
    mt = imputation_metrics(pred, tgt, msk)
    check("metrics score only masked entries",
          abs(mt["mae"] - 2.0) < 1e-9 and abs(mt["rmse"] - 2.0) < 1e-9 and mt["n"] == 1,
          "mae=%.3f n=%d" % (mt["mae"], mt["n"]))
    check("MAPE is in percent", abs(mt["mape"] - 50.0) < 1e-6, "%.2f" % mt["mape"])


def test_splits_and_ema():
    print("\n[splits / EMA]")
    b = load_synthetic(n_nodes=15, n_steps=300, seed=0)
    tr, va, te, sc, info = build_splits(b.x, b.observed_mask, window=24, rho=0.2, seed=0)
    total = sum(info["split_sizes"])
    check("70/10/20 split", abs(info["split_sizes"][0] / total - 0.7) < 0.02
          and abs(info["split_sizes"][2] / total - 0.2) < 0.02, str(info["split_sizes"]))
    check("splits do not overlap in time",
          tr.indices.max() < va.indices.min() and va.indices.max() < te.indices.min())
    item = tr[0]
    check("cond_mask = observed and not eval",
          torch.allclose(item["cond_mask"], item["observed_mask"] * (1 - item["eval_mask"])))
    check("scaler roundtrips", abs(sc.inverse_transform(sc.transform(3.7)) - 3.7) < 1e-6)

    g = graphs(15, 24)
    model = GiFlow(hidden=8, n_mp_layers=1, emb_dim=4, **g)
    ema = EMA(model, decay=0.9999)
    before = next(iter(ema.shadow.parameters())).clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model)
    after = next(iter(ema.shadow.parameters()))
    check("EMA warmup actually moves the shadow",
          float((after - before).abs().max()) > 1e-3,
          "delta %.4f" % float((after - before).abs().max()))


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    test_heat_kernel()
    test_prior()
    test_vector_field_and_flow()
    test_masking_and_metrics()
    test_splits_and_ema()
    print("\n" + "=" * 60)
    print("%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        for f in FAIL:
            print("  FAILED: %s" % f)
    print("=" * 60)
    sys.exit(1 if FAIL else 0)
