# GiFlow — Spatiotemporal Imputation with Graph-Informed Flow Matching

A from-scratch PyTorch implementation of **GiFlow** (Zhang, Einizade, Giraldo, Fink —
*Spatiotemporal Imputation with Graph-Informed Flow Matching*, ICML 2026,
arXiv:2606.06682), built by reading the paper only. No `torch-geometric` dependency —
the GNN layers are dense-adjacency implementations, which is fine at these graph sizes.

> **Status:** implementation complete and tested; results are *not* yet reproducing the
> paper's numbers. See [Current results](#current-results) and
> [Known problem: the filtering factors degenerate](#known-problem-the-filtering-factors-degenerate).

---

## What GiFlow does

Standard diffusion-based imputers start from a problem-agnostic Gaussian prior. GiFlow
instead builds a **graph-informed prior** by heat-kernel filtering the *observable*
signal over the spatial and temporal graphs, which puts the flow's source distribution
much closer to the target and so shortens the transport path:

```
X_tau = exp(-tau_eta L_eta) (X_1 * M) exp(-tau_xi L_xi)          # Eq. (4)
phi_t = (1 - t) X_tau + t X_1                                     # Eq. (9)  linear path
u_t   = X_1 - X_tau                                               # Eq. (10) target field
L     = E_t || M * ( v_t(X_t; theta, M, L) - X_1 + X_tau ) ||^2   # Sec. 3.2
```

`(tau_eta, tau_xi)` are *learned* by minimising Problem (5) on the training split, then
frozen. Sampling is a deterministic Euler integration from `X_tau` to `t = 1`.

## Paper → code map

| Paper | Code |
|---|---|
| Eq. (4) spatiotemporal filtering | `giflow/models/prior.py::GraphInformedPrior.forward` |
| Problem (5) adaptive `tau` | `giflow/models/prior.py::optimize_filtering_factors` |
| Eq. (6) Taylor truncation | `HeatKernel.matrix(mode="taylor")` |
| Prop. 3.1 truncation bound | `prior.py::truncation_bound` (verified in tests) |
| Thm. 3.2 transport cost | `GraphInformedPrior.transport_cost` |
| Eq. (9)/(10) path + field | `giflow/models/giflow.py::GiFlow.loss` |
| Eq. (11) spatial attention | `giflow/models/layers.py::SpatialAttention` |
| Eq. (12)/(13) temporal attention | `layers.py::TemporalAttention` |
| Eq. (14) ST propagation | `layers.py::SpatioTemporalPropagation` |
| Eq. (15) output head | `giflow/models/vector_field.py` |
| Euler sampling, 20 steps | `GiFlow.impute` |
| Point / block missing (Sec. 4) | `giflow/data/masking.py` |
| Synthetic data (Sec. 4.1) | `giflow/data/synthetic.py` |
| Appendix C.3 protocol | `giflow/train.py` |
| Baselines (Appendix C.2) | `giflow/baselines.py` |

GNN layers follow the papers GiFlow cites: `GraphConv` = Morris et al. (2019),
`SGConv` = Wu et al. (2019).

## Layout

```
giflow/
  data/   graph.py  masking.py  synthetic.py  dataset.py  datasets.py
  models/ prior.py  layers.py   vector_field.py  giflow.py
  train.py  metrics.py  baselines.py
scripts/  run.py  benchmark.py
tests/    test_giflow.py
```

## Install

```bash
python -m pip install -r requirements.txt
```

CPU-only PyTorch is enough (see [Hardware](#hardware-notes)).

## Usage

```bash
# smoke test (~1 min)
python scripts/run.py --dataset synthetic --nodes 20 --steps 200 \
    --epochs 3 --tau-epochs 5 --euler-steps 5 --hidden 32 --layers 2

# paper-scale synthetic, Table 1 setting
python scripts/run.py --dataset synthetic --nodes 50 --steps 3000 --sigma 0.1 \
    --missing point --rho 0.2 --threads 12

# real datasets (files must be in data/ first, see below)
python scripts/run.py --dataset air36 --missing block --rho 0.2
python scripts/run.py --dataset aqi    --missing point --rho 0.2
python scripts/run.py --dataset pems08 --missing point --rho 0.2

# 5 trials, as the paper reports (seeds 0..4)
python scripts/run.py --dataset synthetic --seeds 5

# ablations (Tables 4 and 5)
python scripts/run.py --variant fm_gauss          # Gaussian prior
python scripts/run.py --variant gfm               # spatial-only prior
python scripts/run.py --variant tfm               # temporal-only prior
python scripts/run.py --variant no_spatial_attn
python scripts/run.py --variant no_temporal_attn
python scripts/run.py --variant no_st_attn
python scripts/run.py --variant no_propagation

# correctness tests, and a timing benchmark for your machine
python tests/test_giflow.py
python scripts/benchmark.py --threads 12
```

Useful flags: `--max-windows` (cap the dataset), `--stride` (thin overlapping windows),
`--euler-steps`, `--max-tau`, `--normalized-laplacian`, `--prior-renormalize`,
`--threads`, `--skip-baselines`.

## Datasets

Synthetic works out of the box. The three real datasets are **not** included — put the
files in `data/` and the loaders will pick them up:

| Dataset | N | Timesteps | File(s) expected in `data/` | Source |
|---|---|---|---|---|
| Air-36 | 36 | 8760 (1 h) | `small36.h5` | GRIN release (`Graph-Machine-Learning-Group/grin`) |
| AQI | 437 | 8760 (1 h) | `full437.h5` | same |
| PeMS08 | 170 | 17856 (5 min) | `PEMS08.npz` (+ `distance08.csv`) | ASTGNN / STSGCN releases |

Optional real station distances: `air36_dist.npy` / `aqi_dist.npy`. Without them the
spatial graph falls back to a Gaussian kernel on the value profile, which is a weaker
graph than the paper's distance-based one.

## Current results

Synthetic, N=50, R=3000, sigma=0.1, point missing rho=20%, **seed 0 only, no
hyperparameter tuning**, early-stopped at epoch 24 (best epoch 16):

| Method | MAE | RMSE | MAPE |
|---|---|---|---|
| Mean-S | 25.86 | 31.73 | 114.3% |
| Mean-T | 1.001 | 1.303 | 17.1% |
| **Linear** | **0.396** | **0.512** | **5.7%** |
| KNN | 16.90 | 21.91 | 193.0% |
| FP | 17.27 | 22.30 | 206.7% |
| GiFlow (this code) | 1.072 | 1.446 | 12.0% |
| GiFlow (paper, Table 1) | 0.23 | 0.30 | 6.65% |

So **linear interpolation currently beats this implementation**, and the paper's number
is ~4.7x better than what this reproduces. Cause below. Not yet run: any real dataset,
multiple seeds, the hyperparameter search, or the ablations.

What *does* check out:

- Heat kernel matches `scipy.linalg.expm` to ~1e-7; rows sum to 1; symmetric.
- Proposition 3.1's truncation bound holds in every case tested (in float64 — at K=20
  float32 roundoff is larger than the bound itself).
- Theorem 3.2's direction is clear: transport cost **98.2** for the graph-informed prior
  vs **~2540** for a Gaussian prior on the same batch, a ~26x reduction.
- Imputation preserves observed entries exactly; 1-step Euler equals `prior + v_0`.
- Point and block missing hit their target rate exactly; block masks are contiguous.

## Known problem: the filtering factors degenerate

On this synthetic data Problem (5) drives

```
tau_s -> 0.0018   (essentially no spatial filtering)
tau_t -> 9.985    (pinned at the max_tau=10 bound)
```

`tau_t` runs to its bound because **Eq. (5)'s smoothness term
`tr(X_tau^T L_eta X_tau)` only involves the spatial Laplacian** — nothing in the
objective penalises large `tau_xi`. And the alignment term `||X_1 - X_tau||^2` is
genuinely minimised by heavy temporal averaging here, because the synthetic signal is a
slow random walk whose within-window mean is close to every sample in the window.

The result is a prior that is nearly flat in time and unfiltered in space — the opposite
of "graph-informed" — so the vector field has to regenerate all local temporal detail,
and linear interpolation (which exploits exactly that local continuity) wins. The paper
reports `tau` in the 0.6–1.2 range (Fig. 3), which this does not reproduce.

Note this is a property of Eq. (5) *as written* plus this particular signal, not
obviously an implementation bug — on real PM2.5 data, with strong diurnal structure, the
alignment term should bound `tau_xi` at a moderate value. Things to try:

1. `--max-tau 1.5` to force the paper's reported range.
2. `--prior-renormalize` — divides by the identically filtered mask, removing the
   shrinkage bias that zeros at missing entries introduce (a deviation from Eq. (4)).
3. `--normalized-laplacian` — with `L = D - A` and average degree ~6, `tau_s ~ 1` already
   collapses the spatial kernel to a global average, so the paper's `tau` range only
   looks sensible for a normalised Laplacian. A sweep is in the repo history.
4. Run the Appendix C.3 hyperparameter search; only defaults have been used so far.

## Hardware notes

Measured on the development machine — Intel Core i7-13700H (14 cores / 20 threads),
15.7 GB RAM, Intel Iris Xe integrated graphics, **no CUDA GPU**, `torch 2.12.0+cpu`:

| | |
|---|---|
| Model size | 163,841 trainable parameters |
| `tau` optimisation (30 epochs) | 23 s |
| Training epoch, synthetic N=50 | ~30–40 s steady state |
| Full run to early stop (25 epochs) | 65 min |
| Test pass (595 windows, 20 Euler steps) | 36 s |
| Peak RSS | ~0.73 GB |

The per-epoch validation pass runs the full 20-step Euler solver and costs about as much
as the training steps themselves. To cut wall-clock: `--euler-steps 5` (the paper's
Table 9 shows 5 steps loses very little), `--stride` > 1, `--max-windows`, or evaluate
less often. Run `python scripts/benchmark.py` to get projections for your own machine
and for the larger AQI (N=437) and PeMS08 (N=17856 steps) shapes.

## Deviations from the paper, and why

- **EMA warmup.** The paper's EMA decay is 0.9999. With warmup disabled the shadow
  weights sit at initialisation for thousands of steps, so early stopping fires on a
  frozen model. `min(decay, (1+step)/(10+step))` ramps it in.
- **Bounded `tau`.** Eq. (5) leaves `tau_xi` unbounded (see above), so a
  `max_tau`-scaled sigmoid is used by default. `--max-tau 0` restores the unbounded
  softplus.
- **Exact heat kernel by default.** Eq. (6) truncates the Taylor series; an
  eigendecomposition is exact, differentiable in `tau`, and cheap at these sizes.
  `--prior-mode taylor` uses the truncated form. Note the truncation is only usable for
  small `tau * spectral_radius` — at `tau = 1` with `L = D - A` the K=12 error is larger
  than the signal.
- **Training-time extra masking.** Written literally, the loss conditions on `X_1 * M`
  and is scored on the same `M`, which is trivially satisfiable. Following PriSTI/GRIN
  practice, a random subset of the visible entries is hidden each step and the loss is
  scored on what was hidden (`--train-keep-range`).
- **MAPE guard.** Targets with `|x| < 1e-2` are skipped, otherwise the ratio explodes.
- Window stride, batch size and the graph binarisation threshold for the synthetic set
  are not specified in the paper; defaults are stride 1, batch 32, threshold 0.1.

## License / attribution

Implementation code here is mine. The method, and all equation numbering referenced
above, are from the GiFlow paper (arXiv:2606.06682); the authors' own code is at
`github.com/zepengzhang/GiFlow`. The PDF is not redistributed in this repo.
