# GiFlow Research Tracker

Living record for the SEED-IV imputation experiments. Add one row after every
controlled run; keep the default run as the control condition.

## Objective

Given an EEG window with 62 electrode channels and missing observations, recover
only the missing values while preserving the observed values. The main target
experiment is whole-electrode recovery: hide 50% of channels and reconstruct them
from the remaining channels and temporal context.

## Model in formulas

For a window `X` with conditioning mask `M`:

```text
X_obs = X * M
X_0 = exp(-tau_s L_s) X_obs exp(-tau_t L_t)
X_t = (1 - t) X_0 + t X_1
u_t = X_1 - X_0
L_flow = || M_target * (v_theta(X_t, X_obs, M, t) - u_t) ||^2
X_endpoint = X_t + (1 - t) v_theta(...)
L_preserve = || M_cond * (X_endpoint - X_1) ||^2
L_total = L_flow + preservation_weight * L_preserve
```

At inference, Euler integration starts at `X_0` and uses 20 steps by default:

```text
X <- X + (1 / n_steps) * v_theta(X, X_obs, M, t)
```

Observed values are restored at the end. With `--clamp-observed-each-step`,
they are restored after every Euler update as well.

## Current control baseline

Source: committed SEED-IV checkpoint, one subject, point missing 20%, seed 0.

| Condition | MAE | RMSE | MAPE | Status |
|---|---:|---:|---:|---|
| GiFlow | 0.14626 | 0.20604 | 55.60% | reproducible checkpoint |
| Linear interpolation | 0.12004 | 0.17954 | 48.20% | current baseline winner |

Filtering factors in the checkpoint: `tau_s=0.005303`, `tau_t=0.943472`.
All repository correctness checks pass; 51 checks pass across both test scripts.

The new controls also pass a one-epoch synthetic end-to-end smoke:
`channel` masking reported `eval_rate=0.500`, and the combined channel-drop,
spatial-floor, dropout, and per-step-clamp run completed successfully. This is
an interface check, not a quality result.

## Implemented experiment controls

| Control | CLI | Purpose |
|---|---|---|
| SRGDiff-inspired flow | `--variant srg_flow` | Residual velocity correction plus step-aware scale/bias regularization |
| Point missing | `--missing point --rho 0.2` | Existing scattered-value control |
| Block missing | `--missing block --rho 0.2` | Existing contiguous temporal corruption |
| Whole-channel missing | `--missing channel --rho 0.5` | Hide 50% of electrodes and score all their samples |
| Training electrode dropout | `--train-channel-drop-prob 0.5` | Randomly hide electrodes during training |
| Forced spatial smoothing | `--min-tau-s 0.25` | Prevent `tau_s` collapsing to zero |
| Per-step observed clamp | `--clamp-observed-each-step` | Clamp visible entries after each Euler update |
| Preservation off | `--preservation-weight 0` | Ablation for the preservation loss |
| Normalized graph | `--normalized-laplacian` | Alternative Laplacian scaling |
| Prior renormalization | `--prior-renormalize` | Correct shrinkage caused by masked zeros |

Defaults remain unchanged, so old runs are valid control runs.

## Recommended experiment sequence

Use the same subject split, seed, window, epochs, and output naming. Change one
factor at a time.

1. **Control:** existing point-missing configuration.
2. **Mask pattern:** `--missing channel --rho 0.5`.
3. **Training robustness:** channel evaluation plus `--train-channel-drop-prob 0.5`.
4. **Spatial prior:** repeat step 3 with `--min-tau-s 0.25`.
5. **Numerical path:** compare default final-only restoration with
   `--clamp-observed-each-step`.
6. **Loss ablation:** compare `--preservation-weight 0` and `0.1` under the
   same channel-missing setting.
7. **Regularization:** repeat the best configuration with higher dropout, for
   example `--dropout 0.3`, lower learning rate, and higher weight decay.
8. **Reliability:** run the final candidates over seeds 0..4 and report mean/std.

Example command:

```bash
python scripts/run.py --dataset seed4 --seed-root data/seed_iv_one.npz \
  --window 100 --stride 100 --missing channel --rho 0.5 \
  --train-channel-drop-prob 0.5 --min-tau-s 0.25 \
  --clamp-observed-each-step --dropout 0.3 --epochs 300 \
  --seeds 5 --out runs/seed4_channel50_spatial_clamp
```

## Experiment log

Use `pending` until the command has actually run. Never fill in an unmeasured
metric from expectation.

| ID | Change | PCC | NMSE | PSNR | SNR | MAE | RMSE | tau_s | tau_t | Best epoch | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| J0 | Joint reconstructed pair, GiFlow control | 0.7976 | 0.3674 | 40.52 | 4.35 | 4.324 | 10.648 | 0.0377 | 0.000165 | 21 | Euler 20; 4,000 windows/source |
| J1 | Joint reconstructed pair, SRG-flow | 0.7996 | 0.5476 | 38.79 | 2.62 | 6.947 | 12.999 | 0.0398 | 0.000080 | 5 | Euler 5; not directly comparable to J0 |
| C0 | Existing point 20% control | pending | pending | pending | pending | 0.14626 | 0.20604 | 0.0053 | 0.9435 | 1 | old checkpoint; new metrics pending |
| E1 | Channel missing 50% | pending | pending | pending | pending | pending | pending | pending | pending | pending | new evaluation mode |
| E2 | E1 + training channel dropout | pending | pending | pending | pending | pending | pending | pending | pending | pending | robustness training |
| E3 | E2 + `min_tau_s=0.25` | pending | pending | pending | pending | pending | pending | pending | pending | pending | forced spatial prior |
| E4 | E3 + per-step clamp | pending | pending | pending | pending | pending | pending | pending | pending | pending | trajectory ablation |
| E5 | Best candidate, 5 seeds | pending | pending | pending | pending | pending | pending | pending | pending | pending | final comparison |

J0/J1 both used seed 0, 50% channel masking, 0.5 training channel dropout,
the same 4,000-window-per-source split, and RTX 4050. The two roots contain
matched reconstruction variants of the same recordings, not independent
cohorts. Each test metric covers 2.48M hidden entries per source. Both
checkpoints were reevaluated on the same test batches at matched sampler settings:

| Model | Euler | PCC | NMSE | PSNR | SNR | MAE | RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| GiFlow | 5 | 0.7968 | 0.3658 | 40.54 | 4.37 | 4.305 | 10.624 |
| SRG-flow | 5 | 0.7996 | 0.5476 | 38.79 | 2.62 | 6.947 | 12.999 |
| GiFlow | 20 | 0.7976 | 0.3674 | 40.52 | 4.35 | 4.324 | 10.648 |
| SRG-flow | 20 | 0.8014 | 0.6909 | 37.78 | 1.61 | 8.211 | 14.602 |

The GiFlow checkpoint wins on NMSE/PSNR/SNR/MAE/RMSE at both step counts;
SRG-flow PCC is about 0.003 higher. Training checkpoint selection still used
different validation Euler counts, so this remains preliminary until both are
retrained with identical validation settings and multiple seeds.

### GPU diagnostic (synthetic only)

The local RTX 4050 successfully ran matched 50%-channel experiments. The
unchanged control scored MAE `5.2880`, RMSE `6.8859`, MAPE `403.56%`; the combined
new configuration scored MAE `5.4819`, RMSE `7.0000`, MAPE `430.24%`. Spatial
smoothing stayed active (`tau_s=0.4899`), but the combined change was 3.7% worse
on this one synthetic test. Keep this separate from the SEED-IV table and test
the factors one at a time before drawing a conclusion.

## Interpretation rules

- Compare against the same missing pattern; point missing and channel missing are
   different tasks and must not share one leaderboard.
- Report PCC/NMSE/PSNR/SNR, standard deviation across seeds, train/validation
   curves, final `tau_s` and `tau_t`, and best epoch. Joint runs macro-average
   dataset-level scores. Keep legacy MAE/RMSE/MAPE in the history JSON.
- A lower MAE with `tau_s` near zero does not demonstrate useful spatial learning.
- A higher MAE from forced smoothing is still informative: it tests whether the
   current graph is useful for EEG or whether the graph construction needs work.
- The preservation loss changes gradients through shared network parameters; it
   does not directly correct hidden-channel velocities. Per-step clamping changes
   the trajectory itself and is a separate ablation.

The runner now reports PCC, NMSE, PSNR, and the paper's SNR, while retaining
MAE/RMSE/MAPE for continuity. The committed C0 checkpoint predates those
metrics; its PCC/NMSE/PSNR/SNR values remain pending reevaluation.

## Next updates

After each run, append its output directory, exact command, metrics, factors,
and a one-sentence conclusion to the experiment log. Keep this file committed
with the corresponding `summary.json` and `history.json` so results remain
traceable.
