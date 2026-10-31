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
| Forced temporal smoothing | `--min-tau-t <value>` | Prevent `tau_t` collapsing to zero; evaluate floors as a controlled sweep |
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
4. **Spatial prior:** repeat step 3 with candidate `--min-tau-s` floors.
5. **Temporal prior:** repeat step 3 with candidate `--min-tau-t` floors.
6. **Numerical path:** compare default final-only restoration with
   `--clamp-observed-each-step`.
7. **Loss ablation:** compare `--preservation-weight 0` and `0.1` under the
   same channel-missing setting.
8. **Regularization:** repeat the best configuration with higher dropout, for
   example `--dropout 0.3`, lower learning rate, and higher weight decay.
9. **Reliability:** run the final candidates over seeds 0..4 and report mean/std.

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

## Historical raw DEAP + SEED-IV participant-held-out pilot (superseded)

The results in this subsection predate canonical name-keyed evaluation masks.
Retain them only for run provenance; use the matched raw-data comparison below
for current joint-versus-individual conclusions.

Runs `study_subject_giflow_channel50_seed0` and
`study_subject_srgflow_channel50_seed0` completed on the RTX 4050. This is a
one-seed pilot comparison, not a final result. Both trained jointly
on DEAP-RAW and the original SEED-IV MATLAB files using 25 evenly spaced
100-sample windows per participant, 50% whole-channel evaluation masking,
50% training channel dropout, batch size 32, 20 tau-optimization epochs,
10 Euler steps, a maximum of 50 flow epochs, and patience 8. Early stopping
selected GiFlow epoch 10 and SRG-flow epoch 7. Training took 134.7 and 120.4
seconds, respectively, after data preparation. Learned factors were
`tau_s=0.0158, tau_t=0.0272` (GiFlow) and
`tau_s=0.0166, tau_t=0.0293` (SRG-flow). No tau floor was used.

SEED-IV test participants were 7, 10, and 15; DEAP used the deterministic
participant split from split seed 0 (test participants 03, 05, 11, 12, 22,
26). Validation and test participants were disjoint from training. GiFlow's
unchanged checkpoint was reevaluated after participant-level scoring was
added; its pooled metrics matched the original run.

| Model | Source | PCC | NMSE | PSNR (dB) | SNR (dB) | MAE | RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| GiFlow | DEAP-RAW | 0.6328 | 0.5931 | 26.19 | 2.27 | 0.00855 | 0.02781 |
| GiFlow | SEED-IV-MAT | 0.5608 | 0.7242 | 32.87 | 1.40 | 10.4459 | 17.8611 |
| GiFlow | Macro mean | 0.5968 | 0.6587 | 29.53 | 1.84 | 5.2272 | 8.9445 |
| SRG-flow | DEAP-RAW | 0.6315 | 0.5997 | 26.14 | 2.22 | 0.00878 | 0.02797 |
| SRG-flow | SEED-IV-MAT | 0.5938 | 0.7852 | 32.52 | 1.05 | 12.7264 | 18.5986 |
| SRG-flow | Macro mean | 0.6127 | 0.6925 | 29.33 | 1.64 | 6.3676 | 9.3133 |

KNN spatial interpolation had DEAP-RAW NMSE 0.5938 / PCC 0.6320 and
SEED-IV-MAT NMSE 0.6141 / PCC 0.6255, giving macro NMSE 0.6040 / PCC 0.6288.
Thus neither learned variant beat KNN on macro NMSE, and neither should be presented as a win. SRG-flow's
macro PCC was higher than GiFlow's, but its macro NMSE/SNR were worse.
Differences in physical scale (DEAP volts versus SEED-IV source units) make pooled MAE and
PSNR across datasets descriptive only; per-source normalized metrics are the
primary cross-dataset comparison. Participant-level scoring was added after
the GiFlow run and then recomputed from its unchanged checkpoint; its pooled
scores matched the original within 1e-5. Equal-participant mean +/- std PCC /
NMSE were 0.232 +/- 0.390 / 0.650 +/- 0.486 (DEAP) and
0.558 +/- 0.178 / 0.749 +/- 0.302 (SEED-IV) for GiFlow; corresponding
SRG-flow values were 0.226 +/- 0.402 / 0.669 +/- 0.462 and
0.580 +/- 0.116 / 0.798 +/- 0.144. Full run configuration and traces are in
`runs/study_subject_giflow_channel50_seed0/summary.json`,
`runs/study_subject_giflow_channel50_seed0/history.json`, and
`runs/study_subject_giflow_channel50_seed0/baselines.json`.

Matched SRG-flow traces are in
`runs/study_subject_srgflow_channel50_seed0/`. The comparison figure is
`paper/figures/subject_channel50_pilot_comparison.png`. The same channel-mask
comparison must be repeated for multiple seeds and candidate tau floors before
inference.

## Next updates

After each run, append its output directory, exact command, metrics, factors,
and a one-sentence conclusion to the experiment log. Keep this file committed
with the corresponding `summary.json` and `history.json` so results remain
traceable.

### Channel-mask alignment correction

The original joint and individual channel pilots used the same seed but did
not guarantee the same named electrodes were hidden: mask sampling followed
the local/aligned row order. A fixed seed now samples in canonical electrode
name order, making per-electrode masks invariant to row reordering and
union-padding. Regression tests verify this for point, block, channel, and
shared-stream masks. The initial joint-versus-individual scores remain
unmatched exploratory results and are excluded from the comparison below.

### Matched raw-data 50%-channel comparison

Six seed-0 runs completed on the RTX 4050 with raw DEAP BDF and raw SEED-IV
MATLAB sources, participant-held-out splits, canonical electrode masks, 25
evenly sampled 100-sample windows per participant, 50% channel training
dropout, batch size 32, 20 tau-optimization epochs, 10 Euler steps, 50-epoch
cap, and patience 8. The runner estimates each source graph from its training
windows; joint runs average the source graphs. Fixed test groups are DEAP
03/05/11/12/22/26 and SEED-IV 7/10/15. No tau floor was applied.

| Training | Model | Source | PCC | NMSE | PSNR | SNR | tau_s | tau_t | Best epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Joint | GiFlow | DEAP-RAW | 0.6261 | 0.5920 | 26.08 | 2.28 | 0.0194 | 0.0381 | 2 |
| Joint | GiFlow | SEED-IV-MAT | 0.6313 | 0.6333 | 31.90 | 1.98 | 0.0194 | 0.0381 | 2 |
| Joint | SRG-flow | DEAP-RAW | 0.6219 | 0.5979 | 26.04 | 2.23 | 0.0193 | 0.0469 | 2 |
| Joint | SRG-flow | SEED-IV-MAT | 0.6248 | 0.6416 | 31.84 | 1.93 | 0.0193 | 0.0469 | 2 |
| Individual | GiFlow | DEAP-RAW | 0.5942 | 0.6325 | 25.79 | 1.99 | 2.5104 | 0.0528 | 40 |
| Individual | SRG-flow | DEAP-RAW | 0.4247 | 0.8456 | 24.53 | 0.73 | 2.5685 | 0.0543 | 2 |
| Individual | GiFlow | SEED-IV-MAT | 0.5351 | 0.7571 | 31.13 | 1.21 | 0.0476 | 0.0882 | 2 |
| Individual | SRG-flow | SEED-IV-MAT | 0.5409 | 0.7216 | 31.33 | 1.42 | 0.0466 | 0.0792 | 3 |

The joint KNN baseline scored PCC 0.6258 / NMSE 0.5931 on DEAP and
0.6258 / 0.6113 on SEED-IV (pooled NMSE 0.6022). Individual-graph KNN NMSEs
were 0.5964 on DEAP and 0.6097 on SEED-IV. Joint training lowered model NMSE
relative to individual training on both sources for both architectures in this
single split. This is preliminary: the joint graph also differs from the
source-specific individual graph, so the observed gap is not attributable to
training samples alone. Joint GiFlow pooled NMSE 0.6127 and SRG-flow 0.6198
both trail joint KNN. One seed is not enough for a performance claim.

Exact run artifacts:

- `runs/study_subject_matchedraw_joint_giflow_channel50_seed0/`
- `runs/study_subject_matchedraw_joint_srgflow_channel50_seed0/`
- `runs/study_subject_matchedraw_deap_giflow_channel50_seed0/`
- `runs/study_subject_matchedraw_deap_srgflow_channel50_seed0/`
- `runs/study_subject_matchedraw_seed4_giflow_channel50_seed0/`
- `runs/study_subject_matchedraw_seed4_srgflow_channel50_seed0/`

Baseline-matched figures are `paper/figures/joint_subject_channel50.png`,
`paper/figures/deap_individual_subject_channel50.png`, and
`paper/figures/seed4_individual_subject_channel50.png`.

The `incomplete_wrong_root_*` folders contain partial diagnostics from an
incorrect default SEED root and are not part of these results. Pass
`--seed-root data/seed4/eeg_raw_data` explicitly when the parent folder also
contains processed recordings.

### Prior renormalization and channel-only training masks (Kaggle T4)

Question: do the renormalized heat-kernel prior (`--prior-renormalize`) and a
training mask that hides whole electrodes only (`--train-keep-range 1 1`)
close the gap to KNN? Kernel `anshattre/giflow-seed4-improve`, commit
`f8e2fc8`, packed one-subject SEED-IV (`seed_iv_one.npz`, window split, not
the participant-held-out raw-data setup above). All runs: 50% channel
masking, `--train-channel-drop-prob 0.5`, 20 tau epochs, 10 Euler steps,
50-epoch cap, patience 8, seeds 0-2. Each seed also changes which channels are
hidden, so the seed-0 column is the only one matched to the baselines.

| Run | Change | NMSE mean +/- std | PCC mean +/- std | Seed-0 NMSE | tau_s (seed 0) |
|---|---|---:|---:|---:|---:|
| A | paper training config | 0.4445 +/- 0.0354 | 0.7660 +/- 0.0215 | 0.4396 | 0.1030 |
| B | A + renormalized prior | 0.4622 +/- 0.0673 | 0.7375 +/- 0.0436 | 0.5482 | 0.0337 |
| C | A + whole-electrode training mask | 0.4320 +/- 0.0614 | 0.7714 +/- 0.0270 | 0.3967 | 0.1030 |
| D | B + C | 0.4455 +/- 0.0586 | 0.7494 +/- 0.0386 | 0.5141 | 0.0337 |

Seed-0 baselines: KNN NMSE 0.3356 / PCC 0.8177, FP NMSE 0.3139 / PCC 0.8310.

Conclusions (one subject, three seeds):

- No variant beats KNN or FP; the best seed-0 run (C, 0.3967) trails KNN by 18%.
- C lowers mean NMSE by 2.8% versus A, but the difference is smaller than the
  seed standard deviation. Treat it as not established.
- The renormalized prior hurts after training (B, D), although on its own it
  matched KNN in a local prior-only check. Its tau_s fits much lower
  (0.015-0.034), so hidden channels get a near-zero denominator. Next check:
  score the prior alone against the trained flow, and retry with a tau_s floor.
- The raw DEAP + SEED-IV paper numbers were not rerun; that data is not on Kaggle.
