# Joint EEG Reconstruction with Graph-Informed Flow Matching

> **Working draft.** This document separates completed implementation and
> measured pilot results from planned experiments. Do not treat pilot values as
> final claims. The completed GiFlow/SRG-flow comparison is a single-seed
> pilot; repeated seeds, tau-floor study, within-dataset protocol, and
> feature/downstream metrics remain incomplete.

## Abstract

Electroencephalography (EEG) reconstruction from incomplete electrode
measurements is challenging when sensor failures remove entire channels or
contiguous time intervals. We study graph-informed flow matching (GiFlow) and
an SRGDiff-inspired residual-guided flow variant for reconstruction under
multiple structured missingness patterns. We establish a joint DEAP-RAW and
SEED-IV setting by mapping named electrodes to a shared 62-channel space,
normalizing each dataset independently, and sampling each source equally
during training. The evaluation distinguishes participant-held-out
generalization from a separate within-dataset window split, and measures
reconstruction on artificially hidden observations. Matched one-seed 50%-channel pilots
completed on an RTX 4050; both learned variants are preliminary and did not
outperform the KNN interpolation baseline on joint macro NMSE. Joint
training lowered held-out NMSE relative to individual training in all four
model/source comparisons, but these results use one seed and remain
preliminary. Multi-seed results and feature-level evaluations are pending.

## 1. Introduction

EEG systems can lose measurements through electrode disconnection, sensor
failure, or acquisition interruptions. These failure modes differ from
independent pointwise dropout: losing a channel removes its full temporal
trace, while a stream outage removes synchronized intervals across sensors.
Consequently, reconstruction methods should be compared under explicitly
defined masks and participant-independent protocols.

This study evaluates two related but distinct models. GiFlow uses a
graph-informed spatiotemporal heat-kernel prior to initialize a conditional
flow-matching trajectory. SRG-flow retains this prior and adds a learned
residual-direction correction and step-aware affine modulation. SRG-flow is an
adaptation of selected SRGDiff ideas to flow matching; it is not a
reimplementation of the VAE latent-diffusion method in SRGDiff.

The study is designed to answer:

1. How do GiFlow and SRG-flow compare across point, block, shared-stream, and
   50%-channel missingness?
2. Does joint training on DEAP-RAW and SEED-IV improve reconstruction relative
   to individual-dataset training?
3. Do participant-held-out results agree with within-dataset window-level
   results?
4. How do spatial and temporal lower bounds on the learned smoothing factors
   affect validation and test performance?

## 2. Methods

### 2.1 GiFlow conditional path

Let \(X_1\in\mathbb{R}^{N\times R}\) be a complete EEG window, \(M\) its
conditioning mask, and \(L_s,L_t\) the spatial and temporal graph Laplacians.
The graph-informed source is

\[
X_0 = \exp(-\tau_s L_s)(M\odot X_1)\exp(-\tau_t L_t).
\]

The flow-matching path and target velocity are

\[
X_t=(1-t)X_0+tX_1,\qquad u_t=X_1-X_0.
\]

The vector field is trained to match \(u_t\) on the training target mask. An
observed-value preservation term is included, and inference integrates the
learned field from \(X_0\) using Euler steps before restoring observed values.
Spatial and temporal filtering factors are optimized on training windows, then
frozen. Candidate positive lower bounds (`min_tau_s`, `min_tau_t`) are
validation ablations; no optimal floor has yet been established.

### 2.2 SRG-flow variant

SRG-flow augments the GiFlow velocity with a residual correction and
step-dependent affine calibration:

\[
v'_t=\gamma_t(v_t+\lambda_{\mathrm{res}}r_t)+\beta_t.
\]

The residual head is supervised toward the remaining target-velocity error;
the modulation head is regularized toward identity calibration. The source,
linear path, masked flow objective, and Euler solver remain those of GiFlow.
The adaptation must not be described as a diffusion denoiser or as an exact
SRGDiff reproduction.

### 2.3 Data and joint channel mapping

DEAP-RAW BDF recordings are read as 32 named EEG electrodes and resampled to
200 Hz. SEED-IV raw MATLAB recordings contain 62-channel trial arrays; trial
and participant boundaries are retained. The model operates on a shared
62-electrode union. Channels absent from a dataset are zero-padded and marked
unobserved, not interpreted as measured zero-valued EEG. Each dataset has its
own training-split scaler and graph contribution. Joint batches sample the
datasets with equal probability.

For the initial bounded pilot, 25 evenly spaced 100-sample windows were
selected per participant. The full comparative study will use identical
sampling, preprocessing, windows, splits, and random seeds across model
variants.

### 2.4 Missingness conditions

For \(M_{\mathrm{raw}}\) denoting genuinely available samples, artificial
evaluation masks are sampled only within available data. We consider:

- **Point:** independent sample removal at rate \(\rho\).
- **Block:** contiguous time blocks removed within channels.
- **Shared stream:** synchronized contiguous intervals removed across channels.
- **Whole channel:** complete electrode traces removed; the main stress test
  uses \(\rho=0.5\).

Metrics are computed only on artificially removed, genuinely available
entries. A separate training channel-dropout probability is used to test
robustness to additional electrode loss.

### 2.5 Evaluation protocols and baselines

The participant-held-out protocol assigns all windows of a participant to
exactly one split. In the current pilot, SEED-IV participants 7, 10, and 15
form the fixed test set; DEAP participants are assigned deterministically from
the split seed. The separate within-dataset protocol uses non-overlapping
sequential train/validation/test window blocks and is not evidence of
participant-independent generalization.

The runner includes spatial and temporal means, linear interpolation, KNN
spatial interpolation, and feature propagation. Learned models are compared
with identical masks, split seeds, training budgets, and sampler settings.
Joint results report source-specific metrics and a macro-average; participant
means and standard deviations are reported for held-out-subject evaluation.

### 2.6 Metrics

Primary signal-level measures follow the SRGDiff paper's NMSE, Pearson
correlation coefficient (PCC), and signal-to-noise ratio (SNR):

\[
\mathrm{NMSE}=\frac{\sum(\hat X-X)^2}{\sum X^2},\quad
\mathrm{SNR}=10\log_{10}\frac{\sum X^2}{\sum(\hat X-X)^2}.
\]

PCC is pooled over scored entries. PSNR, MAE, RMSE, and MAPE are also retained
for continuity with existing experiments. Since source scales differ between
DEAP and SEED-IV, cross-dataset MAE and PSNR are descriptive; normalized
source-specific measures are more appropriate for the macro comparison.

EEG-FID, frequency-domain/topographic errors, and downstream task accuracy are
not yet implemented or measured in this project. They must be added with
training-only feature models, a specified electrode interpolation protocol,
and dataset-appropriate labels before making claims about feature or task
utility.

## 3. Results

### 3.1 Matched participant-held-out 50%-channel pilot

Six runs compared joint and individual training for both models on seed 0.
They used the same participant split, 25 evenly sampled 100-sample windows
per participant, batch size 32, 20 tau-optimization epochs, 10 Euler steps,
50-epoch cap, patience 8, and 50% training channel dropout. Evaluation hid
50% of available named channels; masks were generated in canonical electrode
order, so the joint and individual runs hide the same electrodes for each
source. Graphs were recomputed from training windows only. No tau floor was
applied. The fixed held-out participants were DEAP 03, 05, 11, 12, 22, 26 and
SEED-IV 7, 10, 15. These are preliminary single-seed results, not final
performance estimates.

| Training | Model | Test source | PCC | NMSE | PSNR (dB) | SNR (dB) | \(\tau_s\) | \(\tau_t\) | Best epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Joint | GiFlow | DEAP-RAW | 0.6261 | 0.5920 | 26.08 | 2.28 | 0.0194 | 0.0381 | 2 |
| Joint | GiFlow | SEED-IV-MAT | 0.6313 | 0.6333 | 31.90 | 1.98 | 0.0194 | 0.0381 | 2 |
| Joint | SRG-flow | DEAP-RAW | 0.6219 | 0.5979 | 26.04 | 2.23 | 0.0193 | 0.0469 | 2 |
| Joint | SRG-flow | SEED-IV-MAT | 0.6248 | 0.6416 | 31.84 | 1.93 | 0.0193 | 0.0469 | 2 |
| Individual | GiFlow | DEAP-RAW | 0.5942 | 0.6325 | 25.79 | 1.99 | 2.5104 | 0.0528 | 40 |
| Individual | SRG-flow | DEAP-RAW | 0.4247 | 0.8456 | 24.53 | 0.73 | 2.5685 | 0.0543 | 2 |
| Individual | GiFlow | SEED-IV-MAT | 0.5351 | 0.7571 | 31.13 | 1.21 | 0.0476 | 0.0882 | 2 |
| Individual | SRG-flow | SEED-IV-MAT | 0.5409 | 0.7216 | 31.33 | 1.42 | 0.0466 | 0.0792 | 3 |
| Joint graph | KNN | DEAP-RAW | 0.6258 | 0.5931 | 26.07 | 2.27 | — | — | — |
| Joint graph | KNN | SEED-IV-MAT | 0.6258 | 0.6113 | 32.06 | 2.14 | — | — | — |
| Joint graph | Feature propagation | DEAP-RAW | 0.6167 | 0.6052 | 25.98 | 2.18 | — | — | — |
| Joint graph | Feature propagation | SEED-IV-MAT | 0.5913 | 0.6506 | 31.78 | 1.87 | — | — | — |

MAE and RMSE are shown separately because the datasets use different physical
scales. MAPE remains in the run histories but is not emphasized because
near-zero EEG samples can make percentage errors unstable.

| Training | Model | Test source | MAE | RMSE |
|---|---|---|---:|---:|
| Joint | GiFlow | DEAP-RAW | 0.009442 | 0.028177 |
| Joint | GiFlow | SEED-IV-MAT | 10.6430 | 16.2994 |
| Joint | SRG-flow | DEAP-RAW | 0.009729 | 0.028316 |
| Joint | SRG-flow | SEED-IV-MAT | 10.8282 | 16.4060 |
| Individual | GiFlow | DEAP-RAW | 0.011409 | 0.029123 |
| Individual | SRG-flow | DEAP-RAW | 0.014691 | 0.033675 |
| Individual | GiFlow | SEED-IV-MAT | 11.8966 | 17.8207 |
| Individual | SRG-flow | SEED-IV-MAT | 11.3516 | 17.3985 |
| Joint graph | KNN | DEAP-RAW | 0.009401 | 0.028201 |
| Joint graph | KNN | SEED-IV-MAT | 10.2885 | 16.0130 |
| Individual graph | KNN | DEAP-RAW | 0.009559 | 0.028279 |
| Individual graph | KNN | SEED-IV-MAT | 10.2461 | 15.9919 |

For this seed, joint training lowered NMSE relative to its individual
counterpart for both models on both sources. GiFlow achieved joint per-source
NMSE 0.5920 (DEAP) and 0.6333 (SEED-IV); SRG-flow achieved 0.5979 and 0.6416.
The joint KNN reference remained stronger on macro NMSE (0.6022) than either
learned model (GiFlow 0.6127; SRG-flow 0.6198). GiFlow slightly beat KNN on
DEAP NMSE alone, but not on SEED-IV; SRG-flow did not beat KNN on either
source. The per-source KNN baselines for the individual-graph runs were
NMSE 0.5964 (DEAP) and 0.6097 (SEED-IV), also better than the corresponding
individual learned models.

The equal-participant mean +/- standard deviation PCC/NMSE for joint GiFlow
was 0.124 +/- 0.421 / 0.691 +/- 0.477 (DEAP) and
0.618 +/- 0.131 / 0.639 +/- 0.145 (SEED-IV). SRG-flow gave
0.115 +/- 0.409 / 0.754 +/- 0.531 and
0.610 +/- 0.137 / 0.648 +/- 0.165, respectively. The very low DEAP
participant-average PCCs reflect substantial participant heterogeneity despite
the pooled values. Learned \(\tau_s\) was small for joint training but larger
in individual DEAP training; a lower bound is therefore an ablation to test,
not an assumed improvement.

Joint-versus-individual is a whole-pipeline comparison: individual runs
estimate a graph from that source's training participants, while joint runs
use the pairwise-averaged graph from both sources' training participants.
Thus the observed gap cannot be attributed solely to adding cross-dataset
training examples. All measurements above are one seed and one fixed split.
DEAP is in volts and SEED-IV retains its source scale, so pooled MAE/PSNR are
descriptive; report normalized metrics per source.

### 3.2 Remaining comparisons

| Experiment | Status |
|---|---|
| Joint/individual GiFlow and SRG-flow, held-out-participant 50%-channel pilot | Complete (one seed) |
| GiFlow vs. SRG-flow across multiple seeds | Pending |
| Point, block, and shared-stream missingness | Pending |
| Within-dataset window-level evaluation | Pending |
| Spatial/temporal tau-floor validation sweep | Pending |
| EEG-FID, spectral/topographic, downstream metrics | Not implemented |

## 4. Discussion and limitations

KNN remains a strong non-parametric reference: joint GiFlow only narrowly
improves DEAP NMSE, while KNN has better joint macro NMSE and outperforms the
individual learned models on both sources. A single seed and bounded
per-participant sampling are insufficient for broad conclusions. The
participant-held-out results have high between-participant dispersion,
particularly on DEAP. The learned spatial/temporal filtering factors also
vary strongly by training regime, motivating the planned validation-only
floor study rather than a claim that a fixed positive floor is beneficial.

The joint model combines datasets with different units, electrode coverage,
and acquisition protocols. Dataset-specific normalization and per-dataset
reporting mitigate but do not remove domain shift. The current pilot does not
include EEG-FID, frequency/topographic, or downstream-task evaluation and
therefore cannot support claims about distributional realism or clinical/task
utility. All final conclusions require matched multi-seed comparisons and the
separate participant-held-out and window-level protocols.

## 5. Figures and reproducibility

The model overview is in
[`../docs/giflow_srgflow_architecture.svg`](../docs/giflow_srgflow_architecture.svg).
Matched joint, DEAP-individual, and SEED-IV-individual NMSE/PCC figures are
[`figures/joint_subject_channel50.png`](figures/joint_subject_channel50.png),
[`figures/deap_individual_subject_channel50.png`](figures/deap_individual_subject_channel50.png),
and
[`figures/seed4_individual_subject_channel50.png`](figures/seed4_individual_subject_channel50.png).
Run arguments, histories, and baselines are preserved under the
`../runs/study_subject_matchedraw_*_channel50_seed0/` directories. The older
unmatched exploratory artifacts are retained separately and are not used for
the tables above.
