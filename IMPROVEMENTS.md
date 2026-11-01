# GiFlow for EEG Channel Recovery: Improvement Log

Reference log of every change made to improve reconstruction under 50%
electrode loss: what was changed, why, and what it did to the metrics.
Detailed per-run tables and provenance are in `RESEARCH_TRACKER.md`.

Last updated: 2026-11-01.

## Summary

**Headline (2026-11-01, literature-comparable setup).** On all 15 SEED-IV
subjects, band-passed 0.5–75 Hz at 200 Hz, with participants 7, 10 and 15
held out, **Anchored Flow with a spherical-spline source + spline blend
reaches NMSE 0.1452 ± 0.0058 (PCC 0.927)**. The paper configuration scores
0.3350 on the same setup, so this is **56.7% lower**. It beats the spherical
spline (0.1496), KNN (0.2905) and FP (0.3216) on every seed, but the margin
over the spline is small (1.7–3.6% per seed). On one subject (window split)
it reaches **0.1273 ± 0.0009**, 11% below the spline (0.1430).

| Setup (filtered, 3 seeds) | Paper config | KNN | FP | Spherical spline | **Ours** |
| --- | --- | --- | --- | --- | --- |
| All subjects, held-out participants | 0.3350 | 0.2905 | 0.3216 | 0.1496 | **0.1452** |
| One subject, window split | — | 0.2843 | 0.2621 | 0.1430 | **0.1273** |

Earlier development results (one subject, unfiltered): Anchored Flow + FP
blend reached 0.2930 ± 0.0130, 34.1% below the manuscript configuration
(0.4445) and 13% below KNN and FP, but not below the spherical spline (0.2720).

- KNN and FP were weak baselines; the spherical spline, built from electrode
  positions, is the baseline to beat.
- Most of the gain came from one fix: the model was never trained on the
  electrodes it is scored on. Fixing that alone cut NMSE by 15%.
- The remaining gains came from starting the flow at a strong classical
  estimate (FP, then a learned per-electrode mixture of heat-kernel, FP and
  KNN) and blending the output with FP.
- With the spline in the mixture, it takes about 55% of the source weight,
  and the trained flow adds a correction on top.

## Task, setup and metrics

The task is whole-electrode recovery: 31 of the 62 SEED-IV electrodes are
hidden for the whole window and must be reconstructed from the other 31.
Only the hidden entries are scored.

| Item | Development setup | Literature-comparable setup |
| --- | --- | --- |
| Data | One SEED-IV subject, packed `X_prc1`, 250 Hz | All 15 subjects (17 recordings) |
| Preprocessing | None | 0.5–75 Hz band-pass per 4 s trial, resampled to 200 Hz |
| Split | Windows 70/10/20 in time | Participants held out: test 7, 10, 15; validation 6, 12 |
| Window | 100 samples | 100 samples, up to 15,000 windows |
| Hidden electrodes | 50%, chosen per seed, fixed across the recording | Same |
| Training | Paper config: channel drop 0.5, 20 tau epochs, 10 Euler steps, 50-epoch cap, patience 8 | Same |
| Seeds | 0, 1, 2 (each seed also hides different electrodes) | 0, 1, 2 |
| Hardware | Kaggle T4 | Kaggle T4 |

All metrics are computed on hidden entries only, in source units. With
prediction X̂, reference X and hidden-entry mask Q:

```
NMSE = sum(Q * (X̂ - X)^2) / sum(Q * X^2)                 (lower is better; primary)
SNR  = 10 * log10( sum(Q * X^2) / sum(Q * (X̂ - X)^2) )    (dB, higher is better)
PSNR = 20 * log10( range(X on Q) / RMSE )                 (dB, higher is better)
PCC  = Pearson correlation over scored entries            (higher is better)
```

Standard deviations are population standard deviations over seeds
(`np.nanstd`, ddof = 0). With three seeds the sample standard deviation
(ddof = 1) is about 22% larger; state which one the paper uses.

## Results timeline

Development setup, mean ± std over seeds 0–2. "vs paper" is the NMSE change
relative to the paper configuration (step 0).

| Step | Change | Flags added | NMSE | PCC | PSNR (dB) | SNR (dB) | vs paper | Commit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | Paper configuration | — | 0.4445 ± 0.0354 | 0.7660 ± 0.0215 | 24.12 ± 0.22 | 3.53 ± 0.34 | — | `ecf4171` |
| 1 | Whole-electrode training mask | `--train-keep-range 1 1` | 0.4320 ± 0.0614 | 0.7714 ± 0.0270 | 24.27 ± 0.45 | 3.69 ± 0.59 | −2.8% | `f8e2fc8` |
| 2 | FP flow source (alone) | `--prior-source fp` | 0.4257 ± 0.0142 | 0.7596 ± 0.0101 | 24.30 ± 0.28 | 3.71 ± 0.14 | −4.2% | `9cb5fcd` |
| 3 | Train on the eval electrodes | `--train-reveal-eval` (+ step 1) | 0.3771 ± 0.0154 | 0.7904 ± 0.0087 | 24.83 ± 0.09 | 4.24 ± 0.18 | −15.2% | `9cb5fcd` |
| 4 | Steps 2 + 3 | both | 0.3538 ± 0.0195 | 0.8087 ± 0.0093 | 25.11 ± 0.09 | 4.52 ± 0.23 | −20.4% | `9cb5fcd` |
| 5 | Blend with FP | `--blend-fp` | 0.3181 ± 0.0172 | 0.8270 ± 0.0110 | 25.57 ± 0.11 | 4.98 ± 0.23 | −28.4% | `77329c4` |
| 6 | **Anchored Flow** (learned source mixture) + blend | `--prior-source mix` | **0.2930 ± 0.0130** | **0.8416 ± 0.0080** | **25.92 ± 0.10** | **5.34 ± 0.19** | **−34.1%** | `241265d` |
| — | KNN baseline | — | 0.3394 ± 0.0096 | 0.8153 | — | — | — | — |
| — | FP baseline | — | 0.3360 ± 0.0185 | 0.8186 | — | — | — | — |

Per-seed NMSE of the best model: 0.2775 / 0.3094 / 0.2921, against
KNN 0.3356 / 0.3526 / 0.3302 and FP 0.3139 / 0.3591 / 0.3350.

Full command for the best model:

```bash
python scripts/run.py --dataset seed4 --seed-root data/seed_iv_one.npz \
  --window 100 --stride 100 --missing channel --rho 0.5 \
  --train-channel-drop-prob 0.5 --tau-epochs 20 --euler-steps 10 \
  --epochs 50 --patience 8 --batch-size 32 --seeds 3 --device cuda \
  --train-keep-range 1 1 --train-reveal-eval --prior-source mix --blend-fp
```

## Results on the literature-comparable setup

All 15 subjects, band-pass 0.5–75 Hz, 200 Hz, held-out participants
7 / 10 / 15 (validation 6 / 12), up to 15,000 windows, seeds 0–2, Kaggle T4.
Baselines are recomputed on every seed's mask.

| Method | NMSE | PCC | PSNR (dB) | SNR (dB) | Per-seed NMSE | Commit |
| --- | --- | --- | --- | --- | --- | --- |
| Paper configuration | 0.3350 ± 0.0183 | 0.8203 | 32.79 | 4.76 | 0.3210 / 0.3232 / 0.3609 | `3a518fe` |
| Anchored Flow (heat/FP/KNN), flow only | 0.2381 ± 0.0106 | 0.8755 | 34.27 | 6.24 | 0.2305 / 0.2306 / 0.2532 | `3a518fe` |
| Anchored Flow + FP blend | 0.2196 ± 0.0047 | 0.8847 | 34.62 | 6.59 | 0.2162 / 0.2164 / 0.2262 | `3a518fe` |
| + ensemble of 3 | 0.2186 ± 0.0038 | 0.8851 | 34.64 | 6.60 | 0.2149 / 0.2170 / 0.2238 | `f3a3b3c` |
| + test-time adaptation (100 steps) | 0.2361 ± 0.0032 | 0.8779 | 34.30 | 6.27 | 0.2345 / 0.2332 / 0.2406 | `f3a3b3c` |
| Anchored Flow with spline source, flow only | 0.1511 ± 0.0051 | 0.9245 | 36.24 | 8.21 | 0.1548 / 0.1547 / 0.1439 | `144d638` |
| **Anchored Flow with spline source + spline blend** | **0.1452 ± 0.0058** | **0.9268** | **36.42** | **8.38** | 0.1457 / 0.1521 / 0.1379 | `144d638` |
| Spherical spline | 0.1496 ± 0.0060 | 0.9245 | — | — | 0.1482 / 0.1576 / 0.1430 | — |
| KNN | 0.2905 ± 0.0069 | 0.8455 | — | — | 0.2809 / 0.2963 / 0.2944 | — |
| FP | 0.3216 ± 0.0076 | 0.8345 | — | — | 0.3109 / 0.3260 / 0.3279 | — |

Same filtering, one subject, window split (commit `144d638`):

| Method | NMSE | PCC | PSNR (dB) | SNR (dB) | Per-seed NMSE |
| --- | --- | --- | --- | --- | --- |
| Anchored Flow with spline source, flow only | 0.1289 ± 0.0007 | 0.9348 | 34.82 | 8.90 | 0.1299 / 0.1286 / 0.1283 |
| **+ spline blend** | **0.1273 ± 0.0009** | **0.9355** | **34.88** | **8.95** | 0.1260 / 0.1283 / 0.1275 |
| Spherical spline | 0.1430 ± 0.0059 | 0.9271 | — | — | 0.1360 / 0.1504 / 0.1427 |
| KNN | 0.2843 ± 0.0050 | 0.8474 | — | — | 0.2781 / 0.2846 / 0.2902 |
| FP | 0.2621 ± 0.0073 | 0.8704 | — | — | 0.2531 / 0.2621 / 0.2710 |

Observations:

- **Held-out participants.** The flow alone (0.1511) is slightly worse than
  the spline (0.1496); only the blend beats it, on every seed (−1.7%, −3.5%,
  −3.6%). Blend weights on the model were low (0.25 / 0.40 / 0.54).
- **Overfitting across participants.** Validation NMSE was best after the
  first epoch on all seeds (e.g. 0.1830 → 0.1968 by epoch 8 on seed 2) while
  training loss kept falling. The flow learns subject-specific corrections
  that do not transfer. On one subject, where train and test share the
  participant, the flow clearly beats the spline (−11%).
- Mixture weights with the spline source: heat 0.07–0.10, FP 0.19–0.22,
  KNN 0.14–0.18, spline 0.54–0.56.
- The ensemble of three gave only −0.5%; test-time adaptation made things
  worse (see "What did not help").

## What each improvement does

### 1. Whole-electrode training mask (`--train-keep-range 1 1`)

- **Problem.** During training the model kept only 30–90% of visible
  entries as context, then also dropped whole electrodes. It saw much sparser
  context than at evaluation, where the other 31 electrodes are fully visible.
- **Change.** Keep all visible points and hide whole electrodes only, so
  training masks match the evaluation pattern.
- **Effect.** NMSE −2.8%, but smaller than the seed spread on its own. It
  matters as part of step 3.

### 2. FP flow source (`--prior-source fp`)

- **Problem.** GiFlow starts the flow at a heat-kernel smoothing of the
  zero-filled signal, which shrinks hidden electrodes towards zero. Transport
  cost E‖X₁ − X₀‖² was about 2,545 on the smoke test.
- **Change.** Start the flow at the feature-propagation (FP) estimate, the
  strongest classical baseline. The flow then learns a correction to FP.
- **Why it is principled.** GiFlow's Theorem 3.2 says a source closer to the
  target lowers the transport cost. FP halved it (about 1,284).
- **Effect.** NMSE −4.2% alone, −6.2% on top of step 3.

### 3. Train on the evaluation electrodes (`--train-reveal-eval`), the key fix

- **Problem.** The evaluation mask was built once over the whole signal, so
  the 31 scored electrodes were hidden in the training windows too. The model
  never had a training target on the electrodes it is scored on.
- **Change.** Training windows show all electrodes; random electrodes are
  hidden as targets during training. Validation and test windows keep the
  evaluation mask, so scoring is unchanged.
- **Effect.** NMSE −15.2% (largest single gain), PCC 0.766 → 0.790.
- **Paper note.** This must be stated in the methods. With a window split it
  is the "electrode fails later in the recording" setting; with a participant
  split it is "the electrode is missing for a new participant", and training
  participants legitimately have all electrodes.

### 4. Blend with FP (`--blend-fp`)

- **Change.** Final output = w × model + (1 − w) × FP. The weight w is the
  least-squares fit on the validation hidden entries, clipped to [0, 1].
  The test split is never used to choose it.
- **Why it works.** The model and FP make different errors, so a weighted
  average cancels part of both. NMSE is a squared error, so averaging helps it
  directly.
- **Effect.** NMSE −10.1% versus step 4 (0.3538 → 0.3181); first result to
  beat both KNN and FP on every seed. Fitted w = 0.35 / 0.35 / 0.44.

### 5. Anchored Flow: learned per-electrode source mixture (`--prior-source mix`)

- **Change.** For each electrode, the flow source is a softmax-weighted
  mixture of three classical estimates: heat-kernel prior, FP and KNN. The
  weights are fit after the tau stage by minimising the hidden-entry transport
  cost, hiding electrodes as in training so every electrode gets a weight.
  They are frozen during flow training.
- **Why it is novel.** GiFlow uses one fixed prior. We learn which classical
  estimator to trust at each electrode, justified by the same transport-cost
  argument. No prior work applies flow matching to EEG channel recovery.
- **Effect.** NMSE 0.3181 → 0.2930 (−7.9%), better on every seed.
- **Insight for the paper.** The learned weights are about heat 0.13,
  FP 0.42, KNN 0.45 on every seed. GiFlow's own heat-kernel prior is the
  weakest source for EEG.

### 6. Literature preprocessing (`--bandpass 0.5 75 --resample-hz 200`)

- **Finding.** 54% of the signal power in the packed data is below 0.5 Hz
  (slow drift). SRGDiff, EMAG and DSF all band-pass 0.5–75 Hz at 200 Hz.
- **Change.** Band-pass and resample each 4 s trial separately (packed trials
  are separate epochs).
- **Effect on baselines** (one subject, seeds 0–2): KNN 0.340 → 0.284,
  FP 0.336 → 0.262. The same filtering applies to every method, so
  comparisons stay fair, and our numbers become comparable with the
  literature. Model results on filtered data are pending.

### 7. Multi-model ensemble (`--ensemble K`)

- K models per seed with different initialisations on the same split and
  masks; their mean is scored, and the mean can be blended with FP.
- **Result** (all subjects, held-out, K = 3): 0.2196 → 0.2186 after the blend
  (−0.5%), at three times the training cost. Not worth it on its own.

### 8. Test-time adaptation (`--tta-steps N`), did not help

- For each test participant, a copy of the vector field is fine-tuned on that
  participant's visible electrodes only (some hidden as targets), then the
  evaluation electrodes are imputed. Aimed at the cross-participant gap.
- A unit test replaces every evaluation entry with noise and checks the
  adapted prediction is unchanged; a control shows the same noise on visible
  entries does change it. TTA cannot see the hidden data.
- **Result** (all subjects, held-out, 100 steps at the default rate): it
  hurt, 0.2196 → 0.2361 after the blend and 0.2381 → 0.2633 for the flow
  alone, on every seed. Fine-tuning on 31 visible electrodes overfits them.

### 9. Spherical spline as baseline and mixture source (`--spline`)

- **Finding.** The spherical spline (Perrin et al., 1989), built from the
  10-05 electrode positions, is far stronger than KNN and FP on this data.
  KNN and FP use a correlation graph; the spline uses head geometry.

  | Data (one subject, seeds 0–2) | Spherical spline | KNN | FP | Best model so far |
  | --- | --- | --- | --- | --- |
  | Unfiltered | **0.2720** (0.2789 / 0.2711 / 0.2661) | 0.3394 | 0.3360 | 0.2930 |
  | Filtered 0.5–75 Hz, 200 Hz | **0.1430** (0.1360 / 0.1504 / 0.1427) | 0.2843 | 0.2621 | 0.1273 (with spline source) |

- **Consequence.** The spline must be in the paper's baseline table; the
  earlier "beats the baselines" claim only held against KNN and FP.
- **Change.** Positions come from MNE's 10-05 montage (CB1/CB2 mapped to
  I1/I2 so all 62 positions are distinct), fitted to a sphere. The spline is
  scored as a baseline and added as a fourth source of the Anchored Flow
  mixture; `--blend-with spline` blends the output with the spline.
- **Smoke test** (filtered, 1 epoch, small subset; not a result): spline
  0.1664, Anchored Flow with spline source 0.1430, plus spline blend 0.1409.
  The fitted mixture put 54% of its weight on the spline.
- **Result** (Kaggle kernel `anshattre/giflow-seed4-spline`, commit
  `144d638`): 0.1273 on one subject (spline 0.1430) and 0.1452 on held-out
  participants (spline 0.1496). Full tables are in "Results on the
  literature-comparable setup" above.

## What did not help

| Change | Result (NMSE, 3 seeds) | Why |
| --- | --- | --- |
| Renormalised heat-kernel prior (`--prior-renormalize`) | 0.4622 vs 0.4445 (worse) | Alone it matched KNN, but its fitted tau_s collapsed to 0.015–0.034, giving hidden electrodes a near-zero denominator and an unstable source. |
| Longer training (150 epochs, patience 15) | 0.3181, identical | Early stopping picked the same epochs (2 / 7 / 2). |
| Test-time adaptation, 100 steps (`--tta-steps 100`) | 0.2361 vs 0.2196 (worse, every seed) | Fine-tuning on the visible electrodes of one participant overfits them. |
| Ensemble of 3 (`--ensemble 3`) | 0.2186 vs 0.2196 (−0.5%) | Members make nearly the same errors; three times the cost. |
| Blend with the mixture source instead of FP (`--blend-with source`) | No gain on smoke test (w = 1.0) | The trained flow already improves on its own source. |
| Training channel dropout on top of point subsampling (earlier E-series) | 2.6% worse | Training context became far sparser than evaluation. |
| Forced spatial smoothing `--min-tau-s 0.25` (earlier E-series) | 0.8% worse | The heat-kernel prior is the weak source either way. |

## Comparison with the literature

Results at 2× super-resolution (half the electrodes missing). Numbers are as
reported by each paper and are **not yet directly comparable** with ours.

| Method | Dataset | Protocol | NMSE | PCC |
| --- | --- | --- | --- | --- |
| EMAG (2026) | SEED-IV | Same subject, trial split | 0.1153 | 0.940 |
| SRGDiff (2025) | SEED-IV | Same subject, random split | 0.1663 | 0.911 |
| SRGDiff (2025) | SEED | Same subject, random split | 0.1632 | 0.910 |
| ESTformer | SEED-IV | Same subject | 0.3448 | 0.811 |
| DSF (2026) | SEED | Held-out subjects, fixed layout | 0.2009 | 0.886 |
| SRGDiff (rerun by DSF) | SEED | Held-out subjects, fixed layout | 0.2749 | 0.859 |
| Spherical spline (DSF table) | SEED | Held-out subjects, fixed layout | 0.2912 | 0.818 |

Why the numbers differ and what we must match before comparing:

1. **Protocol.** SRGDiff drops from 0.163 to 0.275 when tested on held-out
   subjects. Our paper uses held-out participants, so DSF (0.20) is the
   relevant target.
2. **Preprocessing.** All of these band-pass 0.5–75 Hz at 200 Hz; our
   manuscript used raw DEAP / SEED-IV, which is why its KNN NMSE is 0.61.
3. **Window length.** They use 4 s windows (800 samples); we use 100.
4. **Electrode choice.** They hide a fixed, spread-out layout; we hide a
   random 50% per seed.

Sources:
[SRGDiff](https://arxiv.org/html/2510.19166v1) ·
[EMAG](https://arxiv.org/html/2605.29731v1) ·
[DSF](https://arxiv.org/html/2610.24528) ·
[TGSD](https://arxiv.org/html/2606.03998) ·
[ScalpINR](https://arxiv.org/html/2606.23707) ·
[MASER](https://doi.org/10.1109/tnsre.2024.3481886) ·
[GiFlow, ICML 2026](https://arxiv.org/pdf/2606.06682)

## Open runs and next steps

- **Done:** all-subject held-out runs for the paper configuration, Anchored
  Flow + blend, ensemble, TTA and the spline source (tables above).
- **Next:** reduce cross-participant overfitting, since validation NMSE is
  best after one epoch: lower learning rate, stronger regularisation, or
  validation checks within the first epoch. The flow should add more on top
  of the spline than the current 3%.
- **Then:** match the literature protocol fully (fixed electrode layout,
  4 s windows via the channel-token architecture), add electrode
  coordinates, run five seeds, and rerun on the raw DEAP + SEED-IV data used
  in the manuscript.

## Reproducibility

| Commit | Change |
| --- | --- |
| `f8e2fc8` | `--train-keep-range` |
| `9cb5fcd` | `--train-reveal-eval`, `--prior-source fp`, baselines on every seed |
| `77329c4` | `--blend-fp` |
| `241265d` | `--prior-source mix` (Anchored Flow) |
| `fdfd26c` | Subject IDs from packed SEED-IV files (`--split-mode subject`) |
| `d752e25` | `--bandpass`, `--resample-hz` |
| `3a518fe` | `--blend-with` |
| `f3a3b3c` | `--ensemble`, `--tta-steps` |
| `144d638` | `--spline`, `--blend-with spline` (spherical spline) |

Data: `data/seed_iv_one.npz` (Kaggle dataset `anshattre/dataset-111`) and
`data/seed_iv_all.npz` (private Kaggle dataset `anshattre/seed-iv-all-giflow`).
