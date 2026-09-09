# Curated checkpoints

EMA-averaged weights from runs worth keeping, together with the run's
`history.json` (config, per-epoch curve, frozen tau) and `baselines.json`, so
every number is traceable and re-checkable.

## Contents

### `seed4_point_rho0.2/`

SEED-IV, subject `12_20150804`, 3160 windows of 62 channels x 100 samples,
point missing 20%, seed 0, trained on a Kaggle T4.

| | MAE | RMSE | MAPE |
|---|---|---|---|
| GiFlow (this checkpoint) | 0.14626 | 0.20604 | 55.60% |
| Linear interpolation | **0.12004** | **0.17954** | **48.20%** |

Frozen filtering factors: `tau_s = 0.005303`, `tau_t = 0.943472`.

**Read this before using it.** The checkpoint is from **epoch 1** -- validation
MAE bottomed out there (0.1385) and then rose for ten consecutive epochs while
the train loss kept falling, so early stopping kept the epoch-1 weights. It is
an overfitted model that loses to linear interpolation, kept as a reproducible
baseline for the GPU pipeline, not as a good imputer. See the repo README for
the overfitting levers and for why `tau_s` collapsing to ~0.005 switches the
spatial half of the prior off.

## Verifying

`scripts/verify_checkpoint.py` rebuilds the model from `history.json`, loads the
weights, re-runs the test split and compares against the recorded metrics:

```bash
python scripts/verify_checkpoint.py --ckpt-dir checkpoints/seed4_point_rho0.2     --seed-root data/seed_iv_one.npz --window 100 --stride 100
```

`seed4_point_rho0.2` reproduces exactly on CPU from the GPU-trained weights
(MAE 0.14626 both ways, 0.00% relative delta, 88 tensors loaded with none
missing or unexpected). Takes a few minutes on CPU -- the test pass runs the
full 20-step Euler solver over 632 windows.

## Loading manually

```python
import json, torch
from giflow.data.seed_iv import load_seed_iv
from giflow.models.giflow import GiFlow

cfg = json.load(open("checkpoints/seed4_point_rho0.2/history.json"))["config"]
b = load_seed_iv(root="data/seed_iv_one.npz", window=100)
g = b.graphs(100)
model = GiFlow(adj_s=g["adj_s"], lap_s=g["lap_s"], lap_t=g["lap_t"],
               adj_s_norm=g["adj_s_norm"], adj_t_norm=g["adj_t_norm"],
               hidden=cfg["hidden"], n_mp_layers=cfg["n_mp_layers"],
               emb_dim=cfg["emb_dim"], dropout=cfg["dropout"])
model.load_state_dict(torch.load("checkpoints/seed4_point_rho0.2/best.pt",
                                 map_location="cpu"))
model.eval()
pred = model.impute(x_obs, cond_mask, n_steps=20)
```

The graphs and `--window` must match training exactly, or the temporal
Laplacian and the message-passing operators will have the wrong shape.
