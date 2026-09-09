# Curated checkpoints

EMA-averaged weights from runs worth keeping, plus the run's `history.json` and
`summary.json` so the numbers are traceable.

Layout: `checkpoints/<dataset>_<setting>/best.pt` + the two JSON files.

Load with:

```python
import json, torch
from giflow.data.seed_iv import load_seed_iv
from giflow.models.giflow import GiFlow

cfg = json.load(open("checkpoints/<run>/history.json"))["config"]
b = load_seed_iv(root="data/seed_iv_one.npz")
g = b.graphs(cfg["... window used ..."])
model = GiFlow(adj_s=g["adj_s"], lap_s=g["lap_s"], lap_t=g["lap_t"],
               adj_s_norm=g["adj_s_norm"], adj_t_norm=g["adj_t_norm"],
               hidden=cfg["hidden"], n_mp_layers=cfg["n_mp_layers"])
model.load_state_dict(torch.load("checkpoints/<run>/best.pt", map_location="cpu"))
model.eval()
```

The graphs must be built exactly as in training or the shapes will not match.
