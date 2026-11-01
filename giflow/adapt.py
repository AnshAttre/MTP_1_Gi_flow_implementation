"""Inference-time extensions: multi-model ensembles and test-time adaptation.

Ensemble: average the imputations of models trained with different
initialisations on the same split and masks.

Test-time adaptation (TTA): for each test participant, fine-tune a copy of the
vector field on that participant's *visible* electrodes only -- some visible
electrodes are hidden and used as targets -- then impute the evaluation
electrodes. The evaluation entries never enter the adaptation inputs or loss.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data.masking import random_subset_mask
from .metrics import MetricAccumulator


class EnsembleImputer(torch.nn.Module):
    """Mean of several GiFlow imputations; FP and the flow source come from the
    first member, so the ensemble can be blended like a single model."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    @torch.no_grad()
    def impute(self, x_obs, cond_mask, **kw):
        return torch.stack([m.impute(x_obs, cond_mask, **kw) for m in self.models]).mean(0)

    def feature_propagation(self, x_cond, cond_mask):
        return self.models[0].feature_propagation(x_cond, cond_mask)

    def source_sample(self, x_cond, cond_mask, node_mask=None):
        return self.models[0].source_sample(x_cond, cond_mask, node_mask)


def _groups(dataset):
    """Window indices per participant; one group when no participant IDs exist."""
    ids = getattr(dataset, "group_ids", None)
    if ids is None:
        return {"all": list(range(len(dataset)))}
    groups = {}
    for index, group in enumerate(ids):
        groups.setdefault(str(group), []).append(index)
    return groups


def adapt_to_windows(model, dataset, cfg, device, steps: int, lr: float, seed: int = 0):
    """Copy `model` and fine-tune its vector field on visible entries of `dataset`.

    Seeded in a forked RNG so adaptation is reproducible (dropout and mask
    sampling also draw from the global generator) without disturbing it.
    """
    devices = [torch.device(device).index or 0] if torch.device(device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        return _adapt(model, dataset, cfg, device, steps, lr, seed)


def _adapt(model, dataset, cfg, device, steps, lr, seed):
    adapted = copy.deepcopy(model)
    # the evaluated model is usually the EMA shadow, whose weights are frozen
    params = list(adapted.vector_field.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator(device=device).manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    adapted.train()
    done = 0
    while done < steps:
        for batch in loader:
            visible = batch["cond_mask"].to(device)
            # zero the evaluation entries so they cannot reach x_t or the loss
            x_visible = batch["x_true"].to(device) * visible
            cond = random_subset_mask(
                visible, (1.0, 1.0),
                channel_drop_prob=max(cfg.train_channel_drop_prob, 0.5),
                generator=gen,
            )
            target = visible - cond
            if float(target.sum()) == 0:
                continue
            ts = batch.get("timestamps")
            loss = adapted.loss(
                x_visible, cond, target,
                None if ts is None else ts.to(device),
                preservation_weight=cfg.preservation_weight,
                node_mask=batch["node_mask"].to(device),
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            done += 1
            if done >= steps:
                break
    return adapted.eval()


@torch.no_grad()
def _score(imputer, dataset, scaler, cfg, device, accumulators):
    from .train import _timestamps

    for batch in DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False):
        x_true = batch["x_true"].to(device)
        cond = batch["cond_mask"].to(device)
        pred = imputer.impute(
            x_true, cond, n_steps=cfg.euler_steps,
            timestamps=_timestamps(batch, device),
            node_mask=batch["node_mask"].to(device),
        )
        for name, value in pred.items() if isinstance(pred, dict) else (("model", pred),):
            accumulators.setdefault(name, MetricAccumulator()).update(
                scaler.inverse_transform(value), scaler.inverse_transform(x_true),
                batch["eval_mask"].to(device),
            )


class _WithBlend(torch.nn.Module):
    """Returns both the adapted model output and its blend with the reference."""

    def __init__(self, model, weight, reference):
        super().__init__()
        self.model, self.weight, self.reference = model, weight, reference

    @torch.no_grad()
    def impute(self, x_obs, cond_mask, node_mask=None, **kw):
        from .train import _blend_reference

        pred = self.model.impute(x_obs, cond_mask, node_mask=node_mask, **kw)
        out = {"model": pred}
        if self.weight is not None:
            ref = _blend_reference(self.model, x_obs, cond_mask, self.reference)
            if node_mask is not None:
                ref = ref * node_mask[:, :, None]
            out["blend"] = self.weight * pred + (1.0 - self.weight) * ref
        return out


def evaluate_tta(model, test_ds, scaler, cfg, device, blend_weight=None):
    """Pooled test metrics after per-participant adaptation (single dataset)."""
    accumulators = {}
    for offset, (group, indices) in enumerate(sorted(_groups(test_ds).items())):
        subset = Subset(test_ds, indices)
        adapted = adapt_to_windows(
            model, subset, cfg, device, cfg.tta_steps, cfg.tta_lr, seed=cfg.seed + offset
        )
        _score(_WithBlend(adapted, blend_weight, cfg.blend_with), subset, scaler, cfg,
               device, accumulators)
    return {name: acc.compute() for name, acc in accumulators.items()}


def _correlation(m):
    d = np.sqrt(np.clip(np.diag(m), 1e-12, None))
    return m / np.outer(d, d)


def adapted_covariance(subject_covs: dict, x_visible: np.ndarray, visible: np.ndarray,
                       temperature: float) -> np.ndarray:
    """Weighted mean of training-participant covariances for one new participant.

    Weights come from how close each training participant's visible-electrode
    correlation is to the new participant's, computed from the new
    participant's visible electrodes only (x_visible: (W, N, R), hidden rows
    zero). Smaller `temperature` concentrates on the closest participants.
    """
    a = x_visible.transpose(1, 0, 2).reshape(x_visible.shape[1], -1)
    own = _correlation((a @ a.T / a.shape[1])[np.ix_(visible, visible)])
    covs = list(subject_covs.values())
    dist = np.array([np.linalg.norm(_correlation(c[np.ix_(visible, visible)]) - own) for c in covs])
    w = np.exp(-(dist - dist.min()) / (temperature * dist.std() + 1e-9))
    w /= w.sum()
    return sum(wi * c for wi, c in zip(w, covs))


class _WithReference(_WithBlend):
    """Model, blend, and the adapted kriging estimate on its own ("reference")."""

    @torch.no_grad()
    def impute(self, x_obs, cond_mask, node_mask=None, **kw):
        from .train import _blend_reference

        out = super().impute(x_obs, cond_mask, node_mask=node_mask, **kw)
        out["reference"] = _blend_reference(self.model, x_obs, cond_mask, "cov")
        return out


def evaluate_adapted_cov(model, dataset, scaler, cfg, device, subject_covs, temperature,
                         blend_weight=None):
    """Pooled metrics with the covariance source re-weighted per participant.

    For each participant in `dataset`, the model's kriging covariance is replaced
    by `adapted_covariance` from that participant's visible electrodes; the
    flow weights are unchanged. Also scores the adapted kriging on its own.
    """
    original = model.cov_k
    accumulators = {}
    try:
        for group, indices in sorted(_groups(dataset).items()):
            subset = Subset(dataset, indices)
            items = [subset[i] for i in range(len(subset))]
            cond = torch.stack([it["cond_mask"] for it in items]).numpy()
            x_vis = torch.stack([it["x_true"] for it in items]).numpy() * cond
            visible = cond.sum(axis=(0, 2)) > 0
            cov = adapted_covariance(subject_covs, x_vis, visible, temperature)
            model.cov_k = torch.as_tensor(cov, dtype=original.dtype, device=original.device)
            _score(_WithReference(model, blend_weight, cfg.blend_with), subset, scaler, cfg, device,
                   accumulators)
    finally:
        model.cov_k = original
    return {name: acc.compute() for name, acc in accumulators.items()}
