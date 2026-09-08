"""GiFlow: graph-informed flow matching for spatiotemporal imputation.

Conditional path (Eq. 9) and vector field (Eq. 10):

    phi_t(X|Z) = (1 - t) * X_tau + t * X_1        with X_tau = exp(-tau_s L_s) X_1^M exp(-tau_t L_t)
    u_t(X|Z)   = X_1 - X_tau

Training loss (Sec. 3.2), a masked regression onto that constant-in-t target:

    L = E_t, Z  || M * ( v_t(X_t; theta, M, L) - X_1 + X_tau ) ||^2

Sampling: start at X_tau and integrate dX = v_t(X) dt with an Euler solver
(20 steps by default), then paste the observed entries back in. Because the
source is deterministic, no multi-sample averaging is needed; optional Gaussian
noise on the prior enables stochastic sampling for uncertainty estimates.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn

from .prior import GraphInformedPrior
from .vector_field import GiFlowVectorField


class EMA:
    """Exponential moving average of parameters (decay 0.9999 in the paper).

    The nominal decay is ramped in with the usual warmup
    min(decay, (1 + step) / (10 + step)); without it the shadow weights stay at
    their initialisation for the first few thousand steps, which would make
    early-stopping decisions on short runs meaningless.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def _decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self._decay()
        self.step += 1
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)

    def module(self) -> nn.Module:
        return self.shadow


class GiFlow(nn.Module):
    def __init__(
        self,
        adj_s: np.ndarray,
        lap_s: np.ndarray,
        lap_t: np.ndarray,
        adj_s_norm: np.ndarray,
        adj_t_norm: np.ndarray,
        hidden: int = 64,
        emb_dim: int = 32,
        n_mp_layers: int = 4,
        k_hops: int = 2,
        dropout: float = 0.0,
        tau_s: float = 1.0,
        tau_t: float = 1.0,
        prior_mode: str = "exact",
        prior_order: int = 10,
        learn_spatial_prior: bool = True,
        learn_temporal_prior: bool = True,
        prior_renormalize: bool = False,
        max_tau: float | None = 10.0,
        gaussian_prior: bool = False,
        max_len: int = 512,
        n_timestamp_classes: tuple[int, ...] = (),
        use_spatial_attention: bool = True,
        use_temporal_attention: bool = True,
        use_propagation: bool = True,
    ):
        super().__init__()
        self.gaussian_prior = gaussian_prior
        self.prior = GraphInformedPrior(
            lap_s,
            lap_t,
            tau_s=tau_s,
            tau_t=tau_t,
            mode=prior_mode,
            order=prior_order,
            learn_spatial=learn_spatial_prior,
            learn_temporal=learn_temporal_prior,
            renormalize=prior_renormalize,
            max_tau=max_tau,
        )
        self.vector_field = GiFlowVectorField(
            adj_s,
            adj_s_norm,
            adj_t_norm,
            hidden=hidden,
            emb_dim=emb_dim,
            n_mp_layers=n_mp_layers,
            k_hops=k_hops,
            dropout=dropout,
            max_len=max_len,
            n_timestamp_classes=n_timestamp_classes,
            use_spatial_attention=use_spatial_attention,
            use_temporal_attention=use_temporal_attention,
            use_propagation=use_propagation,
        )

    # ------------------------------------------------------------------ prior
    def source_sample(self, x_cond: torch.Tensor, cond_mask: torch.Tensor) -> torch.Tensor:
        """X_0. Either the graph-informed prior (Eq. 4) or the FM-Gauss baseline."""
        if self.gaussian_prior:
            return torch.randn_like(x_cond)
        return self.prior(x_cond, cond_mask)

    # --------------------------------------------------------------- training
    def loss(
        self,
        x_true: torch.Tensor,
        cond_mask: torch.Tensor,
        target_mask: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """x_true: (B,N,R) ground truth (only trusted where target_mask/cond_mask are 1).
        cond_mask: entries visible to the model. target_mask: entries scored."""
        x_cond = x_true * cond_mask
        x0 = self.source_sample(x_cond, cond_mask)
        b = x_true.shape[0]
        t = torch.rand(b, device=x_true.device)
        t_b = t.view(b, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * x_true
        u_t = x_true - x0                                  # Eq. (10)
        v_t = self.vector_field(x_t, x_cond, cond_mask, t, timestamps)
        denom = target_mask.sum().clamp_min(1.0)
        return (((v_t - u_t) * target_mask) ** 2).sum() / denom

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def impute(
        self,
        x_obs: torch.Tensor,
        cond_mask: torch.Tensor,
        n_steps: int = 20,
        timestamps: torch.Tensor | None = None,
        noise_std: float = 0.0,
    ) -> torch.Tensor:
        """Euler integration of dX = v_t(X) dt from t=0 to t=1.

        Returns the imputed signal with observed entries preserved.
        """
        x_cond = x_obs * cond_mask
        x = self.source_sample(x_cond, cond_mask)
        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x + dt * self.vector_field(x, x_cond, cond_mask, t, timestamps)
        return cond_mask * x_obs + (1.0 - cond_mask) * x

    @torch.no_grad()
    def impute_ensemble(
        self,
        x_obs: torch.Tensor,
        cond_mask: torch.Tensor,
        n_samples: int = 8,
        noise_std: float = 0.1,
        n_steps: int = 20,
        timestamps: torch.Tensor | None = None,
    ):
        """Optional stochastic sampling for uncertainty quantification (Sec. 3.1)."""
        samples = torch.stack(
            [
                self.impute(x_obs, cond_mask, n_steps, timestamps, noise_std)
                for _ in range(n_samples)
            ]
        )
        return samples.mean(0), samples.std(0)
