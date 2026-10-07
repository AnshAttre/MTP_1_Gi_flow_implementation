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


class ResidualDirectionModule(nn.Module):
    """Predict a correction to the base flow velocity from the current condition."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, kernel_size=(1, 5), padding=(0, 2)),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_t, x_cond, cond_mask, t):
        time = t[:, None, None, None].expand(-1, 1, *x_t.shape[1:])
        features = torch.stack((x_t, x_cond, cond_mask), dim=1)
        return self.net(torch.cat((features, time), dim=1)).squeeze(1)


class StepAwareModulationModule(nn.Module):
    """Predict a time- and condition-dependent affine velocity calibration."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, hidden, kernel_size=(1, 5), padding=(0, 2)),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_t, x_cond, cond_mask, t):
        time = t[:, None, None, None].expand(-1, 1, *x_t.shape[1:])
        features = torch.stack((x_t, x_cond, cond_mask), dim=1)
        raw = self.net(torch.cat((features, time), dim=1))
        return 1.0 + raw[:, 0], raw[:, 1]


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
        min_tau_s: float = 0.0,
        gaussian_prior: bool = False,
        max_len: int = 512,
        n_timestamp_classes: tuple[int, ...] = (),
        use_spatial_attention: bool = True,
        use_temporal_attention: bool = True,
        use_propagation: bool = True,
        clamp_observed_each_step: bool = False,
        use_srg_guidance: bool = False,
        srg_lambda_res: float = 1.0,
        srg_lambda_smm: float = 0.01,
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
            min_tau_s=min_tau_s,
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
        self.clamp_observed_each_step = clamp_observed_each_step
        self.use_srg_guidance = use_srg_guidance
        self.srg_lambda_res = srg_lambda_res
        self.srg_lambda_smm = srg_lambda_smm
        if use_srg_guidance:
            self.residual_direction = ResidualDirectionModule()
            self.step_modulation = StepAwareModulationModule()

    def guided_velocity(self, x_t, x_cond, cond_mask, t, timestamps=None, node_mask=None):
        velocity = self.vector_field(
            x_t, x_cond, cond_mask, t, timestamps, node_mask
        )
        if not self.use_srg_guidance:
            return velocity
        residual = self.residual_direction(x_t, x_cond, cond_mask, t)
        gamma, beta = self.step_modulation(x_t, x_cond, cond_mask, t)
        guided = gamma * (velocity + self.srg_lambda_res * residual) + beta
        return guided if node_mask is None else guided * node_mask[:, :, None]

    # ------------------------------------------------------------------ prior
    def source_sample(
        self, x_cond: torch.Tensor, cond_mask: torch.Tensor,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """X_0. Either the graph-informed prior (Eq. 4) or the FM-Gauss baseline."""
        if self.gaussian_prior:
            source = torch.randn_like(x_cond)
        else:
            source = self.prior(x_cond, cond_mask)
        return source if node_mask is None else source * node_mask[:, :, None]

    # --------------------------------------------------------------- training
    def loss(
        self,
        x_true: torch.Tensor,
        cond_mask: torch.Tensor,
        target_mask: torch.Tensor,
        timestamps: torch.Tensor | None = None,
        preservation_weight: float = 0.1,
        return_components: bool = False,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """x_true: (B,N,R) ground truth (only trusted where target_mask/cond_mask are 1).
        cond_mask: entries visible to the model. target_mask: entries scored.

        The preservation term scores a one-step endpoint estimate on the
        visible entries against their original values, so the field learns not
        to move data that was supplied as conditioning information.
        """
        x_cond = x_true * cond_mask
        x0 = self.source_sample(x_cond, cond_mask, node_mask)
        b = x_true.shape[0]
        t = torch.rand(b, device=x_true.device)
        t_b = t.view(b, 1, 1)
        x_t = (1.0 - t_b) * x0 + t_b * x_true
        u_t = x_true - x0                                  # Eq. (10)
        target_denom = target_mask.sum().clamp_min(1.0)
        base_velocity = self.vector_field(
            x_t, x_cond, cond_mask, t, timestamps, node_mask
        )
        residual_loss = x_true.new_zeros(())
        smm_regularization = x_true.new_zeros(())
        if self.use_srg_guidance:
            residual = self.residual_direction(x_t, x_cond, cond_mask, t)
            gamma, beta = self.step_modulation(x_t, x_cond, cond_mask, t)
            residual_target = u_t - base_velocity.detach()
            residual_loss = (((residual - residual_target) * target_mask) ** 2).sum() / target_denom
            v_t = gamma * (base_velocity + self.srg_lambda_res * residual) + beta
            modulation_error = (gamma - 1.0) ** 2 + beta ** 2
            if node_mask is None:
                smm_regularization = modulation_error.mean()
            else:
                modulation_mask = node_mask[:, :, None]
                smm_regularization = (
                    (modulation_error * modulation_mask).sum()
                    / (modulation_mask.sum() * x_true.shape[-1]).clamp_min(1.0)
                )
        else:
            v_t = base_velocity
        flow_loss = (((v_t - u_t) * target_mask) ** 2).sum() / target_denom

        # x_t + (1-t)v_t estimates the endpoint x_true; score only values
        # that were visible to the model and must therefore be preserved.
        endpoint = x_t + (1.0 - t_b) * v_t
        cond_denom = cond_mask.sum().clamp_min(1.0)
        preservation_loss = (((endpoint - x_true) * cond_mask) ** 2).sum() / cond_denom
        total = (
            flow_loss
            + preservation_weight * preservation_loss
            + self.srg_lambda_res * residual_loss
            + self.srg_lambda_smm * smm_regularization
        )
        if return_components:
            return total, {
                "flow_loss": flow_loss,
                "preservation_loss": preservation_loss,
                "residual_loss": residual_loss,
                "smm_regularization": smm_regularization,
            }
        return total

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def impute(
        self,
        x_obs: torch.Tensor,
        cond_mask: torch.Tensor,
        n_steps: int = 20,
        timestamps: torch.Tensor | None = None,
        noise_std: float = 0.0,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler integration of dX = v_t(X) dt from t=0 to t=1.

        Returns the imputed signal with observed entries preserved.
        """
        x_cond = x_obs * cond_mask
        x = self.source_sample(x_cond, cond_mask, node_mask)
        if noise_std > 0:
            x = x + noise_std * torch.randn_like(x)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=x.device)
            x = x + dt * self.guided_velocity(
                x, x_cond, cond_mask, t, timestamps, node_mask
            )
            if self.clamp_observed_each_step:
                x = cond_mask * x_obs + (1.0 - cond_mask) * x
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
