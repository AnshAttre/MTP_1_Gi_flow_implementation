"""Graph-informed prior via adaptive spatiotemporal filtering (Section 3.1).

The prior is the joint spatiotemporal heat-kernel filtering of the observable
signal (Eq. 4):

    X_tau = exp(-tau_eta L_eta) @ (X_1 * M) @ exp(-tau_xi L_xi)

which is the matrix form of x_tau = exp(-L_etaxi) vec(X_1 * M) with the Kronecker
sum L_etaxi = tau_xi L_xi (+) tau_eta L_eta.

Two evaluation modes:
  * "exact"  -- eigendecomposition of the (symmetric) Laplacian, so the matrix
                exponential is V diag(exp(-tau * lambda)) V^T. Exact, and cheap
                to differentiate w.r.t. tau. Default.
  * "taylor" -- the truncated Taylor series of Eq. (6) with K hops, for graphs
                too large to eigendecompose (cf. Proposition 3.1).

The filtering factors (tau_eta, tau_xi) are learned by minimising Problem (5) on
the training split, where the complete ground truth is available.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1e-6)
    if y > 20.0:
        return y
    return y + math.log(-math.expm1(-y))


class HeatKernel(nn.Module):
    """exp(-tau * L) for a fixed symmetric L, differentiable in tau."""

    def __init__(self, lap: np.ndarray, mode: str = "exact", order: int = 10):
        super().__init__()
        lap_t = torch.as_tensor(np.asarray(lap), dtype=torch.float32)
        self.n = int(lap_t.shape[0])
        self.mode = mode
        self.order = order
        eigvals = torch.linalg.eigvalsh(lap_t.double())
        self.spectral_radius = float(eigvals.abs().max())
        if mode == "exact":
            evals, evecs = torch.linalg.eigh(lap_t.double())
            self.register_buffer("evals", evals.clamp_min(0).float())
            self.register_buffer("evecs", evecs.float())
        elif mode == "taylor":
            self.register_buffer("lap", lap_t)
        else:
            raise ValueError("unknown mode: " + repr(mode))

    def matrix(self, tau: torch.Tensor) -> torch.Tensor:
        """Dense exp(-tau L) of shape (n, n)."""
        if self.mode == "exact":
            return (self.evecs * torch.exp(-tau * self.evals)) @ self.evecs.T
        acc = torch.eye(self.n, device=self.lap.device, dtype=self.lap.dtype)
        term = acc
        for k in range(1, self.order + 1):
            term = term @ self.lap * (-tau / k)
            acc = acc + term
        return acc

    def truncation_tail(self, tau: float, order: int | None = None) -> float:
        """sum_{k>K} |tau|^k C^k / k! -- one factor of the Proposition 3.1 bound."""
        k_max = self.order if order is None else order
        z = abs(float(tau)) * self.spectral_radius
        head = sum(z ** k / math.factorial(k) for k in range(k_max + 1))
        return max(math.exp(z) - head, 0.0)

    def truncation_head(self, tau: float, order: int | None = None) -> float:
        k_max = self.order if order is None else order
        z = abs(float(tau)) * self.spectral_radius
        return sum(z ** k / math.factorial(k) for k in range(k_max + 1))


def truncation_bound(
    kernel_s: HeatKernel,
    kernel_t: HeatKernel,
    tau_s: float,
    tau_t: float,
    order_s: int,
    order_t: int,
    signal_norm: float,
) -> float:
    """Proposition 3.1 bound on ||X_tau - X_tau^{K_eta,K_xi}||."""
    z_s = abs(tau_s) * kernel_s.spectral_radius
    z_t = abs(tau_t) * kernel_t.spectral_radius
    tail_s = kernel_s.truncation_tail(tau_s, order_s)
    tail_t = kernel_t.truncation_tail(tau_t, order_t)
    return (tail_s * math.exp(z_t) + math.exp(z_s) * tail_t) * signal_norm


class GraphInformedPrior(nn.Module):
    """Adaptive spatiotemporal filtering producing the FM source sample X_0.

    Args:
        lap_s: spatial Laplacian (N, N)
        lap_t: temporal Laplacian (R, R)
        tau_s, tau_t: initial filtering factors
        learn_spatial / learn_temporal: set False to pin a factor at 0, which
            gives the GFM (temporal off) and TFM (spatial off) ablations.
        renormalize: divide by the identically filtered mask. NOT in the paper --
            it removes the shrinkage bias caused by zeros at missing entries.
            Off by default to stay faithful to Eq. (4).
    """

    def __init__(
        self,
        lap_s: np.ndarray,
        lap_t: np.ndarray,
        tau_s: float = 1.0,
        tau_t: float = 1.0,
        mode: str = "exact",
        order: int = 10,
        learn_spatial: bool = True,
        learn_temporal: bool = True,
        renormalize: bool = False,
        max_tau: float | None = 10.0,
    ):
        super().__init__()
        self.kernel_s = HeatKernel(lap_s, mode=mode, order=order)
        self.kernel_t = HeatKernel(lap_t, mode=mode, order=order)
        self.learn_spatial = learn_spatial
        self.learn_temporal = learn_temporal
        self.renormalize = renormalize
        self.max_tau = max_tau
        self.raw_tau_s = nn.Parameter(
            torch.tensor(self._to_raw(tau_s)), requires_grad=learn_spatial
        )
        self.raw_tau_t = nn.Parameter(
            torch.tensor(self._to_raw(tau_t)), requires_grad=learn_temporal
        )
        self.register_buffer(
            "lap_s_dense", torch.as_tensor(np.asarray(lap_s), dtype=torch.float32)
        )

    # tau > 0 is enforced by softplus; with `max_tau` set we instead use a scaled
    # sigmoid, because Eq. (5) leaves tau_xi unbounded (its smoothness term only
    # involves L_eta) and a temporally flat prior can otherwise run away to
    # tau_xi -> inf on slowly drifting signals.
    def _to_raw(self, tau: float) -> float:
        if self.max_tau is None:
            return _inv_softplus(tau)
        frac = min(max(float(tau) / self.max_tau, 1e-6), 1 - 1e-6)
        return math.log(frac / (1.0 - frac))

    def _from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        if self.max_tau is None:
            return F.softplus(raw)
        return self.max_tau * torch.sigmoid(raw)

    @property
    def tau_s(self) -> torch.Tensor:
        if not self.learn_spatial:
            return torch.zeros((), device=self.raw_tau_s.device)
        return self._from_raw(self.raw_tau_s)

    @property
    def tau_t(self) -> torch.Tensor:
        if not self.learn_temporal:
            return torch.zeros((), device=self.raw_tau_t.device)
        return self._from_raw(self.raw_tau_t)

    def forward(self, x_obs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """x_obs: (B, N, R), already multiplied by the conditioning mask."""
        fs = self.kernel_s.matrix(self.tau_s)  # (N, N)
        ft = self.kernel_t.matrix(self.tau_t)  # (R, R)
        out = torch.einsum("ij,bjr->bir", fs, x_obs)
        out = torch.einsum("bir,rs->bis", out, ft)
        if self.renormalize and mask is not None:
            den = torch.einsum("ij,bjr->bir", fs, mask)
            den = torch.einsum("bir,rs->bis", den, ft)
            out = out / den.clamp_min(1e-4)
        return out

    @staticmethod
    def transport_cost(x_true: torch.Tensor, x_prior: torch.Tensor) -> torch.Tensor:
        """Expected quadratic cost E||X_1 - X_0||^2 along the linear path (Thm. 3.2)."""
        return ((x_true - x_prior) ** 2).sum(dim=(1, 2)).mean()

    def objective(
        self,
        x_true: torch.Tensor,
        x_obs: torch.Tensor,
        mask: torch.Tensor,
        alpha_tau: float,
    ) -> torch.Tensor:
        """Problem (5): signal alignment + Laplacian smoothness of the filtered signal."""
        x_prior = self.forward(x_obs, mask)
        align = ((x_true - x_prior) ** 2).sum(dim=(1, 2))
        # tr(X_tau^T L_eta X_tau), summed over the temporal axis
        smooth = torch.einsum("bir,ij,bjr->b", x_prior, self.lap_s_dense, x_prior)
        return (align + alpha_tau * smooth).mean()


def optimize_filtering_factors(
    prior: GraphInformedPrior,
    loader,
    alpha_tau: float = 0.01,
    epochs: int = 100,
    lr: float = 0.05,
    device: str = "cpu",
    verbose: bool = True,
    log_every: int = 20,
):
    """Solve Problem (5) with SGD on the training split (Section 4).

    `loader` yields dicts with x_true (B,N,R), x_obs (B,N,R), mask (B,N,R), where
    mask is the conditioning mask used to build the prior. Once selected, the
    factors stay constant during inference.
    """
    prior = prior.to(device).train()
    params = [p for p in prior.parameters() if p.requires_grad]
    if not params:
        return {"tau_s": float(prior.tau_s.detach()), "tau_t": float(prior.tau_t.detach()), "history": []}
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for batch in loader:
            x_true = batch["x_true"].to(device)
            x_obs = batch["x_obs"].to(device)
            mask = batch["mask"].to(device)
            loss = prior.objective(x_true, x_obs, mask, alpha_tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        avg = tot / max(nb, 1)
        history.append(
            {
                "epoch": ep,
                "obj": avg,
                "tau_s": float(prior.tau_s.detach()),
                "tau_t": float(prior.tau_t.detach()),
            }
        )
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            print(
                "  tau-opt step {:4d}  obj {:12.4f}  tau_s {:.4f}  tau_t {:.4f}".format(
                    ep, avg, float(prior.tau_s.detach()), float(prior.tau_t.detach())
                )
            )
    return {"tau_s": float(prior.tau_s.detach()), "tau_t": float(prior.tau_t.detach()), "history": history}
