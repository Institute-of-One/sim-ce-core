"""Physics-informed residual on top of the Bae-style forward map."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from sim_ce_core.nn.layers import mlp, time_autograd
from sim_ce_core.physio.closed_form import simulate_closed_form_tensors
from sim_ce_core.physio.fit import DEFAULT_FREE, recover_parameters
from sim_ce_core.physio.params import REGION_NAMES, InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import (
    concentrations_to_hu,
    params_to_tensors,
    system_matrix,
)

PinnMode = Literal["physics_only", "neural_only", "hybrid"]


class TimeResidualNet(nn.Module):
    """Maps normalized time in ``[0, 1]`` to a residual in HU, shape ``(T, 3)``."""

    def __init__(self, hidden: int = 32, n_out: int = 3) -> None:
        super().__init__()
        self.net = mlp(1, n_out, hidden, zero_last=True)

    def forward(self, t_norm: Tensor) -> Tensor:
        return self.net(t_norm.reshape(-1, 1))


@dataclass
class PinnFitResult:
    params: PhysioParams
    residual_net: TimeResidualNet | None
    mode: PinnMode
    data_loss: float
    physics_loss: float

    def predict_hu(
        self,
        times_s: np.ndarray,
        protocol: InjectionProtocol,
    ) -> np.ndarray:
        """Full-grid HU prediction ``(T, 3)``."""
        if self.mode == "physics_only" or self.residual_net is None:
            from sim_ce_core.physio.forward import simulate_hu

            return simulate_hu(self.params, protocol, times_s).detach().cpu().numpy()
        t = torch.as_tensor(times_s, dtype=torch.float64)
        t_max = float(t[-1].clamp(min=1e-6).item())
        t_norm = t / t_max
        residual = self.residual_net(t_norm)
        if self.mode == "neural_only":
            return residual.detach().cpu().numpy()
        from sim_ce_core.physio.forward import simulate_hu

        phys = simulate_hu(self.params, protocol, times_s)
        return (phys + residual).detach().cpu().numpy()


def _theta_from_log(
    template: PhysioParams,
    free_params: Sequence[str],
    log_vals: Tensor,
) -> dict[str, Tensor]:
    theta = params_to_tensors(template, dtype=log_vals.dtype, device=log_vals.device)
    for name, log_v in zip(free_params, log_vals, strict=True):
        theta[name] = torch.exp(log_v)
    return theta


def _params_from_log(
    template: PhysioParams,
    free_params: Sequence[str],
    log_vals: Tensor,
) -> PhysioParams:
    updates = {
        name: float(torch.exp(log_v).detach().cpu())
        for name, log_v in zip(free_params, log_vals, strict=True)
    }
    return template.model_copy(update=updates)


def fit_pinn(
    times_s: np.ndarray,
    observed_hu: np.ndarray,
    protocol: InjectionProtocol,
    template: PhysioParams,
    *,
    mode: PinnMode = "hybrid",
    free_params: Sequence[str] = DEFAULT_FREE,
    init: dict[str, float] | None = None,
    hidden: int = 32,
    n_steps: int = 150,
    lr: float = 2e-2,
    physics_weight: float = 1.0,
    region_names: Sequence[str] = ("aorta", "organ"),
    seed: int = 0,
) -> PinnFitResult:
    """Fit physics-only LS, a neural curve, or a hybrid PINN residual."""
    torch.manual_seed(seed)
    cols = [REGION_NAMES.index(name) for name in region_names]
    obs = torch.as_tensor(observed_hu, dtype=torch.float64)
    t = torch.as_tensor(times_s, dtype=torch.float64)
    t_max = float(t[-1].clamp(min=1e-6).item())

    if mode == "physics_only":
        fitted, info = recover_parameters(
            times_s,
            observed_hu,
            protocol,
            template,
            free_params=free_params,
            init=init,
            region_names=region_names,
            max_nfev=80,
        )
        return PinnFitResult(
            params=fitted,
            residual_net=None,
            mode=mode,
            data_loss=float(info["cost"]),
            physics_loss=0.0,
        )

    log_init = [
        float(np.log(init[name] if init is not None else getattr(template, name)))
        for name in free_params
    ]
    log_vals = nn.Parameter(torch.tensor(log_init, dtype=torch.float64))
    residual_net = TimeResidualNet(hidden=hidden).double()
    opt = torch.optim.Adam(
        [
            {"params": [log_vals], "lr": lr},
            {"params": residual_net.parameters(), "lr": lr},
        ],
    )
    data_loss = torch.zeros(())
    phys_loss = torch.zeros(())
    for _ in range(n_steps):
        opt.zero_grad()
        t_var = t.detach().clone().requires_grad_(True)
        t_norm = t_var / t_max
        residual_hu = residual_net(t_norm)
        if mode == "neural_only":
            pred = residual_hu
            phys_loss = pred.new_zeros(())
        else:
            theta = _theta_from_log(template, free_params, log_vals)
            conc = simulate_closed_form_tensors(
                theta, protocol, t, delay_s=template.transit_delay_s
            )
            phys_hu = concentrations_to_hu(conc, theta["iodine_to_hu"])
            pred = phys_hu + residual_hu
            residual_c = residual_hu / theta["iodine_to_hu"]
            dr_dt = time_autograd(residual_c, t_var)
            a_mat, _b = system_matrix(theta)
            ar = residual_c @ a_mat.T
            phys_loss = ((dr_dt - ar) ** 2).mean()
        data_loss = ((pred[:, cols] - obs) ** 2).mean()
        loss = data_loss + physics_weight * phys_loss
        loss.backward()
        opt.step()

    fitted = (
        template
        if mode == "neural_only"
        else _params_from_log(template, free_params, log_vals)
    )
    return PinnFitResult(
        params=fitted,
        residual_net=residual_net.eval(),
        mode=mode,
        data_loss=float(data_loss.detach().cpu()),
        physics_loss=float(phys_loss.detach().cpu()),
    )
