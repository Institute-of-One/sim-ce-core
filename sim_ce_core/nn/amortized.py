"""Simulation-based amortized inference: C_obs → θ̂ (optional AIF-free)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from sim_ce_core.data.synthetic import generate_synthetic
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.nn.layers import mlp
from sim_ce_core.physio.fit import DEFAULT_FREE
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.validate.degrade import Degradation, apply_degradation


class AmortizedInferenceNet(nn.Module):
    """MLP encoder from resampled curves ``(B, C, T)`` to log-parameters."""

    def __init__(
        self, n_channels: int, n_times: int, n_params: int, hidden: int = 32
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.net = mlp(n_channels * n_times, n_params, hidden, n_hidden=2)

    def forward(self, curves: Tensor) -> Tensor:
        return self.net(curves.reshape(curves.shape[0], -1))


def resample_curves(
    times_s: np.ndarray,
    curves_hu: np.ndarray,
    t_grid: np.ndarray,
    *,
    use_aif: bool,
    hu_scale: float,
) -> np.ndarray:
    """Interpolate to a fixed grid and scale. Returns ``(C, T)`` float32-ready."""
    organ = np.interp(t_grid, times_s, curves_hu[:, 1])
    if use_aif:
        aorta = np.interp(t_grid, times_s, curves_hu[:, 0])
        stacked = np.stack([aorta, organ], axis=0)
    else:
        stacked = organ[None, :]
    return stacked / hu_scale


def _draw_free(
    template: PhysioParams,
    free_params: Sequence[str],
    rng: np.random.Generator,
    sigma: float = 0.3,
) -> dict[str, float]:
    updates: dict[str, float] = {}
    for name in free_params:
        loc = np.log(getattr(template, name))
        updates[name] = float(np.exp(rng.normal(loc, sigma)))
    return updates


@dataclass
class AmortizedModel:
    net: AmortizedInferenceNet
    free_params: tuple[str, ...]
    t_grid: np.ndarray
    use_aif: bool
    template: PhysioParams
    hu_scale: float

    def encode(self, series: EnhancementSeries) -> Tensor:
        dose = 1.0
        deg_meta = series.metadata.get("degradation") or {}
        if "dose_scale" in deg_meta:
            dose = float(deg_meta["dose_scale"])
        curves = series.curves_hu / max(dose, 1e-8)
        arr = resample_curves(
            series.times_s,
            curves,
            self.t_grid,
            use_aif=self.use_aif,
            hu_scale=self.hu_scale,
        )
        return torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)

    def infer(self, series: EnhancementSeries) -> PhysioParams:
        self.net.eval()
        with torch.no_grad():
            log_hat = self.net(self.encode(series)).squeeze(0)
        updates = {
            name: float(torch.exp(val).cpu())
            for name, val in zip(self.free_params, log_hat, strict=True)
        }
        return self.template.model_copy(update=updates)


def train_amortized(
    template: PhysioParams,
    protocol: InjectionProtocol,
    *,
    free_params: Sequence[str] = DEFAULT_FREE,
    t_end_s: float = 90.0,
    n_times: int = 64,
    hidden: int = 32,
    n_train: int = 96,
    n_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-3,
    use_aif: bool = True,
    hu_scale: float = 400.0,
    degradations: Sequence[Degradation] | None = None,
    seed: int = 0,
    log_sigma: float = 0.3,
) -> AmortizedModel:
    """Online SBI: sample θ, simulate, degrade, regress log θ."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    t_grid = np.linspace(0.0, t_end_s, n_times, dtype=np.float64)
    n_channels = 2 if use_aif else 1
    net = AmortizedInferenceNet(n_channels, n_times, len(free_params), hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    deg_list = list(degradations) if degradations else [Degradation()]
    n_batches = max(1, n_train // batch_size)
    free = tuple(free_params)

    net.train()
    for epoch in range(n_epochs):
        for batch_idx in range(n_batches):
            xs: list[np.ndarray] = []
            ys: list[list[float]] = []
            for item in range(batch_size):
                updates = _draw_free(template, free, rng, sigma=log_sigma)
                trial = template.model_copy(update=updates)
                series = generate_synthetic(
                    trial,
                    protocol,
                    t_grid,
                    backend="closed_form",
                    noise_sd_hu=0.0,
                    seed=None,
                )
                deg = deg_list[int(rng.integers(0, len(deg_list)))]
                item_seed = seed + epoch * 1_000 + batch_idx * 50 + item
                degraded = apply_degradation(series, deg, seed=item_seed)
                dose = max(deg.dose_scale, 1e-8)
                xs.append(
                    resample_curves(
                        degraded.times_s,
                        degraded.curves_hu / dose,
                        t_grid,
                        use_aif=use_aif,
                        hu_scale=hu_scale,
                    )
                )
                ys.append([float(np.log(updates[name])) for name in free])
            x = torch.as_tensor(np.stack(xs), dtype=torch.float32)
            y = torch.as_tensor(ys, dtype=torch.float32)
            opt.zero_grad()
            pred = net(x)
            loss = ((pred - y) ** 2).mean()
            loss.backward()
            opt.step()

    return AmortizedModel(
        net=net.eval(),
        free_params=free,
        t_grid=t_grid,
        use_aif=use_aif,
        template=template,
        hu_scale=hu_scale,
    )
