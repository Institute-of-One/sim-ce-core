"""Sensitivity of the enhancement curve to physiology: ``J = dC/dtheta``.

The identifiability question this package now asks -- what can a two- to four-phase CT
determine about physiology -- is a question about this matrix and nothing else. If two
parameters move the observable curve in the same direction at every sampled time, no
estimator separates them, however it is built.

Two routes are provided and both are used. The system parameters enter the forward model
through ``system_matrix`` as tensors, so autograd differentiates them exactly. The
transit delay does not: it shifts the simulation grid, and the grid is sampled with
``argmin`` in :func:`~sim_ce_core.physio.system.gather_states`, which has no gradient.
Its column is therefore taken by central difference. The test suite checks the autograd
columns against central differences, so the two routes hold each other honest rather
than one being trusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from sim_ce_core.physio.closed_form import simulate_closed_form_tensors
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.physio.system import concentrations_to_hu, params_to_tensors

#: Physiology examined for identifiability, in a fixed order. ``iodine_to_hu`` is
#: included deliberately: enhancement is ``k * c`` and the bolus enters as ``1 / v_c``,
#: so ``k`` and the central blood volume act on the observable through nearly the same
#: ratio. It is the clearest correlated pair in the model, and a diagnostic that cannot
#: see it is not measuring what it claims to.
DESIGN_PARAMS: tuple[str, ...] = (
    "central_blood_volume_ml",
    "organ_volume_ml",
    "recirculation_volume_ml",
    "cardiac_output_ml_s",
    "organ_flow_fraction",
    "iodine_to_hu",
    "transit_delay_s",
)

#: Enhancement is measured in the aorta and in the organ. The recirculation compartment
#: is a modelling device, not something a scan reports, and including it in the Jacobian
#: would credit the design with information no radiologist ever sees.
OBSERVED_REGIONS: tuple[int, ...] = (0, 1)

#: Parameters differentiated by central difference rather than autograd, with the
#: relative step used. See the module docstring.
_FINITE_DIFFERENCE: dict[str, float] = {"transit_delay_s": 1e-4}


@dataclass(frozen=True)
class Sensitivity:
    """``jacobian`` is ``(T, R, P)``: times, observed regions, parameters."""

    jacobian: Tensor
    times_s: Tensor
    parameter_names: tuple[str, ...]
    region_indices: tuple[int, ...]

    @property
    def flat(self) -> Tensor:
        """``(T*R, P)`` -- one row per scalar measurement the scan actually yields."""
        return self.jacobian.reshape(-1, self.jacobian.shape[-1])


def _curve(
    theta: dict[str, Tensor],
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Tensor,
    *,
    delay_s: float,
    regions: tuple[int, ...],
) -> Tensor:
    """Observable enhancement ``(T, R)`` in HU from a tensor parameter dict."""
    conc = simulate_closed_form_tensors(
        theta, protocol, times_s, delay_s=delay_s, dtype=times_s.dtype
    )
    hu = concentrations_to_hu(conc, theta["iodine_to_hu"])
    return hu[:, list(regions)]


def jacobian(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    parameter_names: tuple[str, ...] = DESIGN_PARAMS,
    regions: tuple[int, ...] = OBSERVED_REGIONS,
    dtype: torch.dtype = torch.float64,
) -> Sensitivity:
    """``dC/dtheta`` at the sampled times, in HU per unit of each parameter."""
    t_eval = torch.as_tensor(times_s, dtype=dtype).reshape(-1)
    if t_eval.numel() == 0:
        raise ValueError("times_s must be non-empty")
    base = params_to_tensors(params, dtype=dtype)

    autograd_names = [n for n in parameter_names if n not in _FINITE_DIFFERENCE]
    unknown = [n for n in autograd_names if n not in base]
    if unknown:
        raise KeyError(f"not tensor parameters of the forward model: {unknown}")

    columns: dict[str, Tensor] = {}
    if autograd_names:
        vector = torch.stack([base[n] for n in autograd_names]).requires_grad_(True)

        def evaluate(vec: Tensor) -> Tensor:
            theta = dict(base)
            for index, name in enumerate(autograd_names):
                theta[name] = vec[index]
            return _curve(
                theta,
                params,
                protocol,
                t_eval,
                delay_s=params.transit_delay_s,
                regions=regions,
            )

        # (T, R, P_autograd)
        jac = torch.autograd.functional.jacobian(evaluate, vector, vectorize=True)
        for index, name in enumerate(autograd_names):
            columns[name] = jac[..., index]

    for name, relative_step in _FINITE_DIFFERENCE.items():
        if name not in parameter_names:
            continue
        value = float(getattr(params, name))
        step = relative_step * max(abs(value), 1.0)
        forward = params.model_copy(update={name: value + step})
        backward = params.model_copy(update={name: max(value - step, 0.0)})
        span = (value + step) - max(value - step, 0.0)
        with torch.no_grad():
            high = _curve(
                params_to_tensors(forward, dtype=dtype),
                forward,
                protocol,
                t_eval,
                delay_s=forward.transit_delay_s,
                regions=regions,
            )
            low = _curve(
                params_to_tensors(backward, dtype=dtype),
                backward,
                protocol,
                t_eval,
                delay_s=backward.transit_delay_s,
                regions=regions,
            )
        columns[name] = (high - low) / span

    stacked = torch.stack([columns[name].detach() for name in parameter_names], dim=-1)
    return Sensitivity(
        jacobian=stacked,
        times_s=t_eval.detach(),
        parameter_names=tuple(parameter_names),
        region_indices=tuple(regions),
    )


def finite_difference_jacobian(
    params: PhysioParams,
    protocol: InjectionProtocol,
    times_s: Any,
    *,
    parameter_names: tuple[str, ...] = DESIGN_PARAMS,
    regions: tuple[int, ...] = OBSERVED_REGIONS,
    relative_step: float = 1e-6,
    dtype: torch.dtype = torch.float64,
) -> Sensitivity:
    """The same matrix by central differences, as an independent check on autograd.

    Slow and only as accurate as the step, which is why it is a reference rather than
    the implementation. It exists so that a mistake in the autograd path -- a parameter
    that silently does not reach the system matrix, say -- shows up as a disagreement
    instead of as a column of plausible numbers.
    """
    t_eval = torch.as_tensor(times_s, dtype=dtype).reshape(-1)
    columns = []
    for name in parameter_names:
        value = float(getattr(params, name))
        step = relative_step * max(abs(value), 1.0)
        low_value = max(value - step, 0.0)
        span = (value + step) - low_value
        pair = []
        for candidate in (value + step, low_value):
            updated = params.model_copy(update={name: candidate})
            with torch.no_grad():
                pair.append(
                    _curve(
                        params_to_tensors(updated, dtype=dtype),
                        updated,
                        protocol,
                        t_eval,
                        delay_s=updated.transit_delay_s,
                        regions=regions,
                    )
                )
        columns.append((pair[0] - pair[1]) / span)
    return Sensitivity(
        jacobian=torch.stack(columns, dim=-1),
        times_s=t_eval,
        parameter_names=tuple(parameter_names),
        region_indices=tuple(regions),
    )
