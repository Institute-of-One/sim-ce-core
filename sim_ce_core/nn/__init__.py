"""Neural modules: PINN residual, Neural-ODE, amortized inference."""

from __future__ import annotations

from sim_ce_core.nn.amortized import (
    AmortizedInferenceNet,
    AmortizedModel,
    train_amortized,
)
from sim_ce_core.nn.neural_ode import NeuralODEResidual, simulate_neural_ode
from sim_ce_core.nn.pinn import PinnFitResult, TimeResidualNet, fit_pinn

__all__ = [
    "AmortizedInferenceNet",
    "AmortizedModel",
    "NeuralODEResidual",
    "PinnFitResult",
    "TimeResidualNet",
    "fit_pinn",
    "simulate_neural_ode",
    "train_amortized",
]
