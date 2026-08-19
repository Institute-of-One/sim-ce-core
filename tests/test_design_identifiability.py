"""What the sampling design determines, and the symmetry that no design can break.

The load-bearing test is the one on the exact scale symmetry. Scaling the three volumes,
the cardiac output and the attenuation constant by a common factor leaves every
enhancement curve unchanged, because the rate constants are ratios ``q / v``, the bolus
enters as ``1 / v_c``, and ``HU = k c`` absorbs the remaining factor.
The model therefore has a one-parameter continuous symmetry, and those five quantities
cannot be recovered separately from enhancement at any sampling density or noise level.

That is a statement about the model rather than about an estimator, which is why it is
checked directly against the forward simulation and not inferred from a fit. The
identifiability diagnostic is then required to find the same thing on its own.
"""

from __future__ import annotations

import pytest
import torch

from sim_ce_core.design.fisher import fisher_information
from sim_ce_core.design.identifiability import analyse
from sim_ce_core.design.sensitivity import (
    DESIGN_PARAMS,
    finite_difference_jacobian,
    jacobian,
)
from sim_ce_core.physio.fit import DEFAULT_FREE
from sim_ce_core.physio.forward import simulate_hu
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams

#: The five parameters the scale symmetry moves together.
SCALED = (
    "central_blood_volume_ml",
    "organ_volume_ml",
    "recirculation_volume_ml",
    "cardiac_output_ml_s",
    "iodine_to_hu",
)

CLINICAL_4_PHASE = [0.0, 35.0, 70.0, 180.0]
DENSE = [float(t) for t in range(0, 200, 10)]


@pytest.fixture
def physiology() -> PhysioParams:
    return PhysioParams(
        central_blood_volume_ml=5000.0,
        organ_volume_ml=1800.0,
        recirculation_volume_ml=25000.0,
        cardiac_output_ml_s=100.0,
        organ_flow_fraction=0.25,
        transit_delay_s=8.0,
        iodine_to_hu=26.0,
    )


@pytest.fixture
def protocol() -> InjectionProtocol:
    return InjectionProtocol(
        concentration_mgi_ml=350.0, volume_ml=100.0, duration_s=30.0
    )


def _values(params: PhysioParams, names: tuple[str, ...]) -> dict[str, float]:
    return {name: float(getattr(params, name)) for name in names}


@pytest.mark.parametrize("factor", [0.5, 1.7, 3.0])
def test_the_forward_model_has_an_exact_scale_symmetry(physiology, protocol, factor):
    """Scaling volumes, flow and attenuation together changes no observable."""
    times = [float(t) for t in range(0, 200, 5)]
    base = simulate_hu(physiology, protocol, times)
    scaled = physiology.model_copy(
        update={name: float(getattr(physiology, name)) * factor for name in SCALED}
    )
    difference = (simulate_hu(scaled, protocol, times) - base).abs().max()
    assert float(difference / base.abs().max()) < 1e-12


def test_the_diagnostic_finds_the_symmetry_without_being_told(physiology, protocol):
    """One null direction, lying on the five parameters the symmetry moves.

    Densely sampled and noise-free in effect: if the deficiency survives this, it is
    structural rather than a shortage of data.
    """
    sensitivity = jacobian(physiology, protocol, DENSE)
    fisher = fisher_information(
        sensitivity, _values(physiology, DESIGN_PARAMS), sigma_hu=1.0
    )
    result = analyse(fisher, DESIGN_PARAMS)

    assert result.numerical_rank == len(DESIGN_PARAMS) - 1
    carries_null = {
        name
        for name, share in zip(DESIGN_PARAMS, result.null_fraction, strict=True)
        if float(share) > 0.05
    }
    assert carries_null == set(SCALED)


def test_the_symmetry_breaks_when_the_other_parameters_are_fixed(physiology, protocol):
    """Holding three of the five fixed makes the remaining two recoverable.

    This is what the existing inverse does, and it is an assumption rather than a
    property of the data: the paper's point is that it has to be stated.
    """
    sensitivity = jacobian(physiology, protocol, DENSE, parameter_names=DEFAULT_FREE)
    fisher = fisher_information(
        sensitivity, _values(physiology, DEFAULT_FREE), sigma_hu=25.0
    )
    result = analyse(fisher, DEFAULT_FREE)
    assert result.numerical_rank == len(DEFAULT_FREE)
    assert not result.rank_deficient
    assert all(verdict == "identifiable" for verdict in result.status)


def test_a_design_with_too_few_measurements_is_rank_deficient(physiology, protocol):
    """Two scalar measurements cannot constrain seven parameters."""
    sensitivity = jacobian(physiology, protocol, [70.0])
    fisher = fisher_information(
        sensitivity, _values(physiology, DESIGN_PARAMS), sigma_hu=10.0
    )
    result = analyse(fisher, DESIGN_PARAMS)
    assert result.rank_deficient
    assert result.numerical_rank <= 2  # one time, two observed regions


def test_clinical_phase_counts_do_not_determine_the_two_fitted_parameters(
    physiology, protocol
):
    """The finding the paper reports, stated as a test so it cannot drift.

    At 25 HU noise the Cramer-Rao bound on both fitted parameters exceeds 20% relative
    for every routine phase count, and falls below it only with dense sampling.
    """
    values = _values(physiology, DEFAULT_FREE)
    bounds = {}
    for label, times in {
        "two": [0.0, 70.0],
        "three": [0.0, 35.0, 70.0],
        "four": CLINICAL_4_PHASE,
        "dense": DENSE,
    }.items():
        sensitivity = jacobian(
            physiology, protocol, times, parameter_names=DEFAULT_FREE
        )
        fisher = fisher_information(sensitivity, values, sigma_hu=25.0)
        bounds[label] = analyse(fisher, DEFAULT_FREE).crlb.max().item()

    assert bounds["two"] > 0.20
    assert bounds["three"] > 0.20
    assert bounds["four"] > 0.20
    assert bounds["dense"] < 0.20
    # More phases never carry less information.
    assert bounds["two"] >= bounds["three"] >= bounds["four"] >= bounds["dense"]


@pytest.mark.parametrize("name", DESIGN_PARAMS)
def test_autograd_and_finite_differences_agree(physiology, protocol, name):
    """Each Jacobian column, computed two ways.

    The transit delay reaches the curve through a simulation grid sampled with
    ``argmin``, which has no gradient, so its column is a central difference in both
    routes; the other six are autograd against a difference. A parameter that silently
    failed to reach the system matrix would show up here as a zero column against a
    non-zero reference rather than as plausible numbers.
    """
    times = CLINICAL_4_PHASE
    auto = jacobian(physiology, protocol, times).jacobian
    numeric = finite_difference_jacobian(physiology, protocol, times).jacobian
    index = DESIGN_PARAMS.index(name)

    reference = numeric[..., index]
    scale = reference.abs().max().clamp_min(1e-12)
    assert float(scale) > 0.0, f"{name} does not move the observable at all"
    assert float((auto[..., index] - reference).abs().max() / scale) < 1e-6


def test_information_is_additive_over_measurements(physiology, protocol):
    """``F(A union B) == F(A) + F(B)`` for disjoint acquisition times."""
    values = _values(physiology, DEFAULT_FREE)

    def information(times):
        sensitivity = jacobian(
            physiology, protocol, times, parameter_names=DEFAULT_FREE
        )
        return fisher_information(sensitivity, values, sigma_hu=10.0)

    first, second = [0.0, 35.0], [70.0, 180.0]
    combined = information(first + second)
    assert torch.allclose(combined, information(first) + information(second), atol=1e-8)
