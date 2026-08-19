"""An estimator has to depend on the data it is given.

The v1 amortized network did not. Across every cell of the robustness sweep -- noise 0
to 25 HU, stride 1 to 4, dose 1.0 to 0.5 -- its parameter error moved by 6e-4, and its
calibration scatter, generated and saved with the rest of the figures, is a flat line
predicting 52 mL/s for true cardiac outputs from 53 to 181. It had collapsed onto the
mean of its training prior and was nevertheless reported as the best method on curve
NRMSE, a metric that rewards a constant prediction when the target amplitude is halved.

Nothing in the suite could have caught that, because every test asked whether the code
ran rather than whether the estimate responded.

The check reads the calibration recorded by the run that produced the paper's numbers,
rather than retraining inside the test. Retraining would either take the two hours the
real budget needs or use a smaller one, and a guard that passes at a budget nobody ships
guards nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FROZEN = Path(__file__).resolve().parent.parent / "paper" / "frozen" / "m2_summary.json"

#: Correlation between truth and prediction below which the estimator is not using its
#: input. A floor on being an estimator at all, not a standard of accuracy: a useful
#: method sits far above it, and the v1 network sat below zero.
RESPONDS_CORRELATION = 0.5

#: Predictions must also vary. A network can correlate weakly and still be nearly
#: constant, and constancy is the specific failure being guarded against.
RESPONDS_SD_RATIO = 0.2


@pytest.fixture
def calibration() -> dict:
    if not FROZEN.exists():
        pytest.skip(f"{FROZEN.name} has not been frozen yet")
    payload = json.loads(FROZEN.read_text(encoding="utf-8"))
    if "amortized_calibration" not in payload:
        pytest.skip(
            "the frozen summary predates the calibration record; re-run "
            "configs/m2_robustness.yaml"
        )
    return payload["amortized_calibration"]


def test_the_amortized_estimate_tracks_the_truth(calibration):
    """Predictions must correlate with the physiology that generated the curve."""
    correlation = float(calibration["correlation"])
    assert correlation > RESPONDS_CORRELATION, (
        f"amortized cardiac output correlates {correlation:+.3f} with the truth; "
        "the network is not using its input"
    )


def test_the_amortized_estimate_is_not_constant(calibration):
    """It must also vary by something comparable to the variation it is shown."""
    ratio = float(calibration["sd_ratio"])
    low, high = calibration["predicted_range"]
    true_low, true_high = calibration["true_range"]
    assert ratio > RESPONDS_SD_RATIO, (
        f"predicted spread is {ratio:.3f} of the true spread "
        f"(predicted {low:.0f}-{high:.0f} mL/s for true "
        f"{true_low:.0f}-{true_high:.0f}); the estimate is nearly constant"
    )
