"""What a sampling design determines, and what it does not.

Reads a Fisher matrix and reports, per parameter, whether the design constrains it at
all. The three verdicts are deliberately coarse:

``identifiable``     the Cramer-Rao bound is inside the useful threshold
``weak``             constrained, but only loosely
``not separable``    the parameter lies in (or very near) the null space of the design

The last verdict is the one the paper is about. A routine two-phase scan yields two
enhancement measurements per region; a seven-parameter physiology cannot be recovered
from that many numbers whatever is fitted to them, and the honest output for such a case
is a refusal rather than an estimate with a plausible-looking interval.

Everything is computed in log-parameter space (see :mod:`sim_ce_core.design.fisher`), so
a Cramer-Rao bound of 0.12 reads as "to within 12%".
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

#: Relative standard error below which a parameter is called identifiable, and above
#: which it is called weak. 0.20 is a judgement, not a law: it is the point past
#: which a physiological parameter stops being usable for the comparisons these
#: curves are fitted for. It is a module constant so the threshold can be varied and
#: reported, rather than buried where a reader would take it for a fact.
IDENTIFIABLE_CRLB = 0.20
WEAK_CRLB = 1.00

#: Singular values below ``RANK_TOLERANCE`` times the largest are treated as zero.
#: That is the usual convention for numerical rank, stated here because the count of
#: identifiable directions depends on it.
RANK_TOLERANCE = 1e-10

#: A parameter is called not separable when more than this share of its own direction
#: lies in the null space of the design. Judged per parameter rather than by presence in
#: a null vector: one unconstrained direction usually has a small component on every
#: parameter, and the looser test condemns the whole vector and distinguishes nothing.
NULL_SHARE = 0.10


@dataclass(frozen=True)
class Identifiability:
    """A design's information content, per parameter and as a whole."""

    parameter_names: tuple[str, ...]
    singular_values: Tensor
    numerical_rank: int
    smallest_singular_value: float
    condition_number: float
    log_det: float
    crlb: Tensor
    correlation: Tensor
    null_fraction: Tensor
    status: tuple[str, ...]

    @property
    def rank_deficient(self) -> bool:
        return self.numerical_rank < len(self.parameter_names)

    @property
    def not_separable(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, verdict in zip(self.parameter_names, self.status, strict=True)
            if verdict == "not separable"
        )

    def worst_pair(self) -> tuple[str, str, float]:
        """The two parameters the design confounds most, and their correlation."""
        corr = self.correlation.clone()
        corr.fill_diagonal_(0.0)
        flat = corr.abs().argmax()
        i, j = divmod(int(flat), corr.shape[0])
        return self.parameter_names[i], self.parameter_names[j], float(corr[i, j])


def analyse(
    fisher: Tensor,
    parameter_names: tuple[str, ...],
    *,
    identifiable_crlb: float = IDENTIFIABLE_CRLB,
    weak_crlb: float = WEAK_CRLB,
    rank_tolerance: float = RANK_TOLERANCE,
    null_share: float = NULL_SHARE,
) -> Identifiability:
    """Diagnostics for one Fisher matrix."""
    if fisher.shape[0] != fisher.shape[1] or fisher.shape[0] != len(parameter_names):
        raise ValueError(
            f"fisher is {tuple(fisher.shape)} for {len(parameter_names)} parameters"
        )
    symmetric = 0.5 * (fisher + fisher.T)
    singular = torch.linalg.svdvals(symmetric)
    largest = float(singular.max())
    cutoff = rank_tolerance * largest if largest > 0 else 0.0
    rank = int((singular > cutoff).sum())
    smallest = float(singular.min())
    # Over the constrained subspace. The textbook condition number is infinite for every
    # rank-deficient design, which is all of the clinical ones, so it separates nothing
    # and cannot serve as the predictor the primary endpoint tests. The ratio across the
    # directions the data does constrain does vary between designs, and is what gets
    # reported and tested.
    constrained = singular[singular > cutoff]
    if constrained.numel():
        condition = float(constrained.max() / constrained.min())
    else:
        condition = float("inf")

    # The pseudo-inverse is what makes a singular design reportable at all: a direction
    # the data does not constrain gets a zero there rather than an exception, and the
    # rank count above is what tells the reader that is what happened.
    covariance = torch.linalg.pinv(symmetric, rtol=rank_tolerance)
    variance = torch.diagonal(covariance).clamp_min(0.0)
    crlb = variance.sqrt()

    scale = crlb.clamp_min(1e-300)
    correlation = covariance / scale.reshape(-1, 1) / scale.reshape(1, -1)

    # A parameter in the null space gets zero pseudo-inverse variance, which reads as
    # a perfectly determined value if the rank is not consulted. What must be measured
    # is how much of each parameter's own direction lies in the null space, not whether
    # it appears in a null vector at all. One unconstrained direction usually has a
    # small component on every parameter, so the looser test condemns all seven.
    eig = torch.linalg.eigh(symmetric)
    null_basis = eig.eigenvectors[:, eig.eigenvalues.abs() <= cutoff]
    if null_basis.numel():
        null_fraction = (null_basis**2).sum(dim=1).clamp(0.0, 1.0)
    else:
        null_fraction = torch.zeros(len(parameter_names), dtype=symmetric.dtype)

    status = []
    for index in range(len(parameter_names)):
        if float(null_fraction[index]) > null_share or variance[index] <= 0:
            status.append("not separable")
        elif float(crlb[index]) <= identifiable_crlb:
            status.append("identifiable")
        elif float(crlb[index]) <= weak_crlb:
            status.append("weak")
        else:
            status.append("not separable")

    logs = torch.log(singular.clamp_min(1e-300))
    log_det = float(logs.sum()) if rank else float("-inf")
    return Identifiability(
        parameter_names=tuple(parameter_names),
        singular_values=singular,
        numerical_rank=rank,
        smallest_singular_value=smallest,
        condition_number=condition,
        log_det=log_det if rank == len(parameter_names) else float("-inf"),
        crlb=crlb,
        correlation=correlation,
        null_fraction=null_fraction,
        status=tuple(status),
    )


def report_rows(result: Identifiability) -> list[dict[str, object]]:
    """The per-parameter table a case-level identifiability report is built from."""
    return [
        {
            "parameter": name,
            "crlb_relative": float(result.crlb[index]),
            "null_fraction": float(result.null_fraction[index]),
            "status": result.status[index],
        }
        for index, name in enumerate(result.parameter_names)
    ]
