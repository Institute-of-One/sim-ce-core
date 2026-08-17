"""Shared time-enhancement series type for synthetic and (later) real loaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sim_ce_core.physio.params import REGION_NAMES


@dataclass(frozen=True)
class EnhancementSeries:
    """Time–enhancement curves with optional AIF and metadata."""

    times_s: np.ndarray
    curves_hu: np.ndarray
    region_names: tuple[str, ...] = REGION_NAMES
    aif_hu: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = np.asarray(self.times_s, dtype=np.float64).reshape(-1)
        curves = np.asarray(self.curves_hu, dtype=np.float64)
        if curves.ndim != 2:
            raise ValueError("curves_hu must have shape (T, R)")
        if curves.shape[0] != times.shape[0]:
            raise ValueError("times_s and curves_hu length mismatch")
        if curves.shape[1] != len(self.region_names):
            raise ValueError("curves_hu columns must match region_names")
        object.__setattr__(self, "times_s", times)
        object.__setattr__(self, "curves_hu", curves)
        if self.aif_hu is None:
            object.__setattr__(self, "aif_hu", curves[:, 0].copy())
        else:
            aif = np.asarray(self.aif_hu, dtype=np.float64).reshape(-1)
            if aif.shape[0] != times.shape[0]:
                raise ValueError("aif_hu length mismatch")
            object.__setattr__(self, "aif_hu", aif)

    def region(self, name: str) -> np.ndarray:
        return self.curves_hu[:, self.region_names.index(name)]
