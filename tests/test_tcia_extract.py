"""Offline helpers for TCIA multi-phase extraction (no network)."""

from __future__ import annotations

import numpy as np

from sim_ce_core.data.tcia import liver_mask, mean_liver_hu, split_phase_volumes


def test_liver_mask_picks_parenchyma_blob() -> None:
    vol = np.full((4, 16, 16), -1000.0)
    vol[:, 4:12, 4:12] = 50.0
    mask = liver_mask(vol)
    assert int(mask.sum()) == 4 * 8 * 8
    assert mean_liver_hu(vol) == 50.0


def test_split_phase_volumes_by_time() -> None:
    a = np.ones((2, 2), dtype=np.float64)
    b = np.full((2, 2), 2.0)
    frames = [
        (10.0, 0.0, a),
        (10.0, 1.0, a),
        (20.0, 0.0, b),
        (20.0, 1.0, b),
    ]
    vols = split_phase_volumes(frames)
    assert len(vols) == 2
    assert vols[0].shape[0] == 2
    assert float(vols[1].mean()) == 2.0
