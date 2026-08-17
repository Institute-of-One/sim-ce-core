"""Tikhonov / SVD deconvolution baseline (organ residue from AIF)."""

from __future__ import annotations

import numpy as np


def convolution_matrix(aif: np.ndarray) -> np.ndarray:
    """Lower-triangular Toeplitz matrix implementing discrete convolution."""
    aif_arr = np.asarray(aif, dtype=np.float64).reshape(-1)
    n = aif_arr.size
    toeplitz = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        toeplitz[i, : i + 1] = aif_arr[i::-1]
    return toeplitz


def tikhonov_deconvolution(
    aif_hu: np.ndarray,
    organ_hu: np.ndarray,
    *,
    lam: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(residue, organ_hat)`` via Tikhonov-regularized least squares."""
    aif = np.asarray(aif_hu, dtype=np.float64).reshape(-1)
    organ = np.asarray(organ_hu, dtype=np.float64).reshape(-1)
    if aif.shape != organ.shape:
        raise ValueError("aif_hu and organ_hu must have the same length")
    design = convolution_matrix(aif)
    gram = design.T @ design + lam * np.eye(aif.size)
    residue = np.linalg.solve(gram, design.T @ organ)
    organ_hat = design @ residue
    return residue, organ_hat


def reconstruct_organ(
    aif_hu: np.ndarray,
    organ_hu: np.ndarray,
    *,
    lam: float = 0.2,
) -> np.ndarray:
    """Organ curve reconstructed from the estimated residue."""
    _residue, organ_hat = tikhonov_deconvolution(aif_hu, organ_hu, lam=lam)
    return organ_hat
