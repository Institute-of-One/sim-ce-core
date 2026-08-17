"""Pluggable dataset selector. Switching primary is a one-field change."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DatasetId = Literal["synthetic", "ctp_brain", "mphase_liver", "dce_mri"]

DATASET_NOTES: dict[str, str] = {
    "synthetic": "In-memory Bae-parameterized generator (no download).",
    "ctp_brain": (
        "UniToBrain brain CTP. Place extracted cases under "
        "{root}/ctp_brain/<case_id>/ (series.npz + metadata.json). "
        "Do not fetch the 80 GB IEEE archive from tests or CI. "
        "License: verify on Zenodo 10.5281/zenodo.5109415 before download."
    ),
    "mphase_liver": (
        "TCIA HCC-TACE-Seg multi-phase liver CT (CC BY 4.0; "
        "DOI 10.7937/TCIA.5FNA-0924). "
        "Minimal v1 subset: 20 baseline patients (PRE + one contrast series). "
        "Extracts live under {root}/mphase_liver/<case_id>/. "
        "Download: python -m sim_ce_core.data.tcia --n 20"
    ),
    "dce_mri": (
        "TCIA QIN-SARCOMA DCE-MRI (H3 only). Same NPZ schema; "
        "MR signal model is a future swap, not the v1 primary."
    ),
}


class DatasetConfig(BaseModel):
    """Local-only dataset handle. Tests and CI never download."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary: DatasetId = "synthetic"
    root: str = "data/raw"
    proxy_root: str = "data/proxy"
    max_cases: int = Field(default=30, gt=0, le=30)
    allow_proxy: bool = False
