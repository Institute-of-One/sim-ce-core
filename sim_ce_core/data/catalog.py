"""Load a cohort from the configured primary dataset. Local files only."""

from __future__ import annotations

from pathlib import Path

from sim_ce_core.data.config import DATASET_NOTES, DatasetConfig
from sim_ce_core.data.ctp import UNITOBRAIN_PROTOCOL, load_ctp_cohort
from sim_ce_core.data.dce import load_dce_cohort
from sim_ce_core.data.mphase import load_mphase_cohort
from sim_ce_core.data.proxy import write_mphase_proxy_cohort, write_proxy_cohort
from sim_ce_core.data.types import EnhancementSeries
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams


class MissingLocalDataError(FileNotFoundError):
    """Raised when the configured root has no extracts and proxy is disabled."""


def load_cohort(
    cfg: DatasetConfig,
    *,
    template: PhysioParams,
    protocol: InjectionProtocol,
    seed: int = 0,
) -> list[EnhancementSeries]:
    """Load up to ``cfg.max_cases`` series for ``cfg.primary``.

    ``synthetic`` is not a folder dataset — callers should use
    ``generate_synthetic``. Real ids read ``{root}/{primary}/``. If that
    folder is empty and ``allow_proxy`` is true, a labeled synthetic proxy
    cohort is written under ``{proxy_root}/{primary}/``.
    """
    if cfg.primary == "synthetic":
        raise ValueError("synthetic has no on-disk cohort; use generate_synthetic")

    raw_root = Path(cfg.root) / cfg.primary
    if cfg.primary == "ctp_brain":
        cohort = load_ctp_cohort(raw_root, max_cases=cfg.max_cases)
    elif cfg.primary == "mphase_liver":
        cohort = load_mphase_cohort(raw_root, max_cases=cfg.max_cases)
    else:
        cohort = load_dce_cohort(raw_root, max_cases=cfg.max_cases)

    if cohort:
        return cohort[: cfg.max_cases]
    if not cfg.allow_proxy:
        note = DATASET_NOTES[cfg.primary].format(root=cfg.root)
        raise MissingLocalDataError(
            f"No extracted cases in {raw_root}. {note} "
            "Or set dataset.allow_proxy: true to write a synthetic proxy cohort."
        )

    proxy_root = Path(cfg.proxy_root) / cfg.primary
    if cfg.primary == "mphase_liver":
        return write_mphase_proxy_cohort(
            proxy_root, template, n_cases=cfg.max_cases, seed=seed
        )
    used_protocol = UNITOBRAIN_PROTOCOL if cfg.primary == "ctp_brain" else protocol
    return write_proxy_cohort(
        proxy_root,
        template,
        used_protocol,
        n_cases=cfg.max_cases,
        seed=seed,
        dataset_id=cfg.primary,
    )
