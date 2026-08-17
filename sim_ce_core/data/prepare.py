"""Print dataset access notes or write a local proxy cohort. No downloads."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sim_ce_core.data.config import DATASET_NOTES, DatasetConfig
from sim_ce_core.data.ctp import UNITOBRAIN_PROTOCOL
from sim_ce_core.data.proxy import write_mphase_proxy_cohort, write_proxy_cohort
from sim_ce_core.physio.params import PhysioParams


def _default_physio() -> PhysioParams:
    return PhysioParams(
        central_blood_volume_ml=1000.0,
        organ_volume_ml=400.0,
        recirculation_volume_ml=2500.0,
        cardiac_output_ml_s=108.3,
        organ_flow_fraction=0.25,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare local extracts or print access notes (never downloads)."
    )
    parser.add_argument(
        "--notes",
        action="store_true",
        help="Print license / access notes for all dataset ids.",
    )
    parser.add_argument(
        "--write-proxy",
        choices=("ctp_brain", "mphase_liver"),
        help="Write a synthetic proxy cohort (labeled source=synthetic_proxy).",
    )
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/proxy"))
    args = parser.parse_args(argv)

    if args.notes or args.write_proxy is None:
        cfg = DatasetConfig()
        for key, note in DATASET_NOTES.items():
            print(f"[{key}] {note.format(root=cfg.root)}")
        if args.write_proxy is None:
            return

    n_cases = max(1, min(args.n, 30))
    template = _default_physio()
    if args.write_proxy == "mphase_liver":
        written = write_mphase_proxy_cohort(
            args.out / "mphase_liver", template, n_cases=n_cases, seed=args.seed
        )
    else:
        written = write_proxy_cohort(
            args.out / "ctp_brain",
            template,
            UNITOBRAIN_PROTOCOL,
            n_cases=n_cases,
            seed=args.seed,
        )
    print(f"wrote {len(written)} proxy cases under {args.out / args.write_proxy}")


if __name__ == "__main__":
    main(sys.argv[1:])
