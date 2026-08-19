"""Reproduce an experiment from a YAML config.

Usage:
    python -m sim_ce_core.experiments.run configs/m1_synthetic.yaml
    python -m sim_ce_core.experiments.run configs/m2_robustness.yaml
    python -m sim_ce_core.experiments.run configs/m3_external.yaml
    python -m sim_ce_core.experiments.run configs/m3_tcia.yaml
    python -m sim_ce_core.experiments.run configs/m3_ablation.yaml
    python -m sim_ce_core.experiments.repro_check
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from sim_ce_core.data.config import DatasetConfig
from sim_ce_core.data.synthetic import default_times_s, generate_synthetic
from sim_ce_core.experiments.ablation import run_ablation_experiment
from sim_ce_core.experiments.external import run_external_experiment
from sim_ce_core.experiments.robustness import run_robustness_experiment
from sim_ce_core.physio.forward import simulate_hu
from sim_ce_core.physio.params import InjectionProtocol, PhysioParams
from sim_ce_core.report.figures import save_enhancement_csv, save_enhancement_plot
from sim_ce_core.repro import seed_everything
from sim_ce_core.validate.metrics import nrmse


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    noise_sd_hu: list[float] = Field(default_factory=lambda: [0.0, 10.0, 25.0])
    subsample_stride: list[int] = Field(default_factory=lambda: [1, 4])
    dose_scale: list[float] = Field(default_factory=lambda: [1.0, 0.5])
    #: Independent noise draws per cell. One is enough to plot a curve and not enough to
    #: compare estimators: a Cramer-Rao bound constrains the spread of an error, and a
    #: single draw has no spread. The v1 sweep ran one, and roughly half its cells came
    #: out below their own bound, which is what one draw does rather than evidence of
    #: anything.
    n_realisations: int = Field(ge=1, default=1)


class AmortizedTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden: int = 32
    n_train: int = 96
    n_epochs: int = 20
    batch_size: int = 16
    n_times: int = 64
    lr: float = 1e-3
    use_aif: bool = True


class PinnTrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden: int = 32
    n_steps: int = 120
    physics_weight: float = 1.0
    mode: Literal["physics_only", "neural_only", "hybrid"] = "hybrid"


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: Literal["forward", "robustness", "external", "ablation"] = "forward"
    seed: int = 0
    backend: Literal["closed_form", "ode"] = "closed_form"
    t_end_s: float = Field(gt=0, default=120.0)
    dt_s: float = Field(gt=0, default=0.5)
    noise_sd_hu: float = Field(ge=0, default=0.0)
    output_dir: str = "outputs/m1_synthetic"
    physiology: PhysioParams
    injection: InjectionProtocol
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    free_params: list[str] = Field(
        default_factory=lambda: ["central_blood_volume_ml", "cardiac_output_ml_s"]
    )
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    amortized: AmortizedTrainConfig = Field(default_factory=AmortizedTrainConfig)
    pinn: PinnTrainConfig = Field(default_factory=PinnTrainConfig)


def to_jsonable(obj: Any) -> Any:
    """Replace non-finite floats so summary.json stays valid JSON."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {key: to_jsonable(val) for key, val in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(val) for val in obj]
    return obj


def load_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(raw)


def run_forward_experiment(cfg: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    seed_everything(cfg.seed)
    times = default_times_s(cfg.t_end_s, cfg.dt_s)
    series = generate_synthetic(
        cfg.physiology,
        cfg.injection,
        times,
        backend=cfg.backend,
        noise_sd_hu=cfg.noise_sd_hu,
        seed=cfg.seed,
    )
    hu_closed = (
        simulate_hu(cfg.physiology, cfg.injection, times, backend="closed_form")
        .detach()
        .cpu()
        .numpy()
    )
    hu_ode = (
        simulate_hu(cfg.physiology, cfg.injection, times, backend="ode")
        .detach()
        .cpu()
        .numpy()
    )
    agree = nrmse(hu_ode, hu_closed)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "enhancement.csv"
    png_path = output_dir / "enhancement.png"
    save_enhancement_csv(series, csv_path)
    save_enhancement_plot(series, png_path)

    return {
        "backend": cfg.backend,
        "dataset": cfg.dataset.primary,
        "n_times": int(times.size),
        "peak_aorta_hu": float(series.region("aorta").max()),
        "peak_organ_hu": float(series.region("organ").max()),
        "closed_form_ode_nrmse": agree,
        "csv": str(csv_path),
        "png": str(png_path),
    }


def run_experiment(cfg: ExperimentConfig, output_dir: Path) -> dict[str, Any]:
    if cfg.experiment == "robustness":
        return run_robustness_experiment(cfg, output_dir)
    if cfg.experiment == "external":
        return run_external_experiment(cfg, output_dir)
    if cfg.experiment == "ablation":
        return run_ablation_experiment(cfg, output_dir)
    return run_forward_experiment(cfg, output_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a sim_ce_core experiment from YAML."
    )
    parser.add_argument("config", type=Path, help="Path to YAML experiment config")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    summary = run_experiment(cfg, Path(cfg.output_dir))
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = to_jsonable(summary)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
