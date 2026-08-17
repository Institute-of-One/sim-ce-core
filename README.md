# sim_ce_core

Differentiable, physics-informed **contrast-enhancement kinetics simulator** —
a modern-AI generalization of Bae's CT contrast-enhancement model.

## Hypothesis (this is an Original Article, not a Technical Note)

> **H1.** A physics-informed neural generalization of Bae's contrast-kinetics model reconstructs
> time–enhancement curves and recovers physiological parameters from **sparse / low-dose / noisy**
> sampling **more robustly** than (a) the classical closed-form Bae model and (b) standard deconvolution,
> validated on ~20–30 public CT-perfusion / multi-phase-CT cases.

## Why this is not a Technical Note

A software-only reimplementation of a known pharmacokinetic model would be a Technical Note.
This project is built around a **testable hypothesis (H1)** plus **external validation on real
public CT data** (minimal 20–30 cases). The paper's spine is synthetic ground-truth recovery,
robustness sweeps under sparse/noisy/low-dose sampling, and a small real-data confirmation —
not the packaging of the simulator alone.

Clinical cohorts stay minimal by design. Everything that can be checked against ground truth
runs on **synthetic data with no download**. Real extracts are local NPZ/JSON; the v1
external set is 20 TCIA HCC-TACE-Seg baseline CTs (CC BY 4.0).

## Bae lineage

The forward core is a reduced, differentiable descendant of Bae's physiology-based
compartmental model of iodinated contrast enhancement:

- Bae, Heiken & Brink, *Radiology* 1998 — computer model of aortic and hepatic CT enhancement
  (injection protocol + physiology → organ-specific time–enhancement curves).
- Bae, *Radiology* 2010 — review of intravenous contrast administration and scan timing.
- Related bridges: Barboriak 2008 (PBPK ↔ DCE-MRI), the Tofts model, and QIBA digital
  reference objects.

Bae's original clinical question — how injection rate, dose, and body size determine
enhancement timing and magnitude — is preserved as the forward map
`(protocol, physiology θ) → C(t)`. This package adds autodiff, a Neural-ODE / PINN residual
(milestone M2), and amortized inference, then tests whether that generalization is *more
robust* than closed-form Bae and deconvolution under degraded sampling (H1).

## Status (milestones)

| Milestone | Scope |
|-----------|--------|
| **M0** | Repo scaffold, CI, hypothesis pinned |
| **M1** | Differentiable Bae-style forward core (closed-form + ODE) + synthetic generator |
| **M2** | PINN / Neural-ODE + amortized inference; synthetic robustness sweeps (Fig 1–2) |
| **M3** | Local CTP / multi-phase loaders, external validation (Fig 3), ablations |
| **M4** (current) | Write-up, figure/CSV freeze, JOSS paper, reproducibility check |

## Installation

Python 3.11 required.

```bash
pip install -e ".[dev]"
```

## Quickstart

Run the test suite (no network, synthetic data only):

```bash
ruff check .
black --check .
pytest
```

Run the M1 forward experiment (synthetic aortic / organ curves):

```bash
python -m sim_ce_core.experiments.run configs/m1_synthetic.yaml
```

Run the M2 robustness sweep (closed-form Bae, deconvolution, PINN, amortized):

```bash
python -m sim_ce_core.experiments.run configs/m2_robustness.yaml
```

Outputs land in `outputs/m1_synthetic/` and `outputs/m2_robustness/`
(`fig1_reconstruction.png`, `fig2_curve_nrmse.png`, `fig2_param_mre.png`,
`robustness_sweep.csv`). The sweep is synthetic only — no patient data is downloaded.

### M3 — local extracts, Fig 3, ablations

Loaders read **local** `series.npz` + `metadata.json` only. Tests and CI never
download. The v1 real-data subset is **20 TCIA HCC-TACE-Seg** baseline CTs
(CC BY 4.0; cite Moawad et al., DOI 10.7937/TCIA.5FNA-0924) — enough for an
original-article external check, not a large cohort.

```bash
# Access notes (no download)
python -m sim_ce_core.data.prepare --notes

# Minimal real TCIA subset (20 patients, ~1.6 GB, resumable)
python -m sim_ce_core.data.tcia --n 20

# Fig 3 on the TCIA extracts (no proxy)
python -m sim_ce_core.experiments.run configs/m3_tcia.yaml

# Ablations on synthetic (physics / neural / hybrid x AIF)
python -m sim_ce_core.experiments.run configs/m3_ablation.yaml
```

Switching the primary dataset is `dataset.primary: ctp_brain | mphase_liver | dce_mri`.
Proxy outputs are tagged `source: synthetic_proxy` so they are not mistaken for
real external validation.

## Repository layout

```
sim_ce_core/           # installable package
  physio/              # Bae-style compartmental forward model
  nn/                  # PINN residual, Neural-ODE, amortized inference
  data/                # synthetic + local CTP / multi-phase / DCE loaders
  validate/            # recovery / curve metrics
  report/              # figure + CSV export
  experiments/run.py   # one-command experiment runner
configs/               # YAML, one file per experiment
tests/
paper/                 # JOSS paper + original-article draft + frozen CSVs
```

## M4 — write-up, freeze, reproducibility

Frozen metrics live in `paper/frozen/manifest.json` (copied from `outputs/` so
they survive `.gitignore`). Regenerating figures:

```bash
python -m sim_ce_core.experiments.run configs/m1_synthetic.yaml
python -m sim_ce_core.experiments.run configs/m2_robustness.yaml
python -m sim_ce_core.experiments.run configs/m3_tcia.yaml
python -m sim_ce_core.experiments.run configs/m3_ablation.yaml
python -m sim_ce_core.experiments.repro_check
```

Manuscripts: `paper/paper.md` (JOSS) and `paper/research_article.md` (PMB / MedPhys draft).

**H1, honestly:** under synthetic stress (25 HU noise, stride 4, half dose) PINN
hybrid improves parameter MRE vs closed-form Bae (0.12 vs 0.27) and amortized
inference improves curve NRMSE (0.30 vs 0.48). On 20 real multi-phase TCIA cases,
closed-form Bae fits sparse phases better than a short PINN (NRMSE 0.045 vs 0.127).
That mixed result is the v1 claim — not a blanket win for the neural residual.

## Citation

See `CITATION.cff`. Please also cite the Bae 1998 / 2010 lineage papers when using the model.

Public repository: [Institute-of-One/sim-ce-core](https://github.com/Institute-of-One/sim-ce-core).
TCIA extracts and local `outputs/` stay on disk; they are not part of the public core.

## License

MIT. See `LICENSE`.
