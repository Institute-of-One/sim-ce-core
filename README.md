# sim_ce_core

A differentiable, physics-informed simulator of CT contrast-enhancement kinetics,
built on a reduced descendant of Bae's compartmental model, together with the
identifiability analysis it was written to support.

## What this study found

**A sampling design's Fisher information predicts which physiological parameters can be
recovered from it, and a closed-form fit very nearly reaches that limit.** The consequence
is that on routine phase patterns there is little left for a better estimator to recover:
the limit is in the sampling, not in the inverse method.

- The reduced model carries an **exact continuous scale symmetry**, so no sampling density,
  however dense, constrains all seven parameters.
- Measured parameter error tracks the Cramér–Rao bound of the design that produced it,
  and the closed-form estimator sits close to it (median ratio 1.07).
- **The neural estimators do not beat the classical one**, and the analysis says why:
  the information a routine design withholds is not recoverable by any unbiased estimator.

This is a negative result about estimators and a positive one about design, and the
mechanism — not the ranking — is the contribution.

### Claims this project does *not* make

Two claims from the v1 draft (2026-08-17) were **withdrawn** once they were tested properly,
and are recorded here so the earlier framing is not mistaken for a result:

| Withdrawn v1 claim | Why it did not survive |
|---|---|
| The PINN hybrid improves parameter error over closed-form Bae (0.12 vs 0.27) | One noise realisation per cell. Over 20 realisations the ordering reverses. |
| Amortized inference gives the lowest curve NRMSE (0.30 vs 0.48) | The network was trained at a thirty-second of its budget and returned a near-constant prediction regardless of its input. |

Novelty is claimed neither for applying physics-informed networks to contrast kinetics
nor for Fisher-information-based experimental design, which is standard.

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
and amortized inference, and uses the resulting differentiability to compute the sensitivity
matrix and Fisher information of a sampling design in closed form — which is what the
identifiability analysis rests on.

## Status (milestones)

| Milestone | Scope |
|-----------|--------|
| **M0** | Repo scaffold, CI |
| **M1** | Differentiable Bae-style forward core (closed-form + ODE) + synthetic generator |
| **M2** | PINN / Neural-ODE + amortized inference; synthetic robustness sweeps (Fig 1–2) |
| **M3** | Local CTP / multi-phase loaders, external validation (Fig 3), ablations |
| **M3.5** | Sensitivity, Fisher information and identifiability of a sampling design |
| **M4** (current) | Write-up, figure/CSV freeze, reproducibility check |

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
python -m sim_ce_core.experiments.identifiability_map

python paper/build_all.py
```

`build_all.py` freezes the runs, resolves every number in the manuscript from those frozen
results, and rebuilds each submission artefact in dependency order. Manuscript:
`paper/manuscript.md`; see `paper/README.md`. A JOSS software description is in
`paper/paper.md`.

**The real-data numbers, read correctly.** Across 20 public multi-phase liver CTs the
closed-form fit reaches a mean curve NRMSE of 0.045 against the hybrid's 0.127, but that
comparison is not what it looks like: **13 of the 20 closed-form fits are exact
interpolations, not accurate fits.** Each is a two-phase case carrying one informative
measurement against two free parameters. On the seven cases that actually constrain the
fit the two are indistinguishable — 0.127 against 0.127. There is no ground-truth
physiology on real data, so this compares curve reconstruction and not parameter recovery.

In the ablation, physics-only and hybrid residuals are equivalent (NRMSE 0.041 and 0.043)
while the neural-only residual fails outright (0.95).

## Citation

See `CITATION.cff`. Please also cite the Bae 1998 / 2010 lineage papers when using the model.

Public repository: [Institute-of-One/sim-ce-core](https://github.com/Institute-of-One/sim-ce-core).
TCIA extracts and local `outputs/` stay on disk; they are not part of the public core.

## License

MIT. See `LICENSE`.
