# sim_ce_core — Project Kickoff Spec

> Differentiable, physics-informed **contrast-enhancement kinetics simulator** —
> a modern-AI generalization of Bae's CT contrast-enhancement model.
> Repo root: `D:\DevGit\sim_ce_core` · Package: `sim_ce_core` · Author: Shuji Yamamoto (sole)

---

## 0. 日本語サマリ（開発方針）

Baeのコンパートメント造影増強モデル（注入条件＋生理→大動脈/臓器の造影曲線を「前向き」に予測）を、
**微分可能・物理制約つきニューラルモデル**として再実装し、
（1）合成データでground-truth回収を示し、（2）公開CT造影データ**最小限20〜30例**で外部妥当性を確認する。
新規性は「Bae系**順問題**シミュレータの微分可能AI一般化＋amortized推論＋オープン実装」。
臨床データは最小限。まず合成データで完結する部分を先に作り、実データは差し替え可能なローダで後付けする。
これにより **Technical Note回避（仮説＋実データ検証）** と **短期完成** を両立する。

---

## 1. One-line pitch

A differentiable, physics-informed neural **forward simulator** of contrast-agent enhancement kinetics
that generalizes Bae's compartmental CT-enhancement model, adds amortized (AIF-free) parameter inference,
and is validated on synthetic ground truth plus a **minimal** public CT-contrast dataset.

## 2. Primary hypothesis (the thing that makes this an Original Article, not a Technical Note)

> **H1.** A physics-informed neural generalization of Bae's contrast-kinetics model reconstructs
> time–enhancement curves and recovers physiological parameters from **sparse / low-dose / noisy**
> sampling **more robustly** than (a) the classical closed-form Bae model and (b) standard deconvolution,
> validated on ~20–30 public CT-perfusion / multi-phase-CT cases.

Secondary hypotheses (optional, add only if H1 lands):
- **H2 (digital twin).** The learned forward model predicts patient- and protocol-specific enhancement
  timing/magnitude as a function of injection rate, dose, and body weight (Bae's original clinical question).
- **H3 (cross-modality).** The same forward core transfers to DCE-MRI enhancement curves with a swapped
  signal model (MR extension; keep as a short section only).

## 3. Scope guardrails (short-term discipline)

- **In scope:** forward simulator, PINN/Neural-ODE augmentation, amortized inference, synthetic validation,
  ONE minimal real-data validation, reproducible configs, figures/CSV export, JOSS-ready packaging.
- **Out of scope (v1):** large cohorts, multi-organ whole-body models, clinical outcome prediction,
  GUI, web app. Resist scope creep — these become "future work."
- **Data volume rule:** download only a **minimal subset** (target ≤ 30 cases, ≤ a few tens of GB).
  Everything ground-truth-checkable runs on **synthetic data with no download**.

## 4. Candidate datasets (pluggable — pick one primary at build time)

| id | dataset | why | access | note |
|----|---------|-----|--------|------|
| `ctp_brain` | UniToBrain (brain CT perfusion) | continuous time–attenuation curves + AIF; best for forward-model fidelity | Zenodo / IEEE DataPort (open) | verify license before DL; take ~20 subjects |
| `mphase_liver` | MCT-LTDiag multi-phase liver CT | **injection metadata recorded** (3 mL/s, 1 mL/kg; AP 20–30s / PVP 60s / DP 180s) → tests Bae's exact prediction | Harvard Dataverse (CC BY-NC-ND) | 4 sparse phases; ~30 cases |
| `dce_mri` | TCIA QIN-SARCOMA (DCE-MRI) | optional MR cross-modality demo only | TCIA (open, 10.29 GB) | not the primary; H3 only |

Design a single `DatasetConfig` so switching primary dataset is a one-line change.
**Recommended primary:** `ctp_brain` for H1 (richest curves, fastest to show forward fidelity);
keep `mphase_liver` wired for H2 (the most Bae-faithful validation).

## 5. Architecture

```
injection protocol + physiology θ
        │
        ▼
[ forward core ]  differentiable compartmental ODE  (Bae-lineage)
        │  C(t) predicted enhancement curve(s)
        ├───────────────► [ PINN / Neural-ODE residual ]  learns model mismatch
        ▼
   observed C_obs(t)  ◄── real data loader (DICOM time series / multi-phase)
        │
        ▼
[ inverse / amortized inference net ]  C_obs → θ̂   (optional AIF-free mode)
        │
        ▼
   metrics · figures · CSV
```

### 5.1 Modules (Python, PyTorch + a differentiable ODE lib e.g. `torchdiffeq`)

- `sim_ce_core/physio/` — classical Bae-style compartmental model as differentiable ODEs
  (central blood volume, organ compartments, recirculation). Closed-form baseline + ODE form.
- `sim_ce_core/nn/` — (a) `pinn.py`: physics-informed residual net constrained by the ODEs;
  (b) `amortized.py`: simulation-based-inference network (train on simulator, infer on real).
- `sim_ce_core/data/` — loaders: `synthetic.py` (ground-truth generator), `ctp.py`, `mphase.py`, `dce.py`.
  All return a common `EnhancementSeries` object (times, curves, AIF, metadata).
- `sim_ce_core/validate/` — recovery metrics (parameter RMSE/bias, curve NRMSE, calibration),
  baselines (closed-form Bae, deconvolution), noise/low-dose degradation sweeps.
- `sim_ce_core/report/` — figure + CSV export (matplotlib; follow accessible palette).
- `experiments/` — YAML configs, one per experiment; fully reproducible (seeded).
- `tests/` — unit tests: ODE conservation, closed-form↔ODE agreement, round-trip recovery on synthetic.

### 5.2 Tech stack

Python 3.11 · PyTorch · torchdiffeq (or diffrax if JAX preferred) · numpy/scipy · pydicom/SimpleITK ·
matplotlib · pydantic (configs) · pytest · ruff+black · `pyproject.toml` · MIT license · JOSS paper.

## 6. Validation plan (this IS the paper's spine)

1. **Synthetic ground truth (no download).** Generate curves from known θ (QIBA-style digital reference
   objects for the MR case; Bae-parameterized bolus for CT). Show exact recovery + calibration.
2. **Robustness sweeps.** Degrade temporal sampling, add noise, reduce dose/CNR. Plot recovery vs degradation
   for: closed-form Bae, deconvolution, and the PINN/amortized model. **H1 = our curve dominates under stress.**
3. **Minimal real data.** Run on ~20–30 cases of the chosen dataset. Report curve reconstruction error and,
   for `mphase_liver`, phase-timing/enhancement prediction vs recorded injection protocol & body weight.
4. **Ablations.** physics-only vs neural-only vs PINN(hybrid); with/without AIF.

## 7. Milestones (short-term, solo)

- **M0 (day 0):** repo scaffold, `pyproject.toml`, CI, this spec committed, hypothesis H1 pinned in README.
- **M1 (wk 1–2):** differentiable Bae forward core + closed-form baseline + synthetic generator + tests green.
- **M2 (wk 2–4):** PINN/Neural-ODE + amortized inference; synthetic recovery + robustness sweeps (Fig 1–2).
- **M3 (wk 4–6):** minimal real-data loader + external validation (Fig 3); ablations.
- **M4 (wk 6–8, current):** write-up + figures/CSV freeze + JOSS software paper + reproducibility check.

## 8. Acceptance criteria (definition of done for v1)

- [x] `pip install -e .` clean; `pytest` green; `ruff`/`black` clean.
- [x] One command reproduces every figure from configs (`python -m sim_ce_core.experiments.run <cfg>`).
- [x] Synthetic recovery: parameter MRE and curve NRMSE reported (sweep cells + TCIA mean±std).
- [x] Robustness sweep: honest mixed result (PINN helps θ under stress; amortized helps curves; real sparse phases favor closed-form Bae).
- [x] Real-data external validation on 20 TCIA HCC-TACE-Seg cases, results + CSV in `paper/frozen/`.
- [x] README states H1, the Bae lineage, and the Technical-Note-avoidance rationale.
- [x] LICENSE (MIT for code), CITATION.cff, JOSS `paper/paper.md` present.

## 9. Deliverables → publication

- **Software:** `sim_ce_core` (GitHub, MIT) → **JOSS** short paper.
- **Research article (Original):** target `Physics in Medicine & Biology` / `Medical Physics` /
  `Physica Medica` / `European Radiology Experimental` / `Frontiers in Physics (Medical Physics)`.
- Cite lineage: Bae 1998 (Radiology, computer model), Bae 2010 (Radiology, review),
  Barboriak 2008 (JMRI, PBPK↔DCE bridge), Tofts model, QIBA DRO, and the chosen dataset's data paper.

## 10. Repo layout to scaffold first

```
D:\DevGit\sim_ce_core\
├─ pyproject.toml
├─ README.md              # H1 + lineage + quickstart
├─ LICENSE                # MIT
├─ CITATION.cff
├─ paper/paper.md         # JOSS paper (plus research_article.md + frozen/)
├─ sim_ce_core/
│  ├─ __init__.py
│  ├─ physio/  ├─ nn/  ├─ data/  ├─ validate/  ├─ report/
│  └─ experiments/run.py
├─ configs/               # YAML experiment configs
├─ tests/
└─ notebooks/             # exploration only, not the source of truth
```

---

## Appendix A — Paste-ready FIRST prompt for Cursor / Claude CLI

```
You are setting up a new research software project at D:\DevGit\sim_ce_core (package name: sim_ce_core).
Read PROJECT_SPEC.md in the repo root as the source of truth, then execute Milestone M0 and start M1.

M0 tasks:
1. Scaffold the repo exactly as in PROJECT_SPEC.md §10 (pyproject.toml with PyTorch, torchdiffeq,
   numpy, scipy, pydantic, matplotlib, pydicom/SimpleITK; dev deps pytest, ruff, black).
2. Add MIT LICENSE, CITATION.cff, and paper/paper.md JOSS stub.
3. Write README.md that states Hypothesis H1 verbatim, the Bae lineage, and the
   Technical-Note-avoidance rationale (hypothesis + real-data validation).
4. Set up pytest and a GitHub Actions CI that runs ruff, black --check, pytest.

M1 (begin): implement sim_ce_core/physio/ as a differentiable Bae-style compartmental
contrast-enhancement forward model (central blood volume + organ compartment + recirculation),
in BOTH a closed-form baseline and a torchdiffeq ODE form, and a sim_ce_core/data/synthetic.py
ground-truth generator. Add tests that (a) check closed-form and ODE forms agree within tolerance,
and (b) round-trip recover known parameters from synthetic curves.

Constraints: Python 3.11, typed, small composable functions, seeded reproducibility, no network calls
in tests, no real patient data downloaded yet (synthetic only). Do NOT expand scope beyond §3.
After scaffolding, print a summary of created files and the exact commands to run tests and the first experiment.
```

## Appendix B — Title options

**Primary (recommended):**
- EN: *Differentiable Contrast Kinetics: A Physics-Informed Neural Generalization of Bae's
  Contrast-Enhancement Model with Amortized Inference, Validated on Public CT Perfusion Data*
- JP: 微分可能な造影動態 ― Baeの造影増強モデルの物理制約つきニューラル一般化とamortized推論、公開CT perfusionでの検証

**Alt 1 (punchy / software-forward):**
- EN: *NeuroBolus: A Differentiable, Physics-Informed Simulator of Contrast-Enhancement Kinetics
  Generalizing Bae's Model*

**Alt 2 (digital-twin / Bae-faithful, for the liver multi-phase route):**
- EN: *Learning Contrast Timing: A Differentiable Bae-Model Digital Twin for Patient- and
  Protocol-Specific CT Enhancement, Validated on Public Multi-phase CT*
