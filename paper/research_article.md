# Differentiable Contrast Kinetics: A Physics-Informed Neural Generalization of Bae's Contrast-Enhancement Model with Amortized Inference, Validated on Public Multi-phase CT

**Target:** *Physics in Medicine & Biology* / *Medical Physics* (Original Article, not a Technical Note)

**Status:** v1 results freeze (2026-08-17). Numbers below match `paper/frozen/manifest.json`.

## Abstract

**Purpose.** To test whether a differentiable, physics-informed generalization of Bae's compartmental contrast-enhancement model reconstructs time–enhancement curves and recovers physiology more robustly than closed-form Bae and deconvolution under sparse, noisy, or low-dose sampling (**H1**), using synthetic ground truth plus a minimal public CT cohort.

**Methods.** A three-compartment linear model (central blood, organ, recirculation) was implemented in closed form and as a `torchdiffeq` ODE. A PINN residual and an amortized inference network were trained on the simulator. Robustness was swept over noise, temporal stride, and dose. External validation used 20 TCIA HCC-TACE-Seg baseline liver CTs (CC BY 4.0).

**Results.** Closed-form and ODE forwards agreed to NRMSE \(4\times10^{-8}\). Under the strongest synthetic stress (25 HU noise, stride 4, half dose), PINN hybrid reduced parameter mean relative error versus closed-form Bae (0.12 vs 0.27); amortized inference had the lowest curve NRMSE (0.30 vs 0.48). Ablations: physics-only ≈ hybrid (NRMSE ≈ 0.04); neural-only failed (≈ 0.95). On 20 real multi-phase cases, closed-form Bae fit sparse phases better than a short-trained PINN (NRMSE 0.045 vs 0.127).

**Conclusions.** H1 is only partly supported: the physics-informed residual helps parameter recovery under synthetic stress, but on very sparse real phases the classical Bae forward remains the more stable curve fit. The software and frozen configs make that honest result reproducible.

## 1. Introduction

Bae's physiology-based model predicts organ-specific CT enhancement from injection protocol and body size [@bae1998aortic; @bae2010intravenous]. A software-only reimplementation would be a Technical Note. This article tests **H1**: a physics-informed neural generalization recovers curves and parameters from sparse / low-dose / noisy samples more robustly than (a) closed-form Bae and (b) deconvolution, on synthetic ground truth and ~20 public multi-phase CTs.

## 2. Methods

**Forward model.** States are iodine concentration in central blood, organ, and recirculation. Injection is a delayed rectangular bolus. The linear system \( \dot{c} = Ac + b I(t) \) is solved by matrix exponential (Van Loan) and by piecewise `dopri5`. Enhancement is \( \mathrm{HU} = k\,c \) with \( k = 26 \) HU·mL/mg I.

**Inverse methods.** Closed-form least squares in log-parameter space; Tikhonov deconvolution of organ from AIF (omitted when \( T < 8 \)); PINN hybrid \( C = C_{\mathrm{phys}}(\theta) + r_\phi(t) \) with a homogeneous-ODE residual penalty; amortized MLP trained on simulator draws.

**Data.** Synthetic curves from known \( \theta \). External set: 20 TCIA HCC-TACE-Seg patients, PRE + one contrast series, liver HU-window ROI [@moawad2021hcc; @moawad2023hcc]. No large-cohort download.

**Reproducibility.** `python -m sim_ce_core.experiments.run <yaml>` regenerates each figure. `python -m sim_ce_core.experiments.repro_check` runs lint, tests, and freeze-file checks.

## 3. Results

**Forward fidelity (Fig. M1).** Peak aorta / organ ≈ 420 / 288 HU. Closed-form vs ODE NRMSE \( 4.0\times 10^{-8} \).

**Robustness (Fig. 1–2).** Clean data: closed-form recovers \( \theta \) to numerical noise. Stressed cell (noise 25 HU, stride 4, dose 0.5):

| Method | Curve NRMSE | Param. MRE |
|--------|-------------|------------|
| Closed-form Bae | 0.48 | 0.27 |
| Deconvolution | 0.50 | — |
| PINN hybrid | 0.48 | **0.12** |
| Amortized | **0.30** | 0.66 |

**Ablation (Fig. 3b).** Physics-only / hybrid ≈ 0.04 NRMSE with or without AIF. Neural-only ≈ 0.95 (40 Adam steps, not a competitive curve fitter).

**External CT (Fig. 3).** 20 cases, source `tcia_hcc_tace_seg`. Closed-form NRMSE 0.045 ± 0.070; PINN hybrid 0.127 ± 0.154. Phase-level predictions: 53 rows. No ground-truth \( \theta \) on real data.

## 4. Discussion

H1 holds for **parameter recovery under synthetic degradation** (PINN) and for **curve reconstruction under the same stress** (amortized). It does **not** hold for short PINN fits on 2–4-phase real CT, where the classical Bae model is more stable. That is the result we report. A larger PINN budget or a population prior may close the real-data gap; it is not claimed here.

Limitations: reduced 3-compartment physiology; liver ROI is an HU window, not the published SEG; injection protocol on TCIA cases is a population default (weight not in the imaging archive); amortized training was short.

## 5. Data and code

- Software: MIT, this repository. Cite `CITATION.cff`.
- TCIA HCC-TACE-Seg: CC BY 4.0 [@moawad2021hcc; @moawad2023hcc].
- Frozen tables: `paper/frozen/`.

## References

See `paper/paper.bib`.
