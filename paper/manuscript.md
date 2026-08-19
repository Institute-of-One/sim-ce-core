<!-- Every number below is a marker resolved from paper/frozen/ at build time by
     paper/build_manuscript.py. The syntax is [[results:<file>:<path>|<format>]]; this
     comment is not resolved, so it can show the notation without naming a real file.
     Do not type a number into this file. If a value is not reachable from a frozen
     file, add it to the freeze rather than typing it here. -->

# Differentiable Contrast Kinetics: A Physics-Informed Neural Generalization of Bae's Contrast-Enhancement Model with Amortized Inference, Validated on Public Multi-phase CT

**Shuji Yamamoto**
Institute of One, LISIT Co., Ltd., Tokyo, Japan
yamamoto@lisit.jp · ORCID 0000-0001-9211-1071

## Abstract

**Purpose.** To test whether a differentiable, physics-informed generalization of Bae's
compartmental contrast-enhancement model reconstructs time–enhancement curves and recovers
physiology more robustly than closed-form Bae and deconvolution under sparse, noisy, or
low-dose sampling (**H1**), using synthetic ground truth plus a minimal public CT cohort.

**Methods.** A three-compartment linear model (central blood, organ, recirculation) was
implemented in closed form and as a `torchdiffeq` ODE. A PINN residual and an amortized
inference network were trained on the simulator. Robustness was swept over noise, temporal
stride, and dose. External validation used
[[results:m3_tcia_summary.json:n_cases]] TCIA HCC-TACE-Seg baseline liver CTs (CC BY 4.0).

**Results.** Closed-form and ODE forwards agreed to NRMSE
[[results:manifest.json:metrics.m1_closed_form_ode_nrmse|sci1]]. Under the strongest
synthetic stress (25 HU noise, stride 4, half dose), the PINN hybrid reduced parameter mean
relative error versus closed-form Bae
([[results:manifest.json:metrics.m2_stressed_pinn_param_mre|.2f]] vs
[[results:manifest.json:metrics.m2_stressed_closed_form_param_mre|.2f]]); amortized inference
had the lowest curve NRMSE
([[results:manifest.json:metrics.m2_stressed_amortized_curve_nrmse|.2f]] vs
[[results:manifest.json:metrics.m2_stressed_closed_form_curve_nrmse|.2f]]). Ablations:
physics-only ≈ hybrid (NRMSE ≈
[[results:manifest.json:metrics.m3_ablation_physics_aif|.2f]]); neural-only failed (≈
[[results:manifest.json:metrics.m3_ablation_neural_aif|.2f]]). On
[[results:m3_tcia_summary.json:n_cases]] real multi-phase cases, closed-form Bae fit sparse
phases better than a short-trained PINN
([[results:m3_tcia_summary.json:curve_nrmse.closed_form.mean|.3f]] vs
[[results:m3_tcia_summary.json:curve_nrmse.pinn_hybrid.mean|.3f]]).

**Conclusions.** H1 is only partly supported: the physics-informed residual helps parameter
recovery under synthetic stress, but on very sparse real phases the classical Bae forward
remains the more stable curve fit. The software and frozen configs make that honest result
reproducible.

**Keywords:** contrast enhancement; pharmacokinetic modelling; physics-informed neural
networks; neural ordinary differential equations; amortized inference; computed tomography.

## 1. Introduction

Bae's physiology-based model predicts organ-specific CT enhancement from injection protocol
and body size [1,2]. A software-only reimplementation would be a technical note. This
article tests **H1**: a physics-informed neural generalization recovers curves and parameters
from sparse / low-dose / noisy samples more robustly than (a) closed-form Bae and (b)
deconvolution, on synthetic ground truth and
[[results:m3_tcia_summary.json:n_cases]] public multi-phase CTs.

## 2. Methods

**Forward model.** States are iodine concentration in central blood, organ, and
recirculation. Injection is a delayed rectangular bolus. The linear system `dc/dt = Ac + bI(t)`
is solved by matrix exponential (Van Loan) and by piecewise `dopri5`. Enhancement is
`HU = k c` with `k = 26` HU·mL/mg I.

**Inverse methods.** Closed-form least squares in log-parameter space; Tikhonov deconvolution
of organ from AIF (omitted when `T < 8`); PINN hybrid `C = C_phys(θ) + r_φ(t)` with a
homogeneous-ODE residual penalty; amortized MLP trained on simulator draws.

**Data.** Synthetic curves from known θ. External set:
[[results:m3_tcia_summary.json:n_cases]] TCIA HCC-TACE-Seg patients, source
`[[results:m3_tcia_summary.json:sources[0]]]`, PRE plus one contrast series, liver HU-window
ROI [3,4]. No large-cohort download.

**Reproducibility.** `python -m sim_ce_core.experiments.run <yaml>` regenerates each figure.
`python -m sim_ce_core.experiments.repro_check` runs lint, tests, and freeze-file checks.

## 3. Results

**Forward fidelity (Fig. 1).** Peak aorta / organ
[[results:m1_summary.json:peak_aorta_hu|.0f]] /
[[results:m1_summary.json:peak_organ_hu|.0f]] HU. Closed-form versus ODE NRMSE
[[results:manifest.json:metrics.m1_closed_form_ode_nrmse|sci1]].

**Robustness (Figs. 2–3).** On clean data the closed form recovers θ to numerical noise. In
the stressed cell (noise 25 HU, stride 4, dose 0.5):

| Method | Curve NRMSE | Parameter MRE |
|---|---|---|
| Closed-form Bae | [[results:m2_summary.json:stressed_cell[method=closed_form].curve_nrmse|.2f]] | [[results:m2_summary.json:stressed_cell[method=closed_form].param_mre|.2f]] |
| Deconvolution | [[results:m2_summary.json:stressed_cell[method=deconvolution].curve_nrmse|.2f]] | — |
| PINN hybrid | [[results:m2_summary.json:stressed_cell[method=pinn_hybrid].curve_nrmse|.2f]] | [[results:m2_summary.json:stressed_cell[method=pinn_hybrid].param_mre|.2f]] |
| Amortized | [[results:m2_summary.json:stressed_cell[method=amortized].curve_nrmse|.2f]] | [[results:m2_summary.json:stressed_cell[method=amortized].param_mre|.2f]] |

**Ablation (Fig. 5).** Physics-only and hybrid reach
[[results:m3_ablation_summary.json:mean_curve_nrmse["physics_only/AIF"]|.2f]] and
[[results:m3_ablation_summary.json:mean_curve_nrmse["hybrid/AIF"]|.2f]] NRMSE with the AIF,
and are unchanged without it. Neural-only reaches
[[results:m3_ablation_summary.json:mean_curve_nrmse["neural_only/AIF"]|.2f]]: at the training
budget used here it is not a competitive curve fitter.

**External CT (Fig. 4).** [[results:m3_tcia_summary.json:n_cases]] cases, source
`[[results:m3_tcia_summary.json:sources[0]]]`. Closed-form NRMSE
[[results:m3_tcia_summary.json:curve_nrmse.closed_form.mean|.3f]] ±
[[results:m3_tcia_summary.json:curve_nrmse.closed_form.std|.3f]]; PINN hybrid
[[results:m3_tcia_summary.json:curve_nrmse.pinn_hybrid.mean|.3f]] ±
[[results:m3_tcia_summary.json:curve_nrmse.pinn_hybrid.std|.3f]]. Phase-level predictions:
[[results:m3_tcia_summary.json:n_phase_rows]] rows. There is no ground-truth θ on real data.

## 4. Discussion

H1 holds for parameter recovery under synthetic degradation (PINN) and for curve
reconstruction under the same stress (amortized). It does not hold for short PINN fits on
two- to four-phase real CT, where the classical Bae model is more stable. That is the result
reported here. A larger PINN budget or a population prior may close the real-data gap; that
is not claimed.

Limitations: reduced three-compartment physiology; the liver ROI is an HU window rather than
the published segmentation; the injection protocol for the TCIA cases is a population default
because body weight is not in the imaging archive; amortized training was short.

## 5. Data and code

- Software: MIT, this repository.
- TCIA HCC-TACE-Seg: CC BY 4.0 [3,4].
- Frozen tables: `paper/frozen/`.

## References

1. K. T. Bae, J. P. Heiken, and J. A. Brink, "Aortic and hepatic contrast medium enhancement at CT. Part I. Prediction with a computer model," *Radiology* **207**(3), 647–655 (1998) [doi:10.1148/radiology.207.3.9609886].
2. K. T. Bae, "Intravenous contrast medium administration and scan timing at CT: considerations and approaches," *Radiology* **256**(1), 32–61 (2010) [doi:10.1148/radiol.10090908].
3. A. M. Moawad, D. Fuentes, A. Morshid, et al., "Multimodality annotated hepatocellular carcinoma data set including pre- and post-TACE with imaging segmentation," *Sci. Data* **10**, 33 (2023) [doi:10.1038/s41597-023-01928-3].
4. A. M. Moawad, D. Fuentes, M. ElBanan, et al., "Multimodality annotated HCC cases with and without advanced imaging segmentation (HCC-TACE-Seg)," The Cancer Imaging Archive (2021) [doi:10.7937/TCIA.5FNA-0924].
