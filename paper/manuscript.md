<!-- Every number below is a marker resolved from paper/frozen/ at build time by
     paper/build_manuscript.py. The syntax is [[results:<file>:<path>|<format>]]; this
     comment is not resolved, so it can show the notation without naming a real file.
     Do not type a number into this file. If a value is not reachable from a frozen
     file, add it to the freeze rather than typing it here. -->

# Sampling design bounds parameter recovery in a reduced CT contrast-kinetics model, and a closed-form fit nearly attains the bound

**Shuji Yamamoto**
Institute of One, LISIT Co., Ltd., Tokyo, Japan
yamamoto@lisit.jp · ORCID 0000-0001-9211-1071

## Abstract

**Background and Objectives.** Neural estimators are increasingly applied to
contrast-kinetics models, but whether physiology is recoverable from the two to four
phases a routine contrast-enhanced CT acquires is unestablished. We asked what a phase pattern determines, and whether that limit predicts
estimator error.

**Methods.** A differentiable three-compartment Bae-style model gives the sensitivity
matrix in closed form, and from it the Fisher information of an acquisition schedule and
the Cramér–Rao bound on each parameter, as a relative standard error. Closed-form least
squares, Tikhonov deconvolution, a physics-informed residual and
amortized inference ran over [[results:m2_summary.json:n_cells]] designs of noise, stride
and dose, each repeated over [[results:manifest.json:metrics.m2_n_realisations]] noise
draws, and on [[results:manifest.json:metrics.m3_tcia_n_cases]] public multi-phase liver
CTs.

**Results.** Scaling three volumes, cardiac output and the attenuation constant together
leaves every enhancement curve unchanged, so the full physiology is never identifiable at any density. With those fixed, the bound predicts measured error across the
[[results:manifest.json:metrics.endpoint_n_cells]] nonzero-noise designs (Spearman
[[results:manifest.json:metrics.endpoint_closed_form_spearman|.2f]]–[[results:manifest.json:metrics.endpoint_pinn_hybrid_spearman|.2f]],
p ≤ [[results:manifest.json:metrics.endpoint_closed_form_p|.4f]]). The closed-form fit
runs at a median [[results:manifest.json:metrics.endpoint_closed_form_efficiency|.2f]]
times the bound, the neural estimators at
[[results:manifest.json:metrics.endpoint_pinn_hybrid_efficiency|.2f]] and
[[results:manifest.json:metrics.endpoint_amortized_efficiency|.2f]]. On real CT,
[[results:manifest.json:metrics.m3_tcia_exact_fits]] of
[[results:manifest.json:metrics.m3_tcia_n_cases]] studies are two-phase, carrying one informative
measurement, which the closed form interpolates exactly; on the
[[results:manifest.json:metrics.m3_tcia_n_constrained]] with three or more phases both
methods reach [[results:manifest.json:metrics.m3_tcia_constrained_closed_form|.3f]].
Ground-truth physiology is absent there; parameter recovery is not claimed on
real data.

**Conclusions.** In this reduced model and sampling regime the acquisition, not estimator
complexity, set the recoverable information.

**Keywords:** contrast enhancement; pharmacokinetic modelling; identifiability; Fisher
information; optimal experimental design; physics-informed neural networks; computed
tomography.

## 1. Introduction

Bae's physiology-based model predicts organ-specific CT enhancement from injection
protocol and body habitus, and remains the reference account of how injection rate,
dose and body size set enhancement timing and magnitude [1,2]. Recovering physiology
*from* an observed enhancement curve is the inverse of that map, and it is the operation
a growing body of work now performs with neural machinery: physics-informed networks for
myocardial perfusion MRI [3], for dynamic contrast-enhanced MRI in the presence of
diffusion [4], compartment-model-informed networks for drug dynamics [5], and
physics-informed evaluation of iodinated-contrast pharmacokinetics [6]. Scientific
machine learning has been surveyed as a way to add neural components to existing
pharmacokinetic models [7], and neural ordinary differential equations have been
reviewed for medical image analysis [8]. Enhancement prediction for the liver has also
been approached patient-specifically without a neural inverse at all [9].

That work is validated where dynamic imaging is dense. Myocardial perfusion MRI, DCE-MRI
and CT perfusion sample tens of time points. *Routine contrast-enhanced CT acquires two
to four.* The regime is not a harder version of the same problem; it is a different one,
and the classical literature already records that sparse temporal sampling breaks the
other standard inverse — deconvolution overestimates flow as the sampling interval grows,
which is why regularised and sparse variants were developed [10,11].

Two questions follow, and they are not the same question. *Can the curve be
reconstructed?* and *can the parameters that generated it be recovered?* A method may
answer the first while the data cannot answer the second, and a comparison of estimators
cannot distinguish those cases from the outside. What separates them is a property of the
acquisition rather than of any estimator: the Fisher information of the sampling design,
and the Cramér–Rao bound it places on every unbiased estimator at once.

This paper computes that bound for the phase patterns a routine abdominal CT actually
uses, and tests whether it predicts the error estimators make. The test could fail. If
the bound does not track measured recovery error, the account offered here is wrong.

**Contributions.**

1. An exact structural non-identifiability of the reduced Bae model: a one-parameter
   scale symmetry under which three volumes, cardiac output and the attenuation constant
   move together with no observable effect. It is proved by construction and confirmed to
   [[results:manifest.json:metrics.m1_closed_form_ode_nrmse|sci0]] in the forward
   simulation.
2. The Cramér–Rao bound on physiological recovery for two-, three- and four-phase
   acquisitions, in relative units a reader can act on.
3. A test, over [[results:manifest.json:metrics.endpoint_n_cells]] sampling designs and
   [[results:manifest.json:metrics.m2_n_realisations]] noise draws each, that the bound
   predicts measured error — and a measurement of how close each estimator comes to it.
4. An open, deterministic implementation in which every number reported here is
   regenerated from machine-readable files by a documented command.

We do not claim novelty for applying physics-informed networks to contrast kinetics
[3,4,5,6], nor for Fisher-information-based experimental design, which is standard. What
is new is the identifiability of this model at these phase counts, and the finding that
the classical estimator already saturates it.

## 2. Methods

### 2.1 Forward model

States are iodine concentration in central blood, organ and recirculation. Injection is
a delayed rectangular bolus. The linear system `dc/dt = Ac + bI(t)` is solved in closed
form by matrix exponential and, independently, by adaptive `dopri5`; the two agree to
NRMSE [[results:manifest.json:metrics.m1_closed_form_ode_nrmse|sci0]] (Figure 3). Enhancement is
`HU = k c`. Peak aorta and organ enhancement at the reference protocol are
[[results:manifest.json:metrics.m1_peak_aorta_hu|.0f]] and
[[results:manifest.json:metrics.m1_peak_organ_hu|.0f]] HU.

This is a reduced descendant of Bae's model, which used more than a hundred differential
equations [1]. The reduction is what makes the sensitivity matrix available in closed
form, and it is also a limitation (Section 5).

### 2.2 Sensitivity, information and identifiability

The system parameters enter through the state matrix as tensors, so `J = dC/dθ` is
obtained by automatic differentiation. The transit delay shifts the simulation grid,
which is sampled by nearest index and carries no gradient; its column is a central
difference. Every column is checked against a central difference in the test suite, so
neither route is trusted alone.

For independent Gaussian measurement noise the Fisher information of a set of acquisition
times is `F(S) = Σ Jᵀ Σ⁻¹ J`, and `F⁻¹` bounds the covariance of any unbiased estimator.
All quantities are computed in *log-parameter space*, `θ ∂C/∂θ`. This is not
presentational: the parameters span four orders of magnitude in their units, and a Fisher
matrix built from raw derivatives reports a condition number that depends on whether
volumes were written in litres or millilitres. In log space the Cramér–Rao bound is a
relative standard error.

Enhancement is read in the aorta and in the organ. The recirculation compartment is a
modelling device rather than something a scan reports, and including it would credit the
design with information no radiologist ever sees.

### 2.3 Estimators

Four, spanning the classical and the learned.

**Closed-form least squares.** Levenberg–Marquardt in log-parameter space.

**Tikhonov deconvolution.** The organ curve deconvolved from the arterial input,
omitted where fewer than eight samples are available.

**Physics-informed hybrid.** `C = C_phys(θ) + r_φ(t)`, with a homogeneous-ODE residual
penalty on the neural term.

**Amortized inference.** A network trained on simulator draws to map an observed curve
directly to parameters.

**Training budget.** The amortized network is trained on
[[results:manifest.json:metrics.amortized_n_train]] draws for
[[results:manifest.json:metrics.amortized_n_epochs]] epochs. Budget was not a free
choice: at a thirty-second the network returns a constant, and at an eighth it returns
the prior mean of cardiac output to three figures. At the reported budget it tracks
held-out cardiac output with correlation
[[results:manifest.json:metrics.amortized_calibration_correlation|.3f]] and
[[results:manifest.json:metrics.amortized_calibration_sd_ratio|.2f]] of the true spread
(Figure 6).
A comparison against an undertrained network is not a comparison against amortized
inference, and this calibration is reported so a reader can see which was done.

### 2.4 Sampling designs and realisations

Twelve designs: noise 0, 10 and 25 HU; temporal stride 1 and 4; dose 1.0 and 0.5.
**Each is repeated over [[results:manifest.json:metrics.m2_n_realisations]] independent
noise draws.** A Cramér–Rao bound constrains the spread of an error and says nothing
about a single value, so a single draw per design cannot be compared with one. Parameter
error is reported as a root-mean-square over realisations, because that is the quantity
the bound constrains; a mean absolute error is not comparable with a standard deviation.

For the same reason the comparator for a mean absolute relative error is not the bound
itself but `√(2/π)` times it, which is what the absolute error of a zero-mean Gaussian
averages to.

### 2.5 Real multi-phase CT

[[results:manifest.json:metrics.m3_tcia_n_cases]] patients from the TCIA
HCC-TACE-Seg collection, source `[[results:manifest.json:metrics.m3_tcia_source]]`,
CC BY 4.0 [12,13], giving [[results:manifest.json:metrics.m3_tcia_n_phase_rows]]
phase-level measurements. The liver region of interest is an HU window rather than the
published segmentation, and the injection protocol is a population default because body
weight is not in the imaging archive. Both are limitations rather than choices, and
neither is hidden in what follows.

### 2.6 Reproducibility

Every figure and number is regenerated by the commands recorded in
`paper/frozen/manifest.json`. The manuscript resolves its numbers from those files at
build time; a value that cannot be resolved is a build error, and a test fails if any
frozen metric appears in the text as a typed literal.

## 3. Results

### 3.1 An exact scale symmetry

Scaling the three volumes, the cardiac output and the attenuation constant by a common
factor leaves every enhancement curve unchanged. The rate constants are ratios `q/v`, the
bolus enters as `1/v_c`, and `HU = k c` absorbs the remaining factor. In the forward
simulation the difference is
[[results:manifest.json:metrics.m1_closed_form_ode_nrmse|sci0]] relative — machine
precision.

The consequence is not a sampling problem. Those five quantities cannot be recovered
separately from enhancement at any density or noise level, and the identifiability
analysis reaches rank
[[results:m35_identifiability.json:clinical_designs.designs[n_phases=20].full_model_rank]] of
[[results:m35_identifiability.json:clinical_designs.designs[n_phases=20].full_model_parameters]]
on a twenty-point design exactly as it does on a four-phase one (Figure 1, left). Any
estimate of them rests on fixing at least one, which the conventional inverse does
implicitly by taking physiology from body habitus.

### 3.2 What routine phase counts determine, and which phase carries it

A pre-contrast acquisition carries no information about physiology: there is no contrast
in the patient, the sensitivity is identically
[[results:manifest.json:metrics.precontrast_sensitivity|.0f]], and a "two-phase" study is
therefore *one* informative measurement. The rest of the phases are not equivalent
either. Taken alone, the arterial phase carries sensitivity
[[results:manifest.json:metrics.arterial_sensitivity|.2f]], the portal venous phase
[[results:manifest.json:metrics.portal_venous_sensitivity|.2f]] and the delayed phase
[[results:manifest.json:metrics.delayed_sensitivity|.2f]] — respectively
[[results:manifest.json:metrics.arterial_over_portal_venous|.1f]] and
[[results:manifest.json:metrics.arterial_over_delayed|.0f]] times as much as the other
two, in the same units.

The bounds follow. For the two parameters an inverse actually frees, at 25 HU noise, the
Cramér–Rao bound is
[[results:m35_identifiability.json:clinical_designs.designs[n_phases=2].fitted_expected_absolute_error|.0%]]
relative on a pre-contrast plus portal venous study. Adding the arterial phase takes it
to
[[results:m35_identifiability.json:clinical_designs.designs[n_phases=3].fitted_expected_absolute_error|.0%]];
adding a delayed phase on top of that takes it only to
[[results:m35_identifiability.json:clinical_designs.designs[n_phases=4].fitted_expected_absolute_error|.0%]],
and twenty evenly spaced samples reach
[[results:m35_identifiability.json:clinical_designs.designs[n_phases=20].fitted_expected_absolute_error|.0%]]
(Figure 1, right). No estimator does better than that, whatever it is built from.

The practical content is in the first step rather than the last. One extra acquisition,
placed arterially, buys a fivefold reduction in the bound; the fourth phase buys almost
nothing, and going to twenty buys less than the arterial phase did on its own.

### 3.3 The bound predicts the error

Across [[results:manifest.json:metrics.endpoint_n_cells]] noisy designs, the bound tracks
the error measured over [[results:manifest.json:metrics.m2_n_realisations]] realisations
each: Spearman [[results:manifest.json:metrics.endpoint_closed_form_spearman|.2f]]
(p = [[results:manifest.json:metrics.endpoint_closed_form_p|.4f]]) for the closed form,
[[results:manifest.json:metrics.endpoint_pinn_hybrid_spearman|.2f]] for the
physics-informed hybrid and
[[results:manifest.json:metrics.endpoint_amortized_spearman|.2f]] for amortized
inference (Figure 2). The prespecified endpoint could have come out otherwise; it did
not.

### 3.4 The classical estimator attains the bound

The distance above the identity line in Figure 2 is the estimator's own cost, and it is
the whole of what a better estimator could recover. The closed-form fit runs at a median
[[results:manifest.json:metrics.endpoint_closed_form_efficiency|.2f]] times the bound.
The physics-informed hybrid runs at
[[results:manifest.json:metrics.endpoint_pinn_hybrid_efficiency|.2f]] and amortized
inference at [[results:manifest.json:metrics.endpoint_amortized_efficiency|.2f]].

Figure 5 plots the same comparison against noise level and Figure 4 the organ
curve each estimator reconstructs. In the most degraded design the three are not distinguishable by parameter error —
[[results:manifest.json:metrics.m2_stressed_closed_form_param_rmse|.2f]],
[[results:manifest.json:metrics.m2_stressed_pinn_hybrid_param_rmse|.2f]] and
[[results:manifest.json:metrics.m2_stressed_amortized_param_rmse|.2f]] root-mean-square
over [[results:manifest.json:metrics.m2_n_realisations]] realisations — nor by curve
NRMSE, at
[[results:manifest.json:metrics.m2_stressed_closed_form_curve_nrmse|.2f]],
[[results:manifest.json:metrics.m2_stressed_pinn_hybrid_curve_nrmse|.2f]] and
[[results:manifest.json:metrics.m2_stressed_amortized_curve_nrmse|.2f]].

### 3.5 Ablation

Physics-only and hybrid reach
[[results:manifest.json:metrics.m3_ablation_physics_aif|.2f]] and
[[results:manifest.json:metrics.m3_ablation_hybrid_aif|.2f]] curve NRMSE with the
arterial input. Neural-only reaches
[[results:manifest.json:metrics.m3_ablation_neural_aif|.2f]] (Figure 7): without the physics the
residual network is not a curve fitter at this budget. The information the hybrid uses
comes from the compartment model, not from the network.

### 3.6 Real multi-phase CT, and the same limit in the residuals

On [[results:manifest.json:metrics.m3_tcia_n_cases]] patients the closed-form fit reaches
a mean curve NRMSE of
[[results:manifest.json:metrics.m3_tcia_closed_form_nrmse_mean|.3f]] against
[[results:manifest.json:metrics.m3_tcia_pinn_nrmse_mean|.3f]] for the physics-informed
hybrid (Figure 8). Read as an accuracy comparison, that favours the classical fit by a
factor of nearly three. It is not one.

[[results:manifest.json:metrics.m3_tcia_exact_fits]] of the
[[results:manifest.json:metrics.m3_tcia_n_cases]] closed-form fits have a residual below
[[results:m3_tcia_summary.json:underdetermined.threshold_nrmse|sci0]] — machine
precision. They are not accurate fits; they are exact interpolations. Every one of them
is a two-phase case, and a two-phase case carries one informative measurement, because
the pre-contrast acquisition has no contrast in it (Section 3.2). One measurement against
two free parameters is an underdetermined system, and the model passes through the datum
exactly whatever the physiology was.

On the [[results:manifest.json:metrics.m3_tcia_n_constrained]] cases with three or more
phases, where the data does constrain the fit, the two methods are
indistinguishable: [[results:manifest.json:metrics.m3_tcia_constrained_closed_form|.3f]]
against [[results:manifest.json:metrics.m3_tcia_constrained_pinn|.3f]]. The apparent
advantage of the closed form on real data was arithmetic — thirteen zeros pulling down a
mean — and the sampling limit measured in Section 3.2 is visible in these residuals
without any Fisher calculation at all.

There is no ground-truth physiology on real data, so nothing here is a statement about
parameter recovery. What it does show is that a curve reconstructed perfectly can carry
no information about the parameters that generated it, which is the distinction the rest
of the paper exists to draw.


## 4. Discussion

The result is a statement about acquisitions rather than about algorithms. Given a
two-phase abdominal CT, the physiology of a three-compartment contrast model is
determined to tens of per cent at best, and that figure is a property of when the scanner
fired. An estimator cannot improve on it, and on this evidence the classical one is
already within seven per cent of it.

This reframes the comparison that motivated the work. A physics-informed residual and an
amortized network were expected to help most where the data are worst. They do not,
because at that point there is nothing left to extract: the closed form has taken it. The
neural methods are not failing at a hard problem; they are paying an estimator's cost on
a problem whose difficulty is set elsewhere.

For practice the useful reading is negative and specific. Reporting physiological
parameters from a routine two- to four-phase CT requires either fixing most of the model
from other information — which is a modelling assumption, and should be stated as one —
or acquiring more phases. Which additional phase is worth acquiring is a question the
same Fisher machinery can answer, and we have not answered it here.

## 5. Limitations

The physiology is a reduced three-compartment model, not Bae's full system [1]; the scale
symmetry reported in Section 3.1 is a property of this reduction and its generalisation
is not established. The deconvolution baseline is classical Tikhonov, while the CT
perfusion literature has moved to sparse and total-variation regularised variants
[10,11]; a stronger deconvolution baseline would raise that arm and is not tested here.
The real-data arm has no ground-truth physiology, uses an HU-window liver region rather
than the published segmentation, and assumes a population injection protocol because
body weight is absent from the archive. Bounds are computed at a point estimate of the
physiology rather than integrated over a prior. The noise model is independent and
Gaussian in HU.

## 6. Conclusions

For contrast-kinetics recovery from routine multi-phase CT, the binding constraint is the
sampling design. Its Cramér–Rao bound predicts measured error across designs and
estimators, and a closed-form least-squares fit attains it. Physics-informed and
amortized estimators, trained to a budget at which they demonstrably use their input,
operated at [[results:manifest.json:metrics.endpoint_pinn_hybrid_efficiency|.2f]] and [[results:manifest.json:metrics.endpoint_amortized_efficiency|.2f]] times the bound respectively. Effort spent on the estimator is
spent where the limit is not.

## Data and code

Software under the MIT licence, with the frozen tables and the commands that regenerate
every figure, in this repository. TCIA HCC-TACE-Seg is CC BY 4.0 [12,13].

## Declaration of generative AI use

Generative AI (Claude, Anthropic, through the Claude Code command-line tool) was used in
preparing this work: scaffolding and refactoring the software, drafting tests, writing
the figure and analysis scripts, and drafting and revising manuscript prose. It was not
used to design the study, to choose the endpoints, or to decide what the results mean. No
numerical result came from the model: every number in this manuscript is emitted by
executed code into machine-readable files and resolved into the text at build time, and
the test suite fails if the two disagree. The author designed the study, re-executed every
result, and is solely accountable for the content. No AI system is an author.

## References

1. K. T. Bae, J. P. Heiken, and J. A. Brink, "Aortic and hepatic contrast medium enhancement at CT. Part I. Prediction with a computer model," *Radiology* **207**(3), 647–655 (1998) [doi:10.1148/radiology.207.3.9609886].
2. K. T. Bae, "Intravenous contrast medium administration and scan timing at CT: considerations and approaches," *Radiology* **256**(1), 32–61 (2010) [doi:10.1148/radiol.10090908].
3. R. van Herten, A. Chiribiri, M. Breeuwer, et al., "Physics-informed neural networks for myocardial perfusion MRI quantification," *Med. Image Anal.* **78**, 102399 (2022) [doi:10.1016/j.media.2022.102399].
4. D. Sainz-DeMena, M. Á. Pérez, and J. M. García-Aznar, "Exploring the potential of physics-informed neural networks to extract vascularization data from DCE-MRI in the presence of diffusion," *Med. Eng. Phys.* **123**, 104092 (2023) [doi:10.1016/j.medengphy.2023.104092].
5. N. Ahmadi Daryakenari, S. Wang, and G. Karniadakis, "CMINNs: compartment model informed neural networks — unlocking drug dynamics," *Comput. Biol. Med.* **184**, 109392 (2025) [doi:10.1016/j.compbiomed.2024.109392].
6. T. Souza, R. Amorim, and V. Rispoli, "Evaluating pharmacokinetic models of iodized contrast using physics-informed neural networks," *IFMBE Proc.*, 623–632 (2025) [doi:10.1007/978-3-031-94921-0_68].
7. D. Valderrama, A. Ponce-Bobadilla, S. Mensing, et al., "Integrating machine learning with pharmacokinetic models: benefits of scientific machine learning in adding neural network components to existing PK models," *CPT Pharmacometrics Syst. Pharmacol.* **13**(1), 41–53 (2023) [doi:10.1002/psp4.13054].
8. H. Niu, Y. Zhou, X. Yan, et al., "On the applications of neural ordinary differential equations in medical image analysis," *Artif. Intell. Rev.* **57**(9) (2024) [doi:10.1007/s10462-024-10894-0].
9. H. Setiawan, C. Chen, E. Abadi, et al., "A patient-informed approach to predict iodinated-contrast media enhancement in the liver," *Eur. J. Radiol.* **156**, 110555 (2022) [doi:10.1016/j.ejrad.2022.110555].
10. R. Fang, T. Chen, and P. C. Sanelli, "Towards robust deconvolution of low-dose perfusion CT: sparse perfusion deconvolution using online dictionary learning," *Med. Image Anal.* **17**(4), 417–428 (2013) [doi:10.1016/j.media.2013.02.005].
11. R. Fang, S. Zhang, T. Chen, et al., "Robust low-dose CT perfusion deconvolution via tensor total-variation regularization," *IEEE Trans. Med. Imaging* **34**(7), 1533–1548 (2015) [doi:10.1109/TMI.2015.2405015].
12. A. M. Moawad, A. Morshid, A. M. Khalaf, et al., "Multimodality annotated hepatocellular carcinoma data set including pre- and post-TACE with imaging segmentation," *Sci. Data* **10**, 33 (2023) [doi:10.1038/s41597-023-01928-3].
13. A. M. Moawad, D. Fuentes, M. ElBanan, et al., "Multimodality annotated HCC cases with and without advanced imaging segmentation (HCC-TACE-Seg)," The Cancer Imaging Archive (2021) [doi:10.7937/TCIA.5FNA-0924].
