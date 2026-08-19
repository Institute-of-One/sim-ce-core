# M3.5 — Identifiability and Adaptive Phase Design

Proposal for approval before implementation. Covers the six items requested: the gap
between PROJECT_SPEC and the code, a critical read of the novelty claims, a single primary
endpoint, the spec revision, the new milestone, and what has to be agreed first.

---

## 1. What already exists, and what the pivot needs

| Needed by the new hypothesis | State |
|---|---|
| Differentiable forward `(protocol, θ) → C(t)` | **Exists.** `simulate_closed_form_tensors(theta: dict[str, Tensor], ...)` is documented "autodiff-friendly in ``theta``" and is already reached through `params_to_tensors`. The Jacobian needs an adapter, not a rewrite. |
| Closed-form vs ODE agreement | **Exists**, NRMSE 4×10⁻⁸ |
| Inverse methods to compare | **Exists**: closed-form LS, Tikhonov deconvolution, PINN hybrid, amortized |
| Degradation sweep (noise × stride × dose) | **Exists**, 12 cells / 48 rows, frozen |
| Ablation harness | **Exists**, physics / neural / hybrid × AIF |
| Real multi-phase cohort | **Exists**, 20 TCIA cases, 53 phase rows |
| Sensitivity matrix, Fisher information, identifiability diagnostics | **Missing** — this is the new work |
| Phase-selection criterion and benchmark | **Missing** |
| Dense CTP reference series | **Does not exist and cannot be produced from this repo** |

`data/raw/` holds `_tcia_hcc_tace_seg` and `mphase_liver`. `ctp_brain` exists **only** under
`data/proxy/`. The 16-case `outputs/m3_external` run is tagged `sources: ['synthetic_proxy']`.

**Phase counts in the real cohort**, from `outputs/m3_tcia/fig3_mphase_phases.csv`:

| phases | cases |
|---|---|
| 2 | 13 |
| 3 | 3 |
| 4 | 2 |
| 5 | 2 |

One of the two phases is `nc` (non-contrast, t = 0, 0 HU). Thirteen of twenty cases
therefore carry **one post-contrast measurement**.

---

## 2. Critical read of the novelty claims

Held after verifying the prior art on Crossref.

**Sound.** The combination claimed in the pivot — Bae-style whole-body kinetics × routine
2–4-phase CT × per-case identifiability × abstention × unobserved-phase prediction ×
patient-specific next-phase recommendation — is not in the literature found. The
identifiability framing also supplies what the v1 result lacked: a **mechanism**. Reporting
that a neural estimator wins on synthetic stress and loses on real phases is a curiosity;
reporting that the design is rank-deficient for the full parameter vector at clinical phase
counts, and that no estimator can recover what the design does not determine, is a finding.

**Must not be claimed.**

- *A PINN generalization of a contrast-kinetics model.* Prior art: van Herten et al.,
  *Med Image Anal* 78, 102399 (2022); Sainz-DeMena et al., *Med Eng Phys* 123, 104092
  (2023); Ahmadi Daryakenari, Wang & Karniadakis, *Comput Biol Med* 184, 109392 (2025);
  Souza, Amorim & Rispoli, *IFMBE Proc*, 623–632 (2025) — the last is titled "Evaluating
  Pharmacokinetic Models of Iodized Contrast Using Physics-Informed Neural Networks".
- *Fisher-information-based optimal sampling.* Textbook optimal experimental design. The
  pivot already says not to claim it. The write-up must **actively cite** the OED
  literature; silence reads as ignorance rather than modesty.
- *"Existing work is all densely sampled."* DCE-MRI has prior work on reduced sampling and
  optimal time-point selection. The honest statement is narrower: the identifiability of a
  Bae-style whole-body model at routine CT phase counts has not been reported.

**Wording.** "Certificate" implies a proof in the formal-methods sense; use
**identifiability report**. Keep "AI proposes; physics verifies; identifiability decides"
out of the title and abstract — a slogan reads as marketing to a physics reviewer. It is
fine as a section heading.

**The clinical hook is real and should be used carefully.** A single-phase radiomics study
that fails to separate response classes is an instance of the same shortage this paper
measures, but radiomics failing is not the same event as a kinetic parameter being
unidentifiable, and the paper must not equate them. Unpublished hospital data cannot enter
this manuscript: its current ethics position — no human participants, no data collected,
public de-identified images only — is an asset worth keeping intact.

---

## 3. Primary endpoint — one

> **Whether the Fisher information of a sampling design predicts which physiological
> parameters are recoverable, tested against the recovery errors already measured in the
> M2 sweep.**

Everything else is secondary. This endpoint is chosen because it is the one that converts
the existing result into a mechanism, it runs entirely on synthetic ground truth, it needs
no new data, and it can fail: if the condition number does not predict parameter error, the
identifiability story is wrong and the paper says so.

Secondary endpoints, in order:

1. **Identifiability at clinical phase counts.** For 2-, 3- and 4-phase designs, which
   parameters are identifiable, weakly identifiable, or not separable.
2. **Phase-selection benchmark.** Fixed clinical timing vs equispaced vs random vs
   population-optimal vs patient-adaptive, at equal acquisition count, on synthetic data.
3. **Held-out phase prediction on real CT.** The 7 cases with ≥3 phases: fit on all but one
   phase, predict the withheld phase, compare with the measured HU.
4. **Identifiability applied to the real cohort.** What fraction of the 20 acquisitions is
   rank-deficient for the full θ.

---

## 4. Changes to the experiment plan

| Proposed | Decision | Reason |
|---|---|---|
| Exp 1 — synthetic identifiability map | **Keep**, becomes primary | No new data; carries the mechanism |
| Exp 2 — phase-selection benchmark | **Keep** | The practical payoff; reuses the simulator |
| Exp 3 — dense CTP retrospective | **Rescope**, do not drop | No dense CTP exists. Its scientific role — predict a time point that was actually measured — is delivered by held-out phase prediction on the ≥3-phase subset. The dense-CTP version becomes stated future work |
| Exp 4 — real multi-phase | **Keep**, scoped | Leave-one-phase-out is only meaningful on the 7 cases with ≥3 phases; the 13 two-phase cases enter the identifiability analysis, not the prediction test |
| 12-arm ablation | **Keep in full** | Cheap once the modules exist, and ablations are what make a methods paper read as research rather than a note |
| Neural phase-selection policy | **Drop** | Optional in the original plan, and it re-centres the paper on the AI |
| Profile likelihood | **Defer** | Bootstrap plus Cramér–Rao covers uncertainty; a third route costs time and adds no endpoint |
| A- / E-optimal comparison | **Keep as a sensitivity** | One paragraph, not an experiment. D-optimal is primary |

**Guard, non-negotiable.** Every config whose output can reach the manuscript sets
`allow_proxy: false`, and the run refuses rather than falling back. An autonomous
implementation given the current `m3_external.yaml` (`allow_proxy: true`) will produce a
complete "real dense CTP" result from synthetic data, and nothing in `summary.json` says
"proxy" except the `sources` field.

---

## 5. New modules — the minimum that carries the endpoint

```
sim_ce_core/design/
  sensitivity.py       J = dC/dtheta by autograd, against a finite-difference reference
  fisher.py            F(S) = sum_i J_i^T Sigma_i^-1 J_i
  identifiability.py   rank, smallest singular value, condition number, CRLB, correlation
  phase_selection.py   argmax_t [log det F(S + t) - log det F(S)], plus A- and E-optimal
sim_ce_core/uncertainty/
  bootstrap.py         case-level resampling intervals
  calibration.py       interval coverage
sim_ce_core/experiments/
  identifiability_map.py
  adaptive_phase.py
```

`certificates.py` folds into `identifiability.py` — the report is a row, not a subsystem.
`profile_likelihood.py` deferred. `retrospective_sparse_ctp.py` is not created; held-out
phase prediction extends the existing M3 runner.

**Tests:** autograd Jacobian against finite differences; a rank-deficient design is
detected; a known non-identifiable parameter pair is reported as such; adaptive selection
reduces expected posterior variance against fixed timing; interval coverage; seeded
reproducibility; no network.

---

## 6. What is being asked

Approval of: the primary endpoint in §3, the experiment changes in §4, the module list in
§5, and the `allow_proxy: false` guard. On approval the implementation spec is written from
this document and can be handed to a separate agent; the analysis above is what makes that
hand-off safe.

**Assumed failure conditions**, stated now rather than discovered later:

- The Fisher condition number does not predict recovery error in the M2 sweep. Then the
  primary endpoint is negative and the paper reports that the diagnostic does not work,
  which is publishable and is the reason for choosing a falsifiable endpoint.
- Patient-adaptive selection does not beat fixed clinical timing at equal count. Plausible:
  clinical phase timing is already the product of decades of tuning. Reported as such.
- Too few real cases carry ≥3 phases for the held-out prediction to separate anything. Then
  it is descriptive, and the identifiability analysis of all 20 carries the real-data arm.
