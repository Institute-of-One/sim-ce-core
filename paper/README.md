# Paper artefacts — IORN-008

Target: *Computer Methods and Programs in Biomedicine* (Elsevier), Original Research.

The venue was chosen for a structural reason. A full recent issue of CMPB contains
research articles, reviews, letters, editorials and correspondence, and **no Technical
Note or Short Communication category** — there is no short-form research bin an editor
could reclassify a methods paper into. Two companion papers were reclassified or desk
rejected on classification rather than on correctness, so this mattered more than impact.
CMPB also states that reporting new computational methodology and distributing
demonstrable software is what it exists for, which is the property that gets a paper
demoted elsewhere.

## Build

Run the experiments, then rebuild every artefact with one command:

```bash
python -m sim_ce_core.experiments.run configs/m1_synthetic.yaml
python -m sim_ce_core.experiments.run configs/m2_robustness.yaml
python -m sim_ce_core.experiments.run configs/m3_tcia.yaml
python -m sim_ce_core.experiments.run configs/m3_ablation.yaml
python -m sim_ce_core.experiments.identifiability_map

python paper/build_all.py     # must end "every artefact rebuilt and checked"
```

`build_all` freezes the runs, collects the figures, resolves the manuscript, renders the
Word file, builds the highlights, the interest declaration and the cover letter, writes
the plain-text title and abstract for the portal, and runs the checks in `--strict`.
Rebuilding one artefact and forgetting another is how a copy goes stale, and every copy
in this programme has gone stale at least once: a cover letter that kept a withdrawn
claim, a submission form three weeks behind the abstract, a kit naming a title the paper
no longer had.

Every number in `paper/manuscript.md` is a `[[results:...]]` marker resolved from
`paper/frozen/` at build time. A marker that cannot be resolved is a build error, and a
test fails if any frozen metric appears in the prose as a typed literal.

## Figure captions

Written here once. The manuscript places the image; the caption is taken from this table.
A caption written twice — as alt text and again as a numbered paragraph — prints twice in
the converted document, in two wordings.

| Fig | Caption |
|---|---|
| Fig 1 | What a routine phase pattern determines. Left: the number of directions each design constrains in the full seven-parameter physiology, which never reaches seven because the model has an exact scale symmetry that no sampling density breaks. Right: the Cramér–Rao bound on the two parameters an inverse frees, at 25 HU noise, on a logarithmic axis with the 20 % relative-error line marked |
| Fig 2 | The primary endpoint. Measured parameter error, averaged over 20 independent noise draws, against the Cramér–Rao bound of the design that produced it, for each estimator. The dashed line is where an efficient unbiased estimator sits; distance above it is the estimator's own cost and is the whole of what a better estimator could recover |
| Fig 3 | The forward model at the reference protocol: aorta, organ and recirculation enhancement against time, closed-form solution and adaptive ODE integration overlaid |
| Fig 4 | Organ curve reconstruction in the most degraded design (25 HU noise, stride 4, half dose), for the closed-form fit, the physics-informed hybrid and Tikhonov deconvolution, against the noise-free truth |
| Fig 5 | Parameter mean relative error against noise level for each estimator, at full dose, over 20 realisations per cell |
| Fig 6 | Amortized inference on held-out synthetic physiology: predicted against true cardiac output, at the training budget the study reports. At a thirty-second of that budget the same panel is a horizontal line |
| Fig 7 | Ablation on synthetic data: curve NRMSE for physics-only, hybrid and neural-only residuals, with and without the arterial input function |
| Fig 8 | Curve NRMSE for each of the 20 public multi-phase liver CTs, closed-form fit against physics-informed hybrid, on a logarithmic axis. The thirteen closed-form points at the floor are exact interpolations rather than accurate fits: each is a two-phase case, carrying one informative measurement against two free parameters. There is no ground-truth physiology on real data, so this compares curve reconstruction and not parameter recovery |

## Data provenance

The external arm is 20 TCIA HCC-TACE-Seg patients, CC BY 4.0. A synthetic proxy cohort is
available to the same loaders and is tagged `synthetic_proxy`; the configuration behind
every number in the paper sets `allow_proxy: false`, and a test asserts that the frozen
provenance reads `tcia_hcc_tace_seg`. Simulated data reported as external validation is
the failure this guards against.
