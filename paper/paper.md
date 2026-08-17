---
title: 'sim_ce_core: a differentiable physics-informed simulator of contrast-enhancement kinetics'
tags:
  - Python
  - medical imaging
  - computed tomography
  - pharmacokinetics
  - contrast enhancement
  - differentiable simulation
authors:
  - name: Shuji Yamamoto
    corresponding: true
    affiliation: '1'
affiliations:
  - name: Institute of One, LISIT Co., Ltd., Tokyo, Japan
    index: 1
date: 17 August 2026
bibliography: paper.bib
---

# Summary

`sim_ce_core` is an open, differentiable simulator of iodinated contrast-enhancement
kinetics for CT. It re-implements a Bae-style compartmental forward model — central
blood volume, an organ compartment, and recirculation — in both a closed-form linear
baseline and a `torchdiffeq` ODE form [@bae1998aortic; @bae2010intravenous]. A
physics-informed residual and an amortized inference network sit on that core.
Researchers get a reproducible map from injection protocol and physiology to
time–enhancement curves, plus inverse recovery of physiological parameters from
sparse or noisy samples.

Every figure in the v1 paper is produced by one command,
`python -m sim_ce_core.experiments.run <config.yaml>`. Tests use synthetic data
only. A minimal public external check uses 20 baseline multiphase liver CTs from
TCIA HCC-TACE-Seg [@moawad2021hcc; @moawad2023hcc] (CC BY 4.0).

# Statement of need

Contrast-enhancement timing is still often handled with heuristic scan delays or
non-differentiable compartment codes that cannot be composed with modern neural
inverse methods. Bae's physiology-based model remains the conceptual standard for
relating cardiac output, body size, and injection protocol to aortic and organ
enhancement, but an open, autodiff-native, tested implementation with synthetic
ground-truth recovery has been missing. `sim_ce_core` fills that gap and is built
to test a scientific hypothesis (robust recovery under sparse / low-dose / noisy
sampling) rather than to ship a technical-note reimplementation.

# Features

- Differentiable 3-compartment Bae-lineage forward model (closed-form and ODE).
- PINN residual (`C = C_physics + r_φ(t)`) and Neural-ODE vector-field residual.
- Amortized simulation-based inference (`C_obs → θ̂`), with or without an AIF.
- Synthetic ground-truth generator, degradation sweeps, and Tikhonov deconvolution.
- Local extract loaders (NPZ/JSON) and a resumable TCIA HCC-TACE-Seg downloader
  capped at 30 cases.
- Seeded YAML experiments and colorblind-safe figure/CSV export.

# Usage

```bash
pip install -e ".[dev]"
pytest
python -m sim_ce_core.experiments.run configs/m1_synthetic.yaml
python -m sim_ce_core.experiments.repro_check
```

Frozen metrics and figure paths live in `paper/frozen/manifest.json`.

# Acknowledgements

This work follows the pharmacokinetic lineage of Bae and colleagues and uses the
TCIA HCC-TACE-Seg collection [@moawad2021hcc; @moawad2023hcc] under CC BY 4.0.

# References
