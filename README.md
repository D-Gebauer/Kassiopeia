# forwardmodel

Forward-model multi-probe cosmological maps — **weak lensing**, **galaxy
clustering**, **CMB lensing**, and the **late-time ISW** temperature
anisotropy — from N-body (or lognormal) HEALPix density shells. Includes tidal
tensors, TATT intrinsic alignments, quadratic galaxy bias, magnification bias,
source clustering, reduced shear, photo-z shift/stretch, shear calibration,
additive shear / PSF leakage, and realistic observational noise.

This is a generalised, importable library distilled from a CosmoGrid + DES Y3 +
ACT DR6 forward-modeling pipeline. The core API is dataset-agnostic (any shells,
any n(z), any mask, any noise); optional submodules add one-line loaders for the
original data products and BaryonForge baryonification.

## Install

```bash
pip install -e .                      # core: numpy, scipy, healpy, pyccl
pip install -e ".[lognormal]"         # + GLASS, for high-z shell extension
pip install -e ".[loaders]"           # + h5py, pyyaml, for dataset loaders
pip install -e ".[baryons]"           # + BaryonForge
pip install -e ".[all,test]"          # everything + pytest
```

## Quick start

```python
import numpy as np
import forwardmodel as fm

# 1. Cosmology (thin pyccl wrapper with cached distance/growth interpolants).
cosmo = fm.Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

# 2. Density shells — particle counts or overdensity, with redshift bounds.
shells = fm.ShellStack.from_counts(counts, z_lower, z_upper)   # (n_shells, npix)

# Optionally extend past the N-body cap with cross-correlated lognormal shells.
shells = shells.extend_lognormal(cosmo, z_edges=np.linspace(3.5, 10, 24))

# 3. Redshift distributions, one NumberDensity per tomographic bin.
source_bins = fm.make_bins(z, [nz0, nz1, nz2, nz3])

# 4. Define probes and project (single pass over the shells).
wl  = fm.WeakLensing(nz=source_bins,
                     ia=fm.TATT(a1=0.7, a2=-0.5, b_ta=0.5),
                     source_clustering_bias=1.0)
gc  = fm.GalaxyClustering(nz=lens_bins,
                          bias=fm.QuadraticBias(b1=[1.4, 1.6, 1.8], b2=0.2, bs=-0.1),
                          magnification=[0.4, 1.0, 1.8])   # number-count slope s
cmb = fm.CMBLensing(z_source=1090.0)
isw = fm.ISW()                                       # late-time ISW (ΔT/T)

maps = fm.ForwardModel(cosmo, shells).project(
    weak_lensing=wl, galaxy_clustering=gc, cmb_lensing=cmb, isw=isw)

# 5. Add observational realism.
rng = np.random.default_rng(0)
wl_obs = fm.WeakLensingObservation(
    mask, nbar=1.5,
    shape_noise=fm.DESShapeNoise(metacal, selection_response=R_S),  # or KiDSShapeNoise,
                                                   # EmpiricalShapeNoise, GaussianShapeNoise, NoShapeNoise
    shear_calibration=[-0.006, -0.020, -0.024, -0.037],
    reduced_shear=True,                            # g = γ/(1−κ)
    additive_shear=(c1, c2),                       # c-term, scalar or per-bin
    psf_leakage=(alpha, e1_psf, e2_psf),           # α·e_psf leakage
).observe(maps.weak_lensing, rng)

gc_counts = fm.GalaxyClusteringObservation(mask, nbar=nbar_lens).observe(
    maps.galaxy_clustering, rng)
cmb_obs = fm.CMBLensingObservation(
    mask=mask_act, noise=(ell, N_L), filter=(ell, F_L)).observe(maps.cmb_lensing, rng)
isw_obs = fm.ISWObservation(
    mask=mask_cmb, noise=(ell, C_TT_plus_N_L),         # CMB primary + instrument
    filter=(ell, (ell <= 100).astype(float))).observe(maps.isw, rng)
```

See [`examples/`](examples/) for runnable scripts (no data required for 01–03).

## What it computes

**3D field operations (per shell)**
- Overdensity from particle counts via the shell map mean — no box size or
  particle number required.
- Tidal tensor `s_ij` and scalar invariant `s²` via spherical-harmonic Poisson
  inversion (`compute_tidal`).
- Quadratic galaxy bias `δ_g = b₁δ + b₂(δ²−⟨δ²⟩) + b_s(s²−⟨s²⟩)`
  (`LinearBias`, `QuadraticBias`).
- TATT intrinsic alignments as the exact field-level operators
  `A₁ s_ij + b_TA·A₁ δ s_ij + A₂ (s s)^TF` with redshift-dependent amplitudes
  (`TATT`, `NLA`). In the 2D single-shell tidal field the density-weighting and
  torquing terms both reduce to `δ s_ij` but keep their distinct `1/D`, `1/D²`
  evolution; the full 3D torquing shape needs line-of-sight tidal components a
  single shell does not carry.

**Line-of-sight projections** (geometric, box-independent weights)
- Weak-lensing convergence κ integrated against the lensing efficiency kernel.
- CMB-lensing convergence integrated to the last-scattering surface.
- **Late-time ISW** temperature anisotropy `ΔT/T = sum_i a_i δ_i / [ℓ(ℓ+1)]`
  with `a_i = 3(H₀/c)³ Ω_m E(z_i)(1−f(z_i)) χ²_i Δχ_i`. Sourced only where
  dark energy makes the potentials decay (`f < 1`); vanishes in matter
  domination. Inverse Laplacian applied once after the shell loop.
- Galaxy and source clustering projected against each tomographic n(z); source
  clustering also biases the convergence signal through the non-isotropic
  effective n(z) (a redshift-ordered single pass of cumulative foreground
  convergence) and adds the IA × source-clustering cross term
  `b_src·A₁·δ·s_ij` (the strict-second-order coupling of the local source
  overdensity to the linear IA field).
- Lens magnification bias `(5s−2)·κ_lens`, the lens n(z) acting as the source plane.
- Intrinsic-alignment shear projected as a spin-2 field.

**Photo-z systematics** (on each `NumberDensity`)
- Mean shift `n(z−Δz)` (`shifted`), width stretch `n(z₀+(z−z₀)/s)` (`stretched`),
  or both at once (`transformed`).

**Observational realism**
- Galaxy clustering: Poisson shot noise, `N ~ Poisson(n̄(1+δ_g)·mask)`.
- Weak lensing: source clustering + optional reduced shear `g = γ/(1−κ)` +
  shape noise (real survey catalogue with randomly rotated galaxies at their true
  positions — `DESShapeNoise` divides by the metacal response `<R_γ>+R_S`,
  `KiDSShapeNoise` normalises by the lensfit weights — or bootstrapped catalogue
  shapes, analytic Gaussian, or none) +
  multiplicative shear calibration + additive shear (c-term) + PSF leakage +
  Kaiser-Squires reconstruction.
- CMB lensing: multipole filter + reconstruction-noise realisation + mask.
- ISW: multipole filter + total temperature-noise realisation (primary CMB +
  instrument, in `ΔT/T`) + mask.

## Design

- **High-level**: `ForwardModel.project(...)` accumulates every requested probe
  in one pass; the `*Observation` classes turn the noiseless maps into mocks.
- **Low-level**: every piece is importable on its own —
  `compute_tidal`, `kappa_to_gamma`/`gamma_to_kappa`, the weight builders in
  `forwardmodel.projections`, the bias/IA models, and the noise functions in
  `forwardmodel.noise`.

### Note on shell normalisation

Shells are projected in **overdensity** form. Count shells are converted per
shell as `δ = counts/⟨counts⟩ − 1` using the map mean, so the projection
weights depend only on shell redshift bounds and the cosmology. This is
algebraically equivalent to the UFalcon count-form prefactors when the shell map
mean equals the theoretical particle count, but needs no box size or particle
number — which is what makes the projection dataset-agnostic.

## Optional submodules

- `forwardmodel.loaders.cosmogrid` — CosmoGrid shells (`load_shells`) and
  cosmology (`load_cosmology`).
- `forwardmodel.loaders.des_y3` — DES Y3 n(z), source/lens catalogues, footprint
  mask, the raw metacal catalogue + selection response from the Y3 mastercat
  (`load_metacal_catalog`), and the DES Y3 systematic priors (`draw_systematics`).
- `forwardmodel.loaders.kids` — KiDS lensfit shape catalogues split into Z_B bins
  (`load_shape_catalog`).
- `forwardmodel.loaders.act_dr6` — ACT DR6 CMB-lensing noise curve, filter, mask.
- `forwardmodel.baryons` — BaryonForge baryonification of count shells
  (`baryonify`), matching halos to shells by redshift.

## Tests

```bash
pytest
```
