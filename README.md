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

**Contents:** [Install](#install) · [Quick start](#quick-start) ·
[Inputs and conventions](#inputs-and-conventions) · [Outputs](#outputs) ·
[Density shells](#density-shells) · [What it computes](#what-it-computes) ·
[Shape noise](#shape-noise-models) ·
[Harmonic projection and rotations](#harmonic-space-projection-and-rotations) ·
[Matching theory codes](#matching-theory-codes) ·
[Dataset loaders](#dataset-loaders) · [Baryonification](#baryonification) ·
[Examples](#examples) · [Approximations](#approximations-and-limitations) ·
[Tests](#tests)

Every public class and function has a docstring with its parameters; this
README describes how the pieces fit together.

## Install

Python ≥ 3.10. The package is not on PyPI; install it from a clone:

```bash
git clone https://github.com/D-Gebauer/ForwardModel.git
cd ForwardModel
pip install -e .                      # core: numpy, scipy, healpy, pyccl
```

Optional features are extras:

| Extra | Adds | Needed for |
|---|---|---|
| `lognormal` | `glass`, `camb` | lognormal shells (`from_lognormal`, `extend_lognormal`, `fill_with_lognormal`) |
| `loaders` | `h5py`, `pyyaml`, `astropy` | `forwardmodel.loaders` (CosmoGrid, DES Y3, KiDS; the Gower Street loader needs none of them, FLAMINGO only `h5py` for local files) |
| `rotation` | `ducc0` | faster `forwardmodel.rotation` (healpy is the fallback) |
| `baryons` | `tqdm`, `numba`, `joblib` | dependencies of BaryonForge, which is installed separately (see [Baryonification](#baryonification)) |
| `test` | `pytest`, `pytest-cov` | the test suite |
| `all` | all of the above except `test` | |

```bash
pip install -e ".[all,test]"          # everything
```

Notes:

- **CAMB** is used by the default `Cosmology(transfer_function="boltzmann_camb")`
  whenever a matter power spectrum is computed, which is the case for lognormal
  shells; the `lognormal` extra installs it. Projecting existing shells needs
  distances and growth only and works with the core install. Alternatively
  build the cosmology with `transfer_function="eisenstein_hu"`.
- **matplotlib** is needed by the example notebooks and is not part of any extra.

## Quick start

`counts`, the n(z) arrays, the masks and the noise inputs below are your data;
[`examples/`](examples/) has runnable versions with synthetic inputs.

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

No N-body lightcone at this cosmology? Build the whole model from lognormal
shells (needs the `lognormal` extra):

```python
model = fm.ForwardModel.from_lognormal(cosmo, nside=512, z_max=3.0, dz=0.1)
maps = model.project(weak_lensing=wl)
```

## Inputs and conventions

- **Maps** are full-sky HEALPix maps in RING ordering (the healpy default), all
  at the `nside` of the shells. Multipole transforms use `lmax = 3·nside − 1`
  unless `lmax` is passed.
- **Shells**: an array `(n_shells, npix)` of particle counts or overdensity
  plus the redshift bounds `z_lower`, `z_upper` of each shell. The dtype is kept
  (count shells are often `uint16`); conversion to overdensity happens per shell.
- **n(z)**: `fm.NumberDensity(z, nz)` per tomographic bin, normalised to unit
  integral on construction. `fm.make_bins(z, [nz0, nz1, ...])` builds a list on
  a shared grid. Between the sample points n(z) is interpolated linearly.
- **Per-bin parameters** (`b1`, `magnification`, `nbar`, `shear_calibration`,
  ...) accept a scalar (all bins) or one value per bin.
- **Masks** are binary `(npix,)` arrays at the working `nside`.
- **`nbar`** is the mean number of galaxies **per pixel**, not per arcmin².
- **Catalogues** (shape noise) give RA/Dec in degrees.
- **Shear** `(gamma1, gamma2)` is the spin-2 field returned by
  `healpy.alm2map_spin`, i.e. in healpy's local (θ, φ) basis. Estimators that
  work in an (RA, Dec) basis may need a sign flip of one component.
- **Units**: `Cosmology.comoving_distance` returns Mpc (not Mpc/h); `m_nu` is
  the summed neutrino mass in eV; the ISW map is the dimensionless ΔT/T.
- **Randomness**: every stochastic step takes a `numpy.random.Generator`, so a
  run is reproducible from its seed.

`Cosmology` takes `Omega_m` or `Omega_c`, `sigma8` or `A_s`, plus `w0`, `wa`,
`m_nu`; anything else goes to pyccl through `extra_ccl_kwargs`. An existing
`pyccl.Cosmology` can be wrapped with `fm.Cosmology.from_ccl(ccl_cosmo)`.

## Outputs

`ForwardModel.project(...)` returns a `ProjectedMaps`; probes that were not
requested are `None`.

| Attribute | Type | Fields |
|---|---|---|
| `maps.weak_lensing` | `WeakLensingMaps` | `kappa`, `gamma1`, `gamma2` (gravitational), `ia_gamma1`, `ia_gamma2`, `source_overdensity`, `ia_kappa`; all `(n_bins, npix)`. `shear(with_ia=True)` returns `(gamma1, gamma2)` including IA. |
| `maps.galaxy_clustering` | `GalaxyClusteringMaps` | `delta_g`, `(n_bins, npix)` |
| `maps.cmb_lensing` | `CMBLensingMaps` | `kappa`, `(npix,)` |
| `maps.isw` | `ISWMaps` | `theta` = ΔT/T, `(npix,)` |

`ia_kappa` is only filled for linear IA on the fast path (see
[project options](#project-options)), otherwise it is `None`.

The observation classes return:

| Call | Returns |
|---|---|
| `WeakLensingObservation(...).observe(maps.weak_lensing, rng)` | `ObservedWeakLensing` with `gamma1`, `gamma2` (observed), `gamma1_signal`, `gamma2_signal` (noiseless calibrated signal), `n_source` (source counts per pixel) and `kappa_recon` (Kaiser-Squires reconstruction; `None` with `reconstruct_kappa=False`), all `(n_bins, npix)` |
| `GalaxyClusteringObservation(...).observe(maps.galaxy_clustering, rng)` | galaxy counts, array `(n_bins, npix)` |
| `CMBLensingObservation(...).observe(maps.cmb_lensing, rng)` | filtered, noisy, masked κ map, `(npix,)` |
| `ISWObservation(...).observe(maps.isw, rng)` | filtered, noisy, masked ΔT/T map, `(npix,)` |

## Density shells

`ShellStack` holds the shells and their redshift bounds.

| Constructor / method | Use |
|---|---|
| `ShellStack.from_counts(counts, z_lower, z_upper)` | particle-count shells from an N-body lightcone |
| `ShellStack.from_overdensity(delta, z_lower, z_upper)` | shells that are already overdensity |
| `ShellStack.from_lognormal(cosmo, z_edges, nside)` | lognormal shells from the cosmology alone (GLASS) |
| `.extend_lognormal(cosmo, z_edges)` | append lognormal shells, e.g. above the N-body redshift cap for CMB lensing |
| `.fill_with_lognormal(cosmo, z_max=...)` | fill only the redshift ranges without shells (`.coverage_gaps()` lists them) |
| `.merge(other)` | combine two stacks and sort by redshift |

Lognormal shells are an independent random realisation: they have the correct
power and cross-correlation among themselves but are not correlated with the
N-body shells across the boundary, and the lognormal approximation is weakest at
low redshift.

### Note on shell normalisation

Shells are projected in **overdensity** form. Count shells are converted per
shell as `δ = counts/⟨counts⟩ − 1` using the map mean, so the projection
weights depend only on shell redshift bounds and the cosmology. This is
algebraically equivalent to the UFalcon count-form prefactors when the shell map
mean equals the theoretical particle count, but needs no box size or particle
number — which is what makes the projection dataset-agnostic.

To use the simulation's absolute amplitude instead, pass
`normalization="box"` with `box_size` (Mpc/h), `n_particles` and `cosmo`; counts
are then divided by the expected `N·V_shell/V_box`.

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
  Kaiser-Squires reconstruction. `exact_composition=True` combines intrinsic
  ellipticity and reduced shear with the exact complex mapping
  `e = (e_s + g)/(1 + g*·e_s)` instead of adding them.
- CMB lensing: multipole filter + reconstruction-noise realisation + mask.
- ISW: multipole filter + total temperature-noise realisation (primary CMB +
  instrument, in `ΔT/T`) + mask.

### Project options

`ForwardModel.project(..., z_max_tidal=3.0, linear_ia_fast=True, verbose=False)`:

- `z_max_tidal`: shells above this redshift skip the tidal terms (quadratic
  bias, IA).
- `linear_ia_fast`: for a purely linear IA model (NLA) without source
  clustering, the IA field is accumulated as a scalar like the convergence and
  turned into shear once per bin. This equals the per-shell tidal computation
  with one spin transform per bin instead of one per shell; `False` forces the
  per-shell path.
- `lmax` (argument of `ForwardModel`) sets the band limit of the tidal and
  Kaiser-Squires transforms.

## Shape-noise models

All share the `ShapeNoise` interface and are passed as
`WeakLensingObservation(shape_noise=...)`.

| Class | Input | Behaviour |
|---|---|---|
| `DESShapeNoise(catalogs, selection_response=R_S)` | per bin: `ra, dec, e1, e2, w` and `R11, R22` (or `R`); raw metacal ellipticities | every catalogue galaxy keeps its position and weight, its ellipticity is rotated by a random angle; pixel shear is `Σ w e / (<R> Σ w)` with `<R> = <R_γ> + R_S` |
| `KiDSShapeNoise(catalogs)` | per bin: `ra, dec, e1, e2, w` (lensfit weight) | same, with the lensfit-weighted mean `Σ w e / Σ w` |
| `EmpiricalShapeNoise(catalog)` | per bin: `(e1, e2, R, w)` | catalogue ellipticities bootstrapped into Poisson-drawn simulated source counts; positions are not from the catalogue |
| `GaussianShapeNoise(sigma_e=0.26)` | none | per-pixel Gaussian noise with variance `σ_e²/N` |
| `NoShapeNoise()` | none | signal only (default) |

With `DESShapeNoise` and `KiDSShapeNoise` the noise follows the real survey's
source density, so `nbar` and the simulated source overdensity are not used for
the noise. Catalogues can be dicts of arrays, structured arrays, loaded `.npz`
files, DataFrames or FITS record arrays; `columns=` renames columns. KiDS has no
per-object response: pass the residual multiplicative bias as
`shear_calibration`.

## Harmonic-space projection and rotations

Every linear weak-lensing field is a weighted sum of shell overdensities, and
the harmonic transform, rotations and Kaiser-Squires are linear. So the shells
can be transformed once and reused for many n(z) variants, IA amplitudes and
random rotations, each costing a matrix product and one synthesis instead of a
new pass over the shells (`forwardmodel.harmonic`, `forwardmodel.rotation`).

```python
from forwardmodel import harmonic, rotation

alms = harmonic.shell_overdensity_alms(shells)            # (n_shells, n_alm), once
W_kappa, W_ia = harmonic.weak_lensing_weights(
    cosmo, source_bins, shells.z_lower, shells.z_upper, ia=fm.NLA(a1=0.7))
kappa_lm = harmonic.combine_alms(alms, W_kappa)           # (n_bins, n_alm)
ia_lm    = harmonic.combine_alms(alms, W_ia)

angles = rotation.random_euler_angles(seed=(2026, 17))    # reproducible from the seed
lmax = 3 * shells.nside - 1
rotated = rotation.rotate_alm(kappa_lm[0] + ia_lm[0], lmax, angles)
gamma1, gamma2 = harmonic.kappa_alm_to_shear(rotated, shells.nside)
```

- The result equals the pixel-space projection of `ForwardModel` up to the
  `map2alm` quadrature error and the precision of the stored alms (`complex64`
  by default).
- Linear IA (NLA) only: no TATT quadratic terms and no source clustering.
- Rotations act on the alms and are exact for band-limited fields; angles are
  healpy's ZYZ Euler convention `(psi, theta, phi)`. Random rotations are
  uniform on SO(3). Store the seed or the angles with each product.
- `rotate_alm` uses `ducc0` when installed and healpy otherwise;
  `rotation.have_ducc0()` reports which.

## Matching theory codes

Three conventions differ between codes; the defaults follow CCL.

- **IA normalisation.** `TATT` / `NLA` take `c1_rhocrit` (default
  `forwardmodel.ia.C1_RHOCRIT = 0.0134`, as in CCL). Use
  `forwardmodel.ia.C1_RHOCRIT_COSMOSIS` to match a CosmoSIS theory prediction at
  equal `a1`.
- **n(z) interpolation.** `NumberDensity(z, nz)` interpolates linearly.
  `NumberDensity.from_spline(z, nz)` reproduces codes that spline the n(z)
  (CosmoSIS `project_2d`).
- **Neutrino mass.** `Cosmology(m_nu=...)` is the summed mass. CosmoGrid stores
  the mass per species, which `loaders.CosmoGridV1.cosmology_from_params`
  converts (`n_massive_nu=3`) and checks against the file's `Om`.

## Dataset loaders

`forwardmodel.loaders` reads the data products the package was built around and
returns the generic objects (`ShellStack`, `NumberDensity` lists, masks,
catalogues). It needs the `loaders` extra. The data are not included, and all
paths are passed explicitly; DES Y3 catalogues, CosmoGrid, the Gower Street
and FLAMINGO simulations are obtained from the respective collaborations and releases.

**`loaders.CosmoGridV1`**: CosmoGridV1 (`.npz` shells, `params.yml`).

| Function | Reads |
|---|---|
| `load_cosmology(params.yml)` | CosmoGrid `params.yml` → `Cosmology` |
| `cosmology_from_params(p)` | same from a parameter dict or a v1.1 `sim_params` row |
| `load_shells(path.npz)` | `.npz` with `shells` `(n_shells, npix)` and `shell_info` (`lower_z`, `upper_z`) → `ShellStack` |
| `load_halo_catalog(path.npz)` | halo lightcone structured array (for baryonification) |

**`loaders.CosmoGridV1_1`**: CosmoGrid v1.1 (`baryonified_shells_v11.h5`).

| Function | Reads |
|---|---|
| `load_shells(path.h5, baryonified=False, metainfo=..., shell_info=...)` | `baryonified_shells_v11.h5` → `ShellStack` |
| `load_params(path.h5)` | the `sim_params` row of a v1.1 file |
| `cosmology_from_params(p)` | `sim_params` row → `Cosmology` (the same function as in `CosmoGridV1`) |

The v1.1 shell files do not contain the shell redshift bounds, so
`load_shells` needs either the v1.1 metainfo file or the bounds themselves:

```python
from forwardmodel.loaders import CosmoGridV1_1

shells, p = CosmoGridV1_1.load_shells(
    "<cosmogrid>/<cosmology>/run_0000/baryonified_shells_v11.h5",
    baryonified=True,                                    # DMO + baryonification increments
    metainfo="<cosmogrid>/CosmoGridV11_bary_Fluri+2022_metainfo.h5",
    return_params=True)
cosmo = CosmoGridV1_1.cosmology_from_params(p)
```

**`loaders.GowerStreet`**: the [Gower Street simulations](http://www.star.ucl.ac.uk/GowerStreetSims/)
(Jeffrey et al. 2024). Needs no extra.

| Function | Reads |
|---|---|
| `load_shells(sim_dir, z_max=None, nside=None)` | `run.XXXXX.lightcone.npy` + `z_values.txt` of one unpacked `simNNNNN/` → `ShellStack` (counts, ordered by redshift) |
| `load_shell_info(sim_dir)` | `z_values.txt` → structured array `step, lower_z, upper_z, lower_com, upper_com` |
| `load_control(sim_dir)` | `control.par` → dict (`dBoxSize` in Mpc/h, `nGrid`, ...) |
| `load_params(sheet.csv, serial)` | one row of the parameter sheet (sheet 1 of the release workbook, exported as CSV) → dict |
| `load_cosmology(sheet.csv, serial)`, `cosmology_from_params(p)` | that row → `Cosmology` |

```python
from forwardmodel.loaders import GowerStreet

cosmo = GowerStreet.load_cosmology("<gower>/sheet1.csv", 400)
shells = GowerStreet.load_shells("<gower>/sim00400", z_max=3.0, nside=1024)
```

The ~100 shells of a simulation take about 10 GB at the native NSIDE 2048, so
`z_max` (load only the shells starting below it) and `nside` (downgrade while
loading by summing counts) keep the stack small. Incomplete shells
(`run.XXXXX.incomplete.npy`) are skipped. `m_nu` in the sheet is the summed
mass. `Omega_m` includes the massive neutrinos for simulations 193 to 791; for
the 192 verification runs it is CDM + baryons and the returned cosmology has no
massive neutrinos (see `cosmology_from_params`).

**`loaders.Flamingo`**: the [FLAMINGO](https://flamingo.strw.leidenuniv.nl/)
hydrodynamical and dark-matter-only simulations, from the
[public data release](https://dataweb.cosma.dur.ac.uk:8443/flamingo/introduction.html).

| Function | Reads |
|---|---|
| `load_shells(maps_dir, redshifts, lightcone=0, field="TotalMass", z_max=None, nside=None, remote=None, cache_dir=None)` | the HEALPix mass maps of one observer → `ShellStack` |
| `load_shell_redshifts(path.txt)` | a run's `shell_redshifts*.txt` → structured array `lower_z, upper_z` |
| `load_cosmology(params.yml)`, `load_params`, `cosmology_from_params` | the `Cosmology` block of a run's SWIFT parameter file → `Cosmology` |

Each shell file of the release holds every map and is an index into ~17 GB of
data, so single maps are best read from the server with the release's
[`hdfstream`](https://pypi.org/project/hdfstream/) client (`pip install hdfstream`)
instead of downloading files. Only the two small text files of the run are
needed locally (download them from the release's file browser):

```python
import hdfstream
from forwardmodel.loaders import Flamingo

run = "FLAMINGO/L1_m9/L1_m9"                         # box / run, see the release docs
cosmo = Flamingo.load_cosmology("flamingo_L1000N1800_hydro_fiducial.yml")
shells = Flamingo.load_shells(
    f"{run}/healpix_maps/nside_4096", "shell_redshifts_z3.txt",
    z_max=2.0, nside=1024,
    remote=hdfstream.open("cosma", "/"), cache_dir="flamingo_cache/L1_m9")
shells = shells.fill_with_lognormal(cosmo, z_max=4.0)   # the 1 Gpc boxes stop at z = 3
```

A map at the reduced NSIDE 4096 is 1.6 GB and takes minutes to transfer, so use
`cache_dir`: each downgraded map is stored there once and reused. Without
`remote`, `maps_dir` is a local copy of the same directory tree (needs `h5py`).
`TotalMass` contains gas, dark matter, stars, black holes and neutrinos, so a
hydro run already includes baryonic feedback and no baryonification is applied;
`field` selects any other mass map (e.g. `DarkMatterMass`). The parameter files
do not contain the power-spectrum normalisation: `sigma8` and `n_s` come from
the table `Flamingo.COSMOLOGIES` (the seven non-decaying cosmologies of the
suite). The decaying-dark-matter runs are not supported.

**`loaders.des_y3`**

| Function | Reads |
|---|---|
| `load_nz(path.h5)` | groups `source` and `lens`, each with `z_mid` and `bin_<i>` → `{"source": [...], "lens": [...]}` |
| `load_source_catalog(path.h5)` | groups `bin_<i>` with `e1, e2` (optionally `R`, `weight`) → input of `EmpiricalShapeNoise` |
| `load_metacal_catalog(mastercat.h5)` | official `Y3_mastercat_*.h5` → `(catalogs, R_S)` for `DESShapeNoise`; slow, cache the result |
| `load_lens_positions(path.h5)` | groups `bin_<i>` with `ra, dec` → `ra, dec, counts_per_bin` |
| `build_mask(ra, dec, nside)`, `mean_per_pixel(counts, mask)` | footprint mask from positions; `nbar` per bin |
| `draw_systematics(rng)` | one draw of all DES Y3 nuisance parameters as a dict |

`draw_systematics` uses the published DES Y3 priors for the photo-z shifts and
shear calibration. The stretch, magnification, additive-shear and PSF-leakage
priors are illustrative placeholders (see the module constants); replace them
for a real analysis.

**`loaders.kids`**: `load_shape_catalog(path.fits)` reads a KiDS lensfit shear
catalogue (KiDS-1000 gold column names by default, `columns=` for other
releases) and splits it into tomographic bins in `Z_B` (`z_edges`), ready for
`KiDSShapeNoise`.

**`loaders.act_dr6`**: `load_products(act_dir)` reads the ACT DR6 baseline
lensing noise curve, multipole filter and mask; `downgrade_mask(mask, nside)`
brings the NSIDE 4096 mask to the working resolution.

## Baryonification

`forwardmodel.baryons` wraps BaryonForge: halos from a lightcone catalogue are
matched to shells by redshift and each count shell is displaced.

```python
from forwardmodel.baryons import baryonify, HaloCatalog, DEFAULT_BARYON_PARAMS
from forwardmodel.loaders import CosmoGridV1

h = CosmoGridV1.load_halo_catalog("<path>/halos.npz")
halos = HaloCatalog.from_cartesian(h["x"], h["y"], h["z"], h["Mvir"], cosmo)
baryonified = baryonify(shells, halos, cosmo, params=DEFAULT_BARYON_PARAMS)
```

`HaloCatalog(ra, dec, M, z)` can also be built directly (degrees, masses in
M_sun). `baryonify` expects count shells and returns a new count stack.

**Installation.** BaryonForge 1.0.0 on PyPI pins `pyccl==2.8.0` and `numpy<2`,
while this package needs `pyccl>=3.0`, so it cannot be a regular dependency.
Install it without its pins; the `baryons` extra provides its remaining
dependencies:

```bash
pip install -e ".[baryons]"
pip install --no-deps BaryonForge
```

`forwardmodel.baryons` patches the few numpy and pyccl APIs that changed since
those pins, so BaryonForge runs against the current versions. Building the
profile interpolator takes about 20 s per call.

## Examples

| File | Shows | Needs |
|---|---|---|
| `01_basic_weak_lensing.py` | shells → convergence and shear | core |
| `02_full_3x2pt_pipeline.py` | WL + clustering + CMB lensing with TATT, quadratic bias, source clustering and noise | `lognormal` |
| `03_building_blocks.py` | the low-level functions without `ForwardModel` | core |
| `04_with_real_data.py` | the loaders with CosmoGrid, DES Y3 and ACT DR6 products | `loaders`, `lognormal`, the data (edit `DATA`) |
| `05_nbody_vs_lognormal_cls.ipynb` | C_ℓ of N-body vs lognormal shells vs theory | `loaders`, `lognormal`, matplotlib, the data (edit `DATA`) |
| `06_forward_model_vs_theory_cls.ipynb` | C_ℓ of all four probes vs CCL | `lognormal`, matplotlib; DES Y3 n(z) optional (synthetic fallback) |

01–03 use synthetic inputs and run in seconds. The notebooks are stored without
outputs; run them to produce the figures.

## Design

- **High-level**: `ForwardModel.project(...)` accumulates every requested probe
  in one pass; the `*Observation` classes turn the noiseless maps into mocks.
- **Low-level**: every piece is importable on its own —
  `compute_tidal`, `kappa_to_gamma`/`gamma_to_kappa`, the weight builders in
  `forwardmodel.projections`, the bias/IA models, and the noise functions in
  `forwardmodel.noise`.

## Approximations and limitations

- Projections are weighted sums over shells along unperturbed lines of sight
  (Born approximation); there is no ray tracing.
- Shells are full-sky HEALPix maps; masks are applied at the observation stage.
- The tidal field is computed per shell in 2D, which fixes the form of the TATT
  quadratic terms (see [What it computes](#what-it-computes)).
- Quadratic bias and IA are skipped above `z_max_tidal`.
- The harmonic-space path supports linear IA only.
- Lognormal fill-in shells are uncorrelated with the N-body shells they adjoin.

## Tests

```bash
pip install -e ".[all,test]"
pytest
```

The 77 tests use small synthetic inputs, need no external data and take under
a minute. Tests that need `glass`, `h5py`, `ducc0` or BaryonForge are skipped
when the package is missing.

## License

MIT, see [LICENSE](LICENSE).
