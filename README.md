# Kassiopeia

**K**appa **A**nd **S**hear from **S**hells, **I**ncluding **O**bservational and
**P**hotometric **E**ffects and **I**ntrinsic **A**lignments

Forward-models weak lensing maps from full-sky HEALPix density shells (N-body
or lognormal). Galaxy clustering, CMB lensing and late-time ISW maps are also
available but less tested, see [Validation](#validation).

Modelled effects:

- intrinsic alignments (`NLA`)
- linear and quadratic galaxy bias, magnification bias
- source clustering and reduced shear
- photo-z shift and stretch
- shear calibration, additive shear, PSF leakage
- shape noise from DES or KiDS catalogues or Gaussian, galaxy shot noise,
  CMB-lensing and ISW noise and filtering
- baryonification of count shells with BaryonForge

Parameters of every class and function are documented in the docstrings.

## Install

Python ≥ 3.10.

```bash
pip install kassiopeia           # core: numpy, scipy, healpy, pyccl
pip install "kassiopeia[all]"    # with all optional features
```

From source, for development:

```bash
git clone https://github.com/D-Gebauer/Kassiopeia.git
cd Kassiopeia
pip install -e ".[all,test]"
```

| Extra | Installs | Needed for |
|---|---|---|
| `lognormal` | `glass`, `camb` | lognormal shells |
| `loaders` | `h5py`, `pyyaml`, `astropy` | `kassiopeia.loaders` |
| `rotation` | `ducc0` | faster alm rotations (healpy is the fallback) |
| `baryons` | `tqdm`, `numba`, `joblib` | BaryonForge, see [Baryonification](#baryonification) |
| `test` | `pytest`, `pytest-cov` | the tests |
| `all` | all except `test` | |

The example notebooks also need matplotlib.

## Usage

`counts`, the n(z) arrays, masks and noise inputs are your data.
[`examples/`](https://github.com/D-Gebauer/Kassiopeia/tree/main/examples) has runnable scripts with synthetic inputs.

```python
import numpy as np
import kassiopeia as fm

cosmo = fm.Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
shells = fm.ShellStack.from_counts(counts, z_lower, z_upper)   # (n_shells, npix)

source_bins = fm.make_bins(z, [nz0, nz1, nz2, nz3])
wl  = fm.WeakLensing(nz=source_bins,
                     ia=fm.NLA(a1=0.7),
                     source_clustering_bias=1.0)
gc  = fm.GalaxyClustering(nz=lens_bins,
                          bias=fm.QuadraticBias(b1=[1.4, 1.6, 1.8], b2=0.2, bs=-0.1),
                          magnification=[0.4, 1.0, 1.8])       # number-count slope s
cmb = fm.CMBLensing(z_source=1090.0)
isw = fm.ISW()

maps = fm.ForwardModel(cosmo, shells).project(
    weak_lensing=wl, galaxy_clustering=gc, cmb_lensing=cmb, isw=isw)

rng = np.random.default_rng(0)
wl_obs = fm.WeakLensingObservation(
    mask, nbar=1.5,
    shape_noise=fm.DESShapeNoise(metacal, selection_response=R_S),
    shear_calibration=[-0.006, -0.020, -0.024, -0.037],
    reduced_shear=True,
    additive_shear=(c1, c2),
    psf_leakage=(alpha, e1_psf, e2_psf),
).observe(maps.weak_lensing, rng)

gc_counts = fm.GalaxyClusteringObservation(mask, nbar=nbar_lens).observe(
    maps.galaxy_clustering, rng)
cmb_obs = fm.CMBLensingObservation(
    mask=mask_act, noise=(ell, N_L), filter=(ell, F_L)).observe(maps.cmb_lensing, rng)
isw_obs = fm.ISWObservation(
    mask=mask_cmb, noise=(ell, C_TT_plus_N_L)).observe(maps.isw, rng)
```

Without N-body shells, the model can be built from lognormal shells alone:

```python
model = fm.ForwardModel.from_lognormal(cosmo, nside=512, z_max=3.0, dz=0.1)
```

`project()` skips the tidal terms (quadratic bias, IA) for shells above
`z_max_tidal` (default 3), and the ISW uses only shells below `ISW(z_max=3)`.

## Conventions

- Maps are full-sky HEALPix in RING ordering at the `nside` of the shells.
- `nbar` is the mean number of galaxies per pixel, not per arcmin².
- Shear `(gamma1, gamma2)` is in healpy's (θ, φ) basis, as returned by
  `healpy.alm2map_spin`. Estimators working in (RA, Dec) may need one
  component flipped.
- `Cosmology.comoving_distance` returns Mpc, not Mpc/h. `m_nu` is the summed
  neutrino mass in eV.
- Per-bin parameters (`b1`, `magnification`, `nbar`, `shear_calibration`, ...)
  take a scalar or one value per bin.
- `NumberDensity(z, nz)` is normalised on construction and interpolated
  linearly. `.shifted`, `.stretched` and `.transformed` apply photo-z shifts
  and stretches.
- Masks are binary arrays at the working `nside`. Catalogue positions are
  RA/Dec in degrees.
- Every random step takes a `numpy.random.Generator`.
- `Cosmology` accepts `Omega_m` or `Omega_c`, `sigma8` or `A_s`, `w0`, `wa`,
  `m_nu`; other pyccl arguments go through `extra_ccl_kwargs`, and
  `Cosmology.from_ccl` wraps an existing `pyccl.Cosmology`. The default
  transfer function is CAMB (in the `lognormal` extra); pass
  `transfer_function="eisenstein_hu"` to avoid it.

## Outputs

`project()` returns a `ProjectedMaps`. Probes that were not requested are `None`.

| Attribute | Fields |
|---|---|
| `maps.weak_lensing` | `kappa`, `gamma1`, `gamma2`, `ia_gamma1`, `ia_gamma2`, `source_overdensity`, each `(n_bins, npix)`; `shear(with_ia=True)` returns the total shear |
| `maps.galaxy_clustering` | `delta_g`, `(n_bins, npix)` |
| `maps.cmb_lensing` | `kappa`, `(npix,)` |
| `maps.isw` | `theta` = ΔT/T, `(npix,)` |

| Observation | Returns |
|---|---|
| `WeakLensingObservation` | `gamma1`, `gamma2` (observed), `gamma1_signal`, `gamma2_signal` (noiseless), `n_source`, `kappa_recon` (Kaiser-Squires) |
| `GalaxyClusteringObservation` | galaxy counts `(n_bins, npix)` |
| `CMBLensingObservation`, `ISWObservation` | filtered, noisy, masked map `(npix,)` |

## Shells

| Constructor / method | Use |
|---|---|
| `ShellStack.from_counts(counts, z_lower, z_upper)` | particle-count shells |
| `ShellStack.from_overdensity(delta, z_lower, z_upper)` | overdensity shells |
| `ShellStack.from_lognormal(cosmo, z_edges, nside)` | lognormal shells (GLASS) |
| `.extend_lognormal(cosmo, z_edges)` | append lognormal shells, e.g. above the N-body redshift limit |
| `.fill_with_lognormal(cosmo, z_max=...)` | fill only the redshift ranges without shells |
| `.merge(other)` | combine two stacks |

Count shells are converted to overdensity with their own map mean. Pass
`normalization="box"` with `box_size` (Mpc/h), `n_particles` and `cosmo` to
use the expected count `N·V_shell/V_box` instead.

Lognormal shells are not correlated with the N-body shells they adjoin.

## Shape noise

Passed as `WeakLensingObservation(shape_noise=...)`.

| Class | Input per bin | Noise |
|---|---|---|
| `DESShapeNoise(catalogs, selection_response=R_S)` | `ra, dec, e1, e2, w`, `R11, R22` (or `R`) | catalogue galaxies at their positions, randomly rotated, divided by the metacal response |
| `KiDSShapeNoise(catalogs)` | `ra, dec, e1, e2, w` | same, with the lensfit-weighted mean |
| `EmpiricalShapeNoise(catalog)` | `e1, e2, R, w` | catalogue ellipticities drawn into simulated source counts |
| `GaussianShapeNoise(sigma_e=0.26)` | | Gaussian with variance `σ_e²/N` per pixel |
| `NoShapeNoise()` | | none (default) |

With `DESShapeNoise` and `KiDSShapeNoise` the noise follows the catalogue's
source density and `nbar` is not used for it. Catalogues can be dicts,
structured arrays, `.npz` files, DataFrames or FITS records; `columns=` renames
columns. For KiDS, pass the multiplicative bias as `shear_calibration`.

## Harmonic-space projection and rotations

For linear IA (NLA) without source clustering, the shells can be transformed
once and reused for many n(z), IA amplitudes and rotations:

```python
from kassiopeia import harmonic, rotation

alms = harmonic.shell_overdensity_alms(shells)            # (n_shells, n_alm)
W_kappa, W_ia = harmonic.weak_lensing_weights(
    cosmo, source_bins, shells.z_lower, shells.z_upper, ia=fm.NLA(a1=0.7))
kappa_lm = harmonic.combine_alms(alms, W_kappa)           # (n_bins, n_alm)
ia_lm    = harmonic.combine_alms(alms, W_ia)

angles = rotation.random_euler_angles(seed=(2026, 17))
lmax = 3 * shells.nside - 1
rotated = rotation.rotate_alm(kappa_lm[0] + ia_lm[0], lmax, angles)
gamma1, gamma2 = harmonic.kappa_alm_to_shear(rotated, shells.nside)
```

Angles are healpy's ZYZ Euler angles `(psi, theta, phi)`, drawn uniformly on
SO(3). Alms are stored as `complex64` by default.

## Loaders

`kassiopeia.loaders` reads specific data products into the objects above.
The data are not included; all paths are passed explicitly.

| Module | Reads |
|---|---|
| `CosmoGridV1` | CosmoGridV1 `.npz` shells, `params.yml`, halo catalogue |
| `CosmoGridV1_1` | CosmoGrid v1.1 `baryonified_shells_v11.h5` (DMO or baryonified) |
| `GowerStreet` | Gower Street simulation directories and parameter sheet |
| `Flamingo` | FLAMINGO lightcone mass maps, local or from the public server |
| `des_y3` | DES Y3 n(z), source and lens catalogues, mask, nuisance-parameter draws |
| `kids` | KiDS lensfit shear catalogues |
| `act_dr6` | ACT DR6 lensing noise curve, filter and mask |

### CosmoGrid

```python
from kassiopeia.loaders import CosmoGridV1, CosmoGridV1_1

cosmo = CosmoGridV1.load_cosmology("<dir>/params.yml")
shells = CosmoGridV1.load_shells("<dir>/compressed_shells.npz")

shells, p = CosmoGridV1_1.load_shells(
    "<cosmogrid>/<cosmology>/run_0000/baryonified_shells_v11.h5",
    baryonified=True,
    metainfo="<cosmogrid>/CosmoGridV11_bary_Fluri+2022_metainfo.h5",
    return_params=True)
cosmo = CosmoGridV1_1.cosmology_from_params(p)
```

The v1.1 files do not contain the shell redshifts, so `metainfo=` (or
`shell_info=`) is required.

### Gower Street

[Data release](http://www.star.ucl.ac.uk/GowerStreetSims/). Unpack a
`simNNNNN.tar.gz` and export sheet 1 of the release workbook as CSV.

```python
from kassiopeia.loaders import GowerStreet

cosmo = GowerStreet.load_cosmology("<gower>/sheet1.csv", 400)
shells = GowerStreet.load_shells("<gower>/sim00400", z_max=3.0, nside=1024)
```

All shells of one simulation take about 10 GB at the native NSIDE 2048. `z_max`
limits the redshift range and `nside` downgrades while loading. For
simulations 1 to 192 the sheet's `Omega_m` is CDM + baryons and the returned
cosmology has no massive neutrinos; from 193 on, `Omega_m` includes them.

### FLAMINGO

[Data release](https://dataweb.cosma.dur.ac.uk:8443/flamingo/introduction.html).
Maps are read from the server with `hdfstream` (`pip install hdfstream`). Only
the run's parameter file and `shell_redshifts*.txt` are needed locally.

```python
import hdfstream
from kassiopeia.loaders import Flamingo

run = "FLAMINGO/L1_m9/L1_m9"
cosmo = Flamingo.load_cosmology("flamingo_L1000N1800_hydro_fiducial.yml")
shells = Flamingo.load_shells(
    f"{run}/healpix_maps/nside_4096", "shell_redshifts_z3.txt",
    z_max=2.0, nside=1024,
    remote=hdfstream.open("cosma", "/"), cache_dir="flamingo_cache/L1_m9")
shells = shells.fill_with_lognormal(cosmo, z_max=4.0)   # the 1 Gpc boxes end at z = 3
```

One map is 1.6 GB and takes minutes to transfer; `cache_dir` stores each
downgraded map for reuse. Without `remote`, `maps_dir` is a local copy of the
same directory tree. `field` selects another mass map than `TotalMass`, e.g.
`DarkMatterMass`. `sigma8` and `n_s` are not in the parameter files and come
from `Flamingo.COSMOLOGIES`. The decaying-dark-matter runs are not supported.

### Surveys

- `des_y3`: `load_nz`, `load_source_catalog`, `load_metacal_catalog` (input of
  `DESShapeNoise`; slow, cache the result), `load_lens_positions`,
  `build_mask`, `mean_per_pixel`, `draw_systematics`. In `draw_systematics`
  only the photo-z shift and shear calibration priors are the published DES Y3
  ones; the others are placeholders.
- `kids`: `load_shape_catalog(path.fits)` splits a lensfit catalogue into
  tomographic bins for `KiDSShapeNoise` (KiDS-1000 column names by default).
- `act_dr6`: `load_products(act_dir)` and `downgrade_mask(mask, nside)`.

## Baryonification

```python
from kassiopeia.baryons import baryonify, HaloCatalog, DEFAULT_BARYON_PARAMS
from kassiopeia.loaders import CosmoGridV1

h = CosmoGridV1.load_halo_catalog("<path>/halos.npz")
halos = HaloCatalog.from_cartesian(h["x"], h["y"], h["z"], h["Mvir"], cosmo)
baryonified = baryonify(shells, halos, cosmo, params=DEFAULT_BARYON_PARAMS)
```

`baryonify` takes count shells and returns a new count stack.
`HaloCatalog(ra, dec, M, z)` takes degrees and masses in M_sun.

The BaryonForge release on PyPI pins `pyccl==2.8.0` and `numpy<2`, which
conflict with this package, so install it without its dependencies:

```bash
pip install "kassiopeia[baryons]"
pip install --no-deps BaryonForge
```

## Examples

| File | Shows | Needs |
|---|---|---|
| `01_basic_weak_lensing.py` | shells to convergence and shear | core |
| `02_full_3x2pt_pipeline.py` | lensing, clustering and CMB lensing with systematics and noise | `lognormal` |
| `03_building_blocks.py` | the low-level functions without `ForwardModel` | core |
| `04_with_real_data.py` | the loaders with CosmoGrid, DES Y3 and ACT DR6 data | `loaders`, `lognormal`, the data |
| `05_nbody_vs_lognormal_cls.ipynb` | C_ℓ of N-body and lognormal shells against theory | `loaders`, `lognormal`, the data |
| `06_forward_model_vs_theory_cls.ipynb` | C_ℓ of all four probes against CCL | `lognormal` |

Set `DATA` in 04 and 05 to your data directory.

## Validation

| Probe | Checked | Not checked |
|---|---|---|
| Weak lensing | convergence C_ℓ against CCL: within 1 % on lognormal shells (06), within a few per cent for ℓ < 150 on N-body shells (05); NLA E-modes at 0.76 to 0.94 of CCL (05) | |
| Galaxy clustering | C_ℓ with linear bias against CCL: within 1 % on lognormal shells (06) | quadratic bias, magnification, N-body shells, cross-spectra with lensing |
| CMB lensing | C_ℓ against CCL: within 1 % on lognormal shells (06) | N-body shells extended with lognormal shells, cross-spectra, noise and filtering against data |
| ISW | C_ℓ against CCL: 0.99 ± 0.01 for ℓ = 4 to 40 on lognormal shells (06) | N-body shells, cross-spectra with other probes, ℓ > 40 |

The numbers are mean ratios over ℓ = 60 to 400 (ISW: 4 to 40) of the
auto-spectrum of one tomographic bin, from the example notebooks 05 and 06.
Treat the ISW as experimental.

## Limitations

- Born approximation, no ray tracing.
- The lensing kernel is evaluated per shell: use shells with Δz ≲ 0.05. With
  Δz = 0.15 the convergence power of low-redshift sources is tens of per cent
  too high.
- The harmonic-space path supports linear IA only.
- Late-time ISW only.

## Tests

```bash
pip install -e ".[all,test]"
pytest
```

Tests need no external data. Those requiring an optional package are skipped
when it is missing.

## License

MIT, see [LICENSE](https://github.com/D-Gebauer/Kassiopeia/blob/main/LICENSE).
