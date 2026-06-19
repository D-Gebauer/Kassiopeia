# Validation against UFalcon

`compare_with_ufalcon.py` cross-checks `forwardmodel` against
[UFalcon](https://cosmo-gitlab.phys.ethz.ch/cosmo_public/ufalcon-published) on
the **same** prebaryonified CosmoGrid density shells. forwardmodel's projection
kernels are ports of UFalcon's continuous-lensing / Dirac-lensing /
continuous-clustering integrals, so this is the natural reference.

## What is compared

Only the effects implemented in **both** pipelines, with everything else off:

| Probe | UFalcon | forwardmodel |
|-------|---------|--------------|
| Weak-lensing convergence κ (per source bin) | `construct_kappa_map(IA=None)` | `WeakLensing(nz=[bin])` |
| CMB-lensing convergence κ | `construct_kappa_cmb_map()` | `CMBLensing(z_source=1090)` |
| Linear galaxy clustering δ_g (per lens bin) | `construct_delta_map(b1)` | `GalaxyClustering(LinearBias(b1))` |

forwardmodel-only physics — TATT IA, quadratic bias, source clustering, shape
noise, baryonification — is disabled.

## The one difference under test: normalisation

UFalcon scales count shells by the box-volume `kappa_prefactor` /
`delta_prefactor` (needs box size + particle count). forwardmodel's **default**
converts each shell to overdensity via its own map mean, `δ = counts/⟨counts⟩ −
1`, which needs neither. The two are algebraically identical when a shell's map
mean equals the theoretical particle count `N·V_shell/V_box`. The comparison
quantifies how well that holds for real lightcone shells (the `slope` /
`std_ratio` columns).

Pass `--normalization box` to switch forwardmodel onto the same
`N·V_shell/V_box` amplitude (via `ShellStack(..., normalization="box",
box_size=900, n_particles=832**3, cosmo=...)`). In that mode every probe matches
UFalcon to **machine precision** (corr/slope/std_ratio = 1.00000, residuals
~1e-6) — proof that the δ-form kernels are an exact reimplementation of
UFalcon's count-form and that normalisation was the only remaining difference.
The default `map_mean` mode is kept because it is self-calibrating, needs no box
metadata, and works for lognormal / real-survey shells; box mode is the choice
when you want the simulation's absolute amplitude.

Both pipelines run on the **same background cosmology**, built from the
CosmoGrid params with massless neutrinos so the pyccl (forwardmodel) and astropy
(UFalcon) distances agree to <1e-6 — isolating the pipeline comparison from
cosmology-library and neutrino-convention differences.

## Running

```bash
# Needs both packages. forwardmodel is in the `fm` conda env; UFalcon is pure
# python and is added to sys.path from the `cosmo` env by default.
/opt/miniconda3/envs/fm/bin/python validation/compare_with_ufalcon.py \
    --data /home/david/Documents/research/NERSC_proposal/data

# Quick check at lower resolution:
... --nside 128 --no-cl

# Exact UFalcon amplitude (slopes -> 1.000 to machine precision):
... --normalization box

# Dump maps + residuals:
... --save-dir validation/out
```

## Reading the output

- **corr** — pixel-wise Pearson correlation; spatial agreement. Expect ≈ 1.
- **slope (fm/uf)**, **std_ratio** — amplitude agreement; this is the
  map-mean vs box-volume normalisation test.
- **rms_resid/std** — residual RMS relative to the UFalcon map std.
- **Cl_ratio** — median C_ℓ ratio over a mid-ℓ band; scale dependence of any
  residual.

Clustering agrees to ~0.1–0.5% (its weight is localised to one shell, so the
per-shell normalisation factors out to a near-constant scalar). Weak lensing and
CMB lensing integrate many shells with the lensing kernel, so the slowly
varying per-shell `⟨counts⟩` vs `N·V_shell/V_box` mismatch shows up as a few-%
amplitude offset, largest for the lowest-z source bin and for CMB lensing.
