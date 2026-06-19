"""Full 3x2pt-style pipeline: weak lensing + galaxy clustering + CMB lensing,
with TATT intrinsic alignments, quadratic galaxy bias, source clustering, and
observational realism (empirical shape noise, Poisson shot noise, CMB noise).

Run:  python examples/02_full_3x2pt_pipeline.py
"""

import numpy as np
import healpy as hp
import forwardmodel as fm

rng = np.random.default_rng(42)
nside = 64
npix = hp.nside2npix(nside)

cosmo = fm.Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

# --- Shells (synthetic), extended to high z with lognormal fields for CMB ---
n_shells = 20
z_edges = np.linspace(0.0, 3.0, n_shells + 1)
counts = rng.poisson(100.0, size=(n_shells, npix)).astype(np.float64)
shells = fm.ShellStack.from_counts(counts, z_edges[:-1], z_edges[1:])
shells = shells.extend_lognormal(cosmo, z_edges=np.linspace(3.0, 8.0, 11), rng=rng)
print(shells)

# --- Redshift distributions ---
z = np.linspace(0.01, 3.0, 200)
source_bins = fm.make_bins(z, [np.exp(-0.5 * ((z - m) / 0.2) ** 2)
                               for m in (0.4, 0.7, 1.0, 1.3)])
lens_bins = fm.make_bins(z, [np.exp(-0.5 * ((z - m) / 0.12) ** 2)
                             for m in (0.3, 0.5, 0.7)])

# --- Probes ---
wl = fm.WeakLensing(
    nz=source_bins,
    ia=fm.TATT(a1=0.7, a2=-0.5, eta1=0.0, eta2=0.0, b_ta=0.5),
    source_clustering_bias=1.0,
)
gc = fm.GalaxyClustering(
    nz=lens_bins,
    bias=fm.QuadraticBias(b1=[1.4, 1.6, 1.8], b2=0.2, bs=-0.1),
)
cmb = fm.CMBLensing(z_source=1090.0)

maps = fm.ForwardModel(cosmo, shells).project(
    weak_lensing=wl, galaxy_clustering=gc, cmb_lensing=cmb)

# --- Observational realism ---
mask = np.ones(npix)
mask[: npix // 5] = 0.0  # a partial footprint

# Empirical shape noise needs a per-bin ellipticity catalogue; here it's fake.
catalog = {b: (rng.normal(0, 0.26, 20000).astype("f4"),
               rng.normal(0, 0.26, 20000).astype("f4"),
               np.ones(20000, "f4"), np.ones(20000, "f4"))
           for b in range(4)}

wl_obs = fm.WeakLensingObservation(
    mask, nbar=1.5, shape_noise=fm.EmpiricalShapeNoise(catalog),
    shear_calibration=[-0.006, -0.020, -0.024, -0.037],
).observe(maps.weak_lensing, rng)

gc_counts = fm.GalaxyClusteringObservation(mask, nbar=8.0).observe(
    maps.galaxy_clustering, rng)

ell = np.arange(3 * nside)
cmb_obs = fm.CMBLensingObservation(
    mask=mask, noise=(ell, 2e-8 * np.ones_like(ell, float)),
    filter=(ell, ((ell >= 2) & (ell <= 2 * nside)).astype(float)),
).observe(maps.cmb_lensing, rng)

print("\nObserved data products:")
print(f"  shear gamma1:   {wl_obs.gamma1.shape}  std={wl_obs.gamma1.std():.3e}")
print(f"  kappa_recon:    {wl_obs.kappa_recon.shape}")
print(f"  galaxy counts:  {gc_counts.shape}  total={int(gc_counts.sum())}")
print(f"  CMB kappa:      {cmb_obs.shape}  std={cmb_obs.std():.3e}")
