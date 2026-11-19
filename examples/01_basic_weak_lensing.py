"""Minimal example: project density shells into a weak-lensing convergence map.

Run with the package installed:  python examples/01_basic_weak_lensing.py

This uses synthetic random count shells so it runs anywhere. Swap in your own
shells (or a loader, see example 04) to use real data.
"""

import numpy as np
import healpy as hp
import kassiopeia as fm

rng = np.random.default_rng(0)
nside = 64
npix = hp.nside2npix(nside)

# 1. Cosmology.
cosmo = fm.Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

# 2. Density shells (here: synthetic particle counts, z = 0 -> 3).
n_shells = 20
z_edges = np.linspace(0.0, 3.0, n_shells + 1)
counts = rng.poisson(100.0, size=(n_shells, npix)).astype(np.float64)
shells = fm.ShellStack.from_counts(counts, z_edges[:-1], z_edges[1:])
print(shells)

# 3. Source redshift distributions (two tomographic bins).
z = np.linspace(0.01, 3.0, 200)
source_bins = fm.make_bins(z, [
    np.exp(-0.5 * ((z - 0.5) / 0.2) ** 2),
    np.exp(-0.5 * ((z - 1.0) / 0.25) ** 2),
])

# 4. Project.
wl = fm.WeakLensing(nz=source_bins)
maps = fm.ForwardModel(cosmo, shells).project(weak_lensing=wl)

kappa = maps.weak_lensing.kappa
print(f"kappa shape: {kappa.shape}")
for b in range(kappa.shape[0]):
    print(f"  bin {b}: kappa std = {kappa[b].std():.4e}")
