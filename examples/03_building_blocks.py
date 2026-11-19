"""Low-level building blocks: use the physics functions directly, without the
ForwardModel orchestrator. Useful for custom pipelines or debugging.

Run:  python examples/03_building_blocks.py
"""

import numpy as np
import healpy as hp
import kassiopeia as fm
from kassiopeia import projections as proj

rng = np.random.default_rng(7)
nside = 64
cosmo = fm.Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

# A single overdensity shell.
cl = np.zeros(40)
cl[2:] = 1.0 / np.arange(2, 40) ** 2
delta = hp.synfast(cl, nside, new=True)

# Tidal tensor and its scalar invariant.
s1, s2, s_sq = fm.compute_tidal(delta)
print(f"tidal s^2 mean = {s_sq.mean():.4e}")

# Quadratic bias applied by hand.
d, d2_ms, s2_ms = fm.quadratic_bias_terms(delta, s_sq)
delta_g = fm.apply_quadratic_bias(d, d2_ms, s2_ms, b1=1.5, b2=0.3, bs=-0.2)
print(f"delta_g std = {delta_g.std():.4e}")

# Convergence -> shear and back.
g1, g2 = fm.kappa_to_gamma(delta)
kE, kB = fm.gamma_to_kappa(g1, g2)
print(f"E/B power ratio = {kB.std() / kE.std():.3e}")

# Build projection weights directly for a custom set of shells.
z = np.linspace(0.01, 2.0, 100)
nz_bins = fm.make_bins(z, [np.exp(-0.5 * ((z - 0.8) / 0.2) ** 2)])
z_lo = np.array([0.0, 0.5, 1.0])
z_hi = np.array([0.5, 1.0, 1.5])
w = proj.lensing_weights(cosmo, nz_bins, z_lo, z_hi)
print(f"lensing weights per shell: {w.ravel()}")

# TATT amplitudes at a given redshift.
A1, A2 = fm.tatt_amplitudes(cosmo, z=0.5, a1=1.0, a2=0.5, eta1=0.0, eta2=0.0)
print(f"A1={A1:.4e}, A2={A2:.4e}")
