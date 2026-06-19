"""Using the dataset loaders with real CosmoGrid / DES Y3 / ACT DR6 products.

This mirrors the original pipeline the package was generalised from. Edit the
paths below to point at your data, then run. Requires the loaders extra:

    pip install forwardmodel[loaders]
"""

import numpy as np
import forwardmodel as fm
from forwardmodel.loaders import cosmogrid, des_y3, act_dr6

DATA = "/path/to/data"  # <-- edit me

rng = np.random.default_rng(42)

# --- Cosmology and shells from CosmoGrid ---
cosmo = cosmogrid.load_cosmology(f"{DATA}/cosmogrid/params.yml")
shells = cosmogrid.load_shells(f"{DATA}/cosmogrid/baryonified_shells.npz")
nside = shells.nside

# Extend past the N-body cap to cover the CMB-lensing kernel.
shells = shells.extend_lognormal(
    cosmo, z_edges=np.linspace(shells.z_max, 10.0, 24), rng=rng)

# --- DES Y3 redshift distributions, mask, catalogue ---
nz = des_y3.load_nz(f"{DATA}/n_z/y3_redshift_distributions.h5")
ra, dec, lens_counts = des_y3.load_lens_positions(f"{DATA}/catalogues/y3_lens_maglim.h5")
mask = des_y3.build_mask(ra, dec, nside)
nbar_lens = des_y3.mean_per_pixel(lens_counts, mask)

source_catalog = des_y3.load_source_catalog(f"{DATA}/catalogues/y3_source_calibrated.h5")
src_counts = [len(source_catalog[i][0]) for i in range(4)]
nbar_source = des_y3.mean_per_pixel(src_counts, mask)

# --- ACT DR6 CMB lensing products ---
act = act_dr6.load_products(f"{DATA}/act_dr6/maps/baseline")
mask_act = act_dr6.downgrade_mask(act["mask"], nside)

# --- Draw one realisation of DES Y3 nuisance parameters ---
sys = des_y3.draw_systematics(rng)
source_bins = [nd.shifted(sys["dz_src"][i]) for i, nd in enumerate(nz["source"])]
lens_bins = [nd.shifted(sys["dz_lens"][i]) for i, nd in enumerate(nz["lens"])]

# --- Probes ---
wl = fm.WeakLensing(
    nz=source_bins,
    ia=fm.TATT(a1=sys["a1"], a2=sys["a2"], eta1=sys["eta1"],
               eta2=sys["eta2"], b_ta=sys["b_ta"]),
    source_clustering_bias=sys["b_src"],
)
gc = fm.GalaxyClustering(
    nz=lens_bins,
    bias=fm.QuadraticBias(b1=sys["b1"], b2=sys["b2"], bs=sys["bs"]),
)
cmb = fm.CMBLensing(z_source=1090.0)

maps = fm.ForwardModel(cosmo, shells).project(
    weak_lensing=wl, galaxy_clustering=gc, cmb_lensing=cmb, verbose=True)

# --- Observe ---
wl_obs = fm.WeakLensingObservation(
    mask, nbar=nbar_source, shape_noise=fm.EmpiricalShapeNoise(source_catalog),
    shear_calibration=sys["m"],
).observe(maps.weak_lensing, rng)
gc_counts = fm.GalaxyClusteringObservation(mask, nbar=nbar_lens).observe(
    maps.galaxy_clustering, rng)
cmb_obs = fm.CMBLensingObservation(
    mask=mask_act, noise=act["noise"], filter=act["filter"],
).observe(maps.cmb_lensing, rng)

print("shear:", wl_obs.gamma1.shape)
print("counts:", gc_counts.shape)
print("cmb:", cmb_obs.shape)
