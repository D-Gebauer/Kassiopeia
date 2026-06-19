"""Loaders for DES Y3 redshift distributions, catalogues, and footprint.

The HDF5 layouts assumed here match the products the pipeline was built on:

* ``y3_redshift_distributions.h5`` — groups ``source`` and ``lens``, each with
  ``z_mid`` and ``bin_<i>`` n(z) datasets.
* ``y3_source_calibrated.h5`` — groups ``bin_<i>`` with ``e1, e2`` (and
  optionally ``R``, ``weight``).
* ``y3_lens_maglim.h5`` — groups ``bin_<i>`` with ``ra, dec`` (and ``weight``).

Pass ``n_source_bins`` / ``n_lens_bins`` to match your file.
"""

from __future__ import annotations

import numpy as np
import healpy as hp

from ..nz import NumberDensity


def load_nz(path, n_source_bins=4, n_lens_bins=6):
    """Load DES Y3 source and lens redshift distributions.

    Returns
    -------
    dict
        ``{"source": [NumberDensity, ...], "lens": [NumberDensity, ...]}``.
    """
    import h5py

    result = {"source": [], "lens": []}
    with h5py.File(path, "r") as f:
        z_src = f["source/z_mid"][:]
        for i in range(n_source_bins):
            nz = f[f"source/bin_{i}"][:]
            result["source"].append(NumberDensity(z_src[:len(nz)], nz))
        z_lens = f["lens/z_mid"][:]
        for i in range(n_lens_bins):
            nz = f[f"lens/bin_{i}"][:]
            result["lens"].append(NumberDensity(z_lens[:len(nz)], nz))
    return result


def load_source_catalog(path, n_bins=4):
    """Load source ellipticities for empirical shape noise.

    Returns
    -------
    dict
        Maps bin index -> ``(e1, e2, R, w)`` float32 arrays, ready to pass to
        :class:`~forwardmodel.noise.EmpiricalShapeNoise`. ``R`` and ``w`` default
        to ones when absent.
    """
    import h5py

    catalog = {}
    with h5py.File(path, "r") as f:
        for i in range(n_bins):
            grp = f[f"bin_{i}"]
            e1 = grp["e1"][:].astype(np.float32)
            e2 = grp["e2"][:].astype(np.float32)
            R = grp["R"][:].astype(np.float32) if "R" in grp else np.ones_like(e1)
            w = (grp["weight"][:].astype(np.float32) if "weight" in grp
                 else np.ones_like(e1))
            catalog[i] = (e1, e2, R, w)
    return catalog


def load_lens_positions(path, n_bins=6):
    """Load lens RA/Dec for footprint construction.

    Returns
    -------
    ra, dec : ndarray
        Concatenated positions across all bins.
    counts_per_bin : list of int
    """
    import h5py

    ra_parts, dec_parts, counts = [], [], []
    with h5py.File(path, "r") as f:
        for i in range(n_bins):
            ra = f[f"bin_{i}/ra"][:]
            dec = f[f"bin_{i}/dec"][:]
            ra_parts.append(ra)
            dec_parts.append(dec)
            counts.append(len(ra))
    return np.concatenate(ra_parts), np.concatenate(dec_parts), counts


def build_mask(ra, dec, nside):
    """Binary footprint mask: pixels containing >= 1 object are set to 1."""
    pix = hp.ang2pix(nside, np.radians(90.0 - dec), np.radians(ra))
    mask = np.zeros(hp.nside2npix(nside))
    mask[np.unique(pix)] = 1.0
    return mask


def mean_per_pixel(counts_per_bin, mask):
    """Mean objects per unmasked pixel per bin (``nbar``)."""
    n_unmasked = mask.sum()
    return np.asarray(counts_per_bin, dtype=float) / n_unmasked


# ---------------------------------------------------------------------------
# DES Y3 systematic priors (3x2pt analysis)
# ---------------------------------------------------------------------------

# Shear calibration m_i — Gaussian(mean, sigma), per source bin.
SHEAR_CAL_PRIORS = [(-0.0063, 0.0091), (-0.0198, 0.0078),
                    (-0.0241, 0.0076), (-0.0369, 0.0076)]
# Source photo-z shifts — Gaussian(mean, sigma).
SOURCE_PZ_PRIORS = [(0.0, 0.018), (0.0, 0.015), (0.0, 0.011), (0.0, 0.017)]
# Lens photo-z shifts — Gaussian(mean, sigma).
LENS_PZ_PRIORS = [(-0.009, 0.007), (-0.035, 0.011), (-0.005, 0.006),
                  (-0.007, 0.006), (0.002, 0.007), (0.002, 0.008)]
# Flat priors.
BIAS_B1_RANGE = (0.8, 3.0)
BIAS_B2_RANGE = (-2.0, 2.0)
BIAS_BS_RANGE = (-2.0, 2.0)
BIAS_BSRC_RANGE = (0.5, 2.0)
IA_A1_RANGE = (-5.0, 5.0)
IA_A2_RANGE = (-5.0, 5.0)
IA_ETA1_RANGE = (-5.0, 5.0)
IA_ETA2_RANGE = (-5.0, 5.0)
IA_BTA_RANGE = (0.0, 2.0)

# --- Illustrative priors for systematics NOT in the fiducial DES Y3 3x2pt
# model (which used mean photo-z shifts and multiplicative shear bias only).
# These are reasonable placeholders to exercise the extra effects, not the
# published DES Y3 values; override them for a real analysis. ---
# Photo-z width "stretch" factors, Gaussian(mean, sigma); 1.0 = no stretch.
SOURCE_STRETCH_PRIORS = [(1.0, 0.05)] * 4
LENS_STRETCH_PRIORS = [(1.0, 0.05)] * 6
# Lens-sample magnification number-count slopes s = d log10 N(<m)/dm (fixed
# per-bin constants in a real analysis; representative values here).
LENS_MAG_SLOPE = [0.43, 0.30, 1.75, 1.94, 1.56, 2.96]
# Additive shear (c-term) per component, Gaussian(mean, sigma).
ADDITIVE_SHEAR_PRIOR = (0.0, 1e-4)
# PSF leakage fraction alpha, Gaussian(mean, sigma).
PSF_LEAKAGE_PRIOR = (0.0, 0.02)


def draw_systematics(rng, n_source_bins=4, n_lens_bins=6):
    """Sample one realisation of all DES Y3 nuisance parameters.

    Returns a dict ready to feed the bias/IA models,
    :class:`WeakLensingObservation`, and the n(z) transforms:

    * ``b1`` (n_lens,), ``b2``, ``bs``, ``b_src`` (n_source,) — bias.
    * ``a1, a2, eta1, eta2, b_ta`` — TATT intrinsic alignments.
    * ``m`` (n_source,) — multiplicative shear bias (``shear_calibration=m``).
    * ``dz_src`` (n_source,), ``dz_lens`` (n_lens,) — photo-z mean shifts
      (:meth:`NumberDensity.shifted`).
    * ``sz_src`` (n_source,), ``sz_lens`` (n_lens,) — photo-z width stretches
      (:meth:`NumberDensity.stretched`); combine with shifts via
      :meth:`NumberDensity.transformed`.
    * ``mag_s`` (n_lens,) — lens magnification slopes
      (:class:`GalaxyClustering` ``magnification``).
    * ``c1, c2`` — additive shear, ``psf_alpha`` — PSF leakage fraction
      (:class:`WeakLensingObservation` ``additive_shear`` / ``psf_leakage``).

    The shift and shear-calibration priors are the published DES Y3 values; the
    stretch, magnification, additive-shear and PSF-leakage priors are
    illustrative placeholders (see the module constants).
    """
    f = rng.uniform
    g = rng.normal
    return {
        "b1": np.array([f(*BIAS_B1_RANGE) for _ in range(n_lens_bins)]),
        "b2": f(*BIAS_B2_RANGE),
        "bs": f(*BIAS_BS_RANGE),
        "b_src": np.array([f(*BIAS_BSRC_RANGE) for _ in range(n_source_bins)]),
        "a1": f(*IA_A1_RANGE),
        "a2": f(*IA_A2_RANGE),
        "eta1": f(*IA_ETA1_RANGE),
        "eta2": f(*IA_ETA2_RANGE),
        "b_ta": f(*IA_BTA_RANGE),
        "m": np.array([g(*SHEAR_CAL_PRIORS[i]) for i in range(n_source_bins)]),
        "dz_src": np.array([g(*SOURCE_PZ_PRIORS[i])
                            for i in range(n_source_bins)]),
        "dz_lens": np.array([g(*LENS_PZ_PRIORS[i])
                             for i in range(n_lens_bins)]),
        "sz_src": np.array([g(*SOURCE_STRETCH_PRIORS[i])
                            for i in range(n_source_bins)]),
        "sz_lens": np.array([g(*LENS_STRETCH_PRIORS[i])
                             for i in range(n_lens_bins)]),
        "mag_s": np.array(LENS_MAG_SLOPE[:n_lens_bins], dtype=float),
        "c1": g(*ADDITIVE_SHEAR_PRIOR),
        "c2": g(*ADDITIVE_SHEAR_PRIOR),
        "psf_alpha": g(*PSF_LEAKAGE_PRIOR),
    }
