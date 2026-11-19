"""Loaders for ACT DR6 CMB-lensing products.

The baseline ACT DR6 lensing release provides a reconstruction noise curve, a
multipole filter, and a high-resolution mask. These helpers read those files
and shape them for :class:`~kassiopeia.observation.CMBLensingObservation`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import healpy as hp

_NOISE = "N_L_kk_act_dr6_lensing_v1_baseline.txt"
_FILTER = "kappa_filter_act_dr6_lensing_v1_baseline.txt"
_MASK = "mask_act_dr6_lensing_v1_healpix_nside_4096_baseline.fits"


def load_products(act_dir, noise_file=_NOISE, filter_file=_FILTER,
                  mask_file=_MASK):
    """Load the ACT DR6 baseline noise curve, filter, and mask.

    Returns
    -------
    dict
        ``noise`` -> ``(ell, N_L)``, ``filter`` -> ``(ell, F_L)``,
        ``mask`` -> HEALPix mask (native NSIDE=4096). Pass ``noise``/``filter``
        straight to :class:`CMBLensingObservation`; downgrade the mask with
        :func:`downgrade_mask` first.
    """
    act_dir = Path(act_dir)
    noise = np.loadtxt(act_dir / noise_file)
    filt = np.loadtxt(act_dir / filter_file)
    mask = hp.read_map(act_dir / mask_file, dtype=np.float64)
    return {
        "noise": (noise[:, 0].astype(int), noise[:, 1]),
        "filter": (filt[:, 0].astype(int), filt[:, 1]),
        "mask": mask,
    }


def downgrade_mask(mask, nside_out, threshold=0.99):
    """Downgrade a high-resolution mask, keeping pixels above ``threshold``."""
    low = hp.ud_grade(mask, nside_out)
    return (low >= threshold).astype(np.float64)
