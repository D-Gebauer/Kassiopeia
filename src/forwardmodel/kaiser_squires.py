"""Kaiser-Squires inversion: convergence <-> shear via spin-2 harmonics.

The exact harmonic-space relation between the spin-0 convergence and the
spin-2 shear E-mode is

    gamma_E_lm = sqrt((l+2)(l-1) / (l(l+1))) * kappa_lm,   l >= 2,

so the convergence is recovered from the shear E-mode via the inverse factor

    kappa_lm = sqrt(l(l+1) / ((l+2)(l-1))) * gamma_E_lm,   l >= 2.

The ``l = 0`` and ``l = 1`` modes are unconstrained by spin-2 data and are set
to zero. Applying the factor exactly (rather than treating it as 1, which is
only correct in the high-l limit) matters at the few-percent level for
``l ~ a few`` and is essential for an unbiased convergence reconstruction at
low multipoles.
"""

from __future__ import annotations

import numpy as np
import healpy as hp


def _kappa_from_gamma_factor(lmax):
    """``f_l = sqrt(l(l+1) / ((l+2)(l-1)))`` for ``l >= 2``; 0 below."""
    ell = np.arange(lmax + 1, dtype=np.float64)
    f = np.zeros(lmax + 1, dtype=np.float64)
    mask = ell >= 2
    L = ell[mask]
    f[mask] = np.sqrt(L * (L + 1.0) / ((L + 2.0) * (L - 1.0)))
    return f


def kappa_to_gamma(kappa_map, lmax=None):
    """Convergence -> shear (forward Kaiser-Squires).

    Applies the exact ``sqrt((l+2)(l-1) / (l(l+1)))`` factor between
    ``kappa_lm`` and the shear E-mode harmonics, with ``l < 2`` zeroed.

    Parameters
    ----------
    kappa_map : ndarray, shape (npix,)
    lmax : int, optional
        Defaults to ``3 * nside - 1``.

    Returns
    -------
    gamma1, gamma2 : ndarray
    """
    nside = hp.npix2nside(len(kappa_map))
    if lmax is None:
        lmax = 3 * nside - 1
    kappa_lm = hp.map2alm(kappa_map, lmax=lmax)
    f = _kappa_from_gamma_factor(lmax)
    inv = np.zeros_like(f)
    np.divide(1.0, f, out=inv, where=f > 0)
    elm = hp.almxfl(kappa_lm, inv)
    blm = np.zeros_like(elm)
    gamma1, gamma2 = hp.alm2map_spin([elm, blm], nside, 2, lmax)
    return gamma1, gamma2


def gamma_to_kappa(gamma1, gamma2, lmax=None):
    """Shear -> convergence E/B decomposition (inverse Kaiser-Squires).

    Applies the exact ``sqrt(l(l+1) / ((l+2)(l-1)))`` factor between the
    shear E/B harmonics and the convergence harmonics, with ``l < 2`` zeroed.

    Parameters
    ----------
    gamma1, gamma2 : ndarray, shape (npix,)
    lmax : int, optional
        Defaults to ``3 * nside - 1``.

    Returns
    -------
    kappa_E, kappa_B : ndarray
    """
    nside = hp.npix2nside(len(gamma1))
    if lmax is None:
        lmax = 3 * nside - 1
    elm, blm = hp.map2alm_spin([gamma1, gamma2], 2, lmax=lmax)
    f = _kappa_from_gamma_factor(lmax)
    elm = hp.almxfl(elm, f)
    blm = hp.almxfl(blm, f)
    kappa_E = hp.alm2map(elm, nside, lmax=lmax)
    kappa_B = hp.alm2map(blm, nside, lmax=lmax)
    return kappa_E, kappa_B
