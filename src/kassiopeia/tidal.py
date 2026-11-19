"""Tidal tensor from a density shell via spherical-harmonic Poisson inversion.

The tidal tensor ``s_ij`` is the traceless Hessian of the gravitational
potential ``phi``, with ``nabla^2 phi = delta``. On the sphere it is a spin-2
field whose E-mode harmonics relate to the density harmonics by a purely
ell-dependent factor::

    s_lm = delta_lm * sqrt((l-1)(l+2) / (l(l+1)))

The sign convention matches :func:`kassiopeia.kaiser_squires.kappa_to_gamma`
so that NLA intrinsic alignments correctly *oppose* gravitational shear.
"""

from __future__ import annotations

import numpy as np
import healpy as hp

# Cache of alm-index -> ell arrays, keyed by lmax.
_ELL_INDEX_CACHE: dict[int, np.ndarray] = {}


def _ell_index(lmax):
    """Array mapping each healpy alm index to its ell value (cached)."""
    cached = _ELL_INDEX_CACHE.get(lmax)
    if cached is not None:
        return cached
    n_alm = hp.Alm.getsize(lmax)
    ell = np.empty(n_alm, dtype=np.float64)
    for m in range(lmax + 1):
        start = hp.Alm.getidx(lmax, m, m)
        n = lmax - m + 1
        ell[start:start + n] = np.arange(m, lmax + 1)
    _ELL_INDEX_CACHE[lmax] = ell
    return ell


def compute_tidal(delta_map, lmax=None):
    """Compute the spin-2 tidal field and its scalar invariant from ``delta``.

    Parameters
    ----------
    delta_map : ndarray, shape (npix,)
        Overdensity field.
    lmax : int, optional
        Maximum multipole. Defaults to ``3 * nside - 1``.

    Returns
    -------
    s1, s2 : ndarray
        Spin-2 tidal field components.
    s_squared : ndarray
        Scalar tidal invariant ``s1^2 + s2^2``.
    """
    nside = hp.npix2nside(len(delta_map))
    if lmax is None:
        lmax = 3 * nside - 1

    delta_lm = hp.map2alm(delta_map, lmax=lmax)
    ell = _ell_index(lmax)

    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.sqrt((ell - 1.0) * (ell + 2.0) / (ell * (ell + 1.0)))
    factor[ell < 2] = 0.0  # kill monopole and dipole

    elm = delta_lm * factor
    blm = np.zeros_like(elm)
    s1, s2 = hp.alm2map_spin([elm, blm], nside, 2, lmax)
    return s1, s2, s1**2 + s2**2
