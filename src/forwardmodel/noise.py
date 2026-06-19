"""Noise models: shape noise, galaxy shot noise, and CMB-lensing noise.

Shape noise comes in three interchangeable flavours, all sharing the
:class:`ShapeNoise` interface so they can be swapped freely on a
:class:`forwardmodel.observation.WeakLensingObservation`:

* :class:`EmpiricalShapeNoise` (default) — sample real catalogue ellipticities,
  randomly rotate them, and average per pixel. Noise variance then scales as
  ``1 / N`` and tracks the source-clustering signal.
* :class:`GaussianShapeNoise` — draw per-pixel shear noise from
  ``N(0, sigma_e^2 / N)`` analytically; no catalogue needed.
* :class:`NoShapeNoise` — signal only.

The functions :func:`poisson_counts` and :func:`apply_cmb_noise` cover galaxy
shot noise and the ACT-style CMB-lensing filter+noise+mask chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import healpy as hp

NoiseTriplet = Tuple[np.ndarray, np.ndarray, np.ndarray]


def _coeff(value, bin_index):
    arr = np.asarray(value)
    return float(arr) if arr.ndim == 0 else float(arr[bin_index])


def _source_counts(nbar, delta_source, mask, rng):
    """Poisson-draw source counts per pixel: ``lambda = nbar (1+delta) mask``."""
    lam = np.maximum(nbar * (1.0 + delta_source) * mask, 0.0)
    return rng.poisson(lam)


# ---------------------------------------------------------------------------
# Shape-noise models
# ---------------------------------------------------------------------------

class ShapeNoise(ABC):
    """Interface for per-bin weak-lensing shape noise."""

    @abstractmethod
    def generate(self, bin_index, nbar, delta_source, mask, rng) -> NoiseTriplet:
        """Return ``(n_src, gamma1_noise, gamma2_noise)`` for one source bin."""
        ...


class NoShapeNoise(ShapeNoise):
    """Signal-only mode: zero shape noise (source counts still drawn)."""

    def generate(self, bin_index, nbar, delta_source, mask, rng):
        n_src = _source_counts(nbar, delta_source, mask, rng)
        z = np.zeros(len(mask))
        return n_src.astype(np.float64), z, z.copy()


class GaussianShapeNoise(ShapeNoise):
    """Analytic per-pixel Gaussian shape noise.

    Parameters
    ----------
    sigma_e : float or array_like
        Per-component intrinsic ellipticity dispersion, scalar or per bin.
        Pixel shear noise is drawn from ``N(0, sigma_e^2 / N)`` for each
        component, where ``N`` is that pixel's source count.
    """

    def __init__(self, sigma_e=0.26):
        self.sigma_e = sigma_e

    def generate(self, bin_index, nbar, delta_source, mask, rng):
        npix = len(mask)
        n_src = _source_counts(nbar, delta_source, mask, rng)
        sigma = _coeff(self.sigma_e, bin_index)
        g1 = np.zeros(npix)
        g2 = np.zeros(npix)
        nonzero = n_src > 0
        std = sigma / np.sqrt(n_src[nonzero])
        g1[nonzero] = rng.normal(0.0, std)
        g2[nonzero] = rng.normal(0.0, std)
        return n_src.astype(np.float64), g1, g2


class EmpiricalShapeNoise(ShapeNoise):
    """Shape noise by sampling and rotating real catalogue ellipticities.

    Parameters
    ----------
    catalog : dict
        Maps bin index -> ``(e1, e2, R, w)`` arrays. ``R`` is the per-object
        shear response and ``w`` the inverse-variance weight; pass arrays of
        ones if unavailable.
    """

    def __init__(self, catalog):
        self.catalog = catalog

    def generate(self, bin_index, nbar, delta_source, mask, rng):
        npix = len(mask)
        e1_cat, e2_cat, R_cat, w_cat = self.catalog[bin_index]
        n_cat = len(e1_cat)
        n_src = _source_counts(nbar, delta_source, mask, rng)
        total = int(n_src.sum())
        if total == 0:
            z = np.zeros(npix)
            return n_src.astype(np.float64), z, z.copy()

        pixel_ids = np.repeat(np.arange(npix, dtype=np.int64), n_src)
        idx = rng.integers(0, n_cat, size=total)
        e1 = e1_cat[idx].astype(np.float64)
        e2 = e2_cat[idx].astype(np.float64)
        w = w_cat[idx].astype(np.float64)
        R = R_cat[idx].astype(np.float64)

        # Random rotation destroys any residual cosmological signal.
        phi = rng.uniform(0.0, 2.0 * np.pi, size=total)
        c, s = np.cos(2.0 * phi), np.sin(2.0 * phi)
        e1_rot = e1 * c - e2 * s
        e2_rot = e1 * s + e2 * c

        g1_sum = np.bincount(pixel_ids, weights=w * e1_rot, minlength=npix)
        g2_sum = np.bincount(pixel_ids, weights=w * e2_rot, minlength=npix)
        w_sum = np.bincount(pixel_ids, weights=w, minlength=npix)
        wR_sum = np.bincount(pixel_ids, weights=w * R, minlength=npix)

        w_safe = np.maximum(w_sum, 1e-30)
        R_mean = np.maximum(wR_sum / w_safe, 1e-30)
        g1 = g1_sum / w_safe / R_mean
        g2 = g2_sum / w_safe / R_mean
        g1[w_sum == 0] = 0.0
        g2[w_sum == 0] = 0.0
        return n_src.astype(np.float64), g1, g2


# ---------------------------------------------------------------------------
# Galaxy shot noise
# ---------------------------------------------------------------------------

def poisson_counts(nbar, delta_g, mask, rng):
    """Poisson galaxy counts: ``N ~ Poisson(nbar (1 + delta_g) mask)``.

    Parameters
    ----------
    nbar : float
        Mean galaxies per pixel for this bin.
    delta_g : ndarray
        Projected galaxy overdensity.
    mask : ndarray
        Binary footprint mask.
    rng : np.random.Generator

    Returns
    -------
    ndarray
        Integer galaxy counts per pixel (as float64).
    """
    lam = np.maximum(nbar * (1.0 + delta_g) * mask, 0.0)
    return rng.poisson(lam).astype(np.float64)


# ---------------------------------------------------------------------------
# CMB-lensing noise (ACT-style)
# ---------------------------------------------------------------------------

def apply_cmb_noise(kappa_cmb, noise_ell=None, noise_nl=None,
                    filter_ell=None, filter_fl=None, mask=None, rng=None,
                    lmax=None):
    """Filter a CMB convergence map, add reconstruction noise, and mask it.

    Parameters
    ----------
    kappa_cmb : ndarray
        Simulated CMB convergence.
    noise_ell, noise_nl : array_like, optional
        Reconstruction noise power spectrum ``N_L^{kappa kappa}``. If omitted,
        no noise is added.
    filter_ell, filter_fl : array_like, optional
        Multipole filter (e.g. zeroing modes outside an analysis range). If
        omitted, no filtering is applied.
    mask : ndarray, optional
        Binary mask at the working resolution. If omitted, no mask is applied.
    rng : np.random.Generator, optional
        Required if a noise spectrum is given.
    lmax : int, optional
        Defaults to ``3 * nside - 1``.

    Returns
    -------
    ndarray
        Filtered, noisy, masked CMB convergence.
    """
    nside = hp.npix2nside(len(kappa_cmb))
    if lmax is None:
        lmax = 3 * nside - 1

    kappa_lm = hp.map2alm(kappa_cmb, lmax=lmax)

    filt = None
    if filter_ell is not None and filter_fl is not None:
        filt = np.zeros(lmax + 1)
        m = min(int(np.max(filter_ell)), lmax)
        filt[:m + 1] = np.asarray(filter_fl)[:m + 1]
        hp.almxfl(kappa_lm, filt, inplace=True)

    total_lm = kappa_lm
    if noise_ell is not None and noise_nl is not None:
        if rng is None:
            raise ValueError("rng is required when adding CMB noise")
        nl = np.zeros(lmax + 1)
        m = min(int(np.max(noise_ell)), lmax)
        nl[:m + 1] = np.asarray(noise_nl)[:m + 1]
        if filt is not None:
            nl *= filt
        # synalm has no public rng hook; seed numpy's global state for repeatability.
        np.random.seed(rng.integers(0, 2**32 - 1))
        noise_lm = hp.synalm(nl, lmax=lmax, new=True)
        total_lm = kappa_lm + noise_lm

    out = hp.alm2map(total_lm, nside, lmax=lmax)
    if mask is not None:
        out = out * mask
    return out
