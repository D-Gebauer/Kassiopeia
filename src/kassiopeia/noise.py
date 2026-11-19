"""Noise models: shape noise, galaxy shot noise, and CMB-lensing noise.

Shape noise comes in several interchangeable flavours, all sharing the
:class:`ShapeNoise` interface so they can be swapped freely on a
:class:`kassiopeia.observation.WeakLensingObservation`:

* :class:`DESShapeNoise` / :class:`KiDSShapeNoise` (recommended for survey
  mocks) — take the real catalogue, keep every galaxy at its position, rotate
  its ellipticity by a random angle and form the survey's own pixel shear
  estimator: weighted mean divided by the metacal response ``<R>`` (DES) or
  the lensfit-weighted mean (KiDS). Both derive from
  :class:`CatalogShapeNoise`.
* :class:`EmpiricalShapeNoise` — bootstrap catalogue ellipticities into
  Poisson-drawn simulated source counts (positions not from the catalogue),
  randomly rotate them, and average per pixel. Noise variance then scales as
  ``1 / N`` and tracks the simulated source-clustering signal.
* :class:`GaussianShapeNoise` — draw per-pixel shear noise from
  ``N(0, sigma_e^2 / N)`` analytically; no catalogue needed.
* :class:`NoShapeNoise` — signal only.

The functions :func:`poisson_counts` and :func:`apply_cmb_noise` cover galaxy
shot noise and the ACT-style CMB-lensing filter+noise+mask chain.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
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
# Survey-catalogue shape noise (real positions, random rotations)
# ---------------------------------------------------------------------------

def _as_bin_dict(catalogs):
    if isinstance(catalogs, Mapping):
        return {int(k): v for k, v in catalogs.items()}
    return dict(enumerate(catalogs))


def _per_bin(value, bin_index):
    arr = np.asarray(value, dtype=np.float64)
    return float(arr) if arr.ndim == 0 else float(arr[bin_index])


@dataclass
class _PixelisedBin:
    """One bin's catalogue binned onto a HEALPix grid (reused across draws)."""

    pix: np.ndarray       # (n_occupied,) occupied pixel indices
    inv: np.ndarray       # (n_gal,) galaxy -> position in ``pix``
    we1: np.ndarray       # (n_gal,) w * e1 (uncalibrated)
    we2: np.ndarray       # (n_gal,) w * e2
    norm: np.ndarray      # (n_occupied,) per-pixel denominator
    counts: np.ndarray    # (n_occupied,) galaxies per occupied pixel
    wsum: np.ndarray      # (n_occupied,) sum of weights per occupied pixel


class CatalogShapeNoise(ShapeNoise):
    """Shape noise from the real survey catalogue, galaxy by galaxy.

    Every catalogue galaxy keeps its position, weight and ellipticity
    modulus; only its orientation is randomised. Its spin-2 ellipticity is
    rotated by a uniform random phase, ``e -> e exp(2 i phi)`` with
    ``phi ~ U[0, pi)``, which erases the cosmological shear (and any
    residual additive bias) while keeping the true ellipticity distribution,
    weight distribution and source density field of the survey. The pixel
    noise is then

    ``gamma_noise(p) = sum_{i in p} w_i e_i^rot / N(p)``

    where the normalisation ``N(p)`` is survey-specific (subclasses):
    ``<R> sum w`` for DES metacalibration, ``sum w`` for KiDS lensfit.

    The source counts returned are the actual galaxy counts per pixel, so
    ``nbar`` and ``delta_source`` passed to :meth:`generate` are ignored (the
    noise follows the real survey's source density, not the simulated one).
    Pixels outside ``mask`` are set to zero.

    Parameters
    ----------
    catalogs : mapping or sequence
        Per tomographic bin (``{bin: catalog}`` or a list), a catalogue
        indexable by column name: dict of arrays, numpy structured array,
        ``np.load``-ed ``.npz``, pandas DataFrame, FITS record array, ...
        RA/Dec are in degrees.
    columns : dict, optional
        Overrides of the catalogue column names, keyed by the canonical names
        (``ra, dec, e1, e2, w`` and the survey-specific ones).
    chunk_size : int
        Galaxies rotated per vectorised chunk; bounds temporary memory
        (~40 bytes per galaxy) for catalogues with 10^8 objects.

    Notes
    -----
    The pixelisation of each bin is built lazily for the ``nside`` of the
    first ``mask`` it sees and cached, so repeated draws only cost one
    rotation and two ``bincount`` per galaxy.
    """

    _default_columns = {"ra": "ra", "dec": "dec", "e1": "e1", "e2": "e2", "w": "w"}

    def __init__(self, catalogs, columns=None, chunk_size=4_000_000):
        self.catalogs = _as_bin_dict(catalogs)
        self.columns = {**self._default_columns, **(columns or {})}
        self.chunk_size = int(chunk_size)
        self._cache = {}

    # ---- catalogue access ------------------------------------------------
    def _col(self, cat, name, dtype=np.float64):
        key = self.columns[name]
        try:
            col = cat[key]
        except (KeyError, ValueError, IndexError) as err:
            raise KeyError(
                f"{type(self).__name__}: catalogue has no column {key!r} "
                f"(canonical {name!r}); pass columns={{{name!r}: ...}}") from err
        return np.asarray(col, dtype=dtype)

    def _has_col(self, cat, name):
        key = self.columns.get(name)
        if key is None:
            return False
        try:
            cat[key]
        except (KeyError, ValueError, IndexError):
            return False
        return True

    @abstractmethod
    def _normalisation(self, bin_index, cat, inv, n_occupied, w, wsum):
        """Per-occupied-pixel denominator of the weighted shear estimator."""
        ...

    # ---- pixelisation ----------------------------------------------------
    def _pixelise(self, bin_index, nside):
        key = (bin_index, nside)
        if key in self._cache:
            return self._cache[key]
        cat = self.catalogs[bin_index]
        ra = self._col(cat, "ra")
        dec = self._col(cat, "dec")
        pix_gal = hp.ang2pix(nside, ra, dec, lonlat=True)
        del ra, dec
        pix, inv = np.unique(pix_gal, return_inverse=True)
        del pix_gal
        inv = inv.astype(np.int32 if len(pix) < 2**31 else np.int64)
        n_occ = len(pix)

        w = self._col(cat, "w")
        e1 = self._col(cat, "e1")
        e2 = self._col(cat, "e2")
        wsum = np.bincount(inv, weights=w, minlength=n_occ)
        norm = np.asarray(self._normalisation(bin_index, cat, inv, n_occ, w, wsum),
                          dtype=np.float64)
        if np.any(norm[wsum > 0] <= 0):
            raise ValueError(f"{type(self).__name__}: non-positive shear-estimator "
                             f"normalisation in bin {bin_index}")
        entry = _PixelisedBin(
            pix=pix, inv=inv,
            we1=(w * e1).astype(np.float32), we2=(w * e2).astype(np.float32),
            norm=norm, counts=np.bincount(inv, minlength=n_occ).astype(np.float64),
            wsum=wsum,
        )
        self._cache[key] = entry
        return entry

    # ---- public API ------------------------------------------------------
    def weight_map(self, bin_index, nside):
        """Full-sky ``sum w`` per pixel (the estimator weights) for one bin."""
        e = self._pixelise(bin_index, nside)
        out = np.zeros(hp.nside2npix(nside))
        out[e.pix] = e.wsum
        return out

    def count_map(self, bin_index, nside):
        """Full-sky number of catalogue galaxies per pixel for one bin."""
        e = self._pixelise(bin_index, nside)
        out = np.zeros(hp.nside2npix(nside))
        out[e.pix] = e.counts
        return out

    def realisation(self, bin_index, nside, rng):
        """One randomly rotated noise realisation, full-sky ``(g1, g2)``."""
        e = self._pixelise(bin_index, nside)
        n_gal, n_occ = len(e.inv), len(e.pix)
        s1 = np.zeros(n_occ)
        s2 = np.zeros(n_occ)
        for lo in range(0, n_gal, self.chunk_size):
            hi = min(lo + self.chunk_size, n_gal)
            ang = rng.uniform(0.0, 2.0 * np.pi, size=hi - lo)  # = 2 phi
            c, s = np.cos(ang), np.sin(ang)
            we1 = e.we1[lo:hi]
            we2 = e.we2[lo:hi]
            inv = e.inv[lo:hi]
            s1 += np.bincount(inv, weights=we1 * c - we2 * s, minlength=n_occ)
            s2 += np.bincount(inv, weights=we1 * s + we2 * c, minlength=n_occ)
        npix = hp.nside2npix(nside)
        g1 = np.zeros(npix)
        g2 = np.zeros(npix)
        ok = e.wsum > 0
        g1[e.pix[ok]] = s1[ok] / e.norm[ok]
        g2[e.pix[ok]] = s2[ok] / e.norm[ok]
        return g1, g2

    def generate(self, bin_index, nbar, delta_source, mask, rng):
        nside = hp.npix2nside(len(mask))
        g1, g2 = self.realisation(bin_index, nside, rng)
        n_src = self.count_map(bin_index, nside)
        outside = np.asarray(mask) == 0
        g1[outside] = 0.0
        g2[outside] = 0.0
        n_src[outside] = 0.0
        return n_src, g1, g2


class DESShapeNoise(CatalogShapeNoise):
    """DES metacalibration shape noise from the real source catalogue.

    The catalogue ellipticities are the *raw* metacal ``e1, e2`` (not yet
    divided by the response). After the random rotation the pixel shear is
    the DES estimator

    ``gamma(p) = sum_{i in p} w_i e_i^rot / (<R> sum_{i in p} w_i)``

    with the bin-wide response ``<R> = <R_gamma> + R_S``: ``<R_gamma>`` is the
    ``w``-weighted mean over the bin of ``(R11 + R22) / 2`` (rotations mix the
    two components, so the trace average is the relevant scalar) and ``R_S``
    the selection response of that bin, as in the DES Y3 cosmic-shear and
    mass-map analyses (Gatti et al. 2021).

    Parameters
    ----------
    catalogs : mapping or sequence
        Per bin: columns ``ra, dec, e1, e2, w`` and either ``R11`` and
        ``R22`` or a per-galaxy scalar ``R``.
        :func:`kassiopeia.loaders.des_y3.load_metacal_catalog` returns this.
    selection_response : float or array_like
        ``R_S`` (scalar or per bin), added to ``<R_gamma>``. Default 0.
    per_pixel_response : bool
        If True, normalise each pixel by its own ``sum w R`` instead of the
        bin-wide ``<R> sum w``. The per-galaxy response is noisy, so this is
        not the DES default; it is provided for comparisons.
    columns, chunk_size
        See :class:`CatalogShapeNoise`.
    """

    _default_columns = {**CatalogShapeNoise._default_columns,
                        "R11": "R11", "R22": "R22", "R": "R"}

    def __init__(self, catalogs, selection_response=0.0, per_pixel_response=False,
                 columns=None, chunk_size=4_000_000):
        super().__init__(catalogs, columns=columns, chunk_size=chunk_size)
        self.selection_response = selection_response
        self.per_pixel_response = per_pixel_response

    def _galaxy_response(self, cat):
        if self._has_col(cat, "R11") and self._has_col(cat, "R22"):
            return 0.5 * (self._col(cat, "R11") + self._col(cat, "R22"))
        if self._has_col(cat, "R"):
            return self._col(cat, "R")
        raise KeyError("DESShapeNoise: catalogue needs 'R11' and 'R22' "
                       "(or a scalar 'R') metacal response columns")

    def mean_response(self, bin_index):
        """Bin-wide ``<R> = <R_gamma>_w + R_S`` used in the normalisation."""
        cat = self.catalogs[bin_index]
        w = self._col(cat, "w")
        return (float(np.sum(w * self._galaxy_response(cat)) / np.sum(w))
                + _per_bin(self.selection_response, bin_index))

    def _normalisation(self, bin_index, cat, inv, n_occupied, w, wsum):
        R_S = _per_bin(self.selection_response, bin_index)
        if self.per_pixel_response:
            wR = np.bincount(inv, weights=w * self._galaxy_response(cat),
                             minlength=n_occupied)
            return wR + R_S * wsum
        return self.mean_response(bin_index) * wsum


class KiDSShapeNoise(CatalogShapeNoise):
    """KiDS lensfit shape noise from the real source catalogue.

    The catalogue ellipticities are the calibrated lensfit ``e1, e2``
    (KiDS has no per-object response; the residual multiplicative bias ``m``
    is applied as ``shear_calibration`` of the observation). After the random
    rotation the pixel shear is the lensfit-weighted mean

    ``gamma(p) = sum_{i in p} w_i e_i^rot / sum_{i in p} w_i``.

    Parameters
    ----------
    catalogs : mapping or sequence
        Per bin: columns ``ra, dec, e1, e2, w`` (``w`` = lensfit ``weight``).
        :func:`kassiopeia.loaders.kids.load_shape_catalog` returns this.
    columns, chunk_size
        See :class:`CatalogShapeNoise`.
    """

    def _normalisation(self, bin_index, cat, inv, n_occupied, w, wsum):
        return wsum


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
