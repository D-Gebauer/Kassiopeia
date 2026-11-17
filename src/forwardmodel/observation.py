"""Observational realism: turn noiseless projected maps into mock observations.

Each probe has an observation class that bundles its footprint mask, noise
model, and any calibration:

* :class:`WeakLensingObservation` — adds IA, optional reduced shear,
  multiplicative shear calibration, additive shear (c-term) and PSF leakage,
  shape noise, and (optionally) a Kaiser-Squires reconstructed convergence.
* :class:`GalaxyClusteringObservation` — Poisson galaxy counts on the mask.
* :class:`CMBLensingObservation` — multipole filter + reconstruction noise + mask.
* :class:`ISWObservation` — multipole filter + temperature noise + mask for the
  late-time ISW ``Delta T / T`` map.

These compose the low-level pieces in :mod:`forwardmodel.noise` and
:mod:`forwardmodel.kaiser_squires`; use those directly for finer control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .noise import (
    ShapeNoise, NoShapeNoise, poisson_counts, apply_cmb_noise,
)
from .kaiser_squires import gamma_to_kappa

# Cap kappa when forming the reduced shear g = gamma / (1 - kappa) so rare
# (near-)critical pixels stay finite; the weak-lensing approximation is invalid
# there anyway.
REDUCED_SHEAR_KAPPA_MAX = 0.9


def compose_ellipticity(e1_s, e2_s, g1, g2):
    """Exact complex reduced-shear composition.

    Implements ``e = (e_s + g) / (1 + conj(g) * e_s)`` componentwise, with
    ``e_s = e1_s + i e2_s`` the intrinsic ellipticity and ``g = g1 + i g2`` the
    (reduced) gravitational shear. This is the physically exact mapping used
    on real ellipticity catalogues; the linearized form ``e = e_s + g`` is
    accurate only when both quantities are small.

    Inputs may be per-object or per-pixel arrays of identical shape.

    Returns
    -------
    e1, e2 : ndarray
    """
    es = np.asarray(e1_s, dtype=np.float64) + 1j * np.asarray(e2_s, dtype=np.float64)
    g = np.asarray(g1, dtype=np.float64) + 1j * np.asarray(g2, dtype=np.float64)
    e = (es + g) / (1.0 + np.conjugate(g) * es)
    return e.real, e.imag


def _coeff(value, bin_index, default=0.0):
    if value is None:
        return default
    arr = np.asarray(value)
    return float(arr) if arr.ndim == 0 else float(arr[bin_index])


def _psf_component(e_psf, bin_index):
    """A PSF-ellipticity map: shared ``(npix,)`` or per-bin ``(n_bins, npix)``."""
    arr = np.asarray(e_psf, dtype=float)
    return arr if arr.ndim == 1 else arr[bin_index]


# ---------------------------------------------------------------------------
# Weak lensing
# ---------------------------------------------------------------------------

@dataclass
class ObservedWeakLensing:
    """Result of :meth:`WeakLensingObservation.observe`, all ``(n_bins, npix)``."""

    gamma1: np.ndarray          # observed shear (signal + noise)
    gamma2: np.ndarray
    gamma1_signal: np.ndarray   # noiseless, calibrated signal shear
    gamma2_signal: np.ndarray
    n_source: np.ndarray        # source counts per pixel
    kappa_recon: Optional[np.ndarray] = None  # KS reconstruction of observed shear


@dataclass
class WeakLensingObservation:
    """Observation model for weak lensing.

    Parameters
    ----------
    mask : ndarray
        Binary footprint mask at the working resolution.
    nbar : float or array_like
        Mean source galaxies per pixel, scalar or per bin.
    shape_noise : ShapeNoise
        Noise model. Defaults to :class:`~forwardmodel.noise.NoShapeNoise`.
        For survey mocks use :class:`~forwardmodel.noise.DESShapeNoise` or
        :class:`~forwardmodel.noise.KiDSShapeNoise` (real catalogue, rotated
        galaxies at their true positions; ``nbar`` and the source overdensity
        are then not used for the noise).
    shear_calibration : float or array_like, optional
        Multiplicative bias ``m``; observed signal is scaled by ``1 + m``.
    reduced_shear : bool
        If True, use the reduced shear ``g = gamma / (1 - kappa)`` for the
        gravitational signal instead of ``gamma`` (a second-order correction;
        IA is added to ``g``). ``1 - kappa`` is floored at
        ``1 - REDUCED_SHEAR_KAPPA_MAX`` for numerical safety.
    exact_composition : bool
        If True, combine intrinsic ellipticity (IA + shape noise) with the
        gravitational reduced shear via the exact complex mapping
        ``e = (e_s + g) / (1 + g*·e_s)`` instead of linear addition. Implies
        ``reduced_shear`` for the gravitational signal. This is the
        physically correct combination for real ellipticity catalogues; the
        difference matters at second order in ``|g|`` and ``|e_s|``.
    additive_shear : tuple, optional
        Additive shear (c-term) ``(c1, c2)``, each a scalar or per-bin array,
        added to the calibrated signal: ``gamma_obs += (c1, c2)``.
    psf_leakage : tuple, optional
        PSF leakage ``(alpha, e1_psf, e2_psf)``: a scalar/per-bin coefficient
        ``alpha`` and PSF-ellipticity maps (shared ``(npix,)`` or per-bin
        ``(n_bins, npix)``). Adds ``alpha * (e1_psf, e2_psf)`` to the signal.
    reconstruct_kappa : bool
        If True, Kaiser-Squires invert the observed shear into a convergence map.
    lmax : int, optional
    """

    mask: np.ndarray
    nbar: object = 1.0
    shape_noise: ShapeNoise = field(default_factory=NoShapeNoise)
    shear_calibration: object = None
    reduced_shear: bool = False
    exact_composition: bool = False
    additive_shear: Optional[tuple] = None
    psf_leakage: Optional[tuple] = None
    reconstruct_kappa: bool = True
    lmax: Optional[int] = None

    def observe(self, wl_maps, rng):
        """Apply IA, calibration, shape noise, and KS to noiseless WL maps.

        Parameters
        ----------
        wl_maps : forwardmodel.probes.WeakLensingMaps
        rng : np.random.Generator

        Returns
        -------
        ObservedWeakLensing
        """
        n_bins = wl_maps.n_bins
        npix = wl_maps.kappa.shape[1]
        mask = self.mask

        g1_sig = np.zeros((n_bins, npix))
        g2_sig = np.zeros((n_bins, npix))
        g1_obs = np.zeros((n_bins, npix))
        g2_obs = np.zeros((n_bins, npix))
        n_src = np.zeros((n_bins, npix))
        kappa_recon = (np.zeros((n_bins, npix))
                       if self.reconstruct_kappa else None)

        for b in range(n_bins):
            # Gravitational shear, optionally as reduced shear g = gamma/(1-kappa).
            # Exact composition implies reduced shear.
            use_reduced = self.reduced_shear or self.exact_composition
            g1 = wl_maps.gamma1[b]
            g2 = wl_maps.gamma2[b]
            if use_reduced:
                denom = 1.0 - np.minimum(wl_maps.kappa[b],
                                         REDUCED_SHEAR_KAPPA_MAX)
                g1 = g1 / denom
                g2 = g2 / denom

            nbar_b = _coeff(self.nbar, b, default=1.0)
            ns, g1n, g2n = self.shape_noise.generate(
                b, nbar_b, wl_maps.source_overdensity[b], mask, rng)
            n_src[b] = ns

            if self.exact_composition:
                # Intrinsic side: IA (intrinsic alignment) + shape-noise draw.
                # Combine with the gravitational reduced shear via the exact
                # complex mapping e = (e_s + g) / (1 + g*·e_s).
                es1 = wl_maps.ia_gamma1[b] + g1n
                es2 = wl_maps.ia_gamma2[b] + g2n
                e1_obs, e2_obs = compose_ellipticity(es1, es2, g1, g2)
                # Noiseless calibrated signal: compose IA with g, no shape noise.
                e1_sig, e2_sig = compose_ellipticity(
                    wl_maps.ia_gamma1[b], wl_maps.ia_gamma2[b], g1, g2)
            else:
                # IA adds linearly to the (reduced) gravitational shear.
                g1 = g1 + wl_maps.ia_gamma1[b]
                g2 = g2 + wl_maps.ia_gamma2[b]
                e1_sig, e2_sig = g1, g2
                e1_obs = g1 + g1n
                e2_obs = g2 + g2n

            # Multiplicative shear calibration applied to the composed signal.
            m = _coeff(self.shear_calibration, b, default=0.0)
            e1_sig = (1.0 + m) * e1_sig
            e2_sig = (1.0 + m) * e2_sig
            e1_obs = (1.0 + m) * e1_obs
            e2_obs = (1.0 + m) * e2_obs
            # Additive shear (c-term).
            if self.additive_shear is not None:
                c1, c2 = self.additive_shear
                dc1 = _coeff(c1, b, default=0.0)
                dc2 = _coeff(c2, b, default=0.0)
                e1_sig = e1_sig + dc1; e2_sig = e2_sig + dc2
                e1_obs = e1_obs + dc1; e2_obs = e2_obs + dc2
            # PSF leakage: alpha * e_psf.
            if self.psf_leakage is not None:
                alpha, e1_psf, e2_psf = self.psf_leakage
                a = _coeff(alpha, b, default=0.0)
                dp1 = a * _psf_component(e1_psf, b)
                dp2 = a * _psf_component(e2_psf, b)
                e1_sig = e1_sig + dp1; e2_sig = e2_sig + dp2
                e1_obs = e1_obs + dp1; e2_obs = e2_obs + dp2

            g1_sig[b] = e1_sig
            g2_sig[b] = e2_sig
            g1_obs[b] = e1_obs
            g2_obs[b] = e2_obs

            if kappa_recon is not None:
                kE, _ = gamma_to_kappa(g1_obs[b], g2_obs[b], lmax=self.lmax)
                kappa_recon[b] = kE

        return ObservedWeakLensing(
            gamma1=g1_obs, gamma2=g2_obs,
            gamma1_signal=g1_sig, gamma2_signal=g2_sig,
            n_source=n_src, kappa_recon=kappa_recon,
        )


# ---------------------------------------------------------------------------
# Galaxy clustering
# ---------------------------------------------------------------------------

@dataclass
class GalaxyClusteringObservation:
    """Observation model for galaxy clustering (Poisson shot noise).

    Parameters
    ----------
    mask : ndarray
        Binary footprint mask.
    nbar : float or array_like
        Mean galaxies per pixel, scalar or per bin.
    """

    mask: np.ndarray
    nbar: object = 1.0

    def observe(self, gc_maps, rng):
        """Return Poisson galaxy counts, shape ``(n_bins, npix)``."""
        n_bins = gc_maps.n_bins
        npix = gc_maps.delta_g.shape[1]
        counts = np.zeros((n_bins, npix))
        for b in range(n_bins):
            nbar_b = _coeff(self.nbar, b, default=1.0)
            counts[b] = poisson_counts(nbar_b, gc_maps.delta_g[b], self.mask, rng)
        return counts


# ---------------------------------------------------------------------------
# CMB lensing
# ---------------------------------------------------------------------------

@dataclass
class CMBLensingObservation:
    """Observation model for CMB lensing.

    Parameters
    ----------
    mask : ndarray, optional
        Binary footprint mask at the working resolution.
    noise : tuple, optional
        ``(ell, N_L)`` reconstruction noise power spectrum.
    filter : tuple, optional
        ``(ell, F_L)`` multipole filter.
    lmax : int, optional
    """

    mask: Optional[np.ndarray] = None
    noise: Optional[tuple] = None
    filter: Optional[tuple] = None
    lmax: Optional[int] = None

    def observe(self, cmb_maps, rng):
        """Return the filtered, noisy, masked CMB convergence map."""
        noise_ell, noise_nl = self.noise if self.noise else (None, None)
        filter_ell, filter_fl = self.filter if self.filter else (None, None)
        return apply_cmb_noise(
            cmb_maps.kappa,
            noise_ell=noise_ell, noise_nl=noise_nl,
            filter_ell=filter_ell, filter_fl=filter_fl,
            mask=self.mask, rng=rng, lmax=self.lmax,
        )


# ---------------------------------------------------------------------------
# ISW
# ---------------------------------------------------------------------------

@dataclass
class ISWObservation:
    """Observation model for the late-time ISW temperature map.

    ISW science is almost always done in cross-correlation with a low-z tracer
    (the auto-spectrum is dwarfed by the primary CMB), so ``noise`` should be
    the *total* CMB-temperature power that the cross-estimator sees on top of
    the ISW signal — primary CMB ``C_L^{TT}`` plus instrument noise, in the
    same dimensionless ``Delta T / T`` units as the signal map. Pass nothing
    for a clean signal-only mock.

    Parameters
    ----------
    mask : ndarray, optional
        Binary footprint mask at the working resolution. ``None`` keeps the
        full sky.
    noise : tuple, optional
        ``(ell, N_L)`` temperature power spectrum (dimensionless ``Delta T/T``)
        to draw a Gaussian realisation from and add to the signal. Should
        bundle whatever the cross-correlation analysis treats as noise
        (primary CMB + instrument). ``None`` disables.
    filter : tuple, optional
        ``(ell, F_L)`` multipole filter. Typical use is a low-pass keeping
        ``L < ~100`` where the late-time ISW lives. ``None`` disables.
    lmax : int, optional
        Defaults to ``3 * nside - 1``.
    """

    mask: Optional[np.ndarray] = None
    noise: Optional[tuple] = None
    filter: Optional[tuple] = None
    lmax: Optional[int] = None

    def observe(self, isw_maps, rng):
        """Return the filtered, noisy, masked ISW ``Delta T / T`` map."""
        noise_ell, noise_nl = self.noise if self.noise else (None, None)
        filter_ell, filter_fl = self.filter if self.filter else (None, None)
        return apply_cmb_noise(
            isw_maps.theta,
            noise_ell=noise_ell, noise_nl=noise_nl,
            filter_ell=filter_ell, filter_fl=filter_fl,
            mask=self.mask, rng=rng, lmax=self.lmax,
        )
