"""Line-of-sight projection weights.

Every linear-in-delta probe (weak-lensing kappa, CMB-lensing kappa, linear
galaxy clustering) projects as ``map_b = sum_i delta_i * W[i, b]``, where the
per-shell weight ``W`` is purely geometric. These weights are the UFalcon
continuous-kernel integrals expressed directly in *overdensity* form — i.e. the
box-volume prefactor of the original count-form implementation has been folded
into a shell comoving-volume factor

    vol_factor_i = (chi_hi^3 - chi_lo^3)            [Mpc^3]

so that

    W_lensing[i, b] = (Omega_m / 2) (H0/c)^3 * vol_factor_i * w_count_lensing
    W_clustering[i, b] = (H0^2 / c^3) / 3 * vol_factor_i * w_count_clustering

This is exactly equivalent to multiplying count shells by the UFalcon
``kappa_prefactor`` / ``delta_prefactor`` when the shell map mean equals the
theoretical particle count, but needs no box size or particle number.

Nonlinear terms (quadratic bias, TATT IA) act on delta directly and use the
n(z) shell-overlap weight ``int_shell n(z) dz`` instead.
"""

from __future__ import annotations

import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d

from .cosmology import C_KM_S


# ---------------------------------------------------------------------------
# n(z) preparation: interpolant + integration breakpoints + normalisation
# ---------------------------------------------------------------------------

def _prep_nz(nd):
    """Build the quadrature helpers for a :class:`NumberDensity`."""
    z = nd.z
    nz = nd.nz
    intpt = interp1d(z, nz, bounds_error=False, fill_value=0.0, kind="linear")
    z_lo = max(float(z[0]), 0.0)
    z_hi = float(z[-1])
    mask = (z > z_lo) & (z < z_hi)
    points = z[mask] if mask.any() else None
    limit = 10 * len(points) if points is not None else 1000
    norm = integrate.quad(intpt, z_lo, z_hi, points=points, limit=limit)[0]
    return {"intpt": intpt, "points": points, "z_lo": z_lo, "z_hi": z_hi,
            "norm": norm, "limit": limit}


# ---------------------------------------------------------------------------
# Weak-lensing weight (UFalcon continuous lensing, fast Simpson mode)
# ---------------------------------------------------------------------------

def _lensing_count_weight(z_lo, z_hi, cosmo, nzd, n1d=13, n2d=512):
    """UFalcon continuous-lensing count-form weight ``num / norm``."""
    if z_lo >= nzd["z_hi"] or nzd["norm"] <= 0:
        return 0.0
    intpt = nzd["intpt"]
    H0c = cosmo.H0_over_c

    z_up = min(z_hi, nzd["z_hi"])
    x, dz = np.linspace(z_lo, z_up, n1d, retstep=True)
    x_safe = np.maximum(x, 1e-4)
    y = np.geomspace(x_safe, nzd["z_hi"] + 1e-12, n2d)  # (n2d, n1d)

    chi_x = cosmo.comoving_distance(x) * H0c            # (n1d,)
    chi_y = cosmo.comoving_distance(y) * H0c            # (n2d, n1d)
    inv_E_x = cosmo.inv_efunc(x)                        # (n1d,)
    nz_y = intpt(y)                                     # (n2d, n1d)

    chi_xy = chi_y - chi_x[np.newaxis, :]
    integrand = np.where(
        chi_y > 0,
        nz_y * chi_x[np.newaxis, :] * chi_xy
        * (1.0 + x)[np.newaxis, :] * inv_E_x[np.newaxis, :] / chi_y,
        0.0,
    )
    integrand = np.nan_to_num(integrand)
    inner = integrate.simpson(integrand, x=y, axis=0)   # (n1d,)
    num = integrate.simpson(inner, dx=dz, axis=0)

    z_mid = 0.5 * (z_lo + z_hi)
    chi_shell = cosmo.comoving_distance_dimensionless(z_lo, z_hi)
    chi_mid = cosmo.comoving_distance_dimensionless(0.0, z_mid)
    norm = chi_shell * nzd["norm"] * chi_mid**2
    return num / norm if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# CMB-lensing weight (UFalcon Dirac lensing)
# ---------------------------------------------------------------------------

def _cmb_count_weight(z_lo, z_hi, cosmo, z_source):
    if z_source <= z_lo:
        return 0.0
    if z_source < z_hi:
        z_hi = z_source
    H0c = cosmo.H0_over_c

    def integrand(x):
        chi_x = cosmo.comoving_distance(x) * H0c
        chi_s = cosmo.comoving_distance(z_source) * H0c
        return chi_x * (chi_s - chi_x) * (1.0 + x) * cosmo.inv_efunc(x)

    xs, dx = np.linspace(z_lo, z_hi, 9, retstep=True)
    num = integrate.simpson(np.array([integrand(x) for x in xs]), dx=dx)

    z_mid = 0.5 * (z_lo + z_hi)
    chi_shell = cosmo.comoving_distance_dimensionless(z_lo, z_hi)
    chi_src = cosmo.comoving_distance_dimensionless(0.0, z_source)
    chi_mid = cosmo.comoving_distance_dimensionless(0.0, z_mid)
    norm = chi_shell * chi_src * chi_mid**2
    return num / norm if norm > 0 else 0.0


# ---------------------------------------------------------------------------
# Linear clustering weight (UFalcon continuous clustering, no bias)
# ---------------------------------------------------------------------------

def _clustering_count_weight(z_lo, z_hi, cosmo, nzd):
    if nzd["norm"] <= 0:
        return 0.0
    z_mid = 0.5 * (z_lo + z_hi)
    chi_mid = cosmo.comoving_distance_dimensionless(0.0, z_mid)
    if chi_mid <= 0:
        return 0.0
    return (cosmo.hubble(z_mid) * nzd["intpt"](z_mid)
            / chi_mid**2 / nzd["norm"])


# ---------------------------------------------------------------------------
# delta-form volume factors (fold out the UFalcon box prefactors)
# ---------------------------------------------------------------------------

def _vol_factor(cosmo, z_lo, z_hi):
    """``chi_hi^3 - chi_lo^3`` in Mpc^3 for a shell."""
    chi_lo = cosmo.comoving_distance(z_lo)
    chi_hi = cosmo.comoving_distance(z_hi)
    return chi_hi**3 - chi_lo**3


# ---------------------------------------------------------------------------
# Public weight builders — all return delta-form weights
# ---------------------------------------------------------------------------

def lensing_weights(cosmo, nz_bins, z_lower, z_upper):
    """Weak-lensing kappa weights, shape ``(n_shells, n_bins)`` (delta-form)."""
    n_shells = len(z_lower)
    n_bins = len(nz_bins)
    nzd = [_prep_nz(nd) for nd in nz_bins]
    pref = 0.5 * cosmo.Omega_m * cosmo.H0_over_c**3
    w = np.zeros((n_shells, n_bins))
    for i in range(n_shells):
        vf = _vol_factor(cosmo, z_lower[i], z_upper[i])
        for b in range(n_bins):
            w[i, b] = pref * vf * _lensing_count_weight(
                z_lower[i], z_upper[i], cosmo, nzd[b])
    return w


def cmb_lensing_weights(cosmo, z_lower, z_upper, z_source=1090.0):
    """CMB-lensing kappa weights, shape ``(n_shells,)`` (delta-form)."""
    n_shells = len(z_lower)
    pref = 0.5 * cosmo.Omega_m * cosmo.H0_over_c**3
    w = np.zeros(n_shells)
    for i in range(n_shells):
        vf = _vol_factor(cosmo, z_lower[i], z_upper[i])
        w[i] = pref * vf * _cmb_count_weight(
            z_lower[i], z_upper[i], cosmo, z_source)
    return w


def lensing_separable_coeffs(cosmo, z_lower, z_upper, n_sub=9):
    """Per-shell ``(alpha, beta)`` for single-source-plane convergence weights.

    The delta-form convergence weight from foreground shell ``i`` to a source
    plane at dimensionless comoving distance ``chi_s`` (``H0/c`` units) is
    ``alpha[i] - beta[i] / chi_s``. This is exactly :func:`cmb_lensing_weights`
    with the source-distance dependence factored out, which it can be because
    the single-plane lensing kernel ``chi_x (chi_s - chi_x) / chi_s`` is linear
    in ``1/chi_s``. The separable form lets a single redshift-ordered pass
    accumulate the convergence to *every* source shell via two running sums
    (``S_alpha = sum delta_i alpha_i``, ``S_beta = sum delta_i beta_i`` over
    foreground shells, then ``kappa = S_alpha - S_beta / chi_s``) — used for
    source clustering's non-isotropic n(z).
    """
    n_shells = len(z_lower)
    H0c = cosmo.H0_over_c
    pref = 0.5 * cosmo.Omega_m * H0c**3
    alpha = np.zeros(n_shells)
    beta = np.zeros(n_shells)
    for i in range(n_shells):
        z_lo, z_hi = float(z_lower[i]), float(z_upper[i])
        x, dx = np.linspace(z_lo, z_hi, n_sub, retstep=True)
        chi_x = cosmo.comoving_distance(x) * H0c
        base = chi_x * (1.0 + x) * cosmo.inv_efunc(x)
        I_a = integrate.simpson(base, dx=dx)            # coeff of chi_s
        I_b = integrate.simpson(base * chi_x, dx=dx)    # coeff of -1
        z_mid = 0.5 * (z_lo + z_hi)
        chi_shell = cosmo.comoving_distance_dimensionless(z_lo, z_hi)
        chi_mid = cosmo.comoving_distance_dimensionless(0.0, z_mid)
        denom = chi_shell * chi_mid**2
        if denom <= 0:
            continue
        c = pref * _vol_factor(cosmo, z_lo, z_hi) / denom
        alpha[i] = c * I_a
        beta[i] = c * I_b
    return alpha, beta


def clustering_weights(cosmo, nz_bins, z_lower, z_upper):
    """Linear-clustering weights, shape ``(n_shells, n_bins)`` (delta-form).

    Excludes the linear bias ``b1`` — multiply per bin afterwards.
    """
    n_shells = len(z_lower)
    n_bins = len(nz_bins)
    nzd = [_prep_nz(nd) for nd in nz_bins]
    pref = (cosmo.H0**2 / C_KM_S**3) / 3.0
    w = np.zeros((n_shells, n_bins))
    for i in range(n_shells):
        vf = _vol_factor(cosmo, z_lower[i], z_upper[i])
        for b in range(n_bins):
            w[i, b] = pref * vf * _clustering_count_weight(
                z_lower[i], z_upper[i], cosmo, nzd[b])
    return w


def isw_weights(cosmo, z_lower, z_upper, n_sub=9):
    """Per-shell ISW temperature weight, shape ``(n_shells,)``.

    The (late-time) ISW temperature anisotropy in Limber + linear theory is

        Theta_lm = sum_i a_i / [l (l+1)] * delta_lm^(i)

    where the per-shell scalar coefficient is

        a_i = 3 (H0/c)^3 Omega_m * E(z_i) * (1 - f(z_i))
              * chi(z_i)^2 * (chi_hi - chi_lo)        [Mpc^3 * Mpc^-3 = 1]

    with chi in Mpc, f the linear growth rate, and the integrand evaluated
    via a short Simpson rule across the shell. Matter domination has
    ``f = 1`` exactly so ``a_i`` vanishes there; the kernel is nonzero only
    where dark energy makes the potentials decay.

    The inverse spherical Laplacian ``1/[l(l+1)]`` is applied globally on
    the accumulated map in the main pass, so this builder only returns the
    scalar redshift kernel.
    """
    n_shells = len(z_lower)
    H0c = cosmo.H0_over_c
    pref = 3.0 * cosmo.Omega_m * H0c**3
    w = np.zeros(n_shells)
    for i in range(n_shells):
        z_lo, z_hi = float(z_lower[i]), float(z_upper[i])
        x, dx = np.linspace(z_lo, z_hi, n_sub, retstep=True)
        chi = cosmo.comoving_distance(x)                       # Mpc
        # d chi = (c / H(z)) dz  ->  H(z)/c d chi = dz, so the dz integral
        # already absorbs the dchi -> dz Jacobian via inv_efunc / H0c.
        dchi_dz = cosmo.inv_efunc(x) / H0c                     # Mpc
        E = cosmo.efunc(x)
        f = cosmo.growth_rate(x)
        integrand = E * (1.0 - f) * chi**2 * dchi_dz           # Mpc^3
        w[i] = pref * integrate.simpson(integrand, dx=dx)
    return w


def overlap_weights(nz_bins, z_lower, z_upper, n_sub=21):
    """n(z) shell-overlap ``int_shell n(z) dz``, shape ``(n_shells, n_bins)``.

    Used for the nonlinear (quadratic-bias, TATT) terms, which act on delta
    directly. n(z) is assumed normalised; integrated on a fine sub-grid so it
    is robust to shells narrower than the n(z) sampling.
    """
    n_shells = len(z_lower)
    n_bins = len(nz_bins)
    w = np.zeros((n_shells, n_bins))
    for b, nd in enumerate(nz_bins):
        for i in range(n_shells):
            z_sub = np.linspace(z_lower[i], z_upper[i], n_sub)
            w[i, b] = np.trapezoid(nd(z_sub), z_sub)
    return w


def effective_redshift(nz_bins, z_lower, z_upper, n_sub=21):
    """n(z)-weighted effective redshift per shell per bin (for IA amplitudes).

    Falls back to the shell midpoint where the bin has no n(z) support.
    """
    n_shells = len(z_lower)
    n_bins = len(nz_bins)
    z_eff = np.zeros((n_shells, n_bins))
    for b, nd in enumerate(nz_bins):
        for i in range(n_shells):
            z_mid = 0.5 * (z_lower[i] + z_upper[i])
            z_sub = np.linspace(z_lower[i], z_upper[i], n_sub)
            nz_sub = nd(z_sub)
            denom = np.trapezoid(nz_sub, z_sub)
            z_eff[i, b] = (np.trapezoid(z_sub * nz_sub, z_sub) / denom
                           if denom > 0 else z_mid)
    # For n(z) >= 0 the weighted mean lies inside the shell. Spline n(z) (NumberDensity.from_spline) can
    # undershoot slightly below zero in the tails, where the overlap is ~0 and the ratio is ill-conditioned
    # (it then leaves the shell, even z < 0): clip to the shell bounds.
    return np.clip(z_eff, np.asarray(z_lower, float)[:, None], np.asarray(z_upper, float)[:, None])
