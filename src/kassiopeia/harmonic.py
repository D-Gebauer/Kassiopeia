"""Harmonic-space weak-lensing projection: many n(z) variants and rotations from one set of shell transforms.

Every linear weak-lensing field is a weighted sum over shells of the shell overdensity,
``kappa_b = sum_i W[i, b] delta_i`` (convergence, ``W`` = lensing weights) and, for linear IA (NLA),
``kappa_IA,b = sum_i A1(z_eff,ib) w_ov,ib delta_i`` whose Kaiser-Squires shear is the IA shear (see
:meth:`ForwardModel.project`, ``linear_ia_fast``). Since the spherical-harmonic transform, rotations and
Kaiser-Squires are all linear, the shells can be transformed **once**:

    delta_lm[i]  = map2alm(delta_i)                                   (n_shell transforms, once)
    kappa_lm[b]  = sum_i W[i, b] delta_lm[i]                          (a matrix product per n(z) variant)
    kappa_lm[b] <- rotate(kappa_lm[b])  or rotate delta_lm once        (exact, see kassiopeia.rotation)
    gamma_b      = KS(kappa_lm[b])                                    (one spin-2 synthesis per bin)

so photo-z shifts per realisation, per-bin IA amplitudes and random rotations cost a matrix product, a
rotation and one synthesis per field, not a new pass over the shells. Equal to the pixel-space projection of
:class:`ForwardModel` up to the healpy ``map2alm`` quadrature error (shells are not band-limited) and the
storage precision of the shell alms. Linear IA only (no TATT quadratic terms, no source clustering).
"""

from __future__ import annotations

import numpy as np
import healpy as hp

from . import projections as proj
from .kaiser_squires import _kappa_from_gamma_factor


def shell_overdensity_alms(shells, lmax=None, iter=3, dtype=np.complex64, verbose=False):
    """``a_lm`` of every shell's overdensity, shape ``(n_shells, n_alm)``.

    Overdensity as in the projection (map-mean or box normalisation of the :class:`ShellStack`).
    Shells without data (``overdensity`` returns ``None``) get zeros. ``dtype=complex64`` halves memory
    (relative precision 1e-7, far below the map2alm quadrature error).
    """
    if lmax is None:
        lmax = 3 * shells.nside - 1
    out = np.zeros((shells.n_shells, hp.Alm.getsize(lmax)), dtype=dtype)
    for i in range(shells.n_shells):
        delta = shells.overdensity(i)
        if delta is None:
            continue
        out[i] = hp.map2alm(delta, lmax=lmax, iter=iter)
        if verbose:
            print(f"  shell alm {i + 1}/{shells.n_shells}", flush=True)
    return out


def weak_lensing_weights(cosmo, nz_bins, z_lower, z_upper, ia=None, z_max_tidal=3.0):
    """Shell weights ``(W_kappa, W_ia)``, each ``(n_shells, n_bins)``, as used by ``ForwardModel.project``.

    ``W_ia[i, b] = A1(z_eff,ib) * int_shell n_b dz`` for a linear ``ia`` (NLA; ``None`` gives zeros), zero for
    shells with mid-point above ``z_max_tidal``.
    """
    w_kappa = proj.lensing_weights(cosmo, nz_bins, z_lower, z_upper)
    w_ia = np.zeros_like(w_kappa)
    if ia is not None:
        if getattr(ia, "a2", 0.0) != 0.0 or getattr(ia, "b_ta", 0.0) != 0.0:
            raise ValueError("harmonic projection supports linear IA (NLA) only")
        w_ov = proj.overlap_weights(nz_bins, z_lower, z_upper)
        z_eff = proj.effective_redshift(nz_bins, z_lower, z_upper)
        z_mid = 0.5 * (np.asarray(z_lower) + np.asarray(z_upper))
        for i in range(len(z_lower)):
            if z_mid[i] > z_max_tidal:
                continue
            for b in range(len(nz_bins)):
                if w_ov[i, b] != 0.0:
                    w_ia[i, b] = ia.amplitudes(cosmo, z_eff[i, b])[0] * w_ov[i, b]
    return w_kappa, w_ia


def combine_alms(shell_alms, weights):
    """``sum_i weights[i, b] shell_alms[i]`` -> ``(n_bins, n_alm)``, as one real matrix product.

    ``weights`` is real ``(n_shells, n_bins)``; the complex alms are multiplied as their float view.
    """
    a = np.ascontiguousarray(shell_alms)
    re = a.view(np.float32 if a.dtype == np.complex64 else np.float64)          # (n_shells, 2 n_alm)
    w = np.asarray(weights, dtype=re.dtype)
    return (w.T @ re).view(a.dtype)


def kappa_alm_to_shear(kappa_alm, nside, lmax=None):
    """Kaiser-Squires shear ``(gamma1, gamma2)`` from convergence ``a_lm`` (``l < 2`` zeroed), as
    :func:`kassiopeia.kaiser_squires.kappa_to_gamma` does after its ``map2alm``."""
    kappa_alm = np.asarray(kappa_alm)
    if lmax is None:
        lmax = hp.Alm.getlmax(kappa_alm.shape[-1])
    f = _kappa_from_gamma_factor(lmax)
    inv = np.zeros_like(f)
    np.divide(1.0, f, out=inv, where=f > 0)
    elm = hp.almxfl(kappa_alm.astype(np.complex128), inv)
    g1, g2 = hp.alm2map_spin([elm, np.zeros_like(elm)], nside, 2, lmax)
    return g1, g2
