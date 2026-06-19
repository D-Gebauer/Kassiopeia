"""ForwardModel: orchestrates the single-pass projection over a shell stack.

This is the high-level entry point. Construct it with a cosmology and a shell
stack, then call :meth:`ForwardModel.project` with any combination of probes::

    fm = ForwardModel(cosmo, shells)
    maps = fm.project(weak_lensing=wl, galaxy_clustering=gc, cmb_lensing=cmb)

All requested probes are accumulated in one redshift-ordered pass over the
shells (ordering lets source clustering build cumulative foreground convergence
for its non-isotropic n(z); every other term is an order-independent sum).
Linear-in-delta contributions use the geometric weights from
:mod:`forwardmodel.projections` (including lens magnification,
``(5s-2) kappa_lens``, when requested);
the nonlinear quadratic-bias and TATT terms compute the tidal field per shell
and use n(z) shell-overlap weights, capped at ``z_max_tidal`` (the lognormal
high-z extension is too crude for nonlinear bias and n(z) overlap is negligible
there anyway).
"""

from __future__ import annotations

import numpy as np

from .bias import QuadraticBias, quadratic_bias_terms
from .shells import ShellStack
from .tidal import compute_tidal
from .kaiser_squires import kappa_to_gamma
from . import projections as proj
from .probes import (
    WeakLensing, GalaxyClustering, CMBLensing, ISW,
    WeakLensingMaps, GalaxyClusteringMaps, CMBLensingMaps, ISWMaps,
    ProjectedMaps,
)
import healpy as hp


class ForwardModel:
    """Project density shells into multi-probe cosmological maps.

    Parameters
    ----------
    cosmo : forwardmodel.Cosmology
    shells : forwardmodel.ShellStack
    lmax : int, optional
        Maximum multipole for tidal/Kaiser-Squires transforms. Defaults to
        ``3 * nside - 1``.
    """

    def __init__(self, cosmo, shells, lmax=None):
        self.cosmo = cosmo
        self.shells = shells
        self.nside = shells.nside
        self.lmax = lmax if lmax is not None else 3 * shells.nside - 1

    @classmethod
    def from_lognormal(cls, cosmo, nside, z_max=None, dz=0.1, z_min=0.0,
                       z_edges=None, rng=None, lmax=None, shift=None):
        """Build a model on a freshly generated lognormal shell stack.

        The "no N-body simulation at this cosmology" entry point: given only a
        cosmology, tile redshift into shells, fill them with cross-correlated
        GLASS lognormal fields, and return a ready-to-:meth:`project` model.
        Equivalent to ``ForwardModel(cosmo, ShellStack.from_lognormal(...))``.
        Requires the optional ``glass`` dependency.

        Provide either an explicit ``z_edges`` array, or ``z_max`` (with ``dz``
        and ``z_min``) to tile ``[z_min, z_max]`` uniformly.

        Parameters
        ----------
        cosmo : forwardmodel.Cosmology
        nside : int
            HEALPix resolution of the generated shells.
        z_max : float, optional
            Upper redshift edge when auto-tiling. Required unless ``z_edges``
            is given.
        dz : float
            Target shell thickness for auto-tiling (default 0.1).
        z_min : float
            Lower redshift edge for auto-tiling (default 0).
        z_edges : array_like, optional
            Explicit bin edges; overrides ``z_max`` / ``dz`` / ``z_min``. Use
            this to coarsen the high-z tail — e.g. for CMB lensing out to
            ``z ~ 1100``, uniform thin bins would mean thousands of shells and
            an O(N^2) spectra cost, so taper to wide bins past ``z ~ 3``.
        rng : np.random.Generator, optional
        lmax : int, optional
            Multipole cap for both shell generation and projection. Defaults to
            ``3 * nside - 1``.
        shift : callable, optional
            Lognormal shift ``shift(z) -> float``.

        Returns
        -------
        ForwardModel
        """
        if z_edges is None:
            if z_max is None:
                raise ValueError("provide either z_edges or z_max")
            n_bins = max(1, int(round((z_max - z_min) / dz)))
            z_edges = np.linspace(z_min, z_max, n_bins + 1)
        shells = ShellStack.from_lognormal(cosmo, z_edges, nside=nside, rng=rng,
                                           lmax=lmax, shift=shift)
        return cls(cosmo, shells, lmax=lmax)

    def project(self, weak_lensing=None, galaxy_clustering=None,
                cmb_lensing=None, isw=None, z_max_tidal=3.0, verbose=False):
        """Project the requested probes in a single pass over the shells.

        Parameters
        ----------
        weak_lensing : WeakLensing, optional
        galaxy_clustering : GalaxyClustering, optional
        cmb_lensing : CMBLensing, optional
        isw : ISW, optional
            Late-time ISW temperature anisotropy.
        z_max_tidal : float
            Shells above this redshift skip the nonlinear tidal/IA terms.
        verbose : bool
            Print per-shell progress.

        Returns
        -------
        ProjectedMaps
        """
        if (weak_lensing is None and galaxy_clustering is None
                and cmb_lensing is None and isw is None):
            raise ValueError("Provide at least one probe to project.")

        cosmo = self.cosmo
        shells = self.shells
        npix = shells.npix
        z_lo, z_hi = shells.z_lower, shells.z_upper

        # -- precompute geometric weights ----------------------------------
        wl_w = gc_lin_w = src_lin_w = cmb_w = None
        gc_overlap_w = src_overlap_w = src_zeff = None
        sc_alpha = sc_beta = sc_overlap = sc_chi_src = None

        if weak_lensing is not None:
            wl_w = proj.lensing_weights(cosmo, weak_lensing.nz, z_lo, z_hi)
            if weak_lensing.has_source_clustering:
                src_lin_w = proj.clustering_weights(
                    cosmo, weak_lensing.nz, z_lo, z_hi)
                # Non-isotropic n(z): single-source-plane convergence coeffs
                # (separable in source distance) + n(z) overlap per shell, for
                # the source-density-weighted lensing signal.
                sc_alpha, sc_beta = proj.lensing_separable_coeffs(
                    cosmo, z_lo, z_hi)
                sc_overlap = proj.overlap_weights(weak_lensing.nz, z_lo, z_hi)
                sc_chi_src = cosmo.comoving_distance_dimensionless(
                    0.0, shells.z_mid)
            if weak_lensing.has_ia:
                src_overlap_w = proj.overlap_weights(weak_lensing.nz, z_lo, z_hi)
                src_zeff = proj.effective_redshift(weak_lensing.nz, z_lo, z_hi)

        gc_mag_w = None
        if galaxy_clustering is not None:
            gc_lin_w = proj.clustering_weights(
                cosmo, galaxy_clustering.nz, z_lo, z_hi)
            bias = galaxy_clustering.bias
            if bias.needs_tidal or bias.needs_quadratic_density:
                gc_overlap_w = proj.overlap_weights(
                    galaxy_clustering.nz, z_lo, z_hi)
            if galaxy_clustering.has_magnification:
                # Magnification convergence: lens n(z) acts as the source plane.
                gc_mag_w = proj.lensing_weights(
                    cosmo, galaxy_clustering.nz, z_lo, z_hi)

        if cmb_lensing is not None:
            cmb_w = proj.cmb_lensing_weights(
                cosmo, z_lo, z_hi, z_source=cmb_lensing.z_source)

        isw_w = None
        if isw is not None:
            isw_w = proj.isw_weights(cosmo, z_lo, z_hi)

        # -- allocate accumulators -----------------------------------------
        if weak_lensing is not None:
            n_src = weak_lensing.n_bins
            acc_kappa = np.zeros((n_src, npix))
            acc_ia1 = np.zeros((n_src, npix))
            acc_ia2 = np.zeros((n_src, npix))
            acc_src_delta = np.zeros((n_src, npix))
            # Running foreground convergence sums for source clustering's
            # non-isotropic n(z) (accumulated in redshift order below).
            sc_cum_alpha = (np.zeros(npix)
                            if weak_lensing.has_source_clustering else None)
            sc_cum_beta = (np.zeros(npix)
                           if weak_lensing.has_source_clustering else None)
        if galaxy_clustering is not None:
            n_lens = galaxy_clustering.n_bins
            acc_delta_g = np.zeros((n_lens, npix))
            acc_kappa_mag = (np.zeros((n_lens, npix))
                             if galaxy_clustering.has_magnification else None)
        if cmb_lensing is not None:
            acc_kappa_cmb = np.zeros(npix)
        if isw is not None:
            acc_isw_src = np.zeros(npix)

        # whether any probe needs the tidal field this run
        gc_tidal = (galaxy_clustering is not None
                    and galaxy_clustering.bias.needs_tidal)
        gc_quad = (galaxy_clustering is not None
                   and galaxy_clustering.bias.needs_quadratic_density)
        wl_ia = weak_lensing is not None and weak_lensing.has_ia

        # -- single pass over shells (redshift-ordered, for source clustering) --
        for i in np.argsort(shells.z_mid):
            delta = shells.overdensity(i)
            if delta is None:
                continue
            z_mid = 0.5 * (z_lo[i] + z_hi[i])
            if verbose:
                print(f"  shell {i + 1}/{shells.n_shells} "
                      f"z=[{z_lo[i]:.3f}, {z_hi[i]:.3f}]")

            # Weak-lensing convergence (linear)
            if weak_lensing is not None:
                for b in range(n_src):
                    if wl_w[i, b] != 0.0:
                        acc_kappa[b] += delta * wl_w[i, b]
                # Source clustering (linear)
                if weak_lensing.has_source_clustering:
                    # Projected source overdensity (feeds the shape-noise counts).
                    for b in range(n_src):
                        if src_lin_w[i, b] != 0.0:
                            bsrc = _coeff(weak_lensing.source_clustering_bias, b)
                            acc_src_delta[b] += delta * (src_lin_w[i, b] * bsrc)
                    # Non-isotropic n(z): sources in this shell are over/under-
                    # dense by (1 + b_src*delta), re-weighting the convergence to
                    # this source plane. kappa_plane uses only the foreground
                    # shells accumulated so far in this redshift-ordered pass.
                    chi_s = sc_chi_src[i]
                    if chi_s > 0.0:
                        kappa_plane = sc_cum_alpha - sc_cum_beta / chi_s
                        sc_signal = delta * kappa_plane
                        for b in range(n_src):
                            w_ov = sc_overlap[i, b]
                            if w_ov != 0.0:
                                bsrc = _coeff(
                                    weak_lensing.source_clustering_bias, b)
                                acc_kappa[b] += (bsrc * w_ov) * sc_signal
                    # Append this shell to the cumulative foreground convergence.
                    sc_cum_alpha += delta * sc_alpha[i]
                    sc_cum_beta += delta * sc_beta[i]

            # CMB lensing (linear)
            if cmb_lensing is not None and cmb_w[i] != 0.0:
                acc_kappa_cmb += delta * cmb_w[i]

            # ISW (linear; inverse-Laplacian applied once after the loop)
            if isw is not None and isw_w[i] != 0.0:
                acc_isw_src += delta * isw_w[i]

            # Linear galaxy clustering
            if galaxy_clustering is not None:
                for b in range(n_lens):
                    if gc_lin_w[i, b] != 0.0:
                        b1 = galaxy_clustering.bias.b1_for(b)
                        acc_delta_g[b] += delta * (gc_lin_w[i, b] * b1)
                # Magnification convergence (linear in delta)
                if acc_kappa_mag is not None:
                    for b in range(n_lens):
                        if gc_mag_w[i, b] != 0.0:
                            acc_kappa_mag[b] += delta * gc_mag_w[i, b]

            # Nonlinear terms (need tidal field), capped in redshift
            if z_mid > z_max_tidal or not (gc_tidal or gc_quad or wl_ia):
                continue

            if gc_tidal or wl_ia:
                s1, s2, s_sq = compute_tidal(delta, lmax=self.lmax)
            else:
                s_sq = None

            # Quadratic galaxy bias
            if gc_quad or gc_tidal:
                if s_sq is None:
                    s_sq_local = np.zeros_like(delta)
                else:
                    s_sq_local = s_sq
                _, delta2_ms, s2_ms = quadratic_bias_terms(delta, s_sq_local)
                for b in range(n_lens):
                    w_ov = gc_overlap_w[i, b]
                    if w_ov == 0.0:
                        continue
                    b2 = galaxy_clustering.bias.b2_for(b)
                    bs = galaxy_clustering.bias.bs_for(b)
                    acc_delta_g[b] += (b2 * delta2_ms + bs * s2_ms) * w_ov

            # TATT intrinsic alignments (exact field-level operators):
            #   I_ij = A1 s_ij + (b_TA A1 + A2) delta s_ij
            # The density-weighting (b_TA A1) and tidal-torquing (A2) operators
            # both project onto delta*s_ij in the 2D single-shell tidal field
            # (the only quadratic spin-2 combination of delta and s_ij), but
            # keep distinct redshift evolution via A1 ~ 1/D, A2 ~ 1/D^2.
            if wl_ia:
                ia = weak_lensing.ia
                ds1 = delta * s1
                ds2 = delta * s2
                ia_sc = weak_lensing.has_source_clustering
                for b in range(n_src):
                    w_ov = src_overlap_w[i, b]
                    if w_ov == 0.0:
                        continue
                    A1, A2 = ia.amplitudes(cosmo, src_zeff[i, b])
                    # IA x source clustering at strict second order: the local
                    # source overdensity reweights only the linear A1*s piece,
                    # adding b_src*A1*delta*s. Applying (1 + b_src*delta) to the
                    # already-quadratic delta*s piece would produce a 3rd-order
                    # term, which we drop for consistency with the perturbative
                    # truncation.
                    bsrc = (_coeff(weak_lensing.source_clustering_bias, b)
                            if ia_sc else 0.0)
                    quad = ia.b_ta * A1 + A2 + bsrc * A1
                    ia1 = A1 * s1 + quad * ds1
                    ia2 = A1 * s2 + quad * ds2
                    acc_ia1[b] += ia1 * w_ov
                    acc_ia2[b] += ia2 * w_ov

        # -- finalise: monopole subtraction + Kaiser-Squires ---------------
        result = ProjectedMaps()

        if weak_lensing is not None:
            for b in range(n_src):
                acc_kappa[b] -= acc_kappa[b].mean()
                acc_ia1[b] -= acc_ia1[b].mean()
                acc_ia2[b] -= acc_ia2[b].mean()
                acc_src_delta[b] -= acc_src_delta[b].mean()
            gamma1 = np.zeros_like(acc_kappa)
            gamma2 = np.zeros_like(acc_kappa)
            for b in range(n_src):
                gamma1[b], gamma2[b] = kappa_to_gamma(acc_kappa[b], lmax=self.lmax)
            result.weak_lensing = WeakLensingMaps(
                kappa=acc_kappa, gamma1=gamma1, gamma2=gamma2,
                ia_gamma1=acc_ia1, ia_gamma2=acc_ia2,
                source_overdensity=acc_src_delta,
            )

        if galaxy_clustering is not None:
            if acc_kappa_mag is not None:
                for b in range(n_lens):
                    s_b = _coeff(galaxy_clustering.magnification, b)
                    acc_delta_g[b] += (5.0 * s_b - 2.0) * acc_kappa_mag[b]
            for b in range(n_lens):
                acc_delta_g[b] -= acc_delta_g[b].mean()
            result.galaxy_clustering = GalaxyClusteringMaps(delta_g=acc_delta_g)

        if cmb_lensing is not None:
            acc_kappa_cmb -= acc_kappa_cmb.mean()
            result.cmb_lensing = CMBLensingMaps(kappa=acc_kappa_cmb)

        if isw is not None:
            # Apply 1 / [l(l+1)] in harmonic space once; the source
            # acc_isw_src already contains sum_i a_i * delta_i in real space.
            acc_isw_src -= acc_isw_src.mean()
            alm = hp.map2alm(acc_isw_src, lmax=self.lmax)
            ell = np.arange(self.lmax + 1)
            inv_lap = np.zeros_like(ell, dtype=float)
            nz = ell >= 1
            inv_lap[nz] = 1.0 / (ell[nz] * (ell[nz] + 1.0))
            theta = hp.alm2map(hp.almxfl(alm, inv_lap), nside=self.nside,
                               lmax=self.lmax)
            theta -= theta.mean()
            result.isw = ISWMaps(theta=theta)

        return result


def _coeff(value, bin_index):
    arr = np.asarray(value)
    return float(arr) if arr.ndim == 0 else float(arr[bin_index])
