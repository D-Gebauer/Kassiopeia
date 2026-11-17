"""Probe definitions and projected-map containers.

A *probe* describes one observable to forward-model from the shells:

* :class:`WeakLensing` — convergence/shear for a set of source bins, with an
  optional intrinsic-alignment model and optional source clustering.
* :class:`GalaxyClustering` — projected galaxy overdensity for lens bins, with
  a linear or quadratic bias model.
* :class:`CMBLensing` — CMB convergence to a single source plane.
* :class:`ISW` — late-time integrated Sachs-Wolfe temperature anisotropy.

:meth:`forwardmodel.ForwardModel.project` consumes probes and returns the
matching ``*Maps`` containers below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence
import numpy as np

from .nz import NumberDensity
from .bias import LinearBias
from .ia import TATT


# ---------------------------------------------------------------------------
# Probe specifications
# ---------------------------------------------------------------------------

@dataclass
class WeakLensing:
    """Weak-lensing probe over one or more source tomographic bins.

    Parameters
    ----------
    nz : sequence of NumberDensity
        Source redshift distributions, one per bin.
    ia : TATT or NLA, optional
        Intrinsic-alignment model. ``None`` disables IA.
    source_clustering_bias : float or array_like, optional
        Linear bias coupling sources to the density field, enabling source
        clustering. Sources cluster with structure, so each line of sight sees a
        different effective n(z); this (a) biases the convergence *signal*,
        (b) adds the IA x source-clustering cross term ``b_src * A1 * delta *
        s_ij`` — the second-order coupling of the local source overdensity to
        the linear intrinsic-alignment field (only when an ``ia`` model is also
        set), and (c) drives density-correlated shape noise through the
        projected source overdensity. ``None`` disables it.
    """

    nz: Sequence[NumberDensity]
    ia: Optional[TATT] = None
    source_clustering_bias: object = None

    @property
    def n_bins(self):
        return len(self.nz)

    @property
    def has_ia(self):
        return self.ia is not None

    @property
    def has_source_clustering(self):
        return self.source_clustering_bias is not None


@dataclass
class GalaxyClustering:
    """Galaxy-clustering probe over one or more lens tomographic bins.

    Parameters
    ----------
    nz : sequence of NumberDensity
        Lens redshift distributions, one per bin.
    bias : LinearBias or QuadraticBias
        Galaxy bias model.
    magnification : float or array_like, optional
        Lens-sample number-count slope ``s = d log10 N(<m) / dm``, scalar or
        per bin. Enables magnification bias: the observed overdensity gains
        ``(5 s - 2) * kappa_lens``, where ``kappa_lens`` is the convergence
        projected with this bin's own n(z) as the source plane. ``None``
        disables it; ``s = 0.4`` gives a vanishing coefficient (no effect).
    """

    nz: Sequence[NumberDensity]
    bias: object = field(default_factory=LinearBias)
    magnification: object = None

    @property
    def n_bins(self):
        return len(self.nz)

    @property
    def has_magnification(self):
        return self.magnification is not None


@dataclass
class CMBLensing:
    """CMB-lensing probe to a single source plane.

    Parameters
    ----------
    z_source : float
        Redshift of the CMB last-scattering surface (default 1090).
    """

    z_source: float = 1090.0


@dataclass
class ISW:
    """Integrated Sachs-Wolfe temperature anisotropy.

    Late-time ISW from time-evolving gravitational potentials along the line of
    sight. In Limber + linear theory the harmonic coefficients are

        Theta_lm = sum_i a_i / [l(l+1)] * delta_lm^(i)

    with the per-shell scalar coefficient supplied by
    :func:`forwardmodel.projections.isw_weights`. Matter domination gives
    ``f = 1`` exactly and the kernel vanishes there — only shells where dark
    energy makes the potentials decay contribute.

    The shell stack only needs to cover the redshift range where ``(1-f)`` is
    appreciable (typically ``z < ~2``); contributions above that are tiny.
    Early ISW (around recombination) is *not* modelled here.
    """

    pass


# ---------------------------------------------------------------------------
# Projected-map containers
# ---------------------------------------------------------------------------

@dataclass
class WeakLensingMaps:
    """Noiseless weak-lensing maps, shape ``(n_bins, npix)`` each.

    Attributes
    ----------
    kappa : ndarray
        Gravitational convergence.
    gamma1, gamma2 : ndarray
        Shear from ``kappa`` via Kaiser-Squires.
    ia_gamma1, ia_gamma2 : ndarray
        Intrinsic-alignment shear (zeros if no IA model).
    source_overdensity : ndarray
        Projected source galaxy overdensity (zeros if source clustering off).
    ia_kappa : ndarray or None
        Scalar IA field ``sum_i A1 w_i delta_i`` whose Kaiser-Squires shear is
        ``ia_gamma1/2`` (only for linear IA on the fast path, else ``None``).
    """

    kappa: np.ndarray
    gamma1: np.ndarray
    gamma2: np.ndarray
    ia_gamma1: np.ndarray
    ia_gamma2: np.ndarray
    source_overdensity: np.ndarray
    ia_kappa: Optional[np.ndarray] = None

    @property
    def n_bins(self):
        return self.kappa.shape[0]

    def shear(self, with_ia=True):
        """Return ``(gamma1, gamma2)``, optionally adding the IA contribution."""
        if with_ia:
            return self.gamma1 + self.ia_gamma1, self.gamma2 + self.ia_gamma2
        return self.gamma1, self.gamma2


@dataclass
class GalaxyClusteringMaps:
    """Projected galaxy overdensity, shape ``(n_bins, npix)``."""

    delta_g: np.ndarray

    @property
    def n_bins(self):
        return self.delta_g.shape[0]


@dataclass
class CMBLensingMaps:
    """Projected CMB convergence, shape ``(npix,)``."""

    kappa: np.ndarray


@dataclass
class ISWMaps:
    """Projected ISW temperature anisotropy ``Delta T / T_CMB``, shape ``(npix,)``."""

    theta: np.ndarray


@dataclass
class ProjectedMaps:
    """Bundle of whichever probes were projected."""

    weak_lensing: Optional[WeakLensingMaps] = None
    galaxy_clustering: Optional[GalaxyClusteringMaps] = None
    cmb_lensing: Optional[CMBLensingMaps] = None
    isw: Optional[ISWMaps] = None
