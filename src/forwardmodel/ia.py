"""Intrinsic-alignment models: NLA and TATT.

Both produce a spin-2 IA field from the tidal field ``s_ij`` and the density
``delta``. The TATT amplitudes follow the standard convention::

    A_1(z) = -a_1 C1 rho_crit Omega_m / D(z)   * ((1+z)/(1+z0))^eta1
    A_2(z) =  5 a_2 C1 rho_crit Omega_m / D(z)^2 * ((1+z)/(1+z0))^eta2

This package uses the exact field-level TATT operator structure (Blazek+19)::

    I_ij = A_1 s_ij + A_1 b_TA (delta s_ij) + A_2 (s_ik s_kj)^TF.

In the 2D single-shell (Limber) tidal field used here, the density-weighting
term ``A_1 b_TA delta s_ij`` and the tidal-torquing term ``A_2 (s_ik s_kj)^TF``
both reduce to ``delta s_ij`` — it is the only quadratic spin-2 combination of
``delta`` and ``s_ij`` (the trace-free square of a 2D tidal tensor with trace
``delta`` is ``delta s_ij``; the strictly trace-free 2D square vanishes). They
are therefore combined as ``(b_TA A_1 + A_2) delta s_ij`` but keep their
distinct redshift evolution through ``A_1 ~ 1/D`` and ``A_2 ~ 1/D^2``. The full
3D torquing shape needs the line-of-sight tidal components, which a single
projected shell does not carry. NLA is the special case ``a_2 = b_TA = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

# C1 * rho_crit in M_sun h^2 / Mpc^3 (standard IA normalisation).
C1_RHOCRIT = 0.0134


@dataclass
class TATT:
    """Tidal Alignment + Tidal Torquing intrinsic-alignment model.

    Parameters
    ----------
    a1 : float
        Linear (NLA) alignment amplitude.
    a2 : float
        Quadratic (torquing) amplitude.
    eta1, eta2 : float
        Redshift power-law evolution exponents for ``a1`` and ``a2``.
    b_ta : float
        Tidal-alignment density weighting for the quadratic term.
    z_pivot : float
        Pivot redshift ``z0`` in the evolution factor (default 0.62, DES Y3).
    """

    a1: float = 0.0
    a2: float = 0.0
    eta1: float = 0.0
    eta2: float = 0.0
    b_ta: float = 0.0
    z_pivot: float = 0.62

    def amplitudes(self, cosmo, z):
        """Return ``(A1, A2)`` at redshift ``z`` for the given cosmology."""
        z = float(z)
        D = cosmo.growth_factor(z)
        Om = cosmo.Omega_m
        evo = (1.0 + z) / (1.0 + self.z_pivot)
        A1 = -self.a1 * C1_RHOCRIT * Om / D * evo**self.eta1
        A2 = 5.0 * self.a2 * C1_RHOCRIT * Om / D**2 * evo**self.eta2
        return A1, A2

    def field(self, s1, s2, delta, cosmo, z):
        """Spin-2 IA field ``(I1, I2)`` for one shell.

        Exact field-level TATT in the 2D tidal field:
        ``I_ij = A1 s_ij + (b_TA A1 + A2) delta s_ij`` (see module docstring).
        ``delta`` is the shell overdensity; pass zeros for the pure-NLA limit.
        """
        A1, A2 = self.amplitudes(cosmo, z)
        quad = self.b_ta * A1 + A2
        if quad == 0.0:
            return A1 * s1, A1 * s2
        return A1 * s1 + quad * (delta * s1), A1 * s2 + quad * (delta * s2)


@dataclass
class NLA(TATT):
    """Nonlinear Linear Alignment model — TATT restricted to the linear term.

    Parameters
    ----------
    a1 : float
        Alignment amplitude.
    eta1 : float
        Redshift evolution exponent.
    z_pivot : float
        Pivot redshift (default 0.62).
    """

    def __init__(self, a1=0.0, eta1=0.0, z_pivot=0.62):
        super().__init__(a1=a1, a2=0.0, eta1=eta1, eta2=0.0, b_ta=0.0,
                         z_pivot=z_pivot)


def tatt_amplitudes(cosmo, z, a1, a2, eta1, eta2, z_pivot=0.62):
    """Standalone TATT amplitude functions ``(A1, A2)`` (low-level API)."""
    return TATT(a1=a1, a2=a2, eta1=eta1, eta2=eta2, z_pivot=z_pivot).amplitudes(
        cosmo, z)
