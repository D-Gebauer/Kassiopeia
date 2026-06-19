"""Cosmology: a thin wrapper around ``pyccl.Cosmology`` with cached interpolants.

The wrapper exists so the rest of the package can ask for comoving distance,
growth, and the dimensionless Hubble rate at arbitrary redshift *fast* — the
projection kernels evaluate these on dense grids and per-element ``pyccl`` calls
are far too slow. On construction we tabulate ``chi(z)``, ``E(z)`` and
``1/E(z)`` once on a dense grid and serve cubic-spline interpolants from then
on.

Construct it either from scalar parameters::

    cosmo = Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)

or by wrapping an existing CCL object::

    cosmo = Cosmology.from_ccl(my_ccl_cosmology)
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl
from scipy.interpolate import interp1d

C_KM_S = 299792.458  # speed of light [km/s]


class Cosmology:
    """A cached cosmology backed by ``pyccl``.

    Parameters
    ----------
    Omega_m : float, optional
        Total matter density. If given, ``Omega_c`` is derived as
        ``Omega_m - Omega_b``. Provide either ``Omega_m`` or ``Omega_c``.
    Omega_c : float, optional
        Cold dark matter density.
    Omega_b : float
        Baryon density.
    h : float
        Dimensionless Hubble parameter ``H0 / (100 km/s/Mpc)``.
    sigma8, n_s : float
        Power-spectrum amplitude and tilt. Provide ``sigma8`` or ``A_s``.
    A_s : float, optional
        Primordial amplitude (alternative to ``sigma8``).
    w0, wa : float
        Dark-energy equation-of-state parameters (default ``-1, 0``).
    m_nu : float
        Summed neutrino mass in eV (default ``0``).
    transfer_function, matter_power_spectrum : str
        Passed through to ``pyccl.Cosmology``.
    extra_ccl_kwargs : dict, optional
        Any other keyword arguments forwarded verbatim to ``pyccl.Cosmology``.
    z_max, n_grid : float, int
        Range and resolution of the cached interpolation grid.
    """

    def __init__(
        self,
        Omega_m=None,
        Omega_c=None,
        Omega_b=0.0486,
        h=0.6774,
        sigma8=None,
        n_s=0.9667,
        A_s=None,
        w0=-1.0,
        wa=0.0,
        m_nu=0.0,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="halofit",
        extra_ccl_kwargs=None,
        z_max=1200.0,
        n_grid=4000,
    ):
        if Omega_c is None:
            if Omega_m is None:
                raise ValueError("Provide either Omega_m or Omega_c.")
            Omega_c = Omega_m - Omega_b
        if (sigma8 is None) == (A_s is None):
            raise ValueError("Provide exactly one of sigma8 or A_s.")

        kwargs = dict(
            Omega_c=Omega_c,
            Omega_b=Omega_b,
            h=h,
            n_s=n_s,
            w0=w0,
            wa=wa,
            m_nu=m_nu,
            transfer_function=transfer_function,
            matter_power_spectrum=matter_power_spectrum,
        )
        if sigma8 is not None:
            kwargs["sigma8"] = sigma8
        else:
            kwargs["A_s"] = A_s
        if m_nu != 0.0:
            kwargs.setdefault("mass_split", "equal")
        if extra_ccl_kwargs:
            kwargs.update(extra_ccl_kwargs)

        ccl_cosmo = ccl.Cosmology(**kwargs)
        self._init_from_ccl(ccl_cosmo, z_max=z_max, n_grid=n_grid)

    # -- alternative constructors -------------------------------------------

    @classmethod
    def from_ccl(cls, ccl_cosmo, z_max=1200.0, n_grid=4000):
        """Wrap an existing ``pyccl.Cosmology`` instance."""
        self = cls.__new__(cls)
        self._init_from_ccl(ccl_cosmo, z_max=z_max, n_grid=n_grid)
        return self

    def _init_from_ccl(self, ccl_cosmo, z_max=1200.0, n_grid=4000):
        self.ccl = ccl_cosmo
        self.h = float(ccl_cosmo["h"])
        self.H0 = 100.0 * self.h  # km/s/Mpc
        self.Omega_m = float(ccl_cosmo["Omega_m"])
        self.Omega_b = float(ccl_cosmo["Omega_b"])
        self.H0_over_c = self.H0 / C_KM_S  # 1/Mpc
        self._z_max = z_max

        # Dense grid: fine at low z (where lensing kernels peak), coarse high z.
        z_lo = np.linspace(0.0, 10.0, 2000)
        z_hi = np.linspace(10.0, z_max, max(n_grid - 2000, 200))[1:]
        z = np.concatenate([z_lo, z_hi])
        a = 1.0 / (1.0 + z)
        chi = ccl.comoving_radial_distance(ccl_cosmo, a)
        E = ccl.h_over_h0(ccl_cosmo, a)
        self._chi = interp1d(z, chi, kind="cubic",
                             bounds_error=False, fill_value=(0.0, chi[-1]))
        self._invE = interp1d(z, 1.0 / E, kind="cubic",
                              bounds_error=False, fill_value=(1.0, 1.0 / E[-1]))
        self._E = interp1d(z, E, kind="cubic",
                           bounds_error=False, fill_value=(1.0, E[-1]))

    # -- accessors ----------------------------------------------------------

    def __getitem__(self, key):
        """Forward dict-style access to the underlying CCL cosmology."""
        return self.ccl[key]

    def comoving_distance(self, z):
        """Comoving radial distance ``chi(z)`` in Mpc (cached)."""
        return self._chi(np.asarray(z, dtype=float))

    def comoving_distance_dimensionless(self, z_lo, z_hi):
        """``(chi(z_hi) - chi(z_lo)) * H0/c`` — dimensionless distance."""
        lo = self._chi(np.asarray(z_lo, dtype=float))
        hi = self._chi(np.asarray(z_hi, dtype=float))
        return (hi - lo) * self.H0_over_c

    def inv_efunc(self, z):
        """``1 / E(z)`` where ``E(z) = H(z)/H0`` (cached)."""
        return self._invE(np.asarray(z, dtype=float))

    def efunc(self, z):
        """``E(z) = H(z) / H0`` (cached)."""
        return self._E(np.asarray(z, dtype=float))

    def hubble(self, z):
        """``H(z)`` in km/s/Mpc (cached)."""
        return self.H0 * self._E(np.asarray(z, dtype=float))

    def growth_factor(self, z):
        """Linear growth factor ``D(z)`` (direct CCL call)."""
        return ccl.growth_factor(self.ccl, 1.0 / (1.0 + np.asarray(z, float)))

    def growth_rate(self, z):
        """Linear growth rate ``f(z) = d ln D / d ln a`` (direct CCL call).

        Equals 1 in matter domination and drops below 1 once dark energy
        kicks in — the deficit ``1 - f`` is what sources the (late-time) ISW.
        """
        return ccl.growth_rate(self.ccl, 1.0 / (1.0 + np.asarray(z, float)))

    def __repr__(self):
        return (f"Cosmology(Omega_m={self.Omega_m:.4f}, "
                f"Omega_b={self.Omega_b:.4f}, h={self.h:.4f})")
