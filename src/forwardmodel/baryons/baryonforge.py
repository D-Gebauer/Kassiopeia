"""BaryonForge baryonification of dark-matter-only density shells.

Wraps BaryonForge's ``BaryonifyShell`` runner so it accepts a generic
:class:`~forwardmodel.ShellStack` and a halo light-cone catalogue. Halos are
matched to shells by redshift, so this works with any shell binning (not just
CosmoGrid's shell ids).

BaryonForge 1.x targets ``numpy<2`` and ``pyccl<3.1``; the compatibility shims
below restore the handful of APIs it relies on, so it runs against the modern
stack this package otherwise uses. Because of those pins it has to be installed
without its dependencies: ``pip install --no-deps BaryonForge`` (the ``baryons``
extra provides the remaining ones).
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import numpy as np
import pyccl as ccl
from scipy.interpolate import CubicSpline

# -- compatibility shims (must run before importing BaryonForge) ------------
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # numpy 2.0 removed np.trapz
from pyccl.halos.concentration import Concentration as _Concentration
if not hasattr(_Concentration, "get_concentration"):
    _Concentration.get_concentration = lambda self, cosmo, M, a: self(cosmo, M, a)


def _accept_mdef(cls):
    """Let a pyccl >= 3 concentration class take BaryonForge's ``mdef=`` (renamed ``mass_def``)."""
    init = cls.__dict__.get("__init__")
    if init is None or getattr(init, "_accepts_mdef", False):
        return
    names = inspect.signature(init).parameters
    if "mdef" in names or "mass_def" not in names:
        return

    @functools.wraps(init)
    def wrapped(self, *args, mdef=None, **kwargs):
        if mdef is not None:
            kwargs.setdefault("mass_def", mdef)
        return init(self, *args, **kwargs)

    wrapped._accepts_mdef = True
    cls.__init__ = wrapped


def _with_subclasses(cls):
    yield cls
    for sub in cls.__subclasses__():
        yield from _with_subclasses(sub)


for _cls in list(_with_subclasses(_Concentration)):
    _accept_mdef(_cls)
_orig_corr3d = ccl.correlation_3d
def _corr3d_compat(cosmo, a_or_r=None, r_or_a=None, **kw):
    if "r" in kw or "a" in kw:
        return _orig_corr3d(cosmo, **kw)
    return _orig_corr3d(cosmo, a=a_or_r, r=r_or_a)  # BaryonForge positional (cosmo, a, r)
ccl.correlation_3d = _corr3d_compat

# Schneider+19 fiducial baryonification parameters.
DEFAULT_BARYON_PARAMS = {
    "epsilon_max": 20, "epsilon": 4.0, "theta_ej": 4, "theta_co": 0.1,
    "M_c": 1e14, "mu_beta": 0.1, "gamma": 2.5, "delta": 7,
    "mu_theta_ej": 0.1, "mu_theta_co": 0.0, "M_theta_ej": 5e13,
    "M_theta_co": 5e13, "M_gamma": 5e13, "M_delta": 5e13, "eta": 0.3,
    "eta_delta": 0.1, "tau": -1.5, "tau_delta": 0, "A": 0.055, "M1": 3e11,
    "epsilon_h": 0.015, "a": 0.3, "n": 2.0, "p": 0.3, "q": 0.707,
    "cdelta": None, "alpha_nt": 0.2, "nu_nt": 0.5, "gamma_nt": 0.5,
    "cutoff": np.inf, "proj_cutoff": 1000,
}


@dataclass
class HaloCatalog:
    """A halo light-cone catalogue in observer frame.

    Parameters
    ----------
    ra, dec : ndarray
        Sky positions in degrees.
    M : ndarray
        Halo masses in physical solar masses (``M_sun``, not ``M_sun/h``).
    z : ndarray
        Halo redshifts.
    """

    ra: np.ndarray
    dec: np.ndarray
    M: np.ndarray
    z: np.ndarray

    @classmethod
    def from_cartesian(cls, x, y, z, mass, cosmo, length_unit_mpc=1e-3,
                       mass_is_per_h=True):
        """Build from observer-centred Cartesian coordinates.

        Parameters
        ----------
        x, y, z : ndarray
            Comoving positions. CosmoGrid stores these in kpc/h, so the default
            ``length_unit_mpc=1e-3`` converts kpc->Mpc; positions are also
            divided by ``h`` to reach physical Mpc.
        mass : ndarray
            Halo mass; divided by ``h`` if ``mass_is_per_h``.
        cosmo : forwardmodel.Cosmology
        """
        h = cosmo.h
        x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
        r = np.sqrt(x * x + y * y + z * z)
        dec = np.degrees(np.arcsin(z / r))
        ra = np.degrees(np.arctan2(y, x)) % 360.0

        chi_mpc = r * length_unit_mpc / h
        z_grid = np.linspace(0.0, 5.0, 4096)
        chi_grid = ccl.comoving_radial_distance(cosmo.ccl, 1.0 / (1.0 + z_grid))
        z_halo = CubicSpline(chi_grid, z_grid)(chi_mpc)

        M = np.asarray(mass, float) / h if mass_is_per_h else np.asarray(mass, float)
        return cls(ra=ra, dec=dec, M=M, z=z_halo)


def _cosmo_dict(cosmo):
    return {
        "Omega_m": cosmo.Omega_m, "Omega_b": cosmo.Omega_b, "h": cosmo.h,
        "sigma8": float(cosmo["sigma8"]), "n_s": float(cosmo["n_s"]),
        "w0": float(cosmo["w0"]),
    }


def _build_model(cosmo, nside, params, z_min=1e-3, z_max=3.5, n_z=16):
    import BaryonForge as bfg
    from BaryonForge.Profiles import (
        Baryonification2D, DarkMatterOnly, DarkMatterBaryon,
    )

    cd = _cosmo_dict(cosmo)
    ccl_cosmo = ccl.Cosmology(
        Omega_c=cd["Omega_m"] - cd["Omega_b"], Omega_b=cd["Omega_b"],
        h=cd["h"], sigma8=cd["sigma8"], n_s=cd["n_s"], w0=cd["w0"],
        matter_power_spectrum="linear",
    )
    ccl_cosmo.compute_sigma()

    DMO = bfg.ConvolvedProfile(DarkMatterOnly(**params), bfg.HealPixel(nside))
    DMB = bfg.ConvolvedProfile(DarkMatterBaryon(**params), bfg.HealPixel(nside))
    model = Baryonification2D(DMO=DMO, DMB=DMB, cosmo=ccl_cosmo,
                              epsilon_max=params["epsilon_max"])
    model.setup_interpolator(
        z_min=z_min, z_max=z_max, N_samples_z=n_z, z_linear_sampling=True,
        R_min=1e-4, R_max=300.0, N_samples_R=2000, verbose=False,
    )
    return model


def baryonify(shells, halos, cosmo, params=None, verbose=True):
    """Baryonify a count :class:`ShellStack` using a halo light-cone catalogue.

    Halos are assigned to shells by redshift (``z_lower <= z_halo < z_upper``);
    each shell is then displaced by BaryonForge's ``BaryonifyShell`` runner.

    Parameters
    ----------
    shells : forwardmodel.ShellStack
        Count shells (dark-matter-only).
    halos : HaloCatalog
        Halo light cone.
    cosmo : forwardmodel.Cosmology
    params : dict, optional
        Baryonification parameters. Defaults to :data:`DEFAULT_BARYON_PARAMS`.
    verbose : bool

    Returns
    -------
    ShellStack
        New count stack with baryonified maps (float32).
    """
    from BaryonForge.utils import HaloLightConeCatalog, LightconeShell
    from BaryonForge.Runners import BaryonifyShell
    from ..shells import ShellStack

    if shells.kind != "counts":
        raise ValueError("baryonify expects a count ShellStack")
    params = {**DEFAULT_BARYON_PARAMS, **(params or {})}
    nside = shells.nside
    cdict = _cosmo_dict(cosmo)

    cat = HaloLightConeCatalog(ra=halos.ra, dec=halos.dec, M=halos.M,
                               z=halos.z, cosmo=cdict)
    model = _build_model(cosmo, nside, params)
    epsilon_max = params["epsilon_max"]

    out = np.empty((shells.n_shells, shells.npix), dtype=np.float32)
    for i in range(shells.n_shells):
        z_lo, z_hi = shells.z_lower[i], shells.z_upper[i]
        in_shell = (halos.z >= z_lo) & (halos.z < z_hi)
        shell_map = np.asarray(shells.data[i], dtype=np.float64)
        if verbose:
            print(f"  shell {i + 1}/{shells.n_shells} "
                  f"z=[{z_lo:.3f}, {z_hi:.3f}] Nhalo={int(in_shell.sum())}")
        if in_shell.sum() == 0:
            out[i] = shell_map.astype(np.float32)
            continue
        runner = BaryonifyShell(
            HaloLightConeCatalog=cat[in_shell],
            LightconeShell=LightconeShell(map=shell_map, cosmo=cdict),
            epsilon_max=epsilon_max, model=model, verbose=verbose,
        )
        out[i] = runner.process().astype(np.float32)

    return ShellStack.from_counts(out, shells.z_lower, shells.z_upper,
                                  nside=nside)
