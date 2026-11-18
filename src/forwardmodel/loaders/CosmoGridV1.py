"""Loaders for CosmoGridV1 N-body lightcone products.

CosmoGridV1 ships density shells as a ``.npz`` with a ``shells`` array of
per-pixel particle counts and a structured ``shell_info`` array carrying the
redshift bounds (``compressed_shells.npz``; ``baryonified_shells.npz`` holds the
baryonified shells in the same layout). ``params.yml`` holds the cosmology.

For CosmoGrid v1.1 (``baryonified_shells_v11.h5``) see
:mod:`forwardmodel.loaders.CosmoGridV1_1`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..cosmology import Cosmology
from ..shells import ShellStack


def cosmology_from_params(p, n_massive_nu=3, check_omega_m=True, rtol_omega_m=1e-4,
                          **overrides):
    """Build a :class:`~forwardmodel.Cosmology` from CosmoGrid parameters.

    Parameters
    ----------
    p : mapping or numpy structured row
        Needs ``O_cdm, Ob, H0, s8, ns``; optional ``w0, wa, m_nu, Om``. Works for a v1
        ``params.yml`` dict and for a v1.1 ``sim_params`` row.
    n_massive_nu : int
        CosmoGrid's ``m_nu`` is the mass **per species** of ``n_massive_nu`` degenerate
        neutrinos (v1 and v1.1: 0.02 eV x 3 = 0.06 eV; ``O_nu`` and ``Om`` include them).
        CCL takes the summed mass, so ``m_nu_total = n_massive_nu * m_nu`` with
        ``mass_split="equal"``. ``n_massive_nu=1`` reproduces the pre-2026-11-04 behaviour,
        which passed the per-species mass as the sum (CCL Omega_m 0.4 % low at the fiducial).
    check_omega_m : bool
        Raise if CCL's total Omega_m differs from the file's ``Om`` by more than
        ``rtol_omega_m`` (relative).
    **overrides
        Forwarded to :class:`Cosmology` (e.g. ``transfer_function``, ``A_s`` instead of s8).
    """
    get = (lambda k, d=None: p[k] if k in p.dtype.names else d) if hasattr(p, "dtype") \
        else (lambda k, d=None: p.get(k, d))
    kwargs = dict(
        Omega_c=float(get("O_cdm")),
        Omega_b=float(get("Ob")),
        h=float(get("H0")) / 100.0,
        sigma8=float(get("s8")),
        n_s=float(get("ns")),
        w0=float(get("w0", -1.0)),
        wa=float(get("wa", 0.0)),
        m_nu=n_massive_nu * float(get("m_nu", 0.0)),
    )
    if "A_s" in overrides:
        kwargs.pop("sigma8")
    kwargs.update(overrides)
    cosmo = Cosmology(**kwargs)
    om_file = get("Om")
    if check_omega_m and om_file is not None:
        rel = abs(cosmo.Omega_m / float(om_file) - 1.0)
        if rel > rtol_omega_m:
            raise ValueError(
                f"CCL Omega_m = {cosmo.Omega_m:.6f} but CosmoGrid Om = {float(om_file):.6f} "
                f"(rel. diff {rel:.1e}); check n_massive_nu / m_nu convention")
    return cosmo


def load_cosmology(params_path, n_massive_nu=3, check_omega_m=True, **overrides):
    """Build a :class:`~forwardmodel.Cosmology` from a CosmoGrid ``params.yml``.

    Parameters
    ----------
    params_path : str or Path
        Path to ``params.yml`` with keys ``O_cdm, Ob, H0, s8, ns, w0, wa, m_nu`` (and ``Om``).
    n_massive_nu, check_omega_m
        See :func:`cosmology_from_params` (``m_nu`` in the file is per species).
    **overrides
        Extra keyword arguments forwarded to :class:`Cosmology` (e.g.
        ``transfer_function``, ``matter_power_spectrum``).
    """
    import yaml

    with open(params_path) as f:
        p = yaml.safe_load(f)
    return cosmology_from_params(p, n_massive_nu=n_massive_nu,
                                 check_omega_m=check_omega_m, **overrides)


def load_shells(path, normalization="map_mean", box_size=None,
                n_particles=None, cosmo=None):
    """Load CosmoGrid count shells into a :class:`ShellStack`.

    Parameters
    ----------
    path : str or Path
        ``.npz`` file with ``shells`` (n_shells, npix) and a ``shell_info``
        structured array providing ``lower_z`` and ``upper_z``.
    normalization : {"map_mean", "box"}
        Count-to-overdensity normalization. Pass ``"box"`` with ``box_size``
        (Mpc/h, e.g. 900), ``n_particles`` (e.g. ``832**3``) and ``cosmo`` for
        the UFalcon/CosmoGrid box-volume amplitude; the default ``"map_mean"``
        normalizes each shell by its own mean and needs none of those.

    Returns
    -------
    ShellStack
    """
    data = np.load(Path(path), allow_pickle=True)
    shells = data["shells"]
    info = data["shell_info"]
    return ShellStack.from_counts(
        shells, info["lower_z"], info["upper_z"],
        normalization=normalization, box_size=box_size,
        n_particles=n_particles, cosmo=cosmo)


def load_halo_catalog(path):
    """Load the CosmoGrid halo lightcone catalogue (structured array).

    Returns the raw structured array (fields ``x, y, z, Mvir, shell_id, ...``);
    used by :mod:`forwardmodel.baryons`.
    """
    data = np.load(Path(path), allow_pickle=True)
    return data["halos"]
