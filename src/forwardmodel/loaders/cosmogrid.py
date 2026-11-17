"""Loaders for CosmoGrid N-body lightcone products.

CosmoGrid ships density shells as a ``.npz`` with a ``shells`` array of
per-pixel particle counts and a structured ``shell_info`` array carrying the
redshift bounds. ``params.yml`` holds the fiducial cosmology.
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


# ---------------------------------------------------------------------------
# CosmoGrid v1.1 (``baryonified_shells_v11.h5``, nside 1024)
# ---------------------------------------------------------------------------

def load_v11_params(path):
    """``sim_params`` row (numpy structured) of a v1.1 ``baryonified_shells_v11.h5``."""
    import h5py

    with h5py.File(Path(path), "r") as f:
        return f["sim_params"][0]


def load_v11_shells(path, baryonified=False, metainfo=None, shell_info=None,
                    return_params=False):
    """Load CosmoGrid v1.1 shells into a :class:`ShellStack` (count form, map-mean normalised).

    The file holds DMO particle counts (``nobaryon_shells/shellNNN``, uint16, nside 1024) and the
    baryonification as **increments** at a subset of pixels (``diff_shell_inds`` / ``diff_shell_vals``):
    baryonified = DMO + increment. Shell redshift bounds are not in the file: pass either the v1.1
    ``metainfo`` file (read from ``shell_info/<path_par>``, ``path_par`` from ``sim_params``) or
    ``shell_info`` (a structured array with ``lower_z, upper_z``).

    Parameters
    ----------
    path : str or Path
        ``.../run_XXXX/baryonified_shells_v11.h5``.
    baryonified : bool
        Add the baryonification increments (float32 counts) instead of the DMO counts (uint16).
    metainfo : str or Path, optional
        v1.1 metainfo file (``CosmoGridV11_bary_Fluri+2022_metainfo.h5``, distributed with the
        v1.1 shells). Required unless ``shell_info`` is given.
    shell_info : structured array, optional
        Shell redshift bounds (``lower_z, upper_z``), one row per shell; takes precedence over
        ``metainfo``.
    return_params : bool
        Also return the ``sim_params`` row (for :func:`cosmology_from_params`).
    """
    import h5py

    if shell_info is None and metainfo is None:
        raise ValueError("load_v11_shells needs the shell redshift bounds: pass metainfo= (path to "
                         "the CosmoGrid v1.1 metainfo .h5) or shell_info=")
    with h5py.File(Path(path), "r") as f:
        p = f["sim_params"][0]
        keys = sorted(f["nobaryon_shells"].keys())
        if baryonified:
            counts = np.empty((len(keys), f["nobaryon_shells"][keys[0]].shape[0]), np.float32)
        else:
            counts = np.empty((len(keys), f["nobaryon_shells"][keys[0]].shape[0]), np.uint16)
        for s, k in enumerate(keys):
            counts[s] = f["nobaryon_shells"][k][:]
            if baryonified:
                counts[s, f["diff_shell_inds"][k][:]] += f["diff_shell_vals"][k][:]
    if shell_info is None:
        with h5py.File(Path(metainfo), "r") as m:
            par = p["path_par"].decode() if isinstance(p["path_par"], bytes) else str(p["path_par"])
            shell_info = m[f"shell_info/{par}"][:]
    if len(shell_info) != len(keys):
        raise ValueError(f"{len(keys)} shells in file, {len(shell_info)} in shell_info")
    stack = ShellStack.from_counts(counts, shell_info["lower_z"].astype(float),
                                   shell_info["upper_z"].astype(float))
    return (stack, p) if return_params else stack
