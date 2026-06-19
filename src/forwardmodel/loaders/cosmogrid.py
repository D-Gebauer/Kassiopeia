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


def load_cosmology(params_path, **overrides):
    """Build a :class:`~forwardmodel.Cosmology` from a CosmoGrid ``params.yml``.

    Parameters
    ----------
    params_path : str or Path
        Path to ``params.yml`` with keys ``O_cdm, Ob, H0, s8, ns, w0, wa, m_nu``.
    **overrides
        Extra keyword arguments forwarded to :class:`Cosmology` (e.g.
        ``transfer_function``, ``matter_power_spectrum``).
    """
    import yaml

    with open(params_path) as f:
        p = yaml.safe_load(f)

    kwargs = dict(
        Omega_c=p["O_cdm"],
        Omega_b=p["Ob"],
        h=p["H0"] / 100.0,
        sigma8=p["s8"],
        n_s=p["ns"],
        w0=p.get("w0", -1.0),
        wa=p.get("wa", 0.0),
        m_nu=p.get("m_nu", 0.0),
    )
    kwargs.update(overrides)
    return Cosmology(**kwargs)


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
