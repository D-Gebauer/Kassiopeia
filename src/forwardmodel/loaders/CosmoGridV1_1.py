"""Loaders for CosmoGrid v1.1 lightcone products (``baryonified_shells_v11.h5``, nside 1024).

One HDF5 file per simulation holds the dark-matter-only particle counts
(``nobaryon_shells/shellNNN``), the baryonification as increments at a subset of
pixels (``diff_shell_inds`` / ``diff_shell_vals``) and the cosmology
(``sim_params``). The shell redshift bounds are in the separate v1.1 metainfo file.

For CosmoGridV1 (``.npz`` shells, ``params.yml``) see
:mod:`forwardmodel.loaders.CosmoGridV1`.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np

from ..shells import ShellStack
from .CosmoGridV1 import cosmology_from_params  # noqa: F401  (same parameter names as v1)


def load_params(path):
    """``sim_params`` row (numpy structured) of a v1.1 ``baryonified_shells_v11.h5``."""
    import h5py

    with h5py.File(Path(path), "r") as f:
        return f["sim_params"][0]


def load_shells(path, baryonified=False, metainfo=None, shell_info=None,
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
        raise ValueError("CosmoGridV1_1.load_shells needs the shell redshift bounds: pass metainfo= (path to "
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
