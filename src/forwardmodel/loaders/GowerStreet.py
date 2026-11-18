"""Loaders for the Gower Street simulations (Jeffrey et al. 2024, arXiv:2403.02314).

Each simulation is one directory (``simNNNNN/``, unpacked from ``simNNNNN.tar.gz``) with

* ``run.XXXXX.lightcone.npy``: one HEALPix map (RING, nside 2048) of particle counts per
  lightcone shell; the larger ``XXXXX``, the closer the shell. ``run.XXXXX.incomplete.npy``
  is a shell that is not fully inside the replicated box and is ignored;
* ``z_values.txt``: the redshift bounds of every shell;
* ``control.par``: the PKDGRAV3 control file (box size, particle number).

The cosmological parameters of all simulations are in sheet 1 of the Gower Street workbook
(linked from the data release); export that sheet as CSV.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

import numpy as np

from ..cosmology import Cosmology
from ..shells import ShellStack
from ._maps import sum_downgrade

_LIGHTCONE = re.compile(r"run\.(\d+)\.lightcone\.npy$")
_SHELL_INFO_DTYPE = [("step", "i4"), ("lower_z", "f8"), ("upper_z", "f8"),
                     ("lower_com", "f8"), ("upper_com", "f8")]


def load_shell_info(sim_dir):
    """Shell redshift bounds from ``z_values.txt``.

    Returns a structured array with one row per PKDGRAV3 output step, in file order (far to
    near): ``step`` (the number in the lightcone file name), ``lower_z``, ``upper_z`` and the
    comoving distances ``lower_com``, ``upper_com`` in Mpc/h as listed in the file. Those
    distances are approximate (the release computes them from the matter density alone; for
    ``w != -1`` they differ from the simulation cosmology at the 0.1 % level), so use
    ``Cosmology.comoving_distance`` when it matters. Rows exist for all steps, also those
    without a lightcone file.
    """
    path = Path(sim_dir)
    if path.is_dir():
        path = path / "z_values.txt"
    with open(path) as f:
        header = [c.strip().lower() for c in f.readline().lstrip("#").split(",")]
        rows = [line.split(",") for line in f if line.strip()]

    def column(name):
        for i, c in enumerate(header):
            if c == name or c.startswith(name + "("):
                return i
        raise ValueError(f"{path}: no column {name!r} in header {header}")

    info = np.empty(len(rows), dtype=_SHELL_INFO_DTYPE)
    for (name, _), source in zip(_SHELL_INFO_DTYPE, ("step", "z_near", "z_far", "cmd_near", "cmd_far")):
        c = column(source)
        info[name] = [float(r[c]) for r in rows]
    info["lower_z"] += 0.0          # the last shell is written as z_near = -0.000000
    info["lower_com"] += 0.0
    return info


def load_control(sim_dir):
    """Parse ``control.par`` into a dict (``dBoxSize`` in Mpc/h, ``nGrid``, ``dOmega0``, ...).

    The file is PKDGRAV3's Python-syntax parameter file. Literal values are converted;
    anything else (e.g. ``w0 = 2.0*math.cos(...)``) is kept as the source string.
    """
    path = Path(sim_dir)
    if path.is_dir():
        path = path / "control.par"
    out = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" not in line:
                continue
            key, val = (x.strip() for x in line.split("=", 1))
            try:
                out[key] = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                out[key] = val
    return out


def load_shells(sim_dir, z_max=None, nside=None, normalization="map_mean", cosmo=None):
    """Load the lightcone shells of one simulation into a :class:`ShellStack` (count form).

    Parameters
    ----------
    sim_dir : str or Path
        Directory with ``run.XXXXX.lightcone.npy``, ``z_values.txt`` (and ``control.par``).
    z_max : float, optional
        Only load the shells that start below ``z_max``. All ~100 shells at nside 2048 take
        about 10 GB, so restrict the range or downgrade when the full stack is not needed.
    nside : int, optional
        Downgrade to this resolution while loading, summing the counts (exact, no smoothing).
        Default: the resolution of the files (2048).
    normalization : {"map_mean", "box"}
        Count-to-overdensity normalization, see :class:`ShellStack`. ``"box"`` needs
        ``cosmo`` and reads the box size and particle number from ``control.par``.
    cosmo : forwardmodel.Cosmology, optional
        For ``normalization="box"``.

    Returns
    -------
    ShellStack
        Shells ordered by increasing redshift. Incomplete shells are left out.
    """
    sim_dir = Path(sim_dir)
    files = {int(m.group(1)): f for f in sim_dir.iterdir()
             if (m := _LIGHTCONE.match(f.name))}
    if not files:
        raise FileNotFoundError(f"no run.XXXXX.lightcone.npy files in {sim_dir}")
    info = load_shell_info(sim_dir)
    missing = sorted(set(files) - set(info["step"].tolist()))
    if missing:
        raise ValueError(f"{sim_dir}: lightcone steps {missing} are not in z_values.txt")
    info = info[np.isin(info["step"], list(files))]
    if z_max is not None:
        info = info[info["lower_z"] < z_max]
    info = info[np.argsort(info["lower_z"])]
    if len(info) == 0:
        raise ValueError(f"{sim_dir}: no lightcone shell below z_max={z_max}")

    counts = None
    for s, step in enumerate(info["step"]):
        m = np.load(files[int(step)])
        if nside is not None:
            m = sum_downgrade(m, nside)
        if counts is None:
            counts = np.empty((len(info), m.size), m.dtype)
        counts[s] = m

    box = {}
    if normalization == "box":
        par = load_control(sim_dir)
        box = dict(box_size=float(par["dBoxSize"]), n_particles=int(par["nGrid"]) ** 3,
                   cosmo=cosmo)
    return ShellStack.from_counts(counts, info["lower_z"], info["upper_z"],
                                  normalization=normalization, **box)


def load_params(csv_path, serial):
    """Parameters of simulation ``serial`` from the workbook's sheet 1, exported as CSV.

    Returns a dict keyed by the sheet's column names (``Omega_m``, ``sigma_8``, ``w``,
    ``Omega_b little_h^2``, ``little_h``, ``n_s``, ``m_nu``, ``Omega_b``, ...); numeric
    cells are floats. ``serial`` is the number in ``simNNNNN`` (1 to 791).
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    for i, row in enumerate(rows):
        if row and row[0].strip() == "Serial Number":
            header, rows = [c.strip() for c in row], rows[i + 1:]
            break
    else:
        raise ValueError(f"{csv_path}: no 'Serial Number' header row")
    for row in rows:
        if row and row[0].strip().isdigit() and int(row[0]) == int(serial):
            out = {}
            for k, v in zip(header, row):
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            out["Serial Number"] = int(serial)
            return out
    raise KeyError(f"simulation {serial} not in {csv_path}")


#: Simulations 1 to 192 are the "verification" runs with a naive treatment of neutrinos.
N_VERIFICATION_RUNS = 192


def cosmology_from_params(p, **overrides):
    """Build a :class:`~forwardmodel.Cosmology` from a row of the parameter sheet.

    ``p`` is the dict of :func:`load_params` (``Omega_m, Omega_b, little_h, sigma_8, n_s, w,
    m_nu, Serial Number``). ``m_nu`` is the summed neutrino mass in eV. The meaning of
    ``Omega_m`` differs between the two sets of runs:

    * simulations 193 to 791: ``Omega_m`` is the total matter density including the massive
      neutrinos, so ``Omega_c = Omega_m - Omega_b - Omega_nu``;
    * simulations 1 to 192 (verification runs): ``Omega_m`` is CDM + baryons and is the whole
      matter content the N-body evolved (``dOmega0`` in ``control.par``). A 0.06 eV neutrino
      entered only the transfer function of the initial conditions. The cosmology returned
      here has no massive neutrinos; pass ``m_nu=0.06`` to add them on top of ``Omega_m``.

    ``**overrides`` are forwarded to :class:`Cosmology` (e.g. ``transfer_function``, or
    ``A_s`` instead of ``sigma_8``).
    """
    h = float(p["little_h"])
    omega_b = float(p["Omega_b"])
    verification = int(p["Serial Number"]) <= N_VERIFICATION_RUNS
    m_nu = 0.0 if verification else float(p["m_nu"])
    kwargs = dict(
        Omega_c=float(p["Omega_m"]) - omega_b - m_nu / (93.14 * h**2),
        Omega_b=omega_b,
        h=h,
        sigma8=float(p["sigma_8"]),
        n_s=float(p["n_s"]),
        w0=float(p["w"]),
        m_nu=m_nu,
    )
    if "A_s" in overrides:
        kwargs.pop("sigma8")
    kwargs.update(overrides)
    return Cosmology(**kwargs)


def load_cosmology(csv_path, serial, **overrides):
    """:class:`~forwardmodel.Cosmology` of simulation ``serial`` from the parameter sheet CSV.

    See :func:`load_params` and :func:`cosmology_from_params`.
    """
    return cosmology_from_params(load_params(csv_path, serial), **overrides)
