"""Loaders for the FLAMINGO simulations (Schaye et al. 2023; data release: Helly et al. 2026).

The release (https://dataweb.cosma.dur.ac.uk:8443/flamingo/) holds, for every run
(``/FLAMINGO/<box>/<run>/``, hydrodynamical and dark-matter-only):

* ``healpix_maps/nside_4096/lightcone<X>_shells/shell_<N>/lightcone<X>.shell_<N>.0.hdf5``:
  the full-sky maps of observer ``X`` in redshift shell ``N``, one HDF5 dataset per quantity
  (``TotalMass``, ``DarkMatterMass``, ``SmoothedGasMass``, ``ComptonY``, ...), RING ordered,
  ``float64``; masses are in 10^10 M_sun per pixel. ``nside_16384`` has the full resolution;
* ``shell_redshifts*.txt``: the redshift bounds of the shells;
* the SWIFT parameter file (``flamingo_*.yml`` or ``params.yml``) with the cosmology.

Each shell file is an index of virtual datasets into ~17 GB data files, so a single map is
best read through the release's ``hdfstream`` service rather than by downloading the files;
:func:`load_shells` reads either local files (``h5py``) or from an ``hdfstream`` directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..cosmology import Cosmology
from ..shells import ShellStack
from ._maps import sum_downgrade

#: Power-spectrum normalisation of the FLAMINGO cosmologies (Schaye et al. 2023, table 4),
#: which the SWIFT parameter files do not contain. ``h``, ``Omega_b`` and ``m_nu`` (summed,
#: eV) identify the cosmology of a parameter file.
COSMOLOGIES = {
    "D3A":             dict(h=0.681, Omega_b=0.0486, m_nu=0.06, sigma8=0.807, n_s=0.967, A_s=2.099e-9),
    "Planck":          dict(h=0.673, Omega_b=0.0494, m_nu=0.06, sigma8=0.812, n_s=0.966, A_s=2.101e-9),
    "PlanckNu0p12Var": dict(h=0.673, Omega_b=0.0496, m_nu=0.12, sigma8=0.800, n_s=0.967, A_s=2.113e-9),
    "PlanckNu0p24Var": dict(h=0.662, Omega_b=0.0510, m_nu=0.24, sigma8=0.772, n_s=0.968, A_s=2.109e-9),
    "PlanckNu0p24Fix": dict(h=0.673, Omega_b=0.0494, m_nu=0.24, sigma8=0.769, n_s=0.966, A_s=2.101e-9),
    "PlanckNu0p48Fix": dict(h=0.673, Omega_b=0.0494, m_nu=0.48, sigma8=0.709, n_s=0.966, A_s=2.101e-9),
    "LS8":             dict(h=0.682, Omega_b=0.0473, m_nu=0.06, sigma8=0.760, n_s=0.965, A_s=1.836e-9),
}


def load_params(yml_path):
    """The ``Cosmology`` block of a run's SWIFT parameter file as a dict.

    Values are floats, or lists of floats for comma-separated entries (``M_nu_eV``,
    ``deg_nu``). Keys as in the file: ``h, Omega_cdm, Omega_b, Omega_lambda, N_ur, N_nu,
    M_nu_eV, deg_nu, ...``.
    """
    out, inside = {}, False
    with open(yml_path) as f:
        for line in f:
            text = line.split("#", 1)[0].rstrip()
            if not text.strip():
                continue
            if not text[0].isspace():                 # a top-level key starts a new block
                inside = text.strip() == "Cosmology:"
                continue
            if inside and ":" in text:
                key, val = (x.strip() for x in text.split(":", 1))
                vals = [float(v) for v in val.strip("[]").split(",") if v.strip()]
                out[key] = vals[0] if len(vals) == 1 else vals
    if not out:
        raise ValueError(f"{yml_path}: no Cosmology block")
    return out


def _neutrino_masses(p):
    """Individual neutrino masses (eV) from ``M_nu_eV`` and the degeneracies ``deg_nu``."""
    masses = np.atleast_1d(p.get("M_nu_eV", []))
    degs = np.atleast_1d(p.get("deg_nu", np.ones(len(masses))))
    return [float(m) for m, d in zip(masses, degs) for _ in range(int(round(d)))]


def cosmology_from_params(p, **overrides):
    """Build a :class:`~kassiopeia.Cosmology` from the dict of :func:`load_params`.

    Densities, ``h`` and the neutrino masses come from the parameter file (``Omega_cdm`` and
    ``Omega_b`` exclude the neutrinos, which are added on top). The file has no power-spectrum
    normalisation: ``sigma8`` and ``n_s`` are taken from :data:`COSMOLOGIES`, matched on
    ``h``, ``Omega_b`` and the summed neutrino mass. Pass ``sigma8=`` (or ``A_s=``) and
    ``n_s=`` for a run that is not in the table; other ``**overrides`` are forwarded to
    :class:`Cosmology`. The decaying-dark-matter runs are not supported.
    """
    if "Omega_dcdmdr_0" in p or "Gamma_dcdm" in p:
        raise ValueError("decaying dark matter runs cannot be described by kassiopeia.Cosmology")
    masses = _neutrino_masses(p)
    kwargs: dict = dict(Omega_c=float(p["Omega_cdm"]), Omega_b=float(p["Omega_b"]), h=float(p["h"]))
    if len(masses) == 1:
        kwargs.update(m_nu=masses[0], extra_ccl_kwargs={"mass_split": "single"})
    elif len(masses) == 3 and len(set(masses)) == 1:
        kwargs.update(m_nu=sum(masses))                          # three degenerate species
    elif masses:
        kwargs.update(m_nu=(masses + [0.0, 0.0])[:3], extra_ccl_kwargs={"mass_split": "list"})

    if not ({"sigma8", "A_s"} & set(overrides)) or "n_s" not in overrides:
        for name, c in COSMOLOGIES.items():
            if (abs(c["h"] - kwargs["h"]) < 1e-3 and abs(c["Omega_b"] - kwargs["Omega_b"]) < 1e-4
                    and abs(c["m_nu"] - sum(masses)) < 5e-3):
                kwargs.update(sigma8=c["sigma8"], n_s=c["n_s"])
                break
        else:
            raise ValueError("parameters match no entry of Flamingo.COSMOLOGIES: pass sigma8= "
                             "(or A_s=) and n_s=")
    if "A_s" in overrides:
        kwargs.pop("sigma8", None)
    if "extra_ccl_kwargs" in overrides:
        overrides = dict(overrides)
        kwargs["extra_ccl_kwargs"] = {**kwargs.get("extra_ccl_kwargs", {}),
                                      **overrides.pop("extra_ccl_kwargs")}
    kwargs.update(overrides)
    return Cosmology(**kwargs)


def load_cosmology(yml_path, **overrides):
    """:class:`~kassiopeia.Cosmology` of a run from its SWIFT parameter file.

    See :func:`load_params` and :func:`cosmology_from_params`.
    """
    return cosmology_from_params(load_params(yml_path), **overrides)


def load_shell_redshifts(path):
    """Shell redshift bounds from a run's ``shell_redshifts*.txt``.

    Returns a structured array (``lower_z``, ``upper_z``); row ``N`` belongs to ``shell_N``.
    The 1 Gpc boxes have ``shell_redshifts_z3.txt`` for observer 0 and
    ``shell_redshifts_z0p5.txt`` for observer 1 (see the ``radius_file`` entries of the
    parameter file); the larger boxes have one ``shell_redshifts.txt``.
    """
    z = np.atleast_2d(np.loadtxt(path, delimiter=","))
    info = np.empty(len(z), dtype=[("lower_z", "f8"), ("upper_z", "f8")])
    info["lower_z"], info["upper_z"] = z[:, 0], z[:, 1]
    return info


def shell_file(maps_dir, shell, lightcone=0):
    """Path of the map file of one shell below ``maps_dir`` (``.../healpix_maps/nside_4096``)."""
    return (f"{str(maps_dir).rstrip('/')}/lightcone{lightcone}_shells/shell_{shell}/"
            f"lightcone{lightcone}.shell_{shell}.0.hdf5")


def load_shells(maps_dir, redshifts, lightcone=0, field="TotalMass", z_max=None, nside=None,
                remote=None, cache_dir=None):
    """Load the mass maps of one lightcone into a :class:`ShellStack`.

    The stack is in "count" form with map-mean normalisation: the overdensity of a shell is
    ``mass / mean(mass) - 1``. ``TotalMass`` includes gas, dark matter, stars, black holes and
    neutrinos, so for a hydrodynamical run it carries the baryonic feedback and no
    baryonification step is needed.

    Parameters
    ----------
    maps_dir : str or Path
        The ``.../healpix_maps/nside_4096`` (or ``nside_16384``) directory of a run: a local
        path, or the path on the server when ``remote`` is given (e.g.
        ``"FLAMINGO/L1_m9/L1_m9/healpix_maps/nside_4096"``).
    redshifts : str, Path or structured array
        The run's ``shell_redshifts*.txt`` (a local file) or the array of
        :func:`load_shell_redshifts`.
    lightcone : int
        Observer index ``X`` of ``lightcone<X>``.
    field : str
        Map to load; any additive mass map (``TotalMass``, ``DarkMatterMass``, ...).
    z_max : float, optional
        Only load the shells that start below ``z_max``.
    nside : int, optional
        Downgrade to this resolution while loading, summing the mass (exact). One map at
        nside 4096 is 1.6 GB as ``float64``; the stack is stored as ``float32``.
    remote : hdfstream directory, optional
        Read from the release's server instead of local files, e.g.
        ``hdfstream.open("cosma", "/")``; ``maps_dir`` is then relative to that directory.
        Any object works whose ``remote[path][field][...]`` returns the map.
    cache_dir : str or Path, optional
        Keep each loaded (downgraded) map as a ``.npy`` file in this directory and reuse it
        on later calls. Recommended with ``remote``: a map takes minutes to transfer.

    Returns
    -------
    ShellStack
        Shells ordered by increasing redshift.
    """
    info = redshifts if isinstance(redshifts, np.ndarray) else load_shell_redshifts(redshifts)
    keep = np.arange(len(info)) if z_max is None else np.flatnonzero(info["lower_z"] < z_max)
    if len(keep) == 0:
        raise ValueError(f"no shell below z_max={z_max}")
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

    maps = None
    for s, shell in enumerate(keep):
        cached = None
        if cache_dir is not None:
            tag = "native" if nside is None else f"nside{nside}"
            cached = cache_dir / f"lightcone{lightcone}_{field}_{tag}_shell{shell}.npy"
        if cached is not None and cached.exists():
            m = np.load(cached)
        else:
            m = _read_map(shell_file(maps_dir, shell, lightcone), field, remote)
            if nside is not None:
                m = sum_downgrade(m, nside)
            m = m.astype(np.float32)
            if cached is not None:
                tmp = cached.with_name(cached.name + ".tmp.npy")     # no partial file on interrupt
                np.save(tmp, m)
                tmp.replace(cached)
        if maps is None:
            maps = np.empty((len(keep), m.size), np.float32)
        maps[s] = m
    return ShellStack.from_counts(maps, info["lower_z"][keep], info["upper_z"][keep])


def _read_map(path, field, remote):
    if remote is not None:
        return np.asarray(remote[path][field][...])
    import h5py

    with h5py.File(path, "r") as f:
        return f[field][...]
