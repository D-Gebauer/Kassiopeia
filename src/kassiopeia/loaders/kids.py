"""Loader for KiDS lensfit shape catalogues (KiDS-1000 / KiDS-Legacy).

Returns per-bin catalogues ready for :class:`kassiopeia.noise.KiDSShapeNoise`.
KiDS ellipticities are already calibrated by lensfit (self-calibration); the
residual multiplicative bias ``m`` per bin enters the observation as
``shear_calibration``, not here.
"""

from __future__ import annotations

import numpy as np

#: KiDS-1000 tomographic binning in the BPZ photo-z ``Z_B``: bin i holds
#: ``edges[i] < Z_B <= edges[i+1]``.
KIDS1000_ZB_EDGES = (0.1, 0.3, 0.5, 0.7, 0.9, 1.2)

#: Column names of the public KiDS-1000 gold shear catalogue
#: (``KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits``). Override for other
#: releases (e.g. KiDS-Legacy) via ``columns=``.
KIDS1000_COLUMNS = {"ra": "ALPHA_J2000", "dec": "DELTA_J2000",
                    "e1": "e1", "e2": "e2", "w": "weight", "z": "Z_B"}


def load_shape_catalog(path, z_edges=KIDS1000_ZB_EDGES, columns=None, hdu=1,
                       drop_zero_weight=True):
    """Load a KiDS shape catalogue split into tomographic bins.

    Parameters
    ----------
    path : str
        FITS shear catalogue.
    z_edges : sequence of float
        Photo-z bin edges in the ``z`` column; bin ``i`` is
        ``z_edges[i] < z <= z_edges[i+1]``. ``None`` returns one bin with all
        galaxies.
    columns : dict, optional
        Overrides of :data:`KIDS1000_COLUMNS` (canonical name -> FITS column).
    hdu : int
        FITS extension holding the table.
    drop_zero_weight : bool
        Drop galaxies with lensfit ``weight == 0`` (they carry no shape
        information and do not enter the estimator anyway).

    Returns
    -------
    dict
        Bin index -> dict with ``ra, dec, e1, e2, w``.
    """
    from astropy.io import fits

    cols = {**KIDS1000_COLUMNS, **(columns or {})}
    with fits.open(path, memmap=True) as hdul:
        data = hdul[hdu].data
        arr = {k: np.asarray(data[cols[k]], dtype=np.float64)
               for k in ("ra", "dec", "e1", "e2", "w")}
        z = (np.asarray(data[cols["z"]], dtype=np.float64)
             if z_edges is not None else None)

    keep = arr["w"] > 0 if drop_zero_weight else np.ones(len(arr["w"]), bool)
    if z_edges is None:
        return {0: {k: v[keep] for k, v in arr.items()}}
    catalogs = {}
    for i, (zlo, zhi) in enumerate(zip(z_edges[:-1], z_edges[1:])):
        sel = keep & (z > zlo) & (z <= zhi)
        catalogs[i] = {k: v[sel] for k, v in arr.items()}
    return catalogs
