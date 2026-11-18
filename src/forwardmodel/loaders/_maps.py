"""HEALPix helpers shared by the simulation loaders."""

import healpy as hp
import numpy as np


def sum_downgrade(m, nside):
    """Downgrade a RING map of an additive quantity (counts, mass) by summing pixels.

    Integer maps are summed as ``uint32``, float maps as ``float64``. Returns ``m`` itself
    if it already has resolution ``nside``.
    """
    nside_in = hp.npix2nside(m.size)
    if nside == nside_in:
        return m
    if nside > nside_in:
        raise ValueError(f"nside={nside} is above the map resolution {nside_in}")
    dtype = np.uint32 if np.issubdtype(m.dtype, np.integer) else np.float64
    nest = hp.reorder(m, r2n=True).reshape(hp.nside2npix(nside), -1)
    return hp.reorder(nest.sum(axis=1, dtype=dtype), n2r=True)
