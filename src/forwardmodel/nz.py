"""Redshift distributions n(z) for tomographic bins.

A :class:`NumberDensity` holds one tomographic bin's n(z), normalised to unit
integral. Photo-z systematics are applied as a mean shift (:meth:`shifted`)
and/or a width stretch (:meth:`stretched`) along the z axis — each returns a
new, renormalised instance, so the originals are never mutated and a single set
of fiducial distributions can seed many realisations. :meth:`transformed`
applies both at once.
"""

from __future__ import annotations

import numpy as np


def _normalize(z, nz):
    norm = np.trapezoid(nz, z)
    return nz / norm if norm > 0 else nz


class NumberDensity:
    """A single tomographic redshift distribution.

    Parameters
    ----------
    z : array_like
        Redshift sample points (monotonically increasing).
    nz : array_like
        n(z) values at ``z``. Stored normalised to ``int n(z) dz = 1``.
    normalize : bool
        Whether to renormalise on construction (default True).
    """

    def __init__(self, z, nz, normalize=True):
        z = np.asarray(z, dtype=float)
        nz = np.asarray(nz, dtype=float)
        if z.shape != nz.shape:
            raise ValueError("z and nz must have the same shape")
        self.z = z
        self.nz = _normalize(z, nz) if normalize else nz

    def shifted(self, dz):
        """Return a copy shifted by photo-z bias ``dz``.

        Implements ``n_shifted(z) = n(z - dz)``, so a positive ``dz`` moves the
        distribution to higher redshift (its mean increases by ``dz``).
        Evaluated on the original grid by linear interpolation with zero
        padding, then renormalised.
        """
        nz_shifted = np.interp(self.z - dz, self.z, self.nz, left=0.0, right=0.0)
        return NumberDensity(self.z, nz_shifted)

    def stretched(self, stretch, z_pivot=None):
        """Return a copy with its width rescaled by ``stretch`` about a pivot.

        Implements ``n_stretched(z) = n(z_pivot + (z - z_pivot) / stretch)``, so
        ``stretch > 1`` widens the distribution (its standard deviation is
        multiplied by ``stretch``) and ``stretch < 1`` narrows it, while the
        pivot redshift is held fixed. ``z_pivot`` defaults to the mean of the
        unstretched distribution, which keeps the mean essentially unchanged.
        Evaluated on the original grid by linear interpolation with zero
        padding, then renormalised.
        """
        if z_pivot is None:
            z_pivot = self.mean_z()
        z_remap = z_pivot + (self.z - z_pivot) / stretch
        nz_stretched = np.interp(z_remap, self.z, self.nz, left=0.0, right=0.0)
        return NumberDensity(self.z, nz_stretched)

    def transformed(self, dz=0.0, stretch=1.0, z_pivot=None):
        """Return a copy stretched about ``z_pivot`` then shifted by ``dz``.

        The standard two-parameter photo-z nuisance: a width rescaling followed
        by a mean shift. Equivalent to ``self.stretched(stretch, z_pivot).shifted(dz)``;
        a no-op for the defaults ``dz=0, stretch=1``.
        """
        return self.stretched(stretch, z_pivot=z_pivot).shifted(dz)

    def mean_z(self):
        """Mean redshift ``int z n(z) dz``."""
        return float(np.trapezoid(self.z * self.nz, self.z))

    def __call__(self, z):
        """Interpolate n(z) at arbitrary ``z`` (zero outside support)."""
        return np.interp(np.asarray(z, dtype=float), self.z, self.nz,
                         left=0.0, right=0.0)

    def __repr__(self):
        return (f"NumberDensity(z=[{self.z[0]:.3f}, {self.z[-1]:.3f}], "
                f"mean_z={self.mean_z():.3f})")


def make_bins(z, nz_list, normalize=True):
    """Convenience: build a list of :class:`NumberDensity` from a shared z grid.

    Parameters
    ----------
    z : array_like
        Common redshift grid.
    nz_list : sequence of array_like
        One n(z) array per tomographic bin.

    Returns
    -------
    list of NumberDensity
    """
    return [NumberDensity(z, nz, normalize=normalize) for nz in nz_list]
