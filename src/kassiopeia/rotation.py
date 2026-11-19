"""Exact rotations of full-sky fields in harmonic space, with reproducible random rotations.

A band-limited field (all forward-model shear maps are: Kaiser-Squires and the tidal field are evaluated at
``lmax``) is rotated *exactly* by rotating its ``a_lm`` with Wigner-D matrices and re-synthesising. Pixel-space
rotations interpolate and lose small-scale power, so they are not offered here.

Angles are healpy's ZYZ Euler convention ``(psi, theta, phi)`` (``healpy.rotate_alm``); ``ducc0.sht.rotate_alm``
uses the same convention (checked to 2e-12) and is ~4x faster single-threaded and multi-threaded, so it is
used when installed (``pip install ducc0``), else healpy.

Random rotations are Haar-uniform on SO(3): ``psi, phi ~ U(0, 2 pi)``, ``cos theta ~ U(-1, 1)``. They are drawn
from a ``numpy.random.SeedSequence`` so that a rotation is reproducible from its seed alone; store the seed (or
the angles) with every product.
"""

from __future__ import annotations

import numpy as np

try:  # optional fast backend
    import ducc0 as _ducc0
except ImportError:  # pragma: no cover - depends on the environment
    _ducc0 = None


def have_ducc0():
    """Whether the ducc0 backend is available."""
    return _ducc0 is not None


def random_euler_angles(seed):
    """Haar-uniform rotation ``(psi, theta, phi)`` from ``seed``.

    Parameters
    ----------
    seed : int, sequence of int, numpy.random.SeedSequence or numpy.random.Generator
        Anything ``numpy.random.default_rng`` accepts. The same seed always gives the same angles.
    """
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    psi, phi = rng.uniform(0.0, 2.0 * np.pi, 2)
    theta = np.arccos(rng.uniform(-1.0, 1.0))
    return float(psi), float(theta), float(phi)


def rotate_alm(alm, lmax, angles, nthreads=0, backend="auto"):
    """Rotate ``alm`` (healpy ordering, ``mmax = lmax``) by ZYZ Euler ``angles = (psi, theta, phi)``.

    Returns a new array (the input is not modified). ``backend``: ``"auto"`` (ducc0 if installed),
    ``"ducc0"`` or ``"healpy"``. ``nthreads=0`` uses all threads the backend sees (``OMP_NUM_THREADS`` for
    ducc0 is not read automatically, pass it explicitly in batch jobs).
    """
    psi, theta, phi = (float(a) for a in angles)
    if backend == "auto":
        backend = "ducc0" if _ducc0 is not None else "healpy"
    if backend == "ducc0":
        if _ducc0 is None:
            raise ImportError("ducc0 is not installed")
        return _ducc0.sht.rotate_alm(np.ascontiguousarray(alm), int(lmax), psi, theta, phi,
                                     nthreads=int(nthreads))
    if backend == "healpy":
        import healpy as hp

        out = np.array(alm, dtype=np.complex128, copy=True)
        hp.rotate_alm(out, psi, theta, phi, lmax=int(lmax))
        return out.astype(np.asarray(alm).dtype, copy=False)
    raise ValueError(f"unknown backend {backend!r}")


def rotation_matrix(angles):
    """3x3 matrix of the ZYZ rotation ``(psi, theta, phi)``: ``R = Rz(phi) Ry(theta) Rz(psi)``.

    For bookkeeping (e.g. mapping a direction through the rotation); the alm rotation is the exact operation.
    """
    psi, theta, phi = angles

    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    c, s = np.cos(theta), np.sin(theta)
    ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return rz(phi) @ ry @ rz(psi)
