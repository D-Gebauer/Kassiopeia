"""Shared fixtures for the kassiopeia test suite."""

import numpy as np
import healpy as hp
import pytest

import kassiopeia as kp


@pytest.fixture(scope="session")
def cosmo():
    return kp.Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)


@pytest.fixture
def nside():
    return 16


@pytest.fixture
def rng():
    return np.random.default_rng(1234)


@pytest.fixture
def count_shells(nside, rng):
    """A small stack of synthetic count shells from z=0 to 3."""
    npix = hp.nside2npix(nside)
    n_sh = 10
    z_edges = np.linspace(0.0, 3.0, n_sh + 1)
    counts = rng.poisson(80.0, size=(n_sh, npix)).astype(np.float64)
    return kp.ShellStack.from_counts(counts, z_edges[:-1], z_edges[1:])


@pytest.fixture
def source_bins():
    z = np.linspace(0.01, 3.0, 120)
    return kp.make_bins(z, [np.exp(-0.5 * ((z - 0.5) / 0.2) ** 2),
                            np.exp(-0.5 * ((z - 1.0) / 0.25) ** 2)])


@pytest.fixture
def lens_bins():
    z = np.linspace(0.01, 3.0, 120)
    return kp.make_bins(z, [np.exp(-0.5 * ((z - 0.3) / 0.15) ** 2),
                            np.exp(-0.5 * ((z - 0.6) / 0.15) ** 2)])
