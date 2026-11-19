"""Tests for the core containers: Cosmology, ShellStack, NumberDensity."""

import numpy as np
import pytest

import kassiopeia as fm


def test_cosmology_distance_monotonic(cosmo):
    z = np.linspace(0.0, 5.0, 50)
    chi = cosmo.comoving_distance(z)
    assert np.all(np.diff(chi) > 0)
    assert chi[0] == pytest.approx(0.0, abs=1e-6)


def test_cosmology_growth_decreasing(cosmo):
    # Growth factor decreases with redshift.
    assert cosmo.growth_factor(0.0) > cosmo.growth_factor(1.0) > cosmo.growth_factor(2.0)


def test_cosmology_from_ccl_roundtrip():
    import pyccl as ccl
    ccl_cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7,
                              sigma8=0.8, n_s=0.96)
    c = fm.Cosmology.from_ccl(ccl_cosmo)
    assert c.Omega_m == pytest.approx(0.30, abs=1e-6)


def test_shellstack_counts_to_overdensity(rng):
    import healpy as hp
    npix = hp.nside2npix(8)
    counts = rng.poisson(100.0, size=(3, npix)).astype(float)
    stack = fm.ShellStack.from_counts(counts, [0.0, 0.1, 0.2], [0.1, 0.2, 0.3])
    delta = stack.overdensity(0)
    assert delta.mean() == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(delta, counts[0] / counts[0].mean() - 1.0)


def test_shellstack_empty_shell_returns_none():
    import healpy as hp
    npix = hp.nside2npix(8)
    stack = fm.ShellStack.from_counts(np.zeros((1, npix)), [0.0], [0.1])
    assert stack.overdensity(0) is None


def test_shellstack_box_normalization_formula(cosmo):
    import healpy as hp
    nside = 8
    npix = hp.nside2npix(nside)
    z_lo, z_hi = [0.0], [0.5]
    box_size, n_particles = 900.0, 256 ** 3
    counts = np.full((1, npix), 7.0)
    stack = fm.ShellStack.from_counts(
        counts, z_lo, z_hi, normalization="box",
        box_size=box_size, n_particles=n_particles, cosmo=cosmo)

    # Reproduce N * V_shell / V_box / npix independently.
    chi_lo = cosmo.comoving_distance(z_lo[0])
    chi_hi = cosmo.comoving_distance(z_hi[0])
    v_shell = (4.0 / 3.0) * np.pi * (chi_hi ** 3 - chi_lo ** 3)
    v_box = (box_size / cosmo.h) ** 3
    expected_mean = n_particles * v_shell / v_box / npix
    np.testing.assert_allclose(stack._box_mean_pix(0), expected_mean, rtol=1e-12)
    np.testing.assert_allclose(stack.overdensity(0), 7.0 / expected_mean - 1.0)


def test_shellstack_box_vs_mapmean_rescaling(cosmo, rng):
    # box and map_mean differ only by the per-shell factor map_mean/box_mean.
    import healpy as hp
    npix = hp.nside2npix(8)
    counts = rng.poisson(500.0, size=(1, npix)).astype(float)
    common = dict(box_size=900.0, n_particles=256 ** 3, cosmo=cosmo)
    box = fm.ShellStack.from_counts(counts, [0.2], [0.4],
                                    normalization="box", **common)
    mm = fm.ShellStack.from_counts(counts, [0.2], [0.4])
    d_box = box.overdensity(0)
    d_mm = mm.overdensity(0)
    factor = counts[0].mean() / box._box_mean_pix(0)
    np.testing.assert_allclose(d_box + 1.0, (d_mm + 1.0) * factor, rtol=1e-10)


def test_shellstack_box_requires_metadata(cosmo):
    import healpy as hp
    counts = np.ones((1, hp.nside2npix(8)))
    with pytest.raises(ValueError, match="box_size"):
        fm.ShellStack.from_counts(counts, [0.0], [0.1], normalization="box")
    with pytest.raises(ValueError, match="count shells"):
        fm.ShellStack(counts, [0.0], [0.1], kind="overdensity",
                      normalization="box", box_size=900.0,
                      n_particles=1, cosmo=cosmo)


def test_shellstack_box_merge_overdensity_realizes_delta(cosmo, rng):
    # Merging a lognormal-style overdensity stack into a box stack must not run
    # the (1+delta) pseudo-count trick through the box mean; it converts first.
    import healpy as hp
    npix = hp.nside2npix(8)
    counts = rng.poisson(500.0, size=(2, npix)).astype(float)
    box = fm.ShellStack.from_counts(counts, [0.0, 0.5], [0.5, 1.0],
                                    normalization="box", box_size=900.0,
                                    n_particles=256 ** 3, cosmo=cosmo)
    delta_ext = np.full((1, npix), 0.3)
    ext = fm.ShellStack.from_overdensity(delta_ext, [1.0], [1.5])
    merged = box.merge(ext)
    assert merged.kind == "overdensity"
    # The appended shell stays exactly its delta (no box-mean division).
    hi = np.argmax(merged.z_lower)
    np.testing.assert_allclose(merged.overdensity(hi), 0.3)
    # The N-body shells keep the box amplitude.
    lo = np.argmin(merged.z_lower)
    np.testing.assert_allclose(merged.overdensity(lo), box.overdensity(0))


def test_coverage_gaps():
    import healpy as hp
    npix = hp.nside2npix(8)
    # N-body shells covering [0.5, 1.0) and [1.0, 1.5) -> contiguous block.
    counts = np.ones((2, npix))
    stack = fm.ShellStack.from_counts(counts, [0.5, 1.0], [1.0, 1.5])
    gaps = stack.coverage_gaps(z_min=0.0, z_max=3.0)
    assert len(gaps) == 2
    assert gaps[0] == pytest.approx((0.0, 0.5))
    assert gaps[1] == pytest.approx((1.5, 3.0))
    # An interior gap is detected too.
    s2 = fm.ShellStack.from_counts(np.ones((2, npix)), [0.0, 2.0], [1.0, 3.0])
    g2 = s2.coverage_gaps(z_min=0.0, z_max=3.0)
    assert g2 == [pytest.approx((1.0, 2.0))]
    # No gaps when fully covered.
    assert stack.coverage_gaps(z_min=0.5, z_max=1.5) == []


def test_fill_with_lognormal_only_fills_gaps(cosmo, rng):
    pytest.importorskip("glass")
    import healpy as hp
    nside = 8
    npix = hp.nside2npix(nside)
    # N-body shells covering [0.5, 1.0); gaps below and above.
    counts = (rng.poisson(100.0, size=(1, npix)) + 1.0).astype(float)
    nbody = fm.ShellStack.from_counts(counts, [0.5], [1.0], nside=nside)
    filled = nbody.fill_with_lognormal(cosmo, z_min=0.0, z_max=1.5, dz=0.25,
                                       rng=rng)
    # Continuous coverage now, no remaining gaps.
    assert filled.coverage_gaps(z_min=0.0, z_max=1.5) == []
    # The original N-body shell survives unchanged (find it by its edges).
    j = np.argmin(np.abs(filled.z_lower - 0.5))
    assert filled.z_upper[j] == pytest.approx(1.0)
    np.testing.assert_allclose(filled.overdensity(j),
                               counts[0] / counts[0].mean() - 1.0)
    # Added shells are real lognormal overdensity (finite, mean ~ 0, delta>-1).
    assert filled.n_shells > nbody.n_shells
    for i in range(filled.n_shells):
        d = filled.overdensity(i)
        assert np.all(np.isfinite(d)) and np.all(d > -1.0 - 1e-9)


def test_shellstack_from_lognormal(cosmo, rng):
    pytest.importorskip("glass")
    nside = 8
    z_edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    stack = fm.ShellStack.from_lognormal(cosmo, z_edges, nside=nside, rng=rng)
    assert stack.kind == "overdensity"
    assert stack.n_shells == len(z_edges) - 1
    assert stack.nside == nside
    assert stack.z_min == pytest.approx(0.0)
    assert stack.z_max == pytest.approx(1.0)
    for i in range(stack.n_shells):
        d = stack.overdensity(i)
        assert np.all(np.isfinite(d)) and np.all(d > -1.0 - 1e-9)


def test_forwardmodel_from_lognormal_autotile_and_project(cosmo, rng, source_bins):
    pytest.importorskip("glass")
    # Auto-tile [0, 1] in dz=0.25 -> 4 shells, then project a WL probe.
    model = fm.ForwardModel.from_lognormal(cosmo, nside=8, z_max=1.0, dz=0.25,
                                           rng=rng)
    assert isinstance(model, fm.ForwardModel)
    assert model.shells.n_shells == 4
    assert model.shells.z_max == pytest.approx(1.0)
    wl = fm.WeakLensing(nz=source_bins)
    maps = model.project(weak_lensing=wl)
    assert maps.weak_lensing.kappa.shape == (wl.n_bins, model.shells.npix)
    assert np.all(np.isfinite(maps.weak_lensing.kappa))


def test_forwardmodel_from_lognormal_requires_edges_or_zmax(cosmo):
    with pytest.raises(ValueError, match="z_edges or z_max"):
        fm.ForwardModel.from_lognormal(cosmo, nside=8)


def test_shellstack_overdensity_passthrough():
    import healpy as hp
    npix = hp.nside2npix(8)
    delta = np.full((1, npix), 0.5)
    stack = fm.ShellStack.from_overdensity(delta, [0.0], [0.1])
    np.testing.assert_array_equal(stack.overdensity(0), delta[0])


def test_shellstack_merge_sorts_by_redshift():
    import healpy as hp
    npix = hp.nside2npix(8)
    a = fm.ShellStack.from_overdensity(np.zeros((1, npix)), [2.0], [3.0])
    b = fm.ShellStack.from_overdensity(np.ones((1, npix)), [0.0], [1.0])
    merged = a.merge(b)
    assert list(merged.z_lower) == [0.0, 2.0]
    # The z=0 shell (the ones) should come first after sorting.
    assert merged.overdensity(0).mean() == pytest.approx(1.0)


def test_numberdensity_normalized():
    z = np.linspace(0.0, 2.0, 100)
    nd = fm.NumberDensity(z, np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2))
    assert np.trapezoid(nd.nz, nd.z) == pytest.approx(1.0, rel=1e-6)


def test_numberdensity_shift_moves_mean():
    z = np.linspace(0.0, 3.0, 300)
    nd = fm.NumberDensity(z, np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2))
    shifted = nd.shifted(0.3)
    assert shifted.mean_z() == pytest.approx(nd.mean_z() + 0.3, abs=0.02)
