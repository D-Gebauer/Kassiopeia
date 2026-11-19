"""Tests for physics building blocks: tidal, Kaiser-Squires, bias, IA."""

import numpy as np
import healpy as hp
import pytest

import kassiopeia as kp


def _smooth_map(nside, rng, lmax=20):
    """A band-limited random map (so SHT round-trips are well-behaved)."""
    cl = np.zeros(lmax + 1)
    cl[2:] = 1.0 / np.arange(2, lmax + 1) ** 2
    return hp.synfast(cl, nside, lmax=lmax, new=True)


def test_kaiser_squires_roundtrip(nside, rng):
    kappa = _smooth_map(nside, rng)
    kappa -= kappa.mean()
    g1, g2 = kp.kappa_to_gamma(kappa)
    kE, kB = kp.gamma_to_kappa(g1, g2)
    # KS is exact in harmonic space; pixelization at low nside leaves a small
    # residual, so require high correlation and a vanishing B-mode.
    corr = np.corrcoef(kE, kappa)[0, 1]
    assert corr > 0.99
    assert np.std(kB) < 0.1 * np.std(kE)


def test_tidal_traceless_invariant_nonneg(nside, rng):
    delta = _smooth_map(nside, rng)
    s1, s2, s_sq = kp.compute_tidal(delta)
    np.testing.assert_allclose(s_sq, s1 ** 2 + s2 ** 2)
    assert np.all(s_sq >= 0)


def test_linear_bias():
    delta = np.array([0.1, -0.2, 0.3])
    np.testing.assert_allclose(kp.apply_linear_bias(delta, 2.0), 2.0 * delta)


def test_quadratic_bias_terms_mean_subtracted():
    delta = np.array([0.1, -0.2, 0.3, 0.0])
    _, d2_ms, s2_ms = kp.quadratic_bias_terms(delta, delta ** 2)
    assert d2_ms.mean() == pytest.approx(0.0, abs=1e-12)


def test_bias_model_flags():
    assert not kp.LinearBias(1.5).needs_tidal
    assert kp.QuadraticBias(b1=1.5, bs=-0.2).needs_tidal
    assert kp.QuadraticBias(b1=1.5, b2=0.3).needs_quadratic_density
    assert not kp.QuadraticBias(b1=1.5).needs_tidal


def test_tatt_amplitude_signs(cosmo):
    # a1 > 0 should give A1 < 0 (NLA opposes lensing).
    A1, A2 = kp.TATT(a1=1.0, a2=1.0).amplitudes(cosmo, 0.5)
    assert A1 < 0
    assert A2 > 0


def test_nla_is_tatt_without_quadratic(cosmo):
    nla = kp.NLA(a1=1.0, eta1=0.5)
    A1, A2 = nla.amplitudes(cosmo, 0.7)
    assert A2 == 0.0
    assert nla.b_ta == 0.0


def _nz_std(nd):
    m = nd.mean_z()
    return float(np.sqrt(np.trapezoid(nd.nz * (nd.z - m) ** 2, nd.z)))


def test_nz_stretch_widens_and_preserves_mean():
    z = np.linspace(0.0, 3.0, 600)
    nd = kp.NumberDensity(z, np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2))
    wide = nd.stretched(1.5)
    narrow = nd.stretched(0.5)
    assert _nz_std(wide) > _nz_std(nd) > _nz_std(narrow)
    # Stretch about the mean leaves the mean essentially unchanged.
    assert wide.mean_z() == pytest.approx(nd.mean_z(), abs=1e-2)
    # A unit stretch is a no-op.
    np.testing.assert_allclose(nd.stretched(1.0).nz, nd.nz, atol=1e-10)


def test_tatt_field_exact_operators(cosmo):
    rng = np.random.default_rng(0)
    s1, s2, delta = (rng.normal(size=64), rng.normal(size=64),
                     rng.normal(size=64))
    tatt = kp.TATT(a1=1.0, a2=0.7, b_ta=0.5)
    A1, A2 = tatt.amplitudes(cosmo, 0.5)
    quad = 0.5 * A1 + A2  # b_TA * A1 + A2
    i1, i2 = tatt.field(s1, s2, delta, cosmo, 0.5)
    np.testing.assert_allclose(i1, A1 * s1 + quad * delta * s1)
    np.testing.assert_allclose(i2, A1 * s2 + quad * delta * s2)
    # NLA limit (a2 = b_ta = 0): pure linear alignment, no density term.
    nla = kp.NLA(a1=1.0)
    A1n, _ = nla.amplitudes(cosmo, 0.5)
    j1, j2 = nla.field(s1, s2, delta, cosmo, 0.5)
    np.testing.assert_allclose(j1, A1n * s1)
    np.testing.assert_allclose(j2, A1n * s2)


def test_separable_coeffs_match_cmb_weights(cosmo):
    from kassiopeia import projections as proj
    z_lo = np.array([0.0, 0.3, 0.6, 0.9])
    z_hi = np.array([0.3, 0.6, 0.9, 1.2])
    alpha, beta = proj.lensing_separable_coeffs(cosmo, z_lo, z_hi)
    # For a source plane behind all shells, alpha - beta/chi_s must reproduce
    # the single-plane (CMB-style) convergence weights exactly.
    z_src = 2.0
    chi_s = cosmo.comoving_distance_dimensionless(0.0, z_src)
    w = proj.cmb_lensing_weights(cosmo, z_lo, z_hi, z_source=z_src)
    np.testing.assert_allclose(alpha - beta / chi_s, w, rtol=1e-8, atol=1e-14)


def test_nz_transform_composes_stretch_and_shift():
    z = np.linspace(0.0, 3.0, 600)
    nd = kp.NumberDensity(z, np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2))
    # Pure shift via transformed matches shifted; mean moves by dz.
    np.testing.assert_allclose(nd.transformed(dz=0.1).nz, nd.shifted(0.1).nz)
    assert nd.transformed(dz=0.1).mean_z() == pytest.approx(
        nd.mean_z() + 0.1, abs=1e-2)
