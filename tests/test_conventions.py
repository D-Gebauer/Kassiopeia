"""Convention switches added for matching CosmoSIS theory and CosmoGrid metadata (2026-11-04)."""
import numpy as np
import pytest

import forwardmodel as fm
from forwardmodel.ia import C1_RHOCRIT, C1_RHOCRIT_COSMOSIS
from forwardmodel.loaders import cosmogrid as cg

CG_FID = {"As": 3.0589e-9, "H0": 67.36, "O_cdm": 0.209277442262, "O_nu": 0.0014225577379999993,
          "Ob": 0.0493, "Om": 0.26, "m_nu": 0.02, "ns": 0.9649, "s8": 0.84, "w0": -1.0, "wa": 0.0}


def test_c1_rhocrit_cosmosis_value_and_scaling():
    # CSL la_model compute_c1_baseline(): 5e-14 h^-2 Msun^-1 Mpc^3 * rho_crit,0
    assert C1_RHOCRIT_COSMOSIS == pytest.approx(0.0138731, rel=1e-5)
    cosmo = fm.Cosmology(Omega_m=0.3, sigma8=0.8)
    a = fm.NLA(a1=1.0).amplitudes(cosmo, 0.5)[0]
    b = fm.NLA(a1=1.0, c1_rhocrit=C1_RHOCRIT_COSMOSIS).amplitudes(cosmo, 0.5)[0]
    assert b / a == pytest.approx(C1_RHOCRIT_COSMOSIS / C1_RHOCRIT)
    t = fm.TATT(a1=1.0, a2=1.0, c1_rhocrit=2 * C1_RHOCRIT).amplitudes(cosmo, 0.5)
    t0 = fm.TATT(a1=1.0, a2=1.0).amplitudes(cosmo, 0.5)
    assert t[0] / t0[0] == pytest.approx(2.0) and t[1] / t0[1] == pytest.approx(2.0)


def test_cosmogrid_mnu_is_per_species():
    cosmo = cg.cosmology_from_params(CG_FID)
    assert cosmo.Omega_m == pytest.approx(0.26, rel=1e-4)
    with pytest.raises(ValueError):
        cg.cosmology_from_params(CG_FID, n_massive_nu=1)       # old behaviour: sum = 0.02 eV
    row = np.array([tuple(CG_FID.values())], dtype=[(k, "f8") for k in CG_FID])[0]
    assert cg.cosmology_from_params(row).Omega_m == pytest.approx(cosmo.Omega_m)


def test_nz_from_spline():
    z = np.linspace(0.0, 2.0, 41)
    nz = np.exp(-((z - 0.8) / 0.3) ** 2)
    nz[0] = 0.0
    s = fm.NumberDensity.from_spline(z, nz)
    assert np.trapezoid(s.nz, s.z) == pytest.approx(1.0)
    assert s.mean_z() == pytest.approx(fm.NumberDensity(z, nz).mean_z(), abs=1e-3)
    # passes through the (normalised) sample points
    lin = fm.NumberDensity(z, nz)
    assert np.allclose(s(z[5:-5]), lin.nz[5:-5], rtol=1e-3)


def test_linear_ia_fast_path_equals_tidal_path():
    import healpy as hp
    rng = np.random.default_rng(1)
    nside = 32
    z_edges = np.linspace(0.0, 1.5, 11)
    delta = rng.normal(0, 0.3, (10, hp.nside2npix(nside)))
    shells = fm.ShellStack.from_overdensity(delta, z_edges[:-1], z_edges[1:])
    cosmo = fm.Cosmology(Omega_m=0.3, sigma8=0.8, transfer_function="eisenstein_hu")
    z = np.linspace(0, 2, 81)
    bins = fm.make_bins(z, [np.exp(-((z - m) / 0.2) ** 2) for m in (0.5, 0.9)])
    wl = fm.WeakLensing(nz=bins, ia=fm.NLA(a1=1.3, eta1=0.5))
    model = fm.ForwardModel(cosmo, shells)
    fast = model.project(weak_lensing=wl).weak_lensing
    slow = model.project(weak_lensing=wl, linear_ia_fast=False).weak_lensing
    assert fast.ia_kappa is not None and slow.ia_kappa is None
    # equal up to the pixel mean: the tidal path subtracts the mean of the IA gamma maps (not of the
    # gravitational gamma); the fast path treats IA like kappa_to_gamma treats kappa
    for a, b in ((fast.ia_gamma1, slow.ia_gamma1), (fast.ia_gamma2, slow.ia_gamma2)):
        d = (a - a.mean(axis=1, keepdims=True)) - (b - b.mean(axis=1, keepdims=True))
        # residual = healpy map2alm quadrature error (white-noise shells are not band-limited): ~1e-5
        assert np.max(np.abs(d)) < 1e-4 * np.max(np.abs(b))
    assert np.allclose(fast.gamma1, slow.gamma1)


def test_rotation_reproducible_and_matrix_matches_alm_rotation():
    import healpy as hp
    from forwardmodel import rotation as rot
    a = rot.random_euler_angles(np.random.SeedSequence(5, spawn_key=(1, 2)))
    assert a == rot.random_euler_angles(np.random.SeedSequence(5, spawn_key=(1, 2)))
    lmax, nside = 16, 16
    v0 = np.array([0.3, -0.5, 0.81]); v0 /= np.linalg.norm(v0)
    # dipole field d(n) = v0 . n -> rotated field must be (R v0) . n
    m = np.dot(v0, hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))
    alm = hp.map2alm(m, lmax=lmax, iter=10)
    r = rot.rotate_alm(alm, lmax, a, backend="healpy")
    v = hp.pix2vec(nside, np.arange(hp.nside2npix(nside)))
    m_rot = hp.alm2map(r, nside, lmax=lmax)
    expect = np.dot(rot.rotation_matrix(a) @ v0, v)
    assert np.allclose(m_rot, expect, atol=1e-6)


@pytest.mark.skipif(not __import__("forwardmodel.rotation", fromlist=["x"]).have_ducc0(), reason="no ducc0")
def test_rotation_ducc0_equals_healpy():
    import healpy as hp
    from forwardmodel import rotation as rot
    lmax = 64
    alm = hp.synalm(np.ones(lmax + 1), lmax=lmax).astype(np.complex128)
    a = rot.random_euler_angles(3)
    assert np.allclose(rot.rotate_alm(alm, lmax, a, backend="ducc0"),
                       rot.rotate_alm(alm, lmax, a, backend="healpy"), rtol=0, atol=1e-10)


def test_harmonic_projection_equals_forward_model():
    import healpy as hp
    from forwardmodel import harmonic as H
    rng = np.random.default_rng(2)
    nside = 32
    z_edges = np.linspace(0.0, 1.5, 11)
    delta = rng.normal(0, 0.3, (10, hp.nside2npix(nside)))
    shells = fm.ShellStack.from_overdensity(delta, z_edges[:-1], z_edges[1:])
    cosmo = fm.Cosmology(Omega_m=0.3, sigma8=0.8, transfer_function="eisenstein_hu")
    z = np.linspace(0, 2, 81)
    bins = fm.make_bins(z, [np.exp(-((z - m) / 0.2) ** 2) for m in (0.5, 0.9)])
    ia = fm.NLA(a1=1.0)
    ref = fm.ForwardModel(cosmo, shells).project(weak_lensing=fm.WeakLensing(nz=bins, ia=ia)).weak_lensing
    alms = H.shell_overdensity_alms(shells, dtype=np.complex128)
    wk, wi = H.weak_lensing_weights(cosmo, bins, shells.z_lower, shells.z_upper, ia=ia)
    kap, kia = H.combine_alms(alms, wk), H.combine_alms(alms, wi)
    for b in range(2):
        g1, g2 = H.kappa_alm_to_shear(kap[b], nside)
        i1, i2 = H.kappa_alm_to_shear(kia[b], nside)
        for x, y in ((g1, ref.gamma1[b]), (g2, ref.gamma2[b]), (i1, ref.ia_gamma1[b]), (i2, ref.ia_gamma2[b])):
            assert np.max(np.abs(x - y)) < 1e-4 * np.max(np.abs(y))


def test_effective_redshift_stays_in_shell_for_negative_nz_tails():
    from forwardmodel import projections as proj
    z = np.linspace(0, 3, 301)
    nz = np.exp(-((z - 0.8) / 0.1) ** 2)
    nz[200:] = -1e-12                                   # tiny negative tail (spline undershoot)
    nd = fm.NumberDensity(z, nz, normalize=False)
    zl, zu = np.linspace(0, 2.9, 30), np.linspace(0.1, 3.0, 30)
    ze = proj.effective_redshift([nd], zl, zu)[:, 0]
    assert np.all(ze >= zl) and np.all(ze <= zu)
