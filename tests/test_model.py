"""End-to-end projection and observation tests."""

import numpy as np
import healpy as hp
import pytest

import kassiopeia as fm


def test_project_weak_lensing_only(cosmo, count_shells, source_bins):
    wl = fm.WeakLensing(nz=source_bins)
    maps = fm.ForwardModel(cosmo, count_shells).project(weak_lensing=wl)
    assert maps.weak_lensing is not None
    assert maps.galaxy_clustering is None
    assert maps.weak_lensing.kappa.shape == (2, count_shells.npix)
    # Convergence is mean-subtracted and non-trivial.
    assert maps.weak_lensing.kappa.std() > 0
    np.testing.assert_allclose(maps.weak_lensing.kappa.mean(axis=1), 0, atol=1e-10)


def test_project_requires_a_probe(cosmo, count_shells):
    with pytest.raises(ValueError):
        fm.ForwardModel(cosmo, count_shells).project()


def test_project_all_probes_finite(cosmo, count_shells, source_bins, lens_bins):
    wl = fm.WeakLensing(nz=source_bins, ia=fm.TATT(a1=1.0, a2=0.5, b_ta=0.5),
                        source_clustering_bias=1.2)
    gc = fm.GalaxyClustering(nz=lens_bins,
                             bias=fm.QuadraticBias(b1=[1.4, 1.6], b2=0.3, bs=-0.2))
    cmb = fm.CMBLensing()
    maps = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=wl, galaxy_clustering=gc, cmb_lensing=cmb)
    assert np.all(np.isfinite(maps.weak_lensing.kappa))
    assert np.all(np.isfinite(maps.weak_lensing.ia_gamma1))
    assert maps.weak_lensing.source_overdensity.std() > 0
    assert np.all(np.isfinite(maps.galaxy_clustering.delta_g))
    assert np.all(np.isfinite(maps.cmb_lensing.kappa))


def test_ia_changes_shear(cosmo, count_shells, source_bins):
    base = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins))
    with_ia = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, ia=fm.TATT(a1=2.0)))
    # IA leaves kappa untouched but adds a non-zero IA shear field.
    np.testing.assert_allclose(base.weak_lensing.kappa, with_ia.weak_lensing.kappa)
    assert with_ia.weak_lensing.ia_gamma1.std() > 0
    assert base.weak_lensing.ia_gamma1.std() == 0


def test_source_clustering_biases_convergence(cosmo, count_shells, source_bins):
    base = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins))
    sc = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, source_clustering_bias=1.5))
    # Non-isotropic n(z): the source-density-weighted kernel changes kappa.
    assert np.all(np.isfinite(sc.weak_lensing.kappa))
    assert not np.allclose(base.weak_lensing.kappa, sc.weak_lensing.kappa)
    # Zero bias -> no correction -> identical to no source clustering.
    zero = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, source_clustering_bias=0.0))
    np.testing.assert_allclose(zero.weak_lensing.kappa, base.weak_lensing.kappa,
                               atol=1e-12)


def test_tatt_density_weighting_changes_ia(cosmo, count_shells, source_bins):
    # With a2 = 0, b_TA used to be inert; exact TATT makes it source the
    # density-weighted (delta*s) operator, so it must change the IA field.
    base = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, ia=fm.TATT(a1=1.0)))
    bta = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, ia=fm.TATT(a1=1.0, b_ta=1.0)))
    assert base.weak_lensing.ia_gamma1.std() > 0
    assert not np.allclose(base.weak_lensing.ia_gamma1,
                           bta.weak_lensing.ia_gamma1)


def test_ia_source_clustering_cross_term(cosmo, count_shells, source_bins):
    ia = fm.TATT(a1=1.5, a2=0.5, b_ta=0.5)
    base = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, ia=ia))
    cross = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, ia=ia,
                                    source_clustering_bias=1.5))
    # IA x SC: local source overdensity reweights the IA field by (1+b_src*delta).
    assert np.all(np.isfinite(cross.weak_lensing.ia_gamma1))
    assert not np.allclose(base.weak_lensing.ia_gamma1,
                           cross.weak_lensing.ia_gamma1)
    # Zero bias -> factor 1 -> IA field unchanged.
    zero = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins, ia=ia,
                                    source_clustering_bias=0.0))
    np.testing.assert_allclose(zero.weak_lensing.ia_gamma1,
                               base.weak_lensing.ia_gamma1, atol=1e-12)


@pytest.mark.parametrize("noise_name", ["empirical", "gaussian", "none"])
def test_weak_lensing_observation(cosmo, count_shells, source_bins, rng, noise_name):
    wl = fm.WeakLensing(nz=source_bins, source_clustering_bias=1.0)
    maps = fm.ForwardModel(cosmo, count_shells).project(weak_lensing=wl)
    mask = np.ones(count_shells.npix)
    if noise_name == "empirical":
        cat = {i: (rng.normal(0, 0.26, 5000).astype("f4"),
                   rng.normal(0, 0.26, 5000).astype("f4"),
                   np.ones(5000, "f4"), np.ones(5000, "f4")) for i in (0, 1)}
        sn = fm.EmpiricalShapeNoise(cat)
    elif noise_name == "gaussian":
        sn = fm.GaussianShapeNoise(0.26)
    else:
        sn = fm.NoShapeNoise()

    obs = fm.WeakLensingObservation(mask, nbar=3.0, shape_noise=sn,
                                    shear_calibration=[-0.01, -0.02]).observe(
        maps.weak_lensing, rng)
    assert obs.gamma1.shape == (2, count_shells.npix)
    assert obs.kappa_recon is not None
    assert obs.n_source.sum() > 0
    assert np.all(np.isfinite(obs.gamma1))
    if noise_name == "none":
        # Signal-only: observed equals calibrated signal.
        np.testing.assert_allclose(obs.gamma1, obs.gamma1_signal)


def test_magnification_bias_adds_convergence(cosmo, count_shells, lens_bins):
    bias = fm.LinearBias(b1=[1.4, 1.6])
    base = fm.ForwardModel(cosmo, count_shells).project(
        galaxy_clustering=fm.GalaxyClustering(nz=lens_bins, bias=bias))
    mag = fm.ForwardModel(cosmo, count_shells).project(
        galaxy_clustering=fm.GalaxyClustering(
            nz=lens_bins, bias=bias, magnification=[1.0, 1.5]))
    assert np.all(np.isfinite(mag.galaxy_clustering.delta_g))
    # A non-zero coefficient (5s - 2) changes the clustering map.
    assert not np.allclose(base.galaxy_clustering.delta_g,
                           mag.galaxy_clustering.delta_g)
    # s = 0.4 -> 5s - 2 = 0, so magnification vanishes exactly.
    null = fm.ForwardModel(cosmo, count_shells).project(
        galaxy_clustering=fm.GalaxyClustering(
            nz=lens_bins, bias=bias, magnification=0.4))
    np.testing.assert_allclose(null.galaxy_clustering.delta_g,
                               base.galaxy_clustering.delta_g, atol=1e-12)


def test_reduced_shear_changes_signal(cosmo, count_shells, source_bins, rng):
    maps = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins))
    mask = np.ones(count_shells.npix)
    plain = fm.WeakLensingObservation(mask, shape_noise=fm.NoShapeNoise()).observe(
        maps.weak_lensing, rng)
    reduced = fm.WeakLensingObservation(
        mask, shape_noise=fm.NoShapeNoise(), reduced_shear=True).observe(
        maps.weak_lensing, rng)
    # g = gamma / (1 - kappa): differs from gamma but only at second order.
    expected = maps.weak_lensing.gamma1 / (
        1.0 - np.minimum(maps.weak_lensing.kappa, 0.9))
    np.testing.assert_allclose(reduced.gamma1_signal, expected, atol=1e-12)
    assert not np.allclose(plain.gamma1_signal, reduced.gamma1_signal)


def test_additive_shear_and_psf_leakage(cosmo, count_shells, source_bins, rng):
    maps = fm.ForwardModel(cosmo, count_shells).project(
        weak_lensing=fm.WeakLensing(nz=source_bins))
    mask = np.ones(count_shells.npix)
    npix = count_shells.npix
    e1_psf = rng.normal(0, 0.02, npix)
    e2_psf = rng.normal(0, 0.02, npix)
    base = fm.WeakLensingObservation(mask, shape_noise=fm.NoShapeNoise()).observe(
        maps.weak_lensing, rng)
    obs = fm.WeakLensingObservation(
        mask, shape_noise=fm.NoShapeNoise(),
        additive_shear=(0.001, -0.002),
        psf_leakage=(0.05, e1_psf, e2_psf),
    ).observe(maps.weak_lensing, rng)
    np.testing.assert_allclose(
        obs.gamma1_signal, base.gamma1_signal + 0.001 + 0.05 * e1_psf, atol=1e-12)
    np.testing.assert_allclose(
        obs.gamma2_signal, base.gamma2_signal - 0.002 + 0.05 * e2_psf, atol=1e-12)


def test_galaxy_clustering_observation(cosmo, count_shells, lens_bins, rng):
    gc = fm.GalaxyClustering(nz=lens_bins, bias=fm.LinearBias(b1=[1.4, 1.6]))
    maps = fm.ForwardModel(cosmo, count_shells).project(galaxy_clustering=gc)
    mask = np.ones(count_shells.npix)
    counts = fm.GalaxyClusteringObservation(mask, nbar=10.0).observe(
        maps.galaxy_clustering, rng)
    assert counts.shape == (2, count_shells.npix)
    assert np.all(counts >= 0)
    assert counts.sum() > 0


def test_isw_basic(cosmo, count_shells):
    maps = fm.ForwardModel(cosmo, count_shells).project(isw=fm.ISW())
    assert maps.isw is not None
    assert maps.cmb_lensing is None
    theta = maps.isw.theta
    assert theta.shape == (count_shells.npix,)
    assert np.all(np.isfinite(theta))
    # ISW couples to delta with a nontrivial kernel and goes through an
    # inverse Laplacian -> a non-degenerate map.
    assert theta.std() > 0
    np.testing.assert_allclose(theta.mean(), 0, atol=1e-10)


def test_isw_ignores_shells_above_z_max(cosmo, nside, rng):
    npix = hp.nside2npix(nside)
    z_edges = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 50.0, 400.0])
    delta = 0.1 * rng.standard_normal((len(z_edges) - 1, npix))
    full = fm.ShellStack.from_overdensity(delta, z_edges[:-1], z_edges[1:])
    low = fm.ShellStack.from_overdensity(delta[:4], z_edges[:4], z_edges[1:5])

    def theta(shells, **kw):
        return fm.ForwardModel(cosmo, shells).project(isw=fm.ISW(**kw)).isw.theta

    np.testing.assert_allclose(theta(full), theta(low), rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(theta(full, z_max=1.0), theta(low, z_max=1.0), rtol=1e-10, atol=1e-18)
    # without the limit the high-redshift shells dominate
    assert theta(full, z_max=None).std() > 3 * theta(low).std()
    assert theta(low, z_max=1.0).std() < theta(low).std()


def test_isw_suppressed_in_einstein_de_sitter(cosmo, count_shells):
    # Einstein-de Sitter (Omega_m = 1) has f = 1, so the kernel (1 - f)
    # vanishes analytically. CCL's numerical growth-rate solver leaves a small
    # residual, but it must be orders of magnitude smaller than the LambdaCDM
    # signal at the same shells.
    eds = fm.Cosmology(Omega_m=1.0, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
    eds_theta = fm.ForwardModel(eds, count_shells).project(isw=fm.ISW()).isw.theta
    lcdm_theta = fm.ForwardModel(cosmo, count_shells).project(
        isw=fm.ISW()).isw.theta
    assert eds_theta.std() < 0.05 * lcdm_theta.std()


def test_isw_signal_correlates_positively_with_low_z_density(cosmo, count_shells):
    # Late-time ISW: at fixed delta, decaying potentials transfer energy to
    # photons traversing overdensities, so Theta_ISW correlates positively
    # with the smoothed low-z density along the line of sight.
    maps = fm.ForwardModel(cosmo, count_shells).project(isw=fm.ISW())
    # Sum the low-z shells (where 1-f is largest) as a proxy for the source.
    z_mid = 0.5 * (count_shells.z_lower + count_shells.z_upper)
    low_z = z_mid < 1.0
    if not low_z.any():
        return
    proxy = np.zeros(count_shells.npix)
    for i in np.where(low_z)[0]:
        d = count_shells.overdensity(i)
        if d is not None:
            proxy += d
    proxy -= proxy.mean()
    theta = maps.isw.theta
    # Per-pixel correlation is small on white-noise shells (inverse Laplacian
    # smears most of the variance to large scales) but its sign must be
    # positive (overdensities -> decaying potentials -> blueshifted photons).
    corr = np.corrcoef(proxy, theta)[0, 1]
    assert corr > 0.0


def test_isw_weights_scale_with_omega_m(count_shells):
    # The kernel a_i ~ Omega_m * (1 - f), and 1-f also grows with smaller
    # Omega_m; net effect is that lower Omega_m universes have larger ISW.
    from kassiopeia.projections import isw_weights
    c_low = fm.Cosmology(Omega_m=0.2, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
    c_hi = fm.Cosmology(Omega_m=0.4, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
    w_low = isw_weights(c_low, count_shells.z_lower, count_shells.z_upper)
    w_hi = isw_weights(c_hi, count_shells.z_lower, count_shells.z_upper)
    # Look at low-z shells where the kernel is meaningful.
    z_mid = 0.5 * (count_shells.z_lower + count_shells.z_upper)
    low_z = z_mid < 1.0
    assert np.sum(np.abs(w_low[low_z])) > np.sum(np.abs(w_hi[low_z]))


def test_cmb_observation_with_noise(cosmo, count_shells, rng, nside):
    maps = fm.ForwardModel(cosmo, count_shells).project(cmb_lensing=fm.CMBLensing())
    ell = np.arange(3 * nside)
    obs = fm.CMBLensingObservation(
        noise=(ell, 1e-7 * np.ones_like(ell, float)),
        filter=(ell, (ell >= 2).astype(float)),
    ).observe(maps.cmb_lensing, rng)
    assert obs.shape == (count_shells.npix,)
    assert np.all(np.isfinite(obs))


def test_isw_observation_signal_only(cosmo, count_shells, rng):
    maps = fm.ForwardModel(cosmo, count_shells).project(isw=fm.ISW())
    obs = fm.ISWObservation().observe(maps.isw, rng)
    assert obs.shape == (count_shells.npix,)
    assert np.all(np.isfinite(obs))
    # apply_cmb_noise does a band-limited map2alm -> alm2map round trip; at
    # low nside the bandlimit truncation costs ~percent precision per pixel,
    # but the maps remain strongly correlated.
    assert np.corrcoef(obs, maps.isw.theta)[0, 1] > 0.999


def test_isw_observation_noise_dominates_signal(cosmo, count_shells, rng, nside):
    # CMB primary on degree scales is ~1e-10 in (Delta T/T)^2; that's orders
    # of magnitude above the late-time ISW auto-spectrum, so a realistic
    # observation has noise >> signal in the auto.
    maps = fm.ForwardModel(cosmo, count_shells).project(isw=fm.ISW())
    ell = np.arange(3 * nside)
    obs = fm.ISWObservation(
        noise=(ell, 1e-8 * np.ones_like(ell, float)),
    ).observe(maps.isw, rng)
    assert np.all(np.isfinite(obs))
    assert obs.std() > 5 * maps.isw.theta.std()


def test_isw_observation_mask(cosmo, count_shells, rng):
    maps = fm.ForwardModel(cosmo, count_shells).project(isw=fm.ISW())
    mask = np.ones(count_shells.npix)
    mask[: count_shells.npix // 3] = 0.0
    obs = fm.ISWObservation(mask=mask).observe(maps.isw, rng)
    np.testing.assert_array_equal(obs[mask == 0.0], 0.0)
    # Unmasked pixels still carry the (band-limited) signal.
    assert np.corrcoef(obs[mask == 1.0],
                       maps.isw.theta[mask == 1.0])[0, 1] > 0.999


def test_isw_observation_low_pass_filter(cosmo, count_shells, rng, nside):
    # Full-sky low-pass filter: high-l C_l of the observed map should vanish.
    maps = fm.ForwardModel(cosmo, count_shells).project(isw=fm.ISW())
    ell = np.arange(3 * nside)
    lpass = (ell <= 20).astype(float)
    obs = fm.ISWObservation(filter=(ell, lpass)).observe(maps.isw, rng)
    cl = hp.anafast(obs, lmax=3 * nside - 1)
    cl_in = hp.anafast(maps.isw.theta, lmax=3 * nside - 1)
    # Below cutoff, the filter is the identity (to SHT-roundtrip precision);
    # above, residual power from map -> alm -> map pixelisation is suppressed
    # by many orders of magnitude relative to the in-band signal.
    np.testing.assert_allclose(cl[:21], cl_in[:21], rtol=0.2)
    assert cl[21:].max() < 1e-6 * cl_in[1:21].max()
