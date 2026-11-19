"""Catalogue-based (DES / KiDS) shape noise: real positions, random rotations."""

import numpy as np
import healpy as hp
import pytest

import kassiopeia as kp


NSIDE = 8


def _catalog(rng, n=4000, nside=NSIDE, coherent=0.0, R=None):
    """Random galaxies on the sky; optional coherent e1 (a fake shear/c-term)."""
    pix = rng.integers(0, hp.nside2npix(nside), n)
    ra, dec = hp.pix2ang(nside, pix, lonlat=True)
    cat = {"ra": ra, "dec": dec,
           "e1": rng.normal(coherent, 0.25, n), "e2": rng.normal(0, 0.25, n),
           "w": rng.uniform(0.2, 2.0, n)}
    if R is not None:
        cat["R11"] = R + rng.normal(0, 0.05, n)
        cat["R22"] = R + rng.normal(0, 0.05, n)
    return cat


def _expected_var(cat, nside, norm_scale=1.0):
    """Exact per-pixel E[g1^2] = sum w^2 |e|^2 / 2 / N^2 under random phases."""
    pix = hp.ang2pix(nside, cat["ra"], cat["dec"], lonlat=True)
    npix = hp.nside2npix(nside)
    num = np.bincount(pix, cat["w"] ** 2 * (cat["e1"] ** 2 + cat["e2"] ** 2) / 2,
                      minlength=npix)
    den = (norm_scale * np.bincount(pix, cat["w"], minlength=npix)) ** 2
    out = np.zeros(npix)
    ok = den > 0
    out[ok] = num[ok] / den[ok]
    return out


def test_kids_single_galaxy_keeps_modulus(rng):
    cat = _catalog(rng, n=50)
    # One galaxy per pixel: put them on distinct pixels.
    pix = rng.choice(hp.nside2npix(NSIDE), 50, replace=False)
    cat["ra"], cat["dec"] = hp.pix2ang(NSIDE, pix, lonlat=True)
    g1, g2 = kp.KiDSShapeNoise([cat]).realisation(0, NSIDE, rng)
    np.testing.assert_allclose(np.hypot(g1[pix], g2[pix]),
                               np.hypot(cat["e1"], cat["e2"]), rtol=1e-6)
    assert np.count_nonzero(g1) == 50


def test_kids_variance_and_zero_mean(rng):
    cat = _catalog(rng, coherent=0.2)       # coherent signal must be erased
    sn = kp.KiDSShapeNoise({0: cat}, chunk_size=700)   # exercise chunking
    draws = np.array([sn.realisation(0, NSIDE, rng) for _ in range(400)])
    var_th = _expected_var(cat, NSIDE)
    occ = var_th > 0
    # Mean over realisations ~ 0 (rotations destroy the coherent e1).
    z = draws[:, 0, occ].mean(0) / np.sqrt(var_th[occ] / len(draws))
    assert abs(z.mean()) < 0.15 and 0.8 < z.std() < 1.2
    # Variance matches the exact random-phase expectation for both components.
    for c in (0, 1):
        ratio = (draws[:, c, occ] ** 2).mean(0).sum() / var_th[occ].sum()
        assert ratio == pytest.approx(1.0, abs=0.03)


def test_des_divides_by_bin_mean_response(rng):
    cat = _catalog(rng, R=0.7)
    des = kp.DESShapeNoise([cat], selection_response=0.01)
    kids = kp.KiDSShapeNoise([cat])
    R = np.sum(cat["w"] * (cat["R11"] + cat["R22"]) / 2) / cat["w"].sum() + 0.01
    assert des.mean_response(0) == pytest.approx(R)
    g_des = des.realisation(0, NSIDE, np.random.default_rng(7))
    g_kids = kids.realisation(0, NSIDE, np.random.default_rng(7))
    np.testing.assert_allclose(g_des, np.asarray(g_kids) / R, rtol=1e-6, atol=1e-12)


def test_des_per_pixel_response_and_scalar_R(rng):
    cat = _catalog(rng, R=0.7)
    pp = kp.DESShapeNoise([cat], per_pixel_response=True)
    g_pp = np.array(pp.realisation(0, NSIDE, np.random.default_rng(3)))
    num = np.array(kp.KiDSShapeNoise([cat]).realisation(0, NSIDE,
                                                        np.random.default_rng(3)))
    pix = hp.ang2pix(NSIDE, cat["ra"], cat["dec"], lonlat=True)
    npix = hp.nside2npix(NSIDE)
    wsum = np.bincount(pix, cat["w"], minlength=npix)
    wR = np.bincount(pix, cat["w"] * (cat["R11"] + cat["R22"]) / 2, minlength=npix)
    occ = wsum > 0
    np.testing.assert_allclose(g_pp[:, occ], num[:, occ] * wsum[occ] / wR[occ],
                               rtol=1e-6)
    # A scalar 'R' column is accepted instead of R11/R22, and columns can be renamed.
    alt = {"RA": cat["ra"], "DEC": cat["dec"], "e_1": cat["e1"], "e_2": cat["e2"],
           "weight": cat["w"], "R": np.full(len(cat["w"]), 0.5)}
    sn = kp.DESShapeNoise([alt], columns={"ra": "RA", "dec": "DEC", "e1": "e_1",
                                          "e2": "e_2", "w": "weight"})
    assert sn.mean_response(0) == pytest.approx(0.5)
    with pytest.raises(KeyError):
        kp.DESShapeNoise([{k: cat[k] for k in ("ra", "dec", "e1", "e2", "w")}]
                         ).realisation(0, NSIDE, rng)


def test_structured_array_catalog(rng):
    cat = _catalog(rng)
    arr = np.zeros(len(cat["w"]), dtype=[(k, ">f4") for k in cat])
    for k in cat:
        arr[k] = cat[k]
    g1, g2 = kp.KiDSShapeNoise([arr]).realisation(0, NSIDE, rng)
    assert np.all(np.isfinite(g1)) and np.count_nonzero(g1) > 0


def test_generate_masks_and_counts(rng):
    cat = _catalog(rng)
    sn = kp.KiDSShapeNoise([cat])
    mask = np.ones(hp.nside2npix(NSIDE))
    mask[: len(mask) // 2] = 0
    n_src, g1, g2 = sn.generate(0, 99.0, None, mask, rng)
    pix = hp.ang2pix(NSIDE, cat["ra"], cat["dec"], lonlat=True)
    counts = np.bincount(pix, minlength=len(mask)) * mask
    np.testing.assert_array_equal(n_src, counts)
    assert np.all(g1[mask == 0] == 0) and np.all(g2[mask == 0] == 0)
    np.testing.assert_allclose(sn.weight_map(0, NSIDE),
                               np.bincount(pix, cat["w"], minlength=len(mask)))


@pytest.mark.parametrize("cls", [kp.DESShapeNoise, kp.KiDSShapeNoise])
def test_catalog_noise_in_observation(cosmo, count_shells, source_bins, rng, nside, cls):
    wl = kp.WeakLensing(nz=source_bins)
    maps = kp.ForwardModel(cosmo, count_shells).project(weak_lensing=wl)
    cats = [_catalog(rng, n=3000, nside=nside, R=0.7) for _ in range(2)]
    obs = kp.WeakLensingObservation(np.ones(count_shells.npix), shape_noise=cls(cats)
                                    ).observe(maps.weak_lensing, rng)
    assert obs.gamma1.shape == (2, count_shells.npix)
    assert np.all(np.isfinite(obs.gamma1))
    assert not np.allclose(obs.gamma1, obs.gamma1_signal)
    assert obs.n_source.sum() == 6000


def test_des_mastercat_loader(tmp_path, rng):
    h5py = pytest.importorskip("h5py")
    n = 5000
    cols = {"ra": rng.uniform(0, 360, n), "dec": rng.uniform(-60, 0, n),
            "e_1": rng.normal(0, 0.2, n), "e_2": rng.normal(0, 0.2, n),
            "R11": rng.normal(0.7, 0.3, n), "R22": rng.normal(0.7, 0.3, n),
            "weight": rng.uniform(5, 60, n)}
    w_sh = {sh: rng.uniform(5, 60, n) for sh in ("1p", "1m", "2p", "2m")}
    sel = {f"select_bin{i + 1}": np.sort(rng.choice(n, 900, replace=False))
           for i in range(2)}
    sel.update({f"select_{sh}_bin{i + 1}": rng.permutation(
        np.sort(rng.choice(n, 880, replace=False))) for i in range(2) for sh in w_sh})
    path = tmp_path / "mastercat.h5"
    with h5py.File(path, "w") as f:
        for k, v in cols.items():
            f[f"catalog/metacal/unsheared/{k}"] = v
        for sh, v in w_sh.items():
            f[f"catalog/metacal/sheared_{sh}/weight"] = v
        for k, v in sel.items():
            f[f"index/{k}"] = v

    from kassiopeia.loaders import des_y3
    cats, R_S = des_y3.load_metacal_catalog(path, n_bins=2, chunk_rows=777)
    for i in range(2):
        idx = sel[f"select_bin{i + 1}"]
        np.testing.assert_array_equal(cats[i]["e1"], cols["e_1"][idx])
        np.testing.assert_array_equal(cats[i]["w"], cols["weight"][idx])
        np.testing.assert_array_equal(cats[i]["R22"], cols["R22"][idx])

        def m(sh, e):
            ii = sel[f"select_{sh}_bin{i + 1}"]
            return np.average(cols[e][ii], weights=w_sh[sh][ii])
        expect = 0.5 * ((m("1p", "e_1") - m("1m", "e_1"))
                        + (m("2p", "e_2") - m("2m", "e_2"))) / 0.02
        assert R_S[i] == pytest.approx(expect, rel=1e-10)
    # Plugs straight into DESShapeNoise.
    g1, _ = kp.DESShapeNoise(cats, selection_response=R_S).realisation(0, 8, rng)
    assert np.all(np.isfinite(g1))
