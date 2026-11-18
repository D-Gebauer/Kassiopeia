"""Simulation loaders, on small synthetic files in the format of the data releases."""
import healpy as hp
import numpy as np
import pytest

import forwardmodel as fm
from forwardmodel.loaders import GowerStreet as gs

NSIDE = 16
Z_VALUES_HEADER = ("# Step,z_far,z_near,delta_z,cmd_far(Mpc/h),cmd_near(Mpc/h),delta_cmd(Mpc/h),"
                   "cmd/box_far,cmd/box_near,delta_cmd/box")
Z_VALUES_ROW = "{step},{z_far:.6f},{z_near:.6f},0.010000,{cmd_far:.6f},{cmd_near:.6f},30.000000,0.1,0.1,0.02"
# step, z_far, z_near of the shells with a lightcone file (far to near)
Z_STEPS = [(95, 0.20, 0.15), (96, 0.15, 0.11), (97, 0.11, 0.07), (98, 0.07, 0.03), (99, 0.03, 0.01),
           (100, 0.01, -0.0)]
SHEET = (
    ",,,Cosmological parameters,,,,,,,Derived parameters,,,\n"
    "Serial Number,Run,Job,Omega_m,sigma_8,w,Omega_b little_h^2,little_h,n_s,m_nu,Omega_b,Comment,"
    "RunTime (hours),WhereRun\n"
    "1,C,1,0.2900404837,0.7657514029,-1.008941473,0.02236562735,0.6673620556,0.949651544,0.06,"
    "0.0502178441,Verification run - naive treatment of neutrinos,19.58333,Wilkes\n"
    "400,L,5,0.3888201691,0.7845464443,-0.793383223,0.02213624137,0.682604961,0.9560925225,"
    "0.08734362307,0.04750780893,,24.78333,Wilkes\n")


@pytest.fixture
def sim_dir(tmp_path):
    rng = np.random.default_rng(1)
    lines = [Z_VALUES_HEADER]
    for step, z_far, z_near in [(94, 0.26, 0.20)] + Z_STEPS:
        lines.append(Z_VALUES_ROW.format(step=step, z_far=z_far, z_near=z_near,
                                         cmd_far=3000 * z_far, cmd_near=3000 * z_near))
    (tmp_path / "z_values.txt").write_text("\n".join(lines) + "\n")
    for step, _, _ in Z_STEPS:
        np.save(tmp_path / f"run.{step:05d}.lightcone.npy",
                rng.poisson(50 + step, hp.nside2npix(NSIDE)).astype(np.uint16))
    np.save(tmp_path / "run.00094.incomplete.npy", np.zeros(hp.nside2npix(NSIDE), np.uint16))
    (tmp_path / "control.par").write_text(
        "import math\n\nachOutName      = \"run\"\n# Initial Condition\n"
        "dBoxSize        = 1250       # Mpc/h\nnGrid           = 1080       # nGrid^3 particles\n"
        "w0              = 2.0*math.cos(2.0995652032775993)  # -1.008941472576396\n")
    return tmp_path


def test_gower_street_shells(sim_dir):
    info = gs.load_shell_info(sim_dir)
    assert list(info["step"]) == [94, 95, 96, 97, 98, 99, 100]
    assert info["lower_z"][-1] == 0.0 and not np.signbit(info["lower_z"][-1])
    assert info["upper_z"][0] == 0.26

    shells = gs.load_shells(sim_dir)
    assert shells.kind == "counts" and shells.data.dtype == np.uint16
    assert shells.n_shells == 6 and shells.nside == NSIDE          # incomplete step 94 left out
    np.testing.assert_allclose(shells.z_lower, [0.0, 0.01, 0.03, 0.07, 0.11, 0.15])
    np.testing.assert_allclose(shells.z_upper, [0.01, 0.03, 0.07, 0.11, 0.15, 0.20])
    np.testing.assert_array_equal(shells.data[0], np.load(sim_dir / "run.00100.lightcone.npy"))

    near = gs.load_shells(sim_dir, z_max=0.05)
    assert near.n_shells == 3 and near.z_max == pytest.approx(0.07)

    low = gs.load_shells(sim_dir, nside=4)
    assert low.nside == 4 and np.issubdtype(low.data.dtype, np.integer)
    np.testing.assert_array_equal(low.data.sum(axis=1), shells.data.sum(axis=1, dtype=np.int64))
    np.testing.assert_allclose(low.data[2], hp.ud_grade(shells.data[2].astype(float), 4, power=-2))

    with pytest.raises(ValueError):
        gs.load_shells(sim_dir, nside=32)


def test_gower_street_box_normalization(sim_dir):
    assert gs.load_control(sim_dir)["dBoxSize"] == 1250
    cosmo = fm.Cosmology(Omega_m=0.3, sigma8=0.8)
    shells = gs.load_shells(sim_dir, normalization="box", cosmo=cosmo)
    assert shells.normalization == "box"
    assert shells.box_size == 1250.0 and shells.n_particles == 1080**3
    assert np.isfinite(shells.overdensity(0)).all()


def test_gower_street_cosmology(tmp_path):
    csv = tmp_path / "sheet1.csv"
    csv.write_text(SHEET)
    p = gs.load_params(csv, 400)
    assert p["Serial Number"] == 400 and p["Run"] == "L" and p["m_nu"] == pytest.approx(0.08734362307)
    with pytest.raises(KeyError):
        gs.load_params(csv, 2)

    # runs 193+: Omega_m includes the massive neutrinos; control.par has dOmega0 = 0.386804 (CDM + b)
    cosmo = gs.load_cosmology(csv, 400)
    assert cosmo.Omega_m == pytest.approx(0.3888201691, rel=1e-5)
    assert float(cosmo.ccl["Omega_c"] + cosmo.ccl["Omega_b"]) == pytest.approx(0.386804, rel=2e-5)
    assert float(np.sum(cosmo.ccl["m_nu"])) == pytest.approx(0.08734362307)
    assert float(cosmo.ccl["w0"]) == pytest.approx(-0.793383223)

    # verification runs: Omega_m = CDM + baryons = dOmega0, no neutrinos in the N-body
    naive = gs.load_cosmology(csv, 1)
    assert naive.Omega_m == pytest.approx(0.2900404837, rel=1e-8)
    assert gs.load_cosmology(csv, 1, m_nu=0.06).Omega_m > naive.Omega_m
