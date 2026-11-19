"""Simulation loaders, on small synthetic files in the format of the data releases."""
import healpy as hp
import numpy as np
import pytest

import kassiopeia as kp
from kassiopeia.loaders import Flamingo as fl, GowerStreet as gs

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
    cosmo = kp.Cosmology(Omega_m=0.3, sigma8=0.8)
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


# ---------------------------------------------------------------------------
# FLAMINGO
# ---------------------------------------------------------------------------

FLAMINGO_YML = """MetaData:
  run_name:   L1000N1800-Hydro-Fiducial

# Cosmological parameters
Cosmology:
  h:              0.681        # Reduced Hubble constant
  a_begin:        0.03125      # Initial scale-factor of the simulation
  Omega_cdm:      0.256011     # Matter density parameter
  Omega_lambda:   0.693922     # Dark-energy density parameter
  Omega_b:        0.048600     # Baryon density parameter
  N_ur:           {N_ur}       # Number of ultra-relativistic species
  N_nu:           1            # Integer number of massive neutrinos.
  M_nu_eV:        {M_nu_eV}    # Comma-separated list of N_nu nonzero neutrino masses in electron-volts
  deg_nu:         {deg_nu}     # Comma-separated list of N_nu neutrino degeneracies

# Parameters governing the time integration
TimeIntegration:
  dt_min:     1e-10
  h:          123.0
"""
FLAMINGO_Z = "# Minimum redshift, Maximum redshift\n0.0006815781, 0.05\n0.05, 0.10\n0.10, 0.15\n0.15, 0.20\n"


@pytest.fixture
def flamingo_run(tmp_path):
    """A run directory in the layout of the data release, with two maps per shell."""
    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(2)
    (tmp_path / "shell_redshifts_z3.txt").write_text(FLAMINGO_Z)
    maps_dir = tmp_path / "healpix_maps" / "nside_4096"
    for shell in range(4):
        d = maps_dir / "lightcone0_shells" / f"shell_{shell}"
        d.mkdir(parents=True)
        with h5py.File(d / f"lightcone0.shell_{shell}.0.hdf5", "w") as f:
            dm = rng.lognormal(0.0, 0.5, hp.nside2npix(NSIDE)) * 10.0 ** shell
            f["DarkMatterMass"] = dm
            f["TotalMass"] = 1.2 * dm
            f["TotalMass"].attrs["pixel_ordering_scheme"] = "ring"
            f.create_group("Shell").attrs["comoving_inner_radius"] = [3.0]
    return tmp_path, maps_dir


def test_flamingo_shells(flamingo_run):
    import h5py

    run, maps_dir = flamingo_run
    info = fl.load_shell_redshifts(run / "shell_redshifts_z3.txt")
    assert len(info) == 4 and info["lower_z"][0] == pytest.approx(0.0006815781)

    shells = fl.load_shells(maps_dir, run / "shell_redshifts_z3.txt")
    assert shells.kind == "counts" and shells.n_shells == 4 and shells.nside == NSIDE
    assert shells.data.dtype == np.float32
    np.testing.assert_allclose(shells.z_upper, [0.05, 0.10, 0.15, 0.20])
    with h5py.File(fl.shell_file(maps_dir, 2), "r") as f:
        truth = f["TotalMass"][...]
    np.testing.assert_allclose(shells.data[2], truth, rtol=1e-6)
    np.testing.assert_allclose(shells.overdensity(2), truth / truth.mean() - 1.0, atol=1e-5)

    dm = fl.load_shells(maps_dir, info, field="DarkMatterMass", z_max=0.12, nside=4)
    assert dm.n_shells == 3 and dm.nside == 4
    np.testing.assert_allclose(dm.data.sum(axis=1), shells.data[:3].sum(axis=1) / 1.2, rtol=1e-5)


def test_flamingo_remote_and_cache(flamingo_run, tmp_path_factory):
    import h5py

    run, maps_dir = flamingo_run
    local = fl.load_shells(maps_dir, run / "shell_redshifts_z3.txt", nside=8)

    class Remote:                      # stands in for hdfstream.open("cosma", "/")
        calls = 0

        def __getitem__(self, path):
            Remote.calls += 1
            with h5py.File(run / path, "r") as f:
                return {k: f[k][...] for k in ("TotalMass", "DarkMatterMass")}

    cache = tmp_path_factory.mktemp("cache")
    kw = dict(redshifts=run / "shell_redshifts_z3.txt", nside=8, remote=Remote(), cache_dir=cache)
    remote = fl.load_shells("healpix_maps/nside_4096", **kw)
    np.testing.assert_array_equal(remote.data, local.data)
    assert Remote.calls == 4 and len(list(cache.glob("lightcone0_TotalMass_nside8_shell*.npy"))) == 4
    again = fl.load_shells("healpix_maps/nside_4096", **kw)          # served from the cache
    np.testing.assert_array_equal(again.data, local.data)
    assert Remote.calls == 4


def test_flamingo_cosmology(tmp_path):
    yml = tmp_path / "flamingo.yml"
    yml.write_text(FLAMINGO_YML.format(N_ur="2.0308", M_nu_eV="0.06", deg_nu="1.0"))
    p = fl.load_params(yml)
    assert p["h"] == 0.681 and p["Omega_cdm"] == 0.256011 and "dt_min" not in p
    cosmo = fl.load_cosmology(yml)
    assert cosmo.Omega_m == pytest.approx(0.306, abs=2e-4)          # CDM + baryons + neutrinos
    assert list(cosmo.ccl["m_nu"]) == [0.06]                        # one massive species
    assert float(cosmo.ccl["sigma8"]) == 0.807 and float(cosmo.ccl["n_s"]) == 0.967
    assert float(fl.load_cosmology(yml, A_s=2.099e-9).ccl["A_s"]) == 2.099e-9

    # three degenerate species of 0.08 eV; this combination is not a FLAMINGO cosmology
    yml.write_text(FLAMINGO_YML.format(N_ur="0.00441", M_nu_eV="0.08", deg_nu="3.0"))
    with pytest.raises(ValueError):
        fl.load_cosmology(yml)
    cosmo = fl.load_cosmology(yml, sigma8=0.8, n_s=0.96)
    np.testing.assert_allclose(cosmo.ccl["m_nu"], [0.08, 0.08, 0.08])

    with pytest.raises(ValueError):                                 # decaying dark matter
        fl.cosmology_from_params({**p, "Omega_cdm": 0.0, "Omega_dcdmdr_0": 0.246, "Gamma_dcdm": 12.0})
