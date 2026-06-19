"""Cross-validate forwardmodel against UFalcon on identical CosmoGrid shells.

UFalcon (https://cosmo-gitlab.phys.ethz.ch/cosmo_public/ufalcon-published) and
forwardmodel share the same line-of-sight projection machinery — in fact
forwardmodel's weight kernels are ports of UFalcon's continuous-lensing /
Dirac-lensing / continuous-clustering integrals. This script runs both on the
same prebaryonified CosmoGrid density shells and compares, restricting to the
effects present in *both* pipelines:

    * weak-lensing convergence kappa   (per source bin)
    * CMB-lensing convergence kappa
    * linear galaxy clustering delta_g (per lens bin)

All forwardmodel-only physics (TATT IA, quadratic bias, source clustering,
shape noise, baryonification) is disabled so the comparison is apples-to-apples.

The one deliberate difference under test is the *normalisation*: UFalcon scales
count shells by the box-volume ``kappa_prefactor`` / ``delta_prefactor``, while
forwardmodel converts each shell to overdensity via its own map mean
(``delta = counts/<counts> - 1``). The two agree exactly when a shell's map mean
equals the theoretical particle count ``N * V_shell / V_box``; this script
quantifies how well that holds for real lightcone shells.

Both pipelines are driven with the *same background cosmology*. UFalcon needs an
astropy cosmology and forwardmodel a pyccl one; we build both from the CosmoGrid
params with massless neutrinos so the two libraries' distances agree to <0.1%,
isolating the pipeline comparison from neutrino-convention differences.

Usage
-----
    python validation/compare_with_ufalcon.py \
        --data /home/david/Documents/research/NERSC_proposal/data \
        [--nside 256]        # downgrade for a quick check (default: native 512)
        [--save-dir out]     # dump maps + residuals
        [--no-cl]            # skip power-spectrum ratios

Run with an interpreter that has both packages. forwardmodel lives in the `fm`
conda env; UFalcon in `cosmo`. UFalcon is pure-python, so by default we add the
`cosmo` env's site-packages to sys.path and run everything under `fm`.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


DEFAULT_DATA = "/home/david/Documents/research/NERSC_proposal/data"
DEFAULT_UFALCON_PATH = "/opt/miniconda3/envs/cosmo/lib/python3.13/site-packages"

# CosmoGrid box specification (from the NERSC_proposal pipeline config).
BOX_SIZE_MPC_H = 900.0   # box side length, comoving Mpc/h
N_GRID = 832             # particles per side; n_particles = N_GRID**3
Z_CMB = 1090.0


# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------

def load_params(data_dir):
    import yaml
    with open(os.path.join(data_dir, "cosmogrid", "params.yml")) as f:
        return yaml.safe_load(f)


def build_cosmologies(p):
    """Return (forwardmodel.Cosmology, astropy cosmology) from CosmoGrid params.

    Massless neutrinos in both so pyccl and astropy distances match closely;
    this is a pipeline comparison, not a neutrino-physics comparison.
    """
    import forwardmodel as fm
    from astropy.cosmology import FlatLambdaCDM, Flatw0waCDM
    import astropy.units as u

    h = p["H0"] / 100.0
    Om0 = p["Om"]
    Ob0 = p["Ob"]
    w0 = p.get("w0", -1.0)
    wa = p.get("wa", 0.0)

    fm_cosmo = fm.Cosmology(
        Omega_c=Om0 - Ob0, Omega_b=Ob0, h=h,
        sigma8=p["s8"], n_s=p["ns"], w0=w0, wa=wa, m_nu=0.0,
        transfer_function="eisenstein_hu",  # distances only; avoids CAMB
    )

    if w0 == -1.0 and wa == 0.0:
        astropy_cosmo = FlatLambdaCDM(H0=p["H0"], Om0=Om0, Ob0=Ob0,
                                      Tcmb0=2.725 * u.K, Neff=3.046, m_nu=0.0 * u.eV)
    else:
        astropy_cosmo = Flatw0waCDM(H0=p["H0"], Om0=Om0, Ob0=Ob0, w0=w0, wa=wa,
                                    Tcmb0=2.725 * u.K, Neff=3.046, m_nu=0.0 * u.eV)
    return fm_cosmo, astropy_cosmo


def check_distance_agreement(fm_cosmo, astropy_cosmo):
    """Print the max fractional difference in chi(z) between the backends."""
    z = np.linspace(0.05, 3.5, 50)
    chi_fm = fm_cosmo.comoving_distance(z)
    chi_ap = astropy_cosmo.comoving_distance(z).value
    frac = np.max(np.abs(chi_fm - chi_ap) / chi_ap)
    print(f"pyccl vs astropy chi(z) max fractional diff: {frac:.2e}")
    return frac


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_shells_raw(data_dir, nside_out=None):
    """Load prebaryonified shells; optionally downgrade counts (power=-2)."""
    import healpy as hp
    d = np.load(os.path.join(data_dir, "cosmogrid", "baryonified_shells.npz"),
                allow_pickle=True)
    maps = d["shells"]
    info = d["shell_info"]
    z_low = info["lower_z"].astype(float)
    z_high = info["upper_z"].astype(float)
    nside = hp.npix2nside(maps.shape[1])
    if nside_out is not None and nside_out != nside:
        maps = np.array([hp.ud_grade(m, nside_out, power=-2) for m in maps])
        nside = nside_out
    return maps, z_low, z_high, nside


def load_nz(data_dir):
    from forwardmodel.loaders import des_y3
    return des_y3.load_nz(os.path.join(data_dir, "n_z",
                                       "y3_redshift_distributions.h5"))


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def compare_maps(uf, fm_map):
    """Pixel-level agreement metrics between two maps."""
    uf = np.asarray(uf, dtype=np.float64)
    fm_map = np.asarray(fm_map, dtype=np.float64)
    corr = np.corrcoef(uf, fm_map)[0, 1]
    denom = np.sum(uf * uf)
    slope = np.sum(uf * fm_map) / denom if denom > 0 else np.nan
    std_ratio = fm_map.std() / uf.std() if uf.std() > 0 else np.nan
    rms_frac = np.std(fm_map - uf) / uf.std() if uf.std() > 0 else np.nan
    return {"corr": corr, "slope(fm/uf)": slope,
            "std_ratio": std_ratio, "rms_resid/std": rms_frac}


def cl_ratio(uf, fm_map, lmax=None):
    """Mean C_ell ratio fm/uf over a mid-ell band (scale-dependence check)."""
    import healpy as hp
    nside = hp.npix2nside(len(uf))
    lmax = lmax or (2 * nside)
    cl_uf = hp.anafast(uf, lmax=lmax)
    cl_fm = hp.anafast(fm_map, lmax=lmax)
    lo, hi = 10, min(lmax, 2 * nside)
    band = (cl_uf[lo:hi] > 0)
    return float(np.median(cl_fm[lo:hi][band] / cl_uf[lo:hi][band]))


def print_row(label, m, clr=None):
    s = (f"  {label:<16} corr={m['corr']:.5f}  slope={m['slope(fm/uf)']:.5f}  "
         f"std_ratio={m['std_ratio']:.5f}  rms_resid/std={m['rms_resid/std']:.2e}")
    if clr is not None:
        s += f"  Cl_ratio={clr:.4f}"
    print(s)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA, help="NERSC_proposal data dir")
    ap.add_argument("--ufalcon-path", default=DEFAULT_UFALCON_PATH,
                    help="site-packages dir containing UFalcon (added to sys.path)")
    ap.add_argument("--nside", type=int, default=None,
                    help="downgrade shells to this nside (default: native)")
    ap.add_argument("--n-source-bins", type=int, default=4)
    ap.add_argument("--n-lens-bins", type=int, default=6)
    ap.add_argument("--lin-bias", type=float, default=1.0,
                    help="linear galaxy bias for the clustering comparison")
    ap.add_argument("--normalization", choices=["map_mean", "box"],
                    default="map_mean",
                    help="forwardmodel count normalization; 'box' uses the same "
                         "N*V_shell/V_box amplitude as UFalcon (slopes -> 1)")
    ap.add_argument("--save-dir", default=None, help="dump maps + residuals here")
    ap.add_argument("--no-cl", action="store_true", help="skip C_ell ratios")
    args = ap.parse_args()

    if args.ufalcon_path and args.ufalcon_path not in sys.path:
        sys.path.insert(0, args.ufalcon_path)

    # UFalcon 2.0 predates SciPy 1.14, which removed integrate.simps (-> simpson).
    import scipy.integrate as _si
    if not hasattr(_si, "simps"):
        _si.simps = _si.simpson

    import healpy as hp
    import forwardmodel as fm
    from UFalcon import construct_maps

    # --- cosmology ---
    p = load_params(args.data)
    fm_cosmo, astropy_cosmo = build_cosmologies(p)
    check_distance_agreement(fm_cosmo, astropy_cosmo)

    boxsize_gpc = BOX_SIZE_MPC_H / fm_cosmo.h / 1000.0  # physical Gpc
    n_particles = N_GRID ** 3

    # --- shells ---
    maps, z_low, z_high, nside = load_shells_raw(args.data, args.nside)
    print(f"shells: {maps.shape[0]} @ nside={nside}, "
          f"z=[{z_low.min():.3f}, {z_high.max():.3f}]")

    if args.normalization == "box":
        shells = fm.ShellStack.from_counts(
            maps, z_low, z_high, nside=nside, normalization="box",
            box_size=BOX_SIZE_MPC_H, n_particles=n_particles, cosmo=fm_cosmo)
    else:
        shells = fm.ShellStack.from_counts(maps, z_low, z_high, nside=nside)
    print(f"forwardmodel normalization: {args.normalization}")
    fwd = fm.ForwardModel(fm_cosmo, shells)

    # UFalcon driver (zf above the max shell z so all shells are used).
    uf = construct_maps.construct_map_cosmogrid(
        maps=maps, z_low=z_low, z_high=z_high, nside=nside,
        boxsize=boxsize_gpc, cosmo=astropy_cosmo, n_particles=n_particles,
        zi=0.0, zf=z_high.max() + 0.1,
    )

    nz = load_nz(args.data)
    save = {}

    def maybe_cl(uf_map, fm_map):
        return None if args.no_cl else cl_ratio(uf_map, fm_map)

    # --- Weak lensing convergence ---
    print("\nWeak-lensing convergence kappa:")
    for b in range(min(args.n_source_bins, len(nz["source"]))):
        nd = nz["source"][b]
        nz_arr = np.column_stack([nd.z, nd.nz])
        uf_k = uf.construct_kappa_map(nz_arr, fast_mode=True, IA=None)
        fm_k = fwd.project(weak_lensing=fm.WeakLensing(nz=[nd])).weak_lensing.kappa[0]
        print_row(f"source bin {b}", compare_maps(uf_k, fm_k), maybe_cl(uf_k, fm_k))
        save[f"wl_kappa_bin{b}_ufalcon"] = uf_k
        save[f"wl_kappa_bin{b}_forwardmodel"] = fm_k

    # --- CMB lensing convergence ---
    print("\nCMB-lensing convergence kappa (z<z_shell_max contribution only):")
    uf_cmb = uf.construct_kappa_cmb_map()
    fm_cmb = fwd.project(cmb_lensing=fm.CMBLensing(z_source=Z_CMB)).cmb_lensing.kappa
    print_row("cmb", compare_maps(uf_cmb, fm_cmb), maybe_cl(uf_cmb, fm_cmb))
    save["cmb_kappa_ufalcon"] = uf_cmb
    save["cmb_kappa_forwardmodel"] = fm_cmb

    # --- Linear galaxy clustering ---
    print(f"\nLinear galaxy clustering delta_g (b1={args.lin_bias}):")
    for b in range(min(args.n_lens_bins, len(nz["lens"]))):
        nd = nz["lens"][b]
        nz_arr = np.column_stack([nd.z, nd.nz])
        uf_d = uf.construct_delta_map(nz_arr, lin_bias=args.lin_bias)
        gc = fm.GalaxyClustering(nz=[nd], bias=fm.LinearBias(b1=args.lin_bias))
        fm_d = fwd.project(galaxy_clustering=gc).galaxy_clustering.delta_g[0]
        print_row(f"lens bin {b}", compare_maps(uf_d, fm_d), maybe_cl(uf_d, fm_d))
        save[f"clustering_bin{b}_ufalcon"] = uf_d
        save[f"clustering_bin{b}_forwardmodel"] = fm_d

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        for k, v in save.items():
            np.save(os.path.join(args.save_dir, f"{k}.npy"), v.astype(np.float32))
        print(f"\nsaved {len(save)} maps to {args.save_dir}")

    print("\nInterpretation: corr -> spatial agreement (should be ~1.0); "
          "slope/std_ratio -> amplitude agreement (the map-mean vs box-volume "
          "normalisation test); Cl_ratio -> scale dependence of any residual.")


if __name__ == "__main__":
    main()
