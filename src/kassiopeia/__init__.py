"""kassiopeia — forward-model multi-probe cosmological maps from density shells.

Generate realistic weak-lensing, galaxy-clustering, and CMB-lensing maps from
N-body (or lognormal) HEALPix density shells, with tidal tensors, TATT
intrinsic alignments, quadratic galaxy bias, and source clustering.

Quick start
-----------
>>> import numpy as np, kassiopeia as kp
>>> cosmo = kp.Cosmology(Omega_m=0.3, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
>>> shells = kp.ShellStack.from_counts(counts, z_lower, z_upper)
>>> wl = kp.WeakLensing(nz=source_bins, ia=kp.TATT(a1=0.5))
>>> maps = kp.ForwardModel(cosmo, shells).project(weak_lensing=wl)

No N-body lightcone for this cosmology? Generate cross-correlated lognormal
shells from the cosmology alone and run the same pipeline (needs ``glass``):

>>> model = kp.ForwardModel.from_lognormal(cosmo, nside=512, z_max=3.0, dz=0.1)
>>> maps = model.project(weak_lensing=wl)
>>> obs = kp.WeakLensingObservation(mask, nbar=0.5,
...           shape_noise=kp.EmpiricalShapeNoise(catalog)).observe(
...           maps.weak_lensing, np.random.default_rng(0))

The high-level objects above cover the common case; every building block
(:func:`compute_tidal`, :func:`kappa_to_gamma`, the weight builders in
:mod:`kassiopeia.projections`, the noise models in :mod:`kassiopeia.noise`,
...) is also importable directly for custom pipelines.

Optional features live behind extras: ``kassiopeia[lognormal]`` (GLASS shell
extension), ``kassiopeia[loaders]`` (CosmoGrid/DES Y3/KiDS/ACT DR6 readers),
``kassiopeia[rotation]`` (ducc0 for fast rotations), and
``kassiopeia[baryons]`` (dependencies of BaryonForge baryonification; see
:mod:`kassiopeia.baryons`).
"""

from importlib.metadata import PackageNotFoundError, version as _version

from .cosmology import Cosmology
from .shells import ShellStack
from .nz import NumberDensity, make_bins
from .bias import (
    LinearBias, QuadraticBias,
    quadratic_bias_terms, apply_quadratic_bias, apply_linear_bias,
)
from .ia import TATT, NLA, tatt_amplitudes
from .tidal import compute_tidal
from .kaiser_squires import kappa_to_gamma, gamma_to_kappa
from .probes import (
    WeakLensing, GalaxyClustering, CMBLensing, ISW,
    WeakLensingMaps, GalaxyClusteringMaps, CMBLensingMaps, ISWMaps,
    ProjectedMaps,
)
from .model import ForwardModel
from .lognormal import lognormal_shells
from .noise import (
    ShapeNoise, NoShapeNoise, GaussianShapeNoise, EmpiricalShapeNoise,
    CatalogShapeNoise, DESShapeNoise, KiDSShapeNoise,
    poisson_counts, apply_cmb_noise,
)
from .observation import (
    WeakLensingObservation, GalaxyClusteringObservation, CMBLensingObservation,
    ISWObservation, ObservedWeakLensing, compose_ellipticity,
)

try:
    __version__ = _version("kassiopeia")
except PackageNotFoundError:   # source tree that was never installed
    __version__ = "unknown"

__all__ = [
    # core
    "Cosmology", "ShellStack", "NumberDensity", "make_bins",
    # bias
    "LinearBias", "QuadraticBias",
    "quadratic_bias_terms", "apply_quadratic_bias", "apply_linear_bias",
    # IA
    "TATT", "NLA", "tatt_amplitudes",
    # physics building blocks
    "compute_tidal", "kappa_to_gamma", "gamma_to_kappa",
    # probes & maps
    "WeakLensing", "GalaxyClustering", "CMBLensing", "ISW",
    "WeakLensingMaps", "GalaxyClusteringMaps", "CMBLensingMaps", "ISWMaps",
    "ProjectedMaps",
    # orchestrator
    "ForwardModel",
    # lognormal shells (no N-body input)
    "lognormal_shells",
    # noise
    "ShapeNoise", "NoShapeNoise", "GaussianShapeNoise", "EmpiricalShapeNoise",
    "CatalogShapeNoise", "DESShapeNoise", "KiDSShapeNoise",
    "poisson_counts", "apply_cmb_noise",
    # observation
    "WeakLensingObservation", "GalaxyClusteringObservation",
    "CMBLensingObservation", "ISWObservation", "ObservedWeakLensing",
    "compose_ellipticity",
    "__version__",
]
