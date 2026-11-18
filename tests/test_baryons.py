"""BaryonForge wrapper: pyccl compatibility shims and an end-to-end baryonification."""

import numpy as np
import healpy as hp
import pytest

import forwardmodel as fm


def test_concentration_accepts_mdef():
    """BaryonForge 1.0 passes ``mdef=`` to the pyccl concentrations (``mass_def`` since pyccl 3)."""
    import forwardmodel.baryons  # noqa: F401  (installs the shims)
    from pyccl.halos import concentration as conc

    assert conc.ConcentrationDiemer15(mdef="200c").mass_def.name == "200c"
    assert conc.ConcentrationConstant(5.0, mdef="200c").mass_def.name == "200c"
    assert conc.ConcentrationDiemer15(mass_def="200c").mass_def.name == "200c"


def test_baryonify_conserves_mass(cosmo):
    pytest.importorskip("BaryonForge")
    from forwardmodel.baryons import baryonify, HaloCatalog

    rng = np.random.default_rng(1)
    nside, n_halo = 32, 20
    z_edges = np.array([0.2, 0.3, 0.4])
    counts = rng.poisson(200, (2, hp.nside2npix(nside))).astype(np.float64)
    shells = fm.ShellStack.from_counts(counts, z_edges[:-1], z_edges[1:])
    halos = HaloCatalog(ra=rng.uniform(0, 360, n_halo),
                        dec=np.degrees(np.arcsin(rng.uniform(-1, 1, n_halo))),
                        M=10 ** rng.uniform(14, 15, n_halo), z=rng.uniform(0.21, 0.39, n_halo))

    out = baryonify(shells, halos, cosmo, verbose=False)

    assert out.kind == "counts" and out.data.shape == counts.shape
    assert np.isfinite(out.data).all()
    assert (out.data != counts).any()                                   # matter was displaced
    assert np.isclose(out.data.sum(dtype=float), counts.sum(), rtol=1e-6)   # and conserved
