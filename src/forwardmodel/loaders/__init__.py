"""Optional dataset loaders for common map/catalogue formats.

These helpers read the specific data products the package was originally built
around, returning the generic ``forwardmodel`` objects (``ShellStack``,
``NumberDensity`` lists, masks, catalogues). They require the ``loaders`` extra
(``pip install forwardmodel[loaders]``): ``h5py``, ``pyyaml`` and ``astropy``.

Submodules:

* :mod:`forwardmodel.loaders.CosmoGridV1` — CosmoGridV1 shells, cosmology, halo catalogue.
* :mod:`forwardmodel.loaders.CosmoGridV1_1` — CosmoGrid v1.1 shells (DMO and baryonified).
* :mod:`forwardmodel.loaders.GowerStreet` — Gower Street simulation shells and cosmology
  (needs no extra).
* :mod:`forwardmodel.loaders.Flamingo` — FLAMINGO lightcone mass maps and cosmology
  (local files or the release's ``hdfstream`` server).
* :mod:`forwardmodel.loaders.des_y3` — DES Y3 n(z), source/lens catalogues, mask.
* :mod:`forwardmodel.loaders.kids` — KiDS lensfit shape catalogues (``astropy``).
* :mod:`forwardmodel.loaders.act_dr6` — ACT DR6 CMB-lensing products.
"""
