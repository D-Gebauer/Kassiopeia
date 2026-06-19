"""Optional dataset loaders for common map/catalogue formats.

These helpers read the specific data products the package was originally built
around, returning the generic ``forwardmodel`` objects (``ShellStack``,
``NumberDensity`` lists, masks, catalogues). They require the ``loaders`` extra
(``pip install forwardmodel[loaders]``): ``h5py`` and ``pyyaml``.

Submodules:

* :mod:`forwardmodel.loaders.cosmogrid` — CosmoGrid shells and cosmology.
* :mod:`forwardmodel.loaders.des_y3` — DES Y3 n(z), source/lens catalogues, mask.
* :mod:`forwardmodel.loaders.act_dr6` — ACT DR6 CMB-lensing products.
"""
