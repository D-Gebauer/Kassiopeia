"""Optional baryonification via BaryonForge.

Displace the matter in dark-matter-only density shells to mimic baryonic
feedback, following Schneider+19. Requires the ``baryons`` extra
(``pip install forwardmodel[baryons]``).

>>> from forwardmodel.baryons import baryonify, DEFAULT_BARYON_PARAMS
>>> baryonified = baryonify(shells, halos, cosmo)

See :func:`forwardmodel.baryons.baryonforge.baryonify`.
"""

from .baryonforge import baryonify, DEFAULT_BARYON_PARAMS, HaloCatalog

__all__ = ["baryonify", "DEFAULT_BARYON_PARAMS", "HaloCatalog"]
