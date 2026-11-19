"""Optional baryonification via BaryonForge.

Displace the matter in dark-matter-only density shells to mimic baryonic
feedback, following Schneider+19. Requires BaryonForge, installed without its
version pins: ``pip install --no-deps BaryonForge`` plus the ``baryons`` extra
(its remaining dependencies).

>>> from kassiopeia.baryons import baryonify, DEFAULT_BARYON_PARAMS
>>> baryonified = baryonify(shells, halos, cosmo)

See :func:`kassiopeia.baryons.baryonforge.baryonify`.
"""

from .baryonforge import baryonify, DEFAULT_BARYON_PARAMS, HaloCatalog

__all__ = ["baryonify", "DEFAULT_BARYON_PARAMS", "HaloCatalog"]
