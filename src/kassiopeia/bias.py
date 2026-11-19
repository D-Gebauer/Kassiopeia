"""Galaxy bias models.

Two field-level models are provided:

* :class:`LinearBias` — ``delta_g = b1 * delta``.
* :class:`QuadraticBias` — adds quadratic density and tidal terms,
  ``delta_g = b1*delta + b2*(delta^2 - <delta^2>) + bs*(s^2 - <s^2>)``.

Each coefficient may be a scalar (shared across tomographic bins) or an array
(one value per bin). The model objects are plugged into
:class:`kassiopeia.probes.GalaxyClustering`; the standalone functions below
expose the same maths for ad-hoc use.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def _coeff(value, bin_index):
    """Return the per-bin coefficient: index into arrays, pass scalars through."""
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return float(arr[bin_index])


@dataclass
class LinearBias:
    """Linear galaxy bias ``delta_g = b1 * delta``.

    Parameters
    ----------
    b1 : float or array_like
        Linear bias, scalar or per tomographic bin.
    """

    b1: object = 1.0

    def b1_for(self, bin_index):
        return _coeff(self.b1, bin_index)

    @property
    def needs_tidal(self):
        return False

    @property
    def needs_quadratic_density(self):
        return False


@dataclass
class QuadraticBias:
    """Quadratic galaxy bias with density and tidal terms.

    ``delta_g = b1*delta + b2*(delta^2 - <delta^2>) + bs*(s^2 - <s^2>)``

    Parameters
    ----------
    b1 : float or array_like
        Linear bias, scalar or per bin.
    b2 : float or array_like
        Quadratic density bias (default 0).
    bs : float or array_like
        Tidal (``s^2``) bias (default 0).
    """

    b1: object = 1.0
    b2: object = 0.0
    bs: object = 0.0

    def b1_for(self, bin_index):
        return _coeff(self.b1, bin_index)

    def b2_for(self, bin_index):
        return _coeff(self.b2, bin_index)

    def bs_for(self, bin_index):
        return _coeff(self.bs, bin_index)

    @property
    def needs_tidal(self):
        return np.any(np.asarray(self.bs) != 0.0)

    @property
    def needs_quadratic_density(self):
        return np.any(np.asarray(self.b2) != 0.0)


# ---------------------------------------------------------------------------
# Standalone field-level functions (low-level API)
# ---------------------------------------------------------------------------

def quadratic_bias_terms(delta, s_squared):
    """Return the mean-subtracted quadratic basis fields.

    Returns ``(delta, delta^2 - <delta^2>, s^2 - <s^2>)`` so a per-bin bias can
    be applied afterwards without recomputing the squares.
    """
    delta2 = delta * delta
    return delta, delta2 - delta2.mean(), s_squared - s_squared.mean()


def apply_quadratic_bias(delta, delta2_ms, s2_ms, b1, b2, bs):
    """``b1*delta + b2*(delta^2 - <delta^2>) + bs*(s^2 - <s^2>)``."""
    return b1 * delta + b2 * delta2_ms + bs * s2_ms


def apply_linear_bias(delta, b1):
    """``b1 * delta``."""
    return b1 * delta
