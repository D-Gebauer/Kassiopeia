"""Density shells: the line-of-sight stack the whole forward model integrates.

A :class:`ShellStack` is a set of full-sky HEALPix maps, one per radial bin,
each tagged with its redshift bounds ``[z_lower, z_upper)``. Shells may be
stored either as **particle counts** (the native N-body lightcone product, e.g.
CosmoGrid/UFalcon) or as **overdensity** ``delta``. Internally everything is
projected in overdensity form: a count shell is converted on the fly via its
own map mean, ``delta_i = counts_i / <counts_i> - 1``.

That single convention is why this package needs no box size or particle count:
those only ever entered the original UFalcon prefactors to undo the count
normalization, and the per-shell map mean does that directly. The projection
weights (see :mod:`kassiopeia.projections`) are therefore purely geometric —
they depend only on the shell redshift bounds and the cosmology.

The map-mean default is robust (self-calibrating, needs no box metadata, works
for lognormal or real-survey shells). For an *exact* UFalcon/CosmoGrid amplitude
match you can opt into ``normalization="box"``, which instead normalizes each
count shell by the theoretical mean ``N * V_shell / V_box`` — the cosmic mean the
box was built with. That requires ``box_size``, ``n_particles`` and a
``cosmo`` (to turn redshift bounds into comoving shell volumes). The two agree
when a shell's realized map mean equals that box expectation; where they differ
(a few percent in the high-z shells of a lightcone) ``"box"`` gives the
amplitude tied to the simulation's known density.
"""

from __future__ import annotations

import numpy as np
import healpy as hp


class ShellStack:
    """A redshift-ordered stack of full-sky density shells.

    Parameters
    ----------
    data : ndarray, shape (n_shells, npix)
        Per-shell HEALPix maps. Either particle counts or overdensity,
        depending on ``kind``. The dtype is preserved (count shells are often
        ``uint16``); conversion to ``float64`` overdensity happens per shell at
        projection time to keep memory down.
    z_lower, z_upper : array_like, shape (n_shells,)
        Redshift bounds of each shell.
    kind : {"counts", "overdensity"}
        How to interpret ``data``.
    nside : int, optional
        HEALPix resolution. Inferred from ``data`` if omitted.
    normalization : {"map_mean", "box"}
        How count shells are turned into overdensity. ``"map_mean"`` (default)
        divides by each shell's own map mean. ``"box"`` divides by the
        theoretical box expectation ``N * V_shell / V_box`` (the UFalcon /
        CosmoGrid amplitude); it requires ``box_size``, ``n_particles`` and
        ``cosmo``. Ignored for overdensity shells.
    box_size : float, optional
        Comoving box side length in **Mpc/h** (CosmoGrid convention). Required
        for ``normalization="box"``.
    n_particles : int, optional
        Total particle number in the box (e.g. ``N_grid**3``). Required for
        ``normalization="box"``.
    cosmo : kassiopeia.Cosmology, optional
        Needed by ``normalization="box"`` to convert redshift bounds into
        comoving shell volumes.
    """

    def __init__(self, data, z_lower, z_upper, kind="counts", nside=None,
                 normalization="map_mean", box_size=None, n_particles=None,
                 cosmo=None):
        if kind not in ("counts", "overdensity"):
            raise ValueError(f"kind must be 'counts' or 'overdensity', got {kind!r}")
        if normalization not in ("map_mean", "box"):
            raise ValueError(
                f"normalization must be 'map_mean' or 'box', got {normalization!r}")
        data = np.asarray(data)
        if data.ndim != 2:
            raise ValueError("data must be 2-D (n_shells, npix)")
        z_lower = np.asarray(z_lower, dtype=float)
        z_upper = np.asarray(z_upper, dtype=float)
        if not (len(z_lower) == len(z_upper) == data.shape[0]):
            raise ValueError("z_lower, z_upper must match the number of shells")

        if nside is None:
            nside = hp.npix2nside(data.shape[1])
        elif hp.nside2npix(nside) != data.shape[1]:
            raise ValueError(f"nside={nside} inconsistent with npix={data.shape[1]}")

        if normalization == "box":
            if kind != "counts":
                raise ValueError(
                    "normalization='box' only applies to count shells")
            if box_size is None or n_particles is None or cosmo is None:
                raise ValueError(
                    "normalization='box' requires box_size, n_particles and cosmo")

        self.data = data
        self.z_lower = z_lower
        self.z_upper = z_upper
        self.kind = kind
        self.nside = int(nside)
        self.normalization = normalization
        self.box_size = box_size
        self.n_particles = n_particles
        self.cosmo = cosmo
        self._box_mean_cache = None

    # -- constructors -------------------------------------------------------

    @classmethod
    def from_counts(cls, counts, z_lower, z_upper, nside=None,
                    normalization="map_mean", box_size=None, n_particles=None,
                    cosmo=None):
        """Build a stack from particle-count shells.

        Pass ``normalization="box"`` together with ``box_size`` (Mpc/h),
        ``n_particles`` and ``cosmo`` for the UFalcon/CosmoGrid box-volume
        amplitude instead of the default per-shell map mean.
        """
        return cls(counts, z_lower, z_upper, kind="counts", nside=nside,
                   normalization=normalization, box_size=box_size,
                   n_particles=n_particles, cosmo=cosmo)

    @classmethod
    def from_overdensity(cls, delta, z_lower, z_upper, nside=None):
        """Build a stack from overdensity (delta) shells."""
        return cls(delta, z_lower, z_upper, kind="overdensity", nside=nside)

    @classmethod
    def from_lognormal(cls, cosmo, z_edges, nside, rng=None, lmax=None,
                       shift=None):
        """Generate a fully lognormal overdensity stack (no N-body input).

        The entry point for running the forward model at a cosmology where no
        N-body lightcone exists: every shell is a *cross-correlated* GLASS
        lognormal realization carrying the correct angular power and
        cross-shell correlation derived from CCL. Sits alongside
        :meth:`from_counts` / :meth:`from_overdensity`; for a ready-to-project
        model in one call see :meth:`ForwardModel.from_lognormal`. Requires the
        optional ``glass`` dependency.

        Parameters
        ----------
        cosmo : kassiopeia.Cosmology
        z_edges : array_like
            Redshift bin edges; ``len(z_edges) - 1`` shells are produced.
        nside : int
            HEALPix resolution (no data to infer it from, so required).
        rng : np.random.Generator, optional
        lmax : int, optional
            Defaults to ``3 * nside - 1``.
        shift : callable, optional
            Lognormal shift ``shift(z) -> float``; defaults to GLASS's standard
            shift of 1.
        """
        from .lognormal import lognormal_shells
        return lognormal_shells(cosmo, z_edges, nside=nside, rng=rng,
                                lmax=lmax, shift=shift)

    # -- properties ---------------------------------------------------------

    @property
    def n_shells(self):
        return self.data.shape[0]

    @property
    def npix(self):
        return self.data.shape[1]

    @property
    def z_mid(self):
        """Midpoint redshift of each shell."""
        return 0.5 * (self.z_lower + self.z_upper)

    @property
    def z_min(self):
        return float(self.z_lower.min())

    @property
    def z_max(self):
        return float(self.z_upper.max())

    # -- per-shell access ---------------------------------------------------

    def _box_mean_pix(self, i):
        """Theoretical per-pixel mean count ``N * V_shell_i / V_box / npix``.

        Volumes are formed in physical comoving Mpc: the shell volume from the
        cosmology's ``chi(z)`` and the box volume from ``box_size`` (Mpc/h)
        divided by ``h``. Cached across shells on first call.
        """
        if self._box_mean_cache is None:
            chi_lo = self.cosmo.comoving_distance(self.z_lower)
            chi_hi = self.cosmo.comoving_distance(self.z_upper)
            v_shell = (4.0 / 3.0) * np.pi * (chi_hi**3 - chi_lo**3)  # Mpc^3
            box_phys = self.box_size / self.cosmo.h                  # Mpc/h -> Mpc
            v_box = box_phys**3
            self._box_mean_cache = (
                self.n_particles * v_shell / v_box / self.npix)
        return float(self._box_mean_cache[i])

    def overdensity(self, i):
        """Return the overdensity field of shell ``i`` as ``float64``.

        For count shells this is ``counts / mean - 1``. With the default
        ``normalization="map_mean"`` the denominator is the shell's own map
        mean; with ``normalization="box"`` it is the box expectation
        ``N * V_shell / V_box`` (per pixel). Returns ``None`` for an empty
        (all-zero) count shell.
        """
        shell = np.asarray(self.data[i], dtype=np.float64)
        if self.kind == "overdensity":
            return shell
        if self.normalization == "box":
            if shell.sum() <= 0:
                return None
            mean = self._box_mean_pix(i)
        else:
            mean = shell.mean()
        if mean <= 0:
            return None
        return shell / mean - 1.0

    def to_overdensity(self):
        """Realize a count stack as an explicit overdensity stack.

        Applies the configured ``normalization`` shell by shell; empty shells
        become all-zero overdensity. Returns ``self`` unchanged if already
        overdensity. Useful before merging a box-normalized stack with
        overdensity (e.g. lognormal) shells.
        """
        if self.kind == "overdensity":
            return self
        delta = np.zeros((self.n_shells, self.npix), dtype=np.float64)
        for i in range(self.n_shells):
            d = self.overdensity(i)
            if d is not None:
                delta[i] = d
        return ShellStack(delta, self.z_lower, self.z_upper,
                          kind="overdensity", nside=self.nside)

    # -- combination --------------------------------------------------------

    def sorted_by_redshift(self):
        """Return a copy with shells sorted by lower redshift edge.

        N-body lightcone products are frequently stored in shell-id order,
        which is *not* monotonic in redshift; downstream projection assumes
        ordering is irrelevant but sorting keeps diagnostics readable.
        """
        order = np.argsort(self.z_lower)
        return ShellStack(self.data[order], self.z_lower[order],
                          self.z_upper[order], kind=self.kind, nside=self.nside,
                          normalization=self.normalization,
                          box_size=self.box_size, n_particles=self.n_particles,
                          cosmo=self.cosmo)

    def merge(self, other):
        """Concatenate with another stack and sort jointly by redshift.

        Both stacks must share ``nside``. If their ``kind`` differs, ``other``
        is converted to match this stack: an overdensity stack appended to a
        count stack is wrapped as pseudo-counts ``1 + delta`` (mean 1), which
        recovers ``delta`` exactly under the count->overdensity conversion.

        That pseudo-count trick assumes map-mean normalization. If this stack
        uses ``normalization="box"`` and ``other`` is a different kind (e.g. a
        lognormal overdensity extension), this stack is first realized as
        explicit overdensity via its box means, so the result is an overdensity
        stack and amplitudes stay correct.
        """
        if other.nside != self.nside:
            raise ValueError("cannot merge stacks with different nside")

        if self.normalization == "box" and other.kind != self.kind:
            return self.to_overdensity().merge(other)

        other_data = other.data
        if other.kind != self.kind:
            if self.kind == "counts":
                # other is overdensity -> pseudo-counts (mean 1)
                other_data = (1.0 + np.asarray(other_data, dtype=np.float64))
            else:
                # other is counts -> overdensity, per shell
                other_data = np.stack([
                    other.overdensity(i) if other.overdensity(i) is not None
                    else np.zeros(other.npix)
                    for i in range(other.n_shells)
                ])

        data = np.concatenate([self.data, other_data], axis=0)
        z_lo = np.concatenate([self.z_lower, other.z_lower])
        z_hi = np.concatenate([self.z_upper, other.z_upper])
        order = np.argsort(z_lo)
        return ShellStack(data[order], z_lo[order], z_hi[order],
                          kind=self.kind, nside=self.nside,
                          normalization=self.normalization,
                          box_size=self.box_size, n_particles=self.n_particles,
                          cosmo=self.cosmo)

    def extend_lognormal(self, cosmo, z_edges, rng=None, lmax=None,
                         shift=None):
        """Append GLASS lognormal shells covering ``z_edges`` and return a new stack.

        Used to extend an N-body lightcone (typically capped near ``z ~ 3.5``)
        out to the redshifts the CMB-lensing kernel and high-z n(z) tails need.
        Requires the optional ``glass`` dependency.

        Parameters
        ----------
        cosmo : kassiopeia.Cosmology
        z_edges : array_like
            Redshift bin edges; ``len - 1`` shells are produced.
        rng : np.random.Generator, optional
        lmax : int, optional
            Defaults to ``3 * nside - 1``.
        shift : callable, optional
            Lognormal shift ``shift(z) -> float``; defaults to GLASS's standard
            shift of 1, matching ``delta = e^{G - s^2/2} - 1``.
        """
        from .lognormal import lognormal_shells
        ln = lognormal_shells(cosmo, z_edges, nside=self.nside, rng=rng,
                              lmax=lmax, shift=shift)
        return self.merge(ln)

    def coverage_gaps(self, z_min=0.0, z_max=None, min_gap=1e-3):
        """Redshift intervals in ``[z_min, z_max]`` not spanned by any shell.

        Existing shells are merged into covered intervals; the complement is
        returned as a list of ``(lo, hi)`` gaps (the low-z range below the first
        shell, any interior gaps, and the high-z tail above the last shell).
        ``z_max`` defaults to the current :attr:`z_max`.
        """
        if z_max is None:
            z_max = self.z_max
        if z_max <= z_min:
            return []

        order = np.argsort(self.z_lower)
        covered = []
        for i in order:
            lo, hi = float(self.z_lower[i]), float(self.z_upper[i])
            if not covered or lo > covered[-1][1] + min_gap:
                covered.append([lo, hi])
            else:
                covered[-1][1] = max(covered[-1][1], hi)

        gaps = []
        cursor = z_min
        for lo, hi in covered:
            if lo - cursor > min_gap:
                gaps.append((cursor, min(lo, z_max)))
            cursor = max(cursor, hi)
            if cursor >= z_max:
                break
        if cursor < z_max - min_gap:
            gaps.append((cursor, z_max))
        return [(a, b) for a, b in gaps if b - a > min_gap]

    def fill_with_lognormal(self, cosmo, z_max=None, z_min=0.0, dz=None,
                            rng=None, lmax=None, shift=None, min_gap=1e-3):
        """Fill *only* the redshift gaps with lognormal shells; keep the rest.

        Where :meth:`extend_lognormal` blindly appends a block of bins, this
        inspects the existing coverage (:meth:`coverage_gaps`) and generates
        cross-correlated lognormal shells only for ranges with no N-body shell:
        below the first shell, between shells, and above the last one. This is
        the "use lognormal only where no N-body data exists" mode.

        Parameters
        ----------
        cosmo : kassiopeia.Cosmology
        z_max : float, optional
            Upper edge to cover. Defaults to the current :attr:`z_max` (fills
            only interior and low-z gaps). Set higher to also build the high-z
            tail the CMB-lensing kernel / n(z) tails need.
        z_min : float
            Lower edge to cover (default 0).
        dz : float, optional
            Target thickness of generated lognormal bins. Defaults to the
            median thickness of the existing shells.
        rng, lmax, shift
            Forwarded to :func:`~kassiopeia.lognormal.lognormal_shells`.
        min_gap : float
            Ignore gaps narrower than this (avoids spurious thin shells from
            floating-point edges).

        Notes
        -----
        The lognormal shells are an *independent* random realization: they
        carry the correct angular power and cross-correlation *among
        themselves*, but are not correlated with the N-body shells across the
        boundary, and the lognormal approximation is weakest at low z where
        structure is most nonlinear. Each disjoint gap is generated in its own
        cross-correlated GLASS call.
        """
        from .lognormal import lognormal_shells

        gaps = self.coverage_gaps(z_min=z_min, z_max=z_max, min_gap=min_gap)
        if not gaps:
            return self

        if dz is None:
            dz = float(np.median(self.z_upper - self.z_lower))
            if not np.isfinite(dz) or dz <= 0:
                dz = 0.1

        result = self
        for a, b in gaps:
            n_bins = max(1, int(round((b - a) / dz)))
            edges = np.linspace(a, b, n_bins + 1)
            ln = lognormal_shells(cosmo, edges, nside=self.nside, rng=rng,
                                  lmax=lmax, shift=shift)
            result = result.merge(ln)
        return result

    def __len__(self):
        return self.n_shells

    def __repr__(self):
        norm = (f", normalization={self.normalization!r}"
                if self.kind == "counts" and self.normalization != "map_mean"
                else "")
        return (f"ShellStack(n_shells={self.n_shells}, nside={self.nside}, "
                f"kind={self.kind!r}{norm}, "
                f"z=[{self.z_min:.3f}, {self.z_max:.3f}])")
