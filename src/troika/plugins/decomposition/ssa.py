"""SSA decomposition with motion-artifact removal, the paper default.

Paper Section III-A. Thin adapter over ``troika.decomposition.ssa``,
``grouping`` and ``motion_removal``; the numerics live there and are tested
directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.decomposition import ssa as ssa_core
from troika.decomposition.grouping import (
    STRATEGIES,
    component_dominant_bins,
    group_eigentriples,
)
from troika.decomposition.motion_removal import select_motion_components
from troika.plugins.base import Decomposition
from troika.registry import register
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = ["SsaDecomposition"]


@dataclass
class SsaParams:
    """Options of the SSA decomposition."""

    L: int = 400  # PAPER
    grouping: str = "frequency_pairing"  # ASSUMPTION A4
    sv_rel_tol: float = 0.1  # ASSUMPTION A4
    freq_tol_bins: int = 2  # ASSUMPTION A4
    match_tol_bins: int = 2  # ASSUMPTION A5
    n_groups_max: int = 0  # ASSUMPTION A4, for wcorr_hclust; 0 = cut by distance

    def __post_init__(self) -> None:
        if self.L < 2:
            raise ValueError(f"L must be >= 2, got {self.L}")
        if self.grouping not in STRATEGIES:
            raise ValueError(
                f"grouping must be one of {list(STRATEGIES)}, got {self.grouping!r}"
            )
        if not 0.0 <= self.sv_rel_tol <= 1.0:
            raise ValueError(f"sv_rel_tol must be in [0, 1], got {self.sv_rel_tol}")
        if self.freq_tol_bins < 0 or self.match_tol_bins < 0:
            raise ValueError("bin tolerances must be >= 0")


@register("decomposition", "ssa")
class SsaDecomposition(Decomposition):
    """Decompose the window, drop the motion components, sum the rest.

    Paper Section III-A, Eqs. (1) to (6). The paper's rule of thumb is that ``L``
    be close to ``M / 2``; with ``M = 1000`` it uses ``L = 400``, so a warning is
    not raised for ``L`` up to ``M / 2``.

    Fallbacks (Blueprint Section 5.3): if every group matches the accelerometer
    the band-passed window is returned unchanged rather than a zero signal, since
    an empty signal would make the spectrum estimator meaningless. If no group
    matches, the sum is the input, which is the identity.
    """

    Params = SsaParams

    def __init__(self, params: SsaParams, static: StaticContext) -> None:
        super().__init__(params, static)
        if params.L > static.M // 2:
            # Paper footnote 2: L should be close to M/2 for a stable decomposition.
            raise ValueError(
                f"L={params.L} exceeds M/2={static.M // 2} for a {static.M}-sample "
                f"window; the paper requires L < M/2"
            )
        self._band = binmap.band_to_bins(0.4, 5.0, static.fs, static.N)

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return the cleansed PPG window, same length as ``x``."""
        y = np.asarray(x, dtype=np.float64)
        M = y.size
        L = int(self.params.L)
        K = M - L + 1

        trajectory = ssa_core.embed(y, L)
        U, s, Vt = ssa_core.svd(trajectory)

        groups, elementary = group_eigentriples(
            U,
            s,
            Vt,
            M,
            strategy=self.params.grouping,
            n_fft=self.static.N,
            fs=self.static.fs,
            sv_rel_tol=self.params.sv_rel_tol,
            freq_tol_bins=self.params.freq_tol_bins,
            n_groups_max=self.params.n_groups_max,
        )

        components = np.stack([elementary[np.asarray(g, dtype=int)].sum(axis=0) for g in groups])
        dominant = component_dominant_bins(components, self.static.N, self.static.fs)
        removed = select_motion_components(dominant, ctx.acc_bins, self.params.match_tol_bins)

        if removed.all():
            # Every component looked like motion; keeping none would leave the
            # estimator nothing to work with (Blueprint Section 5.3 fallback).
            cleansed = y.copy()
            removed = np.zeros_like(removed)
            fallback = "all_removed"
        else:
            cleansed = components[~removed].sum(axis=0)
            fallback = None

        singular_ranges = [(float(s[g].max()), float(s[g].min())) for g in groups]

        return StageOutput(
            data=cleansed,
            diag={
                # generic decomposition schema (Lab Spec Section 2.3)
                "components": components,
                "component_labels": [
                    f"group {i} ({len(g)} triple{'s' if len(g) > 1 else ''}), "
                    f"{binmap.bin_to_bpm(int(dominant[i]), self.static.fs, self.static.N):.0f} BPM"
                    for i, g in enumerate(groups)
                ],
                "component_dominant_bins": dominant,
                "removed_mask": removed,
                # SSA-specific schema
                "trajectory": trajectory,
                "U": U,
                "s": s,
                "Vt": Vt,
                "groups": [list(map(int, g)) for g in groups],
                "group_dominant_bins": dominant,
                "group_singular_range": singular_ranges,
                "L": L,
                "K": K,
                "grouping_strategy": self.params.grouping,
                "elementary": elementary,
                "fallback": fallback,
            },
        )
