"""Second-order temporal difference, the paper default (paper Section III-B)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.preprocessing.temporal_diff import temporal_difference
from troika.plugins.base import TemporalDiff
from troika.registry import register
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = ["DiffTemporal"]


@dataclass
class DiffParams:
    """Options of the temporal difference stage."""

    order: int = 2  # PAPER: second-order difference
    normalize_after: bool = True  # ASSUMPTION A3: z-score before the estimator

    def __post_init__(self) -> None:
        if self.order < 0:
            raise ValueError(f"order must be >= 0, got {self.order}")


@register("temporal_diff", "diff")
class DiffTemporal(TemporalDiff):
    """Differentiate the cleansed PPG, then optionally z-score it.

    Paper Section III-B: differencing keeps the heartbeat fundamental and its
    harmonics while suppressing aperiodic motion artifact. With order 2 and
    M = 1000 the output has M' = 998 samples.
    """

    Params = DiffParams

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return a series of length ``len(x) - order``."""
        out, pre = temporal_difference(
            np.asarray(x, dtype=np.float64),
            order=self.params.order,
            normalize=self.params.normalize_after,
        )
        return StageOutput(
            data=out,
            diag={"pre_normalization": pre, "order": int(self.params.order)},
        )
