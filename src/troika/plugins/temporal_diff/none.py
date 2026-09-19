"""Identity temporal-difference stage, for ablating the difference operation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.plugins.base import TemporalDiff
from troika.registry import register
from troika.types import StageOutput, WindowContext

__all__ = ["NoTemporalDiff"]


@dataclass
class NoneParams:
    """No options."""


@register("temporal_diff", "none")
class NoTemporalDiff(TemporalDiff):
    """Pass the signal through unchanged.

    Not one of the paper's three ablations; ASSUMPTION A18 keeps the temporal
    difference on in all of them. This exists so the operation itself can be
    ablated, which the sensitivity study in Blueprint Section 8.5 asks for.
    """

    Params = NoneParams

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return ``x`` unchanged, as a copy."""
        out = np.array(x, dtype=np.float64, copy=True)
        return StageOutput(data=out, diag={"pre_normalization": out.copy(), "order": 0})
