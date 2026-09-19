"""Identity decomposition: the paper's "without SSA" ablation (Table I row 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.plugins.base import Decomposition
from troika.registry import register
from troika.types import StageOutput, WindowContext

__all__ = ["NoDecomposition"]


@dataclass
class NoneParams:
    """No options."""


@register("decomposition", "none")
class NoDecomposition(Decomposition):
    """Pass the band-passed PPG straight through, removing nothing.

    Paper Section IV-D: removing the signal decomposition part is the first
    ablation, and it makes tracking fail outright on subject 6.
    """

    Params = NoneParams

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return ``x`` unchanged, reported as a single kept component."""
        out = np.array(x, dtype=np.float64, copy=True)
        return StageOutput(
            data=out,
            diag={
                "components": out.reshape(1, -1).copy(),
                "component_labels": ["band-passed PPG"],
                "component_dominant_bins": np.zeros(1, dtype=np.int64),
                "removed_mask": np.zeros(1, dtype=bool),
            },
        )
