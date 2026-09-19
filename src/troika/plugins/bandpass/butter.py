"""Butterworth band-pass plug-in, the paper default (ASSUMPTION A1)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.preprocessing.bandpass import apply_bandpass, frequency_response, make_bandpass
from troika.plugins.base import Bandpass
from troika.registry import register
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = ["ButterBandpass"]


@dataclass
class ButterParams:
    """Options of the Butterworth band-pass."""

    low_hz: float = 0.4  # PAPER
    high_hz: float = 5.0  # PAPER
    order: int = 4  # ASSUMPTION A1
    mode: str = "per_window"  # ASSUMPTION A2: per_window | global

    def __post_init__(self) -> None:
        if not 0 < self.low_hz < self.high_hz:
            raise ValueError(f"need 0 < low_hz < high_hz, got {self.low_hz} and {self.high_hz}")
        if self.order < 1:
            raise ValueError(f"order must be >= 1, got {self.order}")
        if self.mode not in ("per_window", "global"):
            raise ValueError(f"mode must be 'per_window' or 'global', got {self.mode!r}")


@register("bandpass", "butter")
class ButterBandpass(Bandpass):
    """Zero-phase Butterworth band-pass on the stacked PPG and ACC array.

    Paper Section III: PPG and acceleration are band-pass filtered 0.4-5 Hz in
    each time window. ``mode`` selects ASSUMPTION A2: ``per_window`` follows the
    paper and accepts the edge effects at both window ends, ``global`` filters
    the whole recording once and is handled by the pipeline, which then passes
    already-filtered windows through this stage unchanged.
    """

    Params = ButterParams

    def __init__(self, params: ButterParams, static: StaticContext) -> None:
        super().__init__(params, static)
        self.filter = make_bandpass(
            fs=static.fs,
            low_hz=params.low_hz,
            high_hz=params.high_hz,
            design="butter",
            order=params.order,
        )

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Filter a ``(4, M)`` stack of [PPG; ACC x, y, z] and return ``(4, M)``."""
        out = apply_bandpass(np.asarray(x, dtype=np.float64), self.filter)
        return StageOutput(
            data=out,
            diag={
                "filter_response": frequency_response(self.filter),
                "description": self.filter.description,
            },
        )
