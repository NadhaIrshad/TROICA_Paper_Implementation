"""Linear-phase FIR band-pass, the A1 alternative to the Butterworth default."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.preprocessing.bandpass import apply_bandpass, frequency_response, make_bandpass
from troika.plugins.base import Bandpass
from troika.registry import register
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = ["FirBandpass"]


@dataclass
class FirParams:
    """Options of the FIR band-pass.

    ``order`` is the number of taps, not a filter order; it is forced odd so the
    band-pass stays symmetric.

    The default of 301 taps is close to the largest an 8 s window allows: the
    zero-phase pass needs more than three times the filter length, so 1000
    samples cap the design at about 333 taps. Even at 301 the lower edge is
    much softer than the Butterworth default, passing only 39 % of the
    amplitude at 0.5 Hz against 89 %. That is a property of the 0.4 Hz corner,
    not a bug; see docs/assumption_log.md under A1.
    """

    low_hz: float = 0.4  # PAPER
    high_hz: float = 5.0  # PAPER
    order: int = 301  # ASSUMPTION A1: number of taps
    mode: str = "per_window"  # ASSUMPTION A2

    def __post_init__(self) -> None:
        if not 0 < self.low_hz < self.high_hz:
            raise ValueError(f"need 0 < low_hz < high_hz, got {self.low_hz} and {self.high_hz}")
        if self.order < 3:
            raise ValueError(f"order (number of taps) must be >= 3, got {self.order}")
        if self.mode not in ("per_window", "global"):
            raise ValueError(f"mode must be 'per_window' or 'global', got {self.mode!r}")


@register("bandpass", "fir")
class FirBandpass(Bandpass):
    """Zero-phase Hamming-window FIR band-pass (Blueprint Section 5.1, A1)."""

    Params = FirParams

    def __init__(self, params: FirParams, static: StaticContext) -> None:
        super().__init__(params, static)
        self.filter = make_bandpass(
            fs=static.fs,
            low_hz=params.low_hz,
            high_hz=params.high_hz,
            design="fir",
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
