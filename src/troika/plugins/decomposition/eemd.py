"""EEMD decomposition that keeps the modes in the heart-rate band (DEVIATIONS.md D13).

Thin adapter over ``troika.decomposition.eemd``; the numerics live there and are
tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.decomposition.eemd import default_n_imfs, eemd, pulsatile_mask
from troika.plugins.base import Decomposition
from troika.registry import register
from troika.types import StageOutput, WindowContext

__all__ = ["EemdDecomposition"]


@dataclass
class EemdParams:
    """Options of the EEMD decomposition.

    ``low_hz`` and ``high_hz`` are from Pittara and Orphanidou (2018). That paper
    states no EEMD settings, so the rest follow Wu and Huang (2009).
    """

    low_hz: float = 0.5  # EEMD paper: modes peaking below this are noise
    high_hz: float = 4.0  # EEMD paper: modes peaking above this are noise
    ensemble_size: int = 100  # ASSUMPTION (Wu and Huang)
    noise_std: float = 0.2  # ASSUMPTION (Wu and Huang), in standard deviations of the window
    n_sifts: int = 10  # ASSUMPTION (Wu and Huang)
    n_imfs: int = 0  # ASSUMPTION (Wu and Huang); 0 = log2(M) - 1
    seed: int = 0  # the added noise is redrawn from this for every window

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_hz < self.high_hz:
            raise ValueError(f"need 0 <= low_hz < high_hz, got {self.low_hz} and {self.high_hz}")
        if self.ensemble_size < 1:
            raise ValueError(f"ensemble_size must be >= 1, got {self.ensemble_size}")
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if self.n_sifts < 1:
            raise ValueError(f"n_sifts must be >= 1, got {self.n_sifts}")
        if self.n_imfs < 0:
            raise ValueError(f"n_imfs must be >= 0, got {self.n_imfs}")


@register("decomposition", "eemd")
class EemdDecomposition(Decomposition):
    """Decompose the window with EEMD and sum the modes that peak at a pulse rate.

    Pittara and Orphanidou (2018), Section II-B. Unlike the SSA plug-in this
    reads nothing from the accelerometer: a mode is dropped only because its
    peak frequency is outside ``[low_hz, high_hz]``.

    Fallback, as in the SSA plug-in: if no mode peaks in the band the
    band-passed window is returned unchanged rather than a zero signal.
    """

    Params = EemdParams

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return the pulsatile PPG window, same length as ``x``."""
        y = np.asarray(x, dtype=np.float64)
        n_imfs = int(self.params.n_imfs) or default_n_imfs(y.size)
        modes = eemd(
            y,
            ensemble_size=self.params.ensemble_size,
            noise_std=self.params.noise_std,
            n_sifts=self.params.n_sifts,
            n_imfs=n_imfs,
            seed=self.params.seed,
        )
        keep, dominant = pulsatile_mask(
            modes, self.static.fs, self.static.N, self.params.low_hz, self.params.high_hz
        )

        if keep.any():
            cleansed = modes[keep].sum(axis=0)
            removed = ~keep
            fallback = None
        else:
            cleansed = y.copy()
            removed = np.zeros(modes.shape[0], dtype=bool)
            fallback = "none_kept"

        names = [f"IMF {i + 1}" for i in range(n_imfs)] + ["residual"]
        return StageOutput(
            data=cleansed,
            diag={
                # generic decomposition schema (Lab Spec Section 2.3)
                "components": modes,
                "component_labels": [
                    f"{name}, {binmap.bin_to_bpm(int(k), self.static.fs, self.static.N):.0f} BPM"
                    for name, k in zip(names, dominant)
                ],
                "component_dominant_bins": dominant,
                "removed_mask": removed,
                # EEMD-specific
                "n_imfs": n_imfs,
                "ensemble_size": int(self.params.ensemble_size),
                "fallback": fallback,
            },
        )
