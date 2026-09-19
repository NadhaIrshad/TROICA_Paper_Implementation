"""Periodogram spectrum estimation: the paper's "FFT instead of SSR" ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.plugins.base import SpectrumEstimator
from troika.preprocessing.spectrum import periodogram
from troika.registry import register
from troika.types import StageOutput, WindowContext

__all__ = ["FftSpectrum"]


@dataclass
class FftParams:
    """No options; the grid size comes from the shared ``grid.n_fft``."""


@register("spectrum_estimator", "fft")
class FftSpectrum(SpectrumEstimator):
    """Replace SSR with the Periodogram (paper Table I row 3).

    Paper Section II explains why this fails: the Periodogram has high variance
    and serious leakage, so a strong motion peak smears a nearby weak heart-rate
    peak. Reproducing that failure is the point of the ablation.
    """

    Params = FftParams

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return a one-sided power spectrum of length ``N // 2 + 1``."""
        spectrum = periodogram(np.asarray(x, dtype=np.float64), self.static.N)
        return StageOutput(
            data=spectrum,
            diag={
                "iterates": spectrum[None, :].astype(np.complex128),
                "kept_bins": np.arange(spectrum.size, dtype=np.int64),
                "sparsity_per_iter": np.array([spectrum.size], dtype=np.int64),
            },
        )
