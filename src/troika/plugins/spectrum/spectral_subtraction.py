"""Spectral-subtraction spectrum estimation (DEVIATIONS.md D14).

Thin adapter over ``troika.preprocessing.spectral_subtraction``; the numerics
live there and are tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from troika.plugins.base import SpectrumEstimator
from troika.preprocessing.spectral_subtraction import (
    amplitude_spectrum,
    bpm_prior,
    harmonic_sum,
    spectral_subtraction,
)
from troika.registry import register
from troika.types import StageOutput, WindowContext

__all__ = ["SpectralSubtractionSpectrum"]


@dataclass
class SpectralSubtractionParams:
    """Options of the spectral-subtraction estimator (Nowak et al., EUSIPCO 2015)."""

    harmonic_weights: list = field(default_factory=lambda: [1.0, 0.66, 0.33])  # Nowak Eq. 7
    bpm_prior: bool = False  # Nowak's last step; off because the prior was fitted to this dataset's truth
    prior_loc_bpm: float = 130.0  # Nowak
    prior_scale_bpm: float = 30.0  # Nowak
    prior_skew: float = 0.6  # Nowak
    output_power: bool = False  # ASSUMPTION: Nowak tracks the amplitude spectrum; true squares it

    def __post_init__(self) -> None:
        if len(self.harmonic_weights) < 1 or any(float(a) < 0.0 for a in self.harmonic_weights):
            raise ValueError(f"harmonic_weights must be non-negative, got {self.harmonic_weights}")
        if float(self.harmonic_weights[0]) <= 0.0:
            raise ValueError("harmonic_weights[0], the weight of the bin itself, must be > 0")
        if self.prior_scale_bpm <= 0.0:
            raise ValueError(f"prior_scale_bpm must be > 0, got {self.prior_scale_bpm}")


@register("spectrum_estimator", "spectral_subtraction")
class SpectralSubtractionSpectrum(SpectrumEstimator):
    """Subtract the accelerometer's amplitude spectrum from the PPG's.

    Nowak et al. Section 2.2: each spectrum is scaled by its own sum, the
    accelerometer's is subtracted and negatives are clipped (Eqs. 5-6), then
    every bin gains its second and third harmonics (Eq. 7). The accelerometer
    is ``ctx.acc``, the band-passed window, so both spectra cover the same band.

    Not implemented: Nowak's choice between the two PPG channels by spectrum
    variance, since the pipeline carries the one channel of
    ``signal.ppg_channel``.

    Fallback: if the subtraction leaves nothing, the normalised PPG spectrum is
    returned instead of zeros.
    """

    Params = SpectralSubtractionParams

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return a one-sided spectrum of length ``N // 2 + 1``."""
        n_fft = self.static.N
        ppg = amplitude_spectrum(x, n_fft)
        acc = None if ctx.acc is None else amplitude_spectrum(ctx.acc, n_fft)

        plain = spectral_subtraction(ppg, None)
        subtracted = spectral_subtraction(ppg, acc)
        fallback = None
        if float(subtracted.max()) <= 1e-12 * float(plain.max()):  # nothing but rounding left
            subtracted = plain
            fallback = "all_subtracted"

        spectrum = harmonic_sum(subtracted, self.params.harmonic_weights)
        if self.params.bpm_prior:
            spectrum = spectrum * bpm_prior(
                spectrum.size, self.static.fs, n_fft,
                self.params.prior_loc_bpm, self.params.prior_scale_bpm, self.params.prior_skew,
            )
        if self.params.output_power:
            spectrum = spectrum**2

        stages = np.stack([plain, subtracted, spectrum])
        peak = stages.max(axis=1, keepdims=True)
        sparsity = (stages > 0.01 * np.where(peak > 0, peak, 1.0)).sum(axis=1)
        return StageOutput(
            data=spectrum,
            diag={
                # generic schema: the "iterates" are the normalised PPG spectrum,
                # the subtracted spectrum and the final one
                "iterates": stages.astype(np.complex128),
                "kept_bins": np.arange(spectrum.size, dtype=np.int64),
                "sparsity_per_iter": np.asarray(sparsity, dtype=np.int64),
                # method-specific
                "acc_amplitude": None if acc is None else acc.sum(axis=0),
                "fallback": fallback,
            },
        )
