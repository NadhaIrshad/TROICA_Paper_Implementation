"""Spectral subtraction of the accelerometer from the PPG (DEVIATIONS.md D14).

E. Nowak, M. Krainski, M. Rubinski, M. Pazderska and S. A. Raczynski, "Robust
Real-Time PPG-Based Heart Rate Monitoring", EUSIPCO 2015, Section 2.2. Their
equation numbers are cited below as "Nowak Eq.".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import skewnorm

from troika import binmap

__all__ = ["amplitude_spectrum", "spectral_subtraction", "harmonic_sum", "bpm_prior"]


def amplitude_spectrum(x: NDArray[np.float64], n_fft: int) -> NDArray[np.float64]:
    """One-sided amplitude spectrum ``|FFT(x, n_fft)|``, length ``n_fft // 2 + 1``.

    ``x`` may be one signal ``(M,)`` or several ``(c, M)``; the transform runs
    along the last axis.
    """
    return np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64), n=int(n_fft), axis=-1))


def spectral_subtraction(
    ppg: NDArray[np.float64], acc: NDArray[np.float64] | None,
) -> NDArray[np.float64]:
    """PPG amplitude spectrum with the accelerometer's subtracted (Nowak Eqs. 5-6).

    ``y = R(ppg / sum(ppg) - acc / sum(acc))``, where ``acc`` is the sum of the
    three axes' amplitude spectra and ``R`` clips negatives to zero. ``acc`` is
    ``(3, K)`` or already summed ``(K,)``. With no accelerometer, or a silent
    one, the result is the normalised PPG spectrum; an all-zero PPG spectrum
    gives zeros.
    """
    ppg = np.asarray(ppg, dtype=np.float64)
    total = float(ppg.sum())
    if total <= 0.0:
        return np.zeros_like(ppg)
    y = ppg / total
    if acc is not None:
        acc = np.asarray(acc, dtype=np.float64)
        summed = acc.sum(axis=0) if acc.ndim == 2 else acc
        acc_total = float(summed.sum())
        if acc_total > 0.0:
            y = y - summed / acc_total
    return np.maximum(y, 0.0)


def harmonic_sum(y: NDArray[np.float64], weights) -> NDArray[np.float64]:
    """Add each bin's harmonics onto it (Nowak Eq. 7).

    ``out[n] = a1 * y[n] + a2 * y[2n] + a3 * y[3n] + ...`` with ``weights`` as
    ``(a1, a2, a3, ...)``; a harmonic past the end of the spectrum adds nothing.
    """
    y = np.asarray(y, dtype=np.float64)
    out = np.zeros_like(y)
    n = np.arange(y.size)
    for h, a in enumerate(weights, start=1):
        if float(a) == 0.0:
            continue
        idx = h * n
        ok = idx < y.size
        out[ok] += float(a) * y[idx[ok]]
    return out


def bpm_prior(
    n_bins: int, fs: float, n_fft: int, loc_bpm: float, scale_bpm: float, skew: float,
) -> NDArray[np.float64]:
    """Skewed-normal weight for each of the first ``n_bins`` frequency bins.

    Nowak Section 2.2 multiplies the spectrum by a skewed normal with "mean 130
    BPM, standard deviation 30 BPM and skewness 0.6". ASSUMPTION: those are
    read as the location, scale and shape of ``scipy.stats.skewnorm``.
    """
    bpm = np.asarray(binmap.bin_to_bpm(np.arange(int(n_bins)), fs, n_fft), dtype=np.float64)
    return skewnorm.pdf(bpm, float(skew), loc=float(loc_bpm), scale=float(scale_bpm))
