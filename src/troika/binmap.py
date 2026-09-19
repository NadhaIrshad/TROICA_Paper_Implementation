"""Single source of truth for frequency-bin <-> Hz <-> BPM conversions.

Blueprint Section 3. The paper uses 1-based MATLAB bin indices; this code base
uses 0-based indices everywhere and converts only here. Paper Eq. (11) reads
``f = (N_f - 1)/N * fs`` for 1-based ``N_f``; with ``k = N_f - 1`` that is
``f = k * fs / N``.

Never write raw ``fs / N`` arithmetic anywhere else in the package.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "bin_to_hz",
    "bin_to_bpm",
    "hz_to_bin",
    "bpm_to_bin",
    "band_to_bins",
    "bin_width_hz",
    "bin_width_bpm",
]


def bin_width_hz(fs: float, n_fft: int) -> float:
    """Width of one frequency bin in Hz (``fs / N``)."""
    return float(fs) / int(n_fft)


def bin_width_bpm(fs: float, n_fft: int) -> float:
    """Width of one frequency bin in BPM (``60 * fs / N``).

    For fs = 125 Hz and N = 4096 this is 1.8311 BPM, not the "about 1 BPM"
    claimed in Remark 3 of the paper (Blueprint Section 3).
    """
    return 60.0 * float(fs) / int(n_fft)


def bin_to_hz(k: ArrayLike, fs: float, n_fft: int) -> NDArray[np.float64] | float:
    """Convert a 0-based frequency bin index to Hz (paper Eq. 11, 0-based)."""
    k_arr = np.asarray(k, dtype=np.float64)
    out = k_arr * (float(fs) / int(n_fft))
    return float(out) if out.ndim == 0 else out


def bin_to_bpm(k: ArrayLike, fs: float, n_fft: int) -> NDArray[np.float64] | float:
    """Convert a 0-based frequency bin index to BPM (``60 * k * fs / N``)."""
    k_arr = np.asarray(k, dtype=np.float64)
    out = k_arr * (60.0 * float(fs) / int(n_fft))
    return float(out) if out.ndim == 0 else out


def hz_to_bin(f_hz: ArrayLike, fs: float, n_fft: int) -> NDArray[np.int64] | int:
    """Convert Hz to the nearest 0-based frequency bin index."""
    f_arr = np.asarray(f_hz, dtype=np.float64)
    out = np.rint(f_arr * int(n_fft) / float(fs)).astype(np.int64)
    return int(out) if out.ndim == 0 else out


def bpm_to_bin(bpm: ArrayLike, fs: float, n_fft: int) -> NDArray[np.int64] | int:
    """Convert BPM to the nearest 0-based frequency bin index."""
    bpm_arr = np.asarray(bpm, dtype=np.float64)
    out = np.rint(bpm_arr * int(n_fft) / (60.0 * float(fs))).astype(np.int64)
    return int(out) if out.ndim == 0 else out


def band_to_bins(f_lo: float, f_hi: float, fs: float, n_fft: int) -> tuple[int, int]:
    """Inclusive 0-based bin range covering the band ``[f_lo, f_hi]`` Hz.

    The lower edge is rounded up and the upper edge down, so the returned range
    lies strictly inside the requested band. Both edges are clipped to the
    one-sided spectrum ``0 .. n_fft // 2``.
    """
    n = int(n_fft)
    lo = int(np.ceil(float(f_lo) * n / float(fs)))
    hi = int(np.floor(float(f_hi) * n / float(fs)))
    lo = max(0, min(lo, n // 2))
    hi = max(0, min(hi, n // 2))
    if hi < lo:
        hi = lo
    return lo, hi
