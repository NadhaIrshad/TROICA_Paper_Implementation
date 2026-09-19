"""Periodogram and peak picking (Blueprint Sections 5.2 and 5.3).

Paper Section III-A, footnote 3: the Periodogram is used to find the dominant
frequencies of the acceleration data, because the goal is only to select motion
peaks and an advanced estimator is not needed there. The same periodogram gives
each SSA component its dominant bin, and serves as the FFT ablation of the
spectrum estimator (paper Table I row 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

__all__ = ["periodogram", "local_maxima", "dominant_bins", "dominant_bin"]


def periodogram(x: NDArray[np.float64], n_fft: int) -> NDArray[np.float64]:
    """One-sided periodogram, length ``n_fft // 2 + 1``.

    ``P[k] = |FFT(x, n_fft)[k]|^2 / len(x)`` (Blueprint Section 5.2 step 1).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("cannot take the periodogram of an empty signal")
    spec = np.fft.rfft(x, n=int(n_fft))
    return np.abs(spec) ** 2 / x.size


def local_maxima(
    s: NDArray[np.float64], lo: int = 0, hi: int | None = None
) -> NDArray[np.int64]:
    """Bin indices of local maxima of ``s`` within the inclusive range ``[lo, hi]``.

    ASSUMPTION A16: peaks are plain ``scipy.signal.find_peaks`` local maxima,
    with no prominence requirement. Because ``find_peaks`` ignores the array
    edges, an endpoint that dominates its single neighbour is added explicitly;
    otherwise a peak sitting on the edge of a narrow search range would be
    invisible to the tracker.
    """
    s = np.asarray(s, dtype=np.float64)
    hi = s.size - 1 if hi is None else int(hi)
    lo = max(0, int(lo))
    hi = min(int(hi), s.size - 1)
    if hi < lo:
        return np.zeros(0, dtype=np.int64)
    if hi == lo:
        return np.array([lo], dtype=np.int64)

    segment = s[lo : hi + 1]
    peaks, _ = find_peaks(segment)
    found = set(int(p) + lo for p in peaks)
    if segment[0] > segment[1]:
        found.add(lo)
    if segment[-1] > segment[-2]:
        found.add(hi)
    return np.array(sorted(found), dtype=np.int64)


def dominant_bins(
    s: NDArray[np.float64],
    lo: int = 0,
    hi: int | None = None,
    rel_threshold: float = 0.5,
) -> NDArray[np.int64]:
    """Peaks above ``rel_threshold`` times the maximum inside ``[lo, hi]``.

    Paper Section III-A: "the dominant frequencies are the ones corresponding to
    the spectral peaks with amplitude larger than 50 % of the maximum amplitude
    in a given spectrum", applied per acceleration axis.
    """
    s = np.asarray(s, dtype=np.float64)
    peaks = local_maxima(s, lo, hi)
    if peaks.size == 0:
        return peaks
    hi = s.size - 1 if hi is None else min(int(hi), s.size - 1)
    lo = max(0, int(lo))
    ceiling = float(np.max(s[lo : hi + 1]))
    if ceiling <= 0:
        return np.zeros(0, dtype=np.int64)
    return peaks[s[peaks] > float(rel_threshold) * ceiling]


def dominant_bin(
    s: NDArray[np.float64], lo: int = 0, hi: int | None = None
) -> int:
    """Index of the largest value of ``s`` inside the inclusive range ``[lo, hi]``.

    Used to label each decomposition component with one frequency
    (Blueprint Section 5.3, MA component removal step 1).
    """
    s = np.asarray(s, dtype=np.float64)
    hi = s.size - 1 if hi is None else min(int(hi), s.size - 1)
    lo = max(0, int(lo))
    if hi < lo:
        raise ValueError(f"empty search range [{lo}, {hi}]")
    return int(lo + np.argmax(s[lo : hi + 1]))
