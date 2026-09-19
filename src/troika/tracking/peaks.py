"""Peak helpers for spectral peak tracking (Blueprint Section 5.6).

Paper Section III-D.2: "in each search range, we select no more than three
highest peaks with amplitude no less than a threshold eta. In our experiments
eta was set to 30 % of the highest peak amplitude in R0."
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from troika.preprocessing.spectrum import local_maxima

__all__ = ["search_range", "top_peaks"]


def search_range(centre: int, delta: int, n_bins: int, scale: int = 1) -> tuple[int, int]:
    """Inclusive bin range around ``scale * centre``, clipped to the spectrum.

    ``scale = 1`` gives ``R0 = [k_prev - Delta_s, k_prev + Delta_s]``; ``scale =
    2`` gives the first-harmonic range ``R1 = [2(k_prev - Delta_s),
    2(k_prev + Delta_s)]``, which is the 0-based reading of the paper's 1-based
    ``[2(N_prev - Delta_s - 1) + 1, ..., 2(N_prev + Delta_s - 1) + 1]``
    (Blueprint Section 3).
    """
    lo = int(scale) * (int(centre) - int(delta))
    hi = int(scale) * (int(centre) + int(delta))
    lo = max(0, lo)
    hi = min(int(n_bins) - 1, hi)
    return lo, hi


def top_peaks(
    s: NDArray[np.float64],
    lo: int,
    hi: int,
    eta: float,
    max_peaks: int = 3,
) -> NDArray[np.int64]:
    """Up to ``max_peaks`` highest local maxima in ``[lo, hi]`` with height >= ``eta``.

    Returned in descending amplitude order, which is what the Case 1 tie-break
    and the Case 2 candidate list expect. An empty range gives an empty array.
    """
    if hi < lo:
        return np.zeros(0, dtype=np.int64)
    peaks = local_maxima(s, lo, hi)
    if peaks.size == 0:
        return peaks
    peaks = peaks[s[peaks] >= float(eta)]
    if peaks.size == 0:
        return peaks
    order = np.argsort(-s[peaks], kind="stable")
    return peaks[order][: int(max_peaks)]
