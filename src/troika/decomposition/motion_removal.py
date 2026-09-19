"""Motion-artifact identification and removal (Blueprint Sections 5.2-5.3).

Paper Section III-A. Acceleration data tell SSA which of its components are
motion artifact:

1. Take the periodogram of each acceleration axis in the current window and keep
   the peaks above 50 % of that axis's maximum. ``F_acc`` is the union over the
   three axes.
2. Remove from ``F_acc`` every bin within ``+/- Delta`` of the heartbeat
   fundamental and harmonic estimated in the previous window, giving the refined
   set. Without this, a cadence that happens to coincide with the heart rate
   would take the heartbeat with it.
3. Drop every SSA component whose dominant bin lies in the refined set, and sum
   what is left.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.preprocessing.spectrum import dominant_bins, periodogram

__all__ = ["acc_dominant_bins", "refine_acc_bins", "select_motion_components"]


def acc_dominant_bins(
    acc_win: NDArray[np.float64],
    n_fft: int,
    fs: float,
    rel_threshold: float = 0.5,
    restrict_to_band: bool = True,
    f_lo: float = 0.4,
    f_hi: float = 5.0,
    threshold_domain: str = "power",
) -> set[int]:
    """``F_acc``: dominant bins of the three acceleration axes (paper Section III-A).

    The 50 % threshold is applied per axis, as the paper specifies ("50 % of the
    maximum amplitude in a given spectrum"), and the results are unioned.
    ``restrict_to_band`` is ASSUMPTION A19; the signal is band-passed anyway, so
    restricting mainly guards against out-of-band leakage dominating an axis.
    ``threshold_domain`` is ASSUMPTION A21, whether the paper's 50 % is of
    amplitude or of periodogram power.
    """
    acc_win = np.atleast_2d(np.asarray(acc_win, dtype=np.float64))
    if restrict_to_band:  # ASSUMPTION A19
        lo, hi = binmap.band_to_bins(f_lo, f_hi, fs, n_fft)
    else:
        lo, hi = 0, n_fft // 2

    out: set[int] = set()
    for axis in acc_win:
        spec = periodogram(axis, n_fft)
        out.update(
            int(b) for b in dominant_bins(spec, lo, hi, rel_threshold, threshold_domain)
        )
    return out


def refine_acc_bins(
    f_acc: set[int],
    prev_bin: int | None,
    delta: int = 10,
    n_harmonics: int = 2,
    n_fft: int = 4096,
) -> set[int]:
    """Remove the neighbourhood of the previous heart rate from ``F_acc``.

    Paper Section III-A: with ``N_p`` the fundamental and harmonic locations of
    the heartbeat estimated in the previous window, the set
    ``{N_p - Delta, ..., N_p + Delta}`` is excluded, giving the refined set.

    ``n_harmonics`` is ASSUMPTION A6: 1 uses the fundamental only, 2 adds the
    first harmonic, which is the default reading of "fundamental and harmonic".
    For the first window there is no previous estimate and the set is unchanged.
    """
    if prev_bin is None:
        return set(f_acc)

    protected: set[int] = set()
    for harmonic in range(1, int(n_harmonics) + 1):  # ASSUMPTION A6
        centre = int(prev_bin) * harmonic
        if centre > n_fft // 2:
            break
        protected.update(range(centre - int(delta), centre + int(delta) + 1))
    return {b for b in f_acc if b not in protected}


def select_motion_components(
    dominant: NDArray[np.int64],
    f_acc_refined: set[int],
    match_tol_bins: int = 2,
) -> NDArray[np.bool_]:
    """Mask of components to drop as motion artifact.

    Blueprint Section 5.3 step 2 with ASSUMPTION A5: membership in the refined
    set is tested as ``min_a |d_p - a| <= match_tol_bins`` rather than exact
    equality, because the accelerometer's peak and the PPG component's peak need
    not land on the same bin. Setting the tolerance to 0 recovers exact matching.
    """
    dominant = np.asarray(dominant, dtype=np.int64)
    if not f_acc_refined:
        return np.zeros(dominant.size, dtype=bool)

    acc = np.fromiter(sorted(f_acc_refined), dtype=np.int64)
    distance = np.abs(dominant[:, None] - acc[None, :]).min(axis=1)
    return distance <= int(match_tol_bins)  # ASSUMPTION A5
