"""Verification rules (paper Section III-D.3, Eqs. 16-18).

Two rules protect the selected peak.

*Rule 1* limits how far the estimate may move between two successive windows,
because the change in heart rate rarely exceeds 10 BPM. With ``theta = 6`` bins,
which is about 11 BPM on the paper's grid, a larger jump is replaced by a step
of ``tau = 2`` bins in the same direction (Eq. 16).

*Rule 2* prevents losing the heart rate for a long time. If the selected bin
equals the previous one for ``h`` successive windows, the estimate is nudged by
``2 * N_Trend`` and the search range is widened to ``Delta_s = 20`` (Eqs. 17 and
18). ``N_Trend`` comes from a third-order polynomial fit over the previous 20
estimated BPM values.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

__all__ = ["limit_jump", "predict_trend"]


def limit_jump(k_b: int, k_prev: int, theta: int = 6, tau: int = 2) -> tuple[int, bool]:
    """Rule 1 (paper Eq. 16). Returns ``(k_cur, fired)``.

    ``theta = 6`` bins is about 11 BPM at ``fs = 125``, ``N = 4096``, which is
    the paper's stated reasoning for the value.
    """
    delta = int(k_b) - int(k_prev)
    if delta >= int(theta):
        return int(k_prev) + int(tau), True
    if delta <= -int(theta):
        return int(k_prev) - int(tau), True
    return int(k_b), False


def predict_trend(
    bpm_history: list[float] | NDArray[np.float64],
    poly_order: int = 3,
    max_history: int = 20,
    threshold_bpm: float = 3.0,
) -> tuple[int, float]:
    """Rule 2's direction ``N_Trend`` in {-1, 0, +1} (paper Eq. 18).

    A polynomial of ``poly_order`` is fitted to the last ``max_history``
    estimated BPM values and extrapolated one step ahead. The direction is +1
    when the prediction exceeds the last estimate by ``threshold_bpm``, -1 when
    it falls short by as much, and 0 otherwise.

    Returns ``(trend, bpm_predict)``. With fewer than ``poly_order + 1`` points
    the fit is underdetermined, so the trend is 0.
    """
    history = np.asarray(list(bpm_history), dtype=np.float64)
    if history.size < int(poly_order) + 1:
        last = float(history[-1]) if history.size else float("nan")
        return 0, last

    recent = history[-int(max_history) :]
    x = np.arange(recent.size, dtype=np.float64)
    with warnings.catch_warnings():
        # A short or flat history makes the Vandermonde matrix ill conditioned;
        # the fit is still usable and the paper prescribes no alternative.
        # NumPy 1.x exposes RankWarning at ``np.RankWarning``; NumPy 2.x
        # moved it to ``np.exceptions.RankWarning``.
        rank_warning = getattr(getattr(np, "exceptions", None), "RankWarning", None)
        if rank_warning is None:  # NumPy 1.x
            rank_warning = np.RankWarning
        warnings.simplefilter("ignore", rank_warning)
        coefficients = np.polyfit(x, recent, int(poly_order))

    bpm_predict = float(np.polyval(coefficients, float(recent.size)))
    bpm_prev = float(recent[-1])

    if bpm_predict - bpm_prev >= float(threshold_bpm):
        return 1, bpm_predict
    if bpm_predict - bpm_prev <= -float(threshold_bpm):
        return -1, bpm_predict
    return 0, bpm_predict
