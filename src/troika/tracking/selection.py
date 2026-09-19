"""Peak selection, Cases 1 to 3 (paper Section III-D.2, Eqs. 13-15).

Given the previous estimate ``k_prev``, a search range is set for the heart-rate
fundamental and another for its first harmonic. In each range at most three
peaks above ``eta`` are kept, and the selected bin follows three cases:

* **Case 1** (Eq. 13): a peak pair in harmonic relation exists, so the
  fundamental of that pair is the answer.
* **Case 2** (Eq. 14): peaks exist but no harmonic pair, so the candidate
  closest to ``k_prev`` wins. Candidates are the fundamental-range peaks
  together with the harmonic-range peaks halved.
* **Case 3** (Eq. 15): no peaks at all, so the previous bin is kept, because a
  heart-rate peak usually holds its location between overlapping windows.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from troika.tracking.peaks import search_range, top_peaks

__all__ = ["select_bin"]


def select_bin(
    s: NDArray[np.float64],
    k_prev: int,
    delta_s: int = 16,
    eta_frac: float = 0.30,
    max_peaks: int = 3,
    harm_tol: int = 2,
    pair_tiebreak: str = "max_power",
    half_rounding: str = "round",
) -> tuple[int, int, dict[str, Any]]:
    """Return ``(k_b, case, info)`` for one window.

    ``eta`` is 30 % of the highest peak in ``R0`` and the same absolute value is
    applied in ``R1`` (ASSUMPTION A9); when ``R0`` holds no peak at all the
    threshold falls back to 30 % of ``R1``'s own maximum, so a visible harmonic
    is not discarded for want of a fundamental.
    """
    s = np.asarray(s, dtype=np.float64)
    n_bins = s.size

    r0 = search_range(k_prev, delta_s, n_bins, scale=1)
    r1 = search_range(k_prev, delta_s, n_bins, scale=2)

    # eta comes from R0 (paper Section III-D.2); A9 reuses it for R1.
    r0_peaks_any = top_peaks(s, r0[0], r0[1], -np.inf, max_peaks=n_bins)
    if r0_peaks_any.size:
        eta = float(eta_frac) * float(s[r0_peaks_any].max())
    else:
        r1_peaks_any = top_peaks(s, r1[0], r1[1], -np.inf, max_peaks=n_bins)
        eta = float(eta_frac) * float(s[r1_peaks_any].max()) if r1_peaks_any.size else 0.0

    p0 = top_peaks(s, r0[0], r0[1], eta, max_peaks)
    p1 = top_peaks(s, r1[0], r1[1], eta, max_peaks)

    info: dict[str, Any] = {
        "R0": r0,
        "R1": r1,
        "eta": eta,
        "P0": p0,
        "P1": p1,
        "pairs": [],
        "candidates": np.zeros(0, dtype=np.int64),
    }

    # --- Case 1: a harmonic pair (paper Eq. 13) -----------------------------
    pairs = [
        (int(a), int(b))
        for a in p0
        for b in p1
        if abs(int(b) - 2 * int(a)) <= int(harm_tol)  # ASSUMPTION A7
    ]
    if pairs:
        if pair_tiebreak == "max_power":  # ASSUMPTION A8
            best = max(pairs, key=lambda ab: s[ab[0]] + s[ab[1]])
        elif pair_tiebreak == "closest_to_prev":
            best = min(pairs, key=lambda ab: abs(ab[0] - int(k_prev)))
        else:
            raise ValueError(
                f"pair_tiebreak must be 'max_power' or 'closest_to_prev', "
                f"got {pair_tiebreak!r}"
            )
        info["pairs"] = pairs
        info["candidates"] = np.array([best[0]], dtype=np.int64)
        return int(best[0]), 1, info

    # --- Case 3: nothing anywhere (paper Eq. 15) ----------------------------
    if p0.size == 0 and p1.size == 0:
        info["candidates"] = np.array([int(k_prev)], dtype=np.int64)
        return int(k_prev), 3, info

    # --- Case 2: closest candidate (paper Eq. 14) ---------------------------
    if half_rounding == "round":  # ASSUMPTION A17
        halved = np.rint(p1 / 2.0).astype(np.int64)
    elif half_rounding == "floor":
        halved = np.floor(p1 / 2.0).astype(np.int64)
    else:
        raise ValueError(f"half_rounding must be 'round' or 'floor', got {half_rounding!r}")

    candidates = np.concatenate([p0, halved]).astype(np.int64)
    info["candidates"] = candidates
    # Ties go to the stronger peak, since candidates keep descending-power order.
    k_b = int(candidates[int(np.argmin(np.abs(candidates - int(k_prev))))])
    return k_b, 2, info
