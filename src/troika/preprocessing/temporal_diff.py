"""Temporal difference operation (Blueprint Section 5.4).

Paper Section III-B: the cleansed PPG signal from SSA is temporally
differentiated before SSR. For a periodic series the first-order difference
keeps the fundamental and the harmonic frequencies, and so does the second-order
difference; in the experiments the second order was used. Motion artifact is
generally aperiodic, except for rhythmic hand swing, so differencing makes the
heartbeat peaks more prominent and suppresses random spectral fluctuation.

Normalisation after differencing is ASSUMPTION A3. The paper never states it,
but Fig. 9 plots signals normalised to zero mean and unit variance, and it is
what makes lambda = 0.1 mean the same thing across subjects.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["temporal_difference"]


def temporal_difference(
    x: NDArray[np.float64], order: int = 2, normalize: bool = True
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Difference ``x`` ``order`` times, optionally z-scoring the result.

    Returns ``(out, pre_normalization)``. With the paper's ``order = 2`` and
    ``M = 1000`` samples per window the output has ``M' = 998`` samples, which is
    the length the SSR dictionary must be built for (Blueprint Section 11).

    A constant input has zero variance after differencing; it is returned as
    zeros rather than dividing by zero.
    """
    x = np.asarray(x, dtype=np.float64)
    order = int(order)
    if order < 0:
        raise ValueError(f"difference order must be >= 0, got {order}")
    if x.size <= order:
        raise ValueError(f"signal of length {x.size} is too short for order {order}")

    diffed = np.diff(x, n=order) if order > 0 else x.copy()
    pre = diffed.copy()

    if not normalize:  # ASSUMPTION A3
        return diffed, pre

    centred = diffed - diffed.mean()
    scale = float(np.std(centred))
    if scale <= 0:
        return np.zeros_like(centred), pre
    return centred / scale, pre
