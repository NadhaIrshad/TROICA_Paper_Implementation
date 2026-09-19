"""Sliding-window bookkeeping (Blueprint Sections 2 and 5.1).

Paper Section III: a window of ``T`` seconds slides over the signals with step
``S``; in the experiments ``T = 8`` and ``S = 2``, so successive windows overlap
by 6 s. The dataset readme describes the same geometry: the first estimate comes
from samples 1..1000, the second from 251..1250.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from numpy.typing import NDArray

__all__ = ["n_windows", "iter_windows", "window_bounds", "window_starts_s"]


def _lengths(fs: float, window_s: float, step_s: float) -> tuple[int, int]:
    win = int(round(float(window_s) * float(fs)))
    hop = int(round(float(step_s) * float(fs)))
    if win <= 0 or hop <= 0:
        raise ValueError(f"window and step must be positive, got {window_s=} {step_s=}")
    return win, hop


def n_windows(n_samples: int, fs: float, window_s: float, step_s: float) -> int:
    """Number of complete windows in a recording.

    ``W = floor((n - T*fs) / (S*fs)) + 1`` (Blueprint Section 2). Verified to
    equal ``len(BPM0)`` exactly for all 12 training and all 10 test recordings.
    Returns 0 when the recording is shorter than one window.
    """
    win, hop = _lengths(fs, window_s, step_s)
    if int(n_samples) < win:
        return 0
    return (int(n_samples) - win) // hop + 1


def window_bounds(w: int, fs: float, window_s: float, step_s: float) -> tuple[int, int]:
    """Half-open sample range ``[start, stop)`` of the 0-based window ``w``."""
    win, hop = _lengths(fs, window_s, step_s)
    start = int(w) * hop
    return start, start + win


def iter_windows(
    n_samples: int, fs: float, window_s: float, step_s: float
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(w, start, stop)`` for every complete window, in order.

    The pipeline is stateful across windows, so this must be consumed in order.
    """
    win, hop = _lengths(fs, window_s, step_s)
    for w in range(n_windows(n_samples, fs, window_s, step_s)):
        start = w * hop
        yield w, start, start + win


def window_starts_s(
    n_samples: int, fs: float, window_s: float, step_s: float
) -> NDArray[np.float64]:
    """Start time in seconds of every window, shape ``(W,)``."""
    count = n_windows(n_samples, fs, window_s, step_s)
    return np.arange(count, dtype=np.float64) * float(step_s)
