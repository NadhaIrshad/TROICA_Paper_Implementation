"""Hidden-Markov path tracking over the heart-rate band (DEVIATIONS.md D12).

The states are the frequency bins of the heart-rate band. Between windows the
rate moves by a Gaussian step, with a small uniform probability of jumping
anywhere; in each window every SSR peak candidate raises the likelihood of the
bins around it in proportion to its score. :func:`viterbi_step` carries the
log-probability of the best path ending in each bin from one window to the next,
so the tracker stays causal: it reads the end of the best path so far.

Positions here are 0-based offsets into the band, not FFT bins; the plug-in does
the conversion through ``troika.binmap``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

__all__ = ["transition_log", "emission_log", "viterbi_step"]


def transition_log(n_states: int, sigma_bins: float, jump_prob: float) -> NDArray[np.float64]:
    """Log transition matrix, ``[i, j]`` for a move from state ``i`` to ``j``.

    Each row is ``(1 - jump_prob)`` of a Gaussian step of ``sigma_bins``,
    renormalised inside the band, plus ``jump_prob`` spread uniformly. The
    uniform part is what lets a path leave a wrong first window; ``jump_prob``
    must be positive so that no move has zero probability.
    """
    idx = np.arange(int(n_states), dtype=np.float64)
    gauss = np.exp(-0.5 * ((idx[None, :] - idx[:, None]) / float(sigma_bins)) ** 2)
    gauss /= gauss.sum(axis=1, keepdims=True)
    return np.log((1.0 - float(jump_prob)) * gauss + float(jump_prob) / int(n_states))


def emission_log(
    n_states: int, positions: NDArray[np.int64], scores: NDArray[np.float64],
    sigma_bins: float, floor: float,
) -> NDArray[np.float64]:
    """Log likelihood of one window's candidates under each state.

    A candidate at ``positions[c]`` with score ``scores[c]`` contributes a
    Gaussian bump of ``sigma_bins``; a state takes its largest bump plus
    ``floor``, which keeps states with no candidate nearby possible.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.size == 0:
        return np.full(int(n_states), np.log(float(floor)))
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    idx = np.arange(int(n_states), dtype=np.float64)
    bumps = np.exp(-0.5 * ((idx[None, :] - positions[:, None]) / float(sigma_bins)) ** 2)
    return np.log(float(floor) + (scores[:, None] * bumps).max(axis=0))


def viterbi_step(
    log_delta: NDArray[np.float64], log_trans: NDArray[np.float64],
    log_emit: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Advance the best-path log-probabilities by one window.

    ``log_delta[i]`` is the log-probability of the best path ending in state
    ``i``. The result is shifted so its maximum is 0, which keeps a long
    recording from underflowing and does not change which state wins.
    """
    best = (np.asarray(log_delta, dtype=np.float64)[:, None] + log_trans).max(axis=0)
    out = best + log_emit
    return out - out.max()
