"""Feature extraction shared by the XGBoost tracker and its training script.

The feature vector describes a *candidate spectral peak*, never the ECG label.
This makes it possible to train a model to rank the small set of peaks produced
by SSR while preserving TROIKA's SSA -> TD -> SSR stages unchanged.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import peak_prominences

from troika import binmap
from troika.preprocessing.spectrum import local_maxima, periodogram
from troika.types import WindowContext

FEATURE_NAMES = (
    "candidate_bpm",
    "candidate_power_relative",
    "candidate_prominence_relative",
    "candidate_rank",
    "harmonic_power_relative",
    "harmonic_to_fundamental",
    "acc_power_relative",
    "acc_harmonic_power_relative",
    "distance_to_acc_peak_bins",
    "delta_previous_bpm",
    "delta_history_mean_bpm",
    "history_std_bpm",
    "history_slope_bpm",
)


def candidate_bins(
    spectrum: NDArray[np.float64], fs: float, n_fft: int, low_bpm: float,
    high_bpm: float, count: int,
) -> NDArray[np.int64]:
    """Return up to ``count`` strongest local SSR maxima in the HR band."""
    s = np.asarray(spectrum, dtype=np.float64)
    lo = int(binmap.bpm_to_bin(low_bpm, fs, n_fft))
    hi = min(s.size - 1, int(binmap.bpm_to_bin(high_bpm, fs, n_fft)))
    peaks = local_maxima(s, lo, hi)
    if peaks.size == 0:
        # A degenerate spectrum still needs one deterministic candidate.
        return np.array([lo + int(np.argmax(s[lo : hi + 1]))], dtype=np.int64)
    return peaks[np.argsort(-s[peaks], kind="stable")[: int(count)]].astype(np.int64)


def label_candidates(
    candidates: NDArray[np.int64], gt_bpm: float, fs: float, n_fft: int,
    tol_bpm: float = 5.0,
) -> tuple[NDArray[np.int64], bool]:
    """Training labels for one window: 1 for the candidate nearest the truth.

    Returns ``(labels, covered)``. ``covered`` is False when no candidate lies
    within ``tol_bpm`` of the ECG rate; such a window has no right answer to
    learn from and the training script drops it. Training only: no plug-in may
    call this (Lab Spec Section 2.4).
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    bpm = np.asarray(binmap.bin_to_bpm(candidates, fs, n_fft), dtype=float)
    error = np.abs(bpm - float(gt_bpm))
    winner = int(np.argmin(error))
    labels = (np.arange(candidates.size) == winner).astype(np.int64)
    return labels, bool(error[winner] <= tol_bpm)


def guarded_choice(
    candidates: NDArray[np.int64], scores: NDArray[np.float64], prev_bin: int | None,
    pending: tuple[int, int] | None, max_jump_bins: int, patience: int,
) -> tuple[int, tuple[int, int] | None, bool]:
    """Pick a candidate, refusing a far jump until it has won ``patience`` times.

    The learned counterpart of paper Section III-D.3: a heart rate cannot move
    more than a few BPM between windows 2 s apart, so a top-scoring candidate
    more than ``max_jump_bins`` from ``prev_bin`` is held back. It is accepted
    only once a candidate within ``max_jump_bins`` of the same far location has
    been the top score for ``patience`` consecutive windows, which still lets
    the tracker leave a wrong initialisation. While held, the estimate is the
    best-scoring candidate near ``prev_bin``, or ``prev_bin`` itself.

    ``pending`` is ``(far_bin, wins_so_far)`` from the previous window. Returns
    ``(bin, new_pending, held)``. ``patience <= 1`` disables the guard.
    """
    candidates = np.asarray(candidates, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    best = int(candidates[int(np.argmax(scores))])
    if patience <= 1 or prev_bin is None or abs(best - prev_bin) <= max_jump_bins:
        return best, None, False

    same_place = pending is not None and abs(best - pending[0]) <= max_jump_bins
    wins = pending[1] + 1 if same_place else 1
    if wins >= patience:
        return best, None, False

    near = np.abs(candidates - prev_bin) <= max_jump_bins
    if near.any():
        fallback = int(candidates[near][int(np.argmax(scores[near]))])
    else:
        fallback = int(prev_bin)
    return fallback, (best, wins), True


def feature_matrix(
    spectrum: NDArray[np.float64], ctx: WindowContext, candidates: NDArray[np.int64],
    fs: float, n_fft: int,
) -> NDArray[np.float64]:
    """Create one truth-free feature row per candidate, in ``FEATURE_NAMES`` order."""
    s = np.asarray(spectrum, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.int64)
    scale = max(float(np.max(s)), np.finfo(float).eps)
    prominence = peak_prominences(s, candidates)[0] if candidates.size else np.zeros(0)

    acc_relative = np.zeros(candidates.size)
    acc_harmonic_relative = np.zeros(candidates.size)
    if ctx.acc is not None:
        acc_specs = np.asarray([periodogram(axis, n_fft) for axis in ctx.acc])
        acc_scale = max(float(np.max(acc_specs)), np.finfo(float).eps)
        acc_relative = np.max(acc_specs[:, candidates], axis=0) / acc_scale
        harmonic = np.minimum(2 * candidates, acc_specs.shape[1] - 1)
        acc_harmonic_relative = np.max(acc_specs[:, harmonic], axis=0) / acc_scale

    acc_bins = sorted(ctx.acc_bins_raw)
    distance = (
        np.min(np.abs(candidates[:, None] - np.asarray(acc_bins)[None, :]), axis=1)
        if acc_bins else np.full(candidates.size, n_fft // 2, dtype=float)
    )
    history = np.asarray(ctx.bpm_history, dtype=float)
    previous = float(binmap.bin_to_bpm(ctx.prev_bin, fs, n_fft)) if ctx.prev_bin is not None else 0.0
    mean = float(history[-5:].mean()) if history.size else previous
    std = float(history[-5:].std()) if history.size > 1 else 0.0
    slope = float(history[-1] - history[-2]) if history.size > 1 else 0.0
    bpm = np.asarray(binmap.bin_to_bpm(candidates, fs, n_fft), dtype=float)
    harmonic = np.minimum(2 * candidates, s.size - 1)

    return np.column_stack((
        bpm,
        s[candidates] / scale,
        prominence / scale,
        np.arange(candidates.size, dtype=float),
        s[harmonic] / scale,
        s[harmonic] / np.maximum(s[candidates], np.finfo(float).eps),
        acc_relative,
        acc_harmonic_relative,
        distance,
        bpm - previous,
        bpm - mean,
        np.full(candidates.size, std),
        np.full(candidates.size, slope),
    )).astype(np.float64)
