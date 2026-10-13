"""Ensemble empirical mode decomposition (DEVIATIONS.md D13).

M. Pittara and C. Orphanidou, "Robust Estimation of Pulse Rate from a Wrist-type
PPG During Intensive Exercise", EMBC 2018, Section II-B: the PPG window is
decomposed with EEMD, the power spectral density of every mode is taken with a
Hamming window, and the modes whose peak frequency lies in 0.5-4 Hz are summed
into the pulsatile signal. No accelerometer is used.

That paper gives no EEMD settings and refers to Z. Wu and N. E. Huang, "Ensemble
Empirical Mode Decomposition: A Noise-Assisted Data Analysis Method", 2009. The
defaults here are that paper's: a fixed number of sifting passes, ``log2(M) - 1``
modes, white noise of 0.2 standard deviations and an ensemble of 100.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from troika import binmap
from troika.preprocessing.spectrum import periodogram

__all__ = ["default_n_imfs", "emd", "eemd", "pulsatile_mask"]


def default_n_imfs(n_samples: int) -> int:
    """Number of modes Wu and Huang extract from ``n_samples`` points: ``log2(M) - 1``."""
    return max(1, int(np.log2(int(n_samples))) - 1)


def _extrema(x: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Indices of the interior local maxima and minima of ``x``."""
    d = np.diff(x)
    maxima = np.flatnonzero((d[:-1] > 0) & (d[1:] <= 0)) + 1
    minima = np.flatnonzero((d[:-1] < 0) & (d[1:] >= 0)) + 1
    return maxima, minima


def _envelope(x: NDArray[np.float64], idx: NDArray[np.int64], upper: bool) -> NDArray[np.float64]:
    """Cubic spline through the extrema ``idx`` of ``x``, extended to both ends.

    Each end value is the straight line through the two nearest extrema,
    replaced by the end sample itself when that lies outside the line (Wu and
    Huang's end treatment), so the envelope never cuts into the data there.
    """
    n = x.size
    t = idx.astype(np.float64)
    v = x[idx]
    left = v[0] - (v[1] - v[0]) * t[0] / (t[1] - t[0])
    right = v[-1] + (v[-1] - v[-2]) * (n - 1 - t[-1]) / (t[-1] - t[-2])
    pick = max if upper else min
    knots = np.concatenate(([0.0], t, [n - 1.0]))
    values = np.concatenate(([pick(left, x[0])], v, [pick(right, x[-1])]))
    return CubicSpline(knots, values)(np.arange(n, dtype=np.float64))


def emd(x: NDArray[np.float64], n_imfs: int, n_sifts: int = 10) -> NDArray[np.float64]:
    """Empirical mode decomposition with a fixed number of sifting passes.

    Returns ``(n_imfs + 1, M)``: the intrinsic mode functions, fastest first,
    then the residual. The rows always sum to ``x``. When the residual runs out
    of extrema before ``n_imfs`` modes are found, the remaining rows are zero.
    """
    x = np.asarray(x, dtype=np.float64)
    modes = np.zeros((int(n_imfs) + 1, x.size))
    residual = x.copy()
    for m in range(int(n_imfs)):
        h = residual.copy()
        complete = True
        for _ in range(int(n_sifts)):
            maxima, minima = _extrema(h)
            if maxima.size < 2 or minima.size < 2:
                complete = False
                break
            h = h - 0.5 * (_envelope(h, maxima, True) + _envelope(h, minima, False))
        if not complete:
            break
        modes[m] = h
        residual = residual - h
    modes[-1] = residual
    return modes


def eemd(
    x: NDArray[np.float64],
    ensemble_size: int = 100,
    noise_std: float = 0.2,
    n_sifts: int = 10,
    n_imfs: int | None = None,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Ensemble mean of :func:`emd` over noise-added copies of ``x``.

    ``noise_std`` is in units of the standard deviation of ``x``. The noise is
    drawn from ``seed`` on every call, so the same window always decomposes the
    same way. The rows sum to ``x`` plus the ensemble mean of the noise, which
    shrinks as ``noise_std / sqrt(ensemble_size)``. ``noise_std = 0`` is plain
    EMD.
    """
    x = np.asarray(x, dtype=np.float64)
    n_imfs = default_n_imfs(x.size) if n_imfs is None else int(n_imfs)
    scale = float(np.std(x))
    if scale == 0.0 or noise_std == 0.0:
        return emd(x, n_imfs, n_sifts)

    rng = np.random.default_rng(int(seed))
    z = x / scale
    total = np.zeros((n_imfs + 1, x.size))
    for _ in range(int(ensemble_size)):
        total += emd(z + float(noise_std) * rng.standard_normal(x.size), n_imfs, n_sifts)
    return total * (scale / int(ensemble_size))


def pulsatile_mask(
    modes: NDArray[np.float64], fs: float, n_fft: int, low_hz: float, high_hz: float,
) -> tuple[NDArray[np.bool_], NDArray[np.int64]]:
    """Which modes belong to the pulse, and every mode's peak-frequency bin.

    A mode is kept when the highest peak of its Hamming-windowed periodogram
    lies in ``[low_hz, high_hz]`` (Pittara and Orphanidou, Section II-B).
    """
    modes = np.asarray(modes, dtype=np.float64)
    window = np.hamming(modes.shape[1])
    dominant = np.array([int(np.argmax(periodogram(mode * window, n_fft))) for mode in modes], dtype=np.int64)
    lo, hi = binmap.band_to_bins(low_hz, high_hz, fs, n_fft)
    return (dominant >= lo) & (dominant <= hi), dominant
