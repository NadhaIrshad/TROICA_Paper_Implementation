"""Singular spectrum analysis (Blueprint Section 5.3, paper Section III-A).

SSA decomposes a time series into oscillatory components and noise in four
steps: Embedding, SVD, Grouping and Reconstruction. Grouping lives in
``grouping.py``; this module holds the other three plus the perfect-
reconstruction identity they must satisfy.

Notation follows the paper: the window of ``M`` samples is embedded into an
``L x K`` trajectory matrix with ``K = M - L + 1``. The paper sets ``L = 400``
for ``M = 1000``, following the rule of thumb that ``L`` be close to ``M / 2``.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "embed",
    "svd",
    "hankelize",
    "group_matrices",
    "reconstruct_groups",
    "elementary_series",
    "wcorr",
]


def embed(y: NDArray[np.float64], L: int) -> NDArray[np.float64]:
    """Build the ``L x K`` Hankel trajectory matrix (paper Eq. 2).

    ``Y[i, j] = y[i + j]`` with ``K = M - L + 1``. The result is a read-only
    view when possible, so embedding a 1000-sample window costs nothing.
    """
    y = np.asarray(y, dtype=np.float64)
    M = y.size
    L = int(L)
    if not 1 <= L <= M:
        raise ValueError(f"need 1 <= L <= M, got L={L} and M={M}")
    K = M - L + 1
    return np.lib.stride_tricks.sliding_window_view(y, K)[:L]


def svd(
    Y: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Singular value decomposition of the trajectory matrix (paper Eq. 3).

    Returns ``(U, s, Vt)`` with ``d = min(L, K)`` triples, ordered by decreasing
    singular value.
    """
    return np.linalg.svd(np.asarray(Y, dtype=np.float64), full_matrices=False)


@lru_cache(maxsize=16)
def _diagonal_counts(L: int, K: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Anti-diagonal index map and element counts for an ``L x K`` matrix.

    Cached per shape: the pipeline hankelises hundreds of matrices per window,
    all of the same shape.
    """
    i, j = np.indices((L, K))
    idx = (i + j).ravel()
    return idx, np.bincount(idx)


def hankelize(Y_I: NDArray[np.float64], M: int | None = None) -> NDArray[np.float64]:
    """Diagonal averaging: turn a matrix back into a series of length ``M``.

    Paper Section III-A, Reconstruction step. ``y[n]`` is the mean of
    ``Y_I[i, j]`` over all ``(i, j)`` with ``i + j = n``.
    """
    Y_I = np.asarray(Y_I, dtype=np.float64)
    L, K = Y_I.shape
    idx, counts = _diagonal_counts(L, K)
    out = np.bincount(idx, weights=Y_I.ravel()) / counts
    if M is not None and out.size != int(M):
        raise ValueError(f"hankelize produced {out.size} samples, expected {M}")
    return out


def group_matrices(
    U: NDArray[np.float64],
    s: NDArray[np.float64],
    Vt: NDArray[np.float64],
    groups: list[list[int]],
) -> list[NDArray[np.float64]]:
    """Rebuild one ``L x K`` matrix per group (paper Eq. 4).

    ``Y_I = sum_{t in I} sigma_t u_t v_t^T``.
    """
    out = []
    for members in groups:
        idx = np.asarray(members, dtype=int)
        out.append((U[:, idx] * s[idx]) @ Vt[idx, :])
    return out


def reconstruct_groups(
    U: NDArray[np.float64],
    s: NDArray[np.float64],
    Vt: NDArray[np.float64],
    groups: list[list[int]],
    M: int,
) -> NDArray[np.float64]:
    """Reconstructed series per group, shape ``(g, M)`` (paper Eq. 5).

    When the groups partition all ``d`` eigentriples, the rows sum back to the
    original series to within floating-point error. That identity is the M3
    acceptance test.

    Diagonal averaging is linear, so a group's series is the sum of its members'
    elementary series; that is what makes the fast convolution path usable here.
    """
    if not groups:
        return np.zeros((0, int(M)), dtype=np.float64)
    elementary = elementary_series(U, s, Vt, M)
    return np.stack([elementary[np.asarray(g, dtype=int)].sum(axis=0) for g in groups])


def elementary_series(
    U: NDArray[np.float64],
    s: NDArray[np.float64],
    Vt: NDArray[np.float64],
    M: int,
    indices: NDArray[np.int64] | list[int] | None = None,
) -> NDArray[np.float64]:
    """Reconstructed series of individual eigentriples, shape ``(len(indices), M)``.

    Used by the grouping strategies and by the w-correlation plot.

    Computed as a convolution rather than by hankelising each rank-one matrix.
    Diagonal averaging sums ``sigma * u_i * v_j`` over ``i + j = n``, and that
    sum is exactly ``sigma * (u * v)[n]``, so all ``d`` series come from one
    batched FFT instead of ``d`` dense ``L x K`` passes. For the paper's
    ``L = 400``, ``M = 1000`` this is roughly 40 times faster and agrees with
    the direct form to floating-point error.
    """
    if indices is None:
        idx = np.arange(s.size)
    else:
        idx = np.asarray(list(indices), dtype=int)
    if idx.size == 0:
        return np.zeros((0, int(M)), dtype=np.float64)

    L, K = U.shape[0], Vt.shape[1]
    length = L + K - 1
    if length != int(M):
        raise ValueError(f"L + K - 1 = {length} does not match M = {M}")

    n_fft = int(1 << (length - 1).bit_length())
    left = np.fft.rfft(U[:, idx].T, n=n_fft, axis=-1)
    right = np.fft.rfft(Vt[idx, :], n=n_fft, axis=-1)
    conv = np.fft.irfft(left * right, n=n_fft, axis=-1)[:, :length]

    _, counts = _diagonal_counts(L, K)
    return conv * s[idx][:, None] / counts


def wcorr(series: NDArray[np.float64], L: int, K: int) -> NDArray[np.float64]:
    """Weighted-correlation matrix of reconstructed series (Golyandina et al.).

    The weight of sample ``n`` is the number of times it appears in the
    trajectory matrix, ``min(n + 1, L, K, M - n)``. Two series with a
    w-correlation near 1 are considered separable poorly and belong together.
    """
    series = np.asarray(series, dtype=np.float64)
    M = series.shape[1]
    Ls, Ks = min(L, K), max(L, K)
    n = np.arange(M)
    weights = np.minimum(np.minimum(n + 1, Ls), np.minimum(Ks, M - n)).astype(np.float64)

    inner = (series * weights) @ series.T
    norms = np.sqrt(np.diag(inner))
    norms = np.where(norms > 0, norms, 1.0)
    out = inner / np.outer(norms, norms)
    return np.clip(out, -1.0, 1.0)
