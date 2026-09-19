"""Pruned DFT dictionary for sparse signal reconstruction (paper Eqs. 9 and 12).

The SSR model is ``y = Phi x + v`` with ``Phi[m, n] = exp(j 2 pi m n / N)``,
``m = 0..M-1``, ``n = 0..N-1`` (paper Eq. 9). With ``N = 4096`` and
``M' = 998`` that matrix is large, so the paper prunes it: after band-pass
filtering only bins inside the pass band, widened by the filter's transition
band, can be non-zero. Paper Eq. 12, in 1-based MATLAB indices:

    I_Phi = [ f_lo*N/fs + 1 - df1 ,  f_hi*N/fs + 1 + df2 ]
          U [ N - f_hi*N/fs + 1 - df2 ,  N - f_lo*N/fs + 1 + df1 ]
    with  df1 = f_lo*N/fs - 1   and   df2 = 2*N/fs

For ``N = 4096``, ``fs = 125``, ``f_lo = 0.4`` and ``f_hi = 5`` this gives
1-based 2..230 and 3868..4096, which is 0-based 1..229 and 3867..4095: 458
columns instead of 4096. The kept set is symmetric under ``k -> N - k`` and
excludes DC, as a real-valued input requires.

This module holds the only cached global state in the package, as
``CLAUDE.md`` allows: the dictionary and its Gram matrix depend on
``(M, N, fs, band)`` alone and cost far too much to rebuild every window.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray

__all__ = ["kept_bins", "get_dictionary", "clear_cache"]


def kept_bins(
    N: int, fs: float, f_lo: float = 0.4, f_hi: float = 5.0
) -> NDArray[np.int64]:
    """Column indices kept by the Eq. 12 pruning, 0-based and sorted.

    ASSUMPTION A12: the fractional bounds are rounded outward-safe, with the
    lower bound up and the upper bound down, before converting to 0-based.
    """
    N = int(N)
    lo_pos = float(f_lo) * N / float(fs)
    hi_pos = float(f_hi) * N / float(fs)
    df1 = lo_pos - 1.0
    df2 = 2.0 * N / float(fs)

    # 1-based bounds straight from Eq. 12, then ceil/floor (A12), then 0-based.
    pos = (
        int(np.ceil(lo_pos + 1.0 - df1)) - 1,
        int(np.floor(hi_pos + 1.0 + df2)) - 1,
    )
    neg = (
        int(np.ceil(N - hi_pos + 1.0 - df2)) - 1,
        int(np.floor(N - lo_pos + 1.0 + df1)) - 1,
    )

    bins = set()
    for lo, hi in (pos, neg):
        bins.update(range(max(lo, 0), min(hi, N - 1) + 1))
    bins.discard(0)  # DC carries no heart rate and breaks the k -> N-k symmetry
    return np.fromiter(sorted(bins), dtype=np.int64)


@lru_cache(maxsize=8)
def _build(
    M: int, N: int, fs: float, f_lo: float, f_hi: float, prune: bool
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.int64]]:
    if prune:
        bins = kept_bins(N, fs, f_lo, f_hi)
    else:
        bins = np.arange(N, dtype=np.int64)
    m = np.arange(M, dtype=np.float64)[:, None]
    Phi = np.exp(2j * np.pi * m * bins[None, :].astype(np.float64) / float(N))
    G = Phi.conj().T @ Phi
    return Phi, G, bins


def get_dictionary(
    M: int,
    N: int,
    fs: float = 125.0,
    f_lo: float = 0.4,
    f_hi: float = 5.0,
    prune: bool = True,
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.int64]]:
    """Return ``(Phi, G, bins)``, cached per parameter set.

    ``Phi`` is ``(M, n_cols)`` complex, ``G = Phi^H Phi`` is ``(n_cols, n_cols)``
    and ``bins`` holds the 0-based grid index of each kept column. The arrays are
    shared, so callers must not modify them in place.
    """
    Phi, G, bins = _build(int(M), int(N), float(fs), float(f_lo), float(f_hi), bool(prune))
    return Phi, G, bins


def clear_cache() -> None:
    """Drop the cached dictionaries. Test helper."""
    _build.cache_clear()
