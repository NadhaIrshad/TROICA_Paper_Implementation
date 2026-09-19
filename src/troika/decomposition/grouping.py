"""SSA grouping strategies (ASSUMPTION A4).

Paper Section III-A says only that after the SVD "the grouping is automatically
finished by clustering singular values as described in [23, pp. 66]", and that
the rank-one matrices in a group share a common characteristic, such as their
reconstructed components having the same frequency or exhibiting a harmonic
relation. The exact algorithm is not given, so three strategies are provided and
the choice is a config option.

``frequency_pairing``
    The default. A real sinusoid inside a trajectory matrix produces a pair of
    eigentriples with nearly equal singular values and the same dominant
    frequency, so consecutive eigentriples are merged when both hold.

``wcorr_hclust``
    The classical SSA approach: hierarchical clustering of the weighted-
    correlation matrix of the elementary reconstructions.

``singleton``
    Every eigentriple is its own group. A baseline, and the most literal reading
    of "no grouping".
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.decomposition.ssa import elementary_series, wcorr
from troika.preprocessing.spectrum import dominant_bin, periodogram

__all__ = ["group_eigentriples", "STRATEGIES", "component_dominant_bins"]

STRATEGIES = ("frequency_pairing", "wcorr_hclust", "singleton")


def component_dominant_bins(
    series: NDArray[np.float64],
    n_fft: int,
    fs: float,
    f_lo: float = 0.4,
    f_hi: float = 5.0,
) -> NDArray[np.int64]:
    """Dominant frequency bin of each row of ``series``, restricted to the band.

    Blueprint Section 5.3: the dominant bin of a component is the argmax of its
    periodogram within 0.4-5 Hz, which is where the band-pass has left energy.
    """
    lo, hi = binmap.band_to_bins(f_lo, f_hi, fs, n_fft)
    return np.array(
        [dominant_bin(periodogram(row, n_fft), lo, hi) for row in series],
        dtype=np.int64,
    )


def _frequency_pairing(
    s: NDArray[np.float64],
    bins: NDArray[np.int64],
    sv_rel_tol: float,
    freq_tol_bins: int,
) -> list[list[int]]:
    """Merge consecutive eigentriples with close singular values and frequencies.

    A sinusoid appears as an adjacent pair of eigentriples whose singular values
    differ by little and whose reconstructions peak at the same bin, which is
    exactly what this captures. Anything else stays a singleton.
    """
    d = s.size
    groups: list[list[int]] = []
    current = [0]
    for t in range(1, d):
        prev = current[-1]
        scale = max(abs(s[prev]), abs(s[t]), 1e-300)
        close_sv = abs(s[prev] - s[t]) / scale < sv_rel_tol
        close_f = abs(int(bins[prev]) - int(bins[t])) <= freq_tol_bins
        if close_sv and close_f:
            current.append(t)
        else:
            groups.append(current)
            current = [t]
    groups.append(current)
    return groups


def _wcorr_hclust(
    series: NDArray[np.float64],
    L: int,
    K: int,
    n_groups_max: int,
    threshold: float = 0.5,
) -> list[list[int]]:
    """Average-linkage clustering on the w-correlation distance ``1 - |w|``.

    With ``n_groups_max = 0`` the cut is by distance, so the number of groups
    follows the data; otherwise exactly that many clusters are formed. Groups
    are returned sorted by their strongest member, keeping the dominant
    components first.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = series.shape[0]
    if d < 2:
        return [[0]] if d == 1 else []

    W = np.abs(wcorr(series, L, K))
    distance = 1.0 - W
    np.fill_diagonal(distance, 0.0)
    distance = np.clip((distance + distance.T) / 2.0, 0.0, None)

    Z = linkage(squareform(distance, checks=False), method="average")
    if n_groups_max and n_groups_max > 0:
        labels = fcluster(Z, t=min(int(n_groups_max), d), criterion="maxclust")
    else:
        labels = fcluster(Z, t=threshold, criterion="distance")

    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(int(index))
    return [groups[key] for key in sorted(groups, key=lambda k: min(groups[k]))]


def group_eigentriples(
    U: NDArray[np.float64],
    s: NDArray[np.float64],
    Vt: NDArray[np.float64],
    M: int,
    strategy: str = "frequency_pairing",
    *,
    n_fft: int = 4096,
    fs: float = 125.0,
    sv_rel_tol: float = 0.1,
    freq_tol_bins: int = 2,
    n_groups_max: int = 0,
    f_lo: float = 0.4,
    f_hi: float = 5.0,
    series: NDArray[np.float64] | None = None,
) -> tuple[list[list[int]], NDArray[np.float64]]:
    """Partition the ``d`` eigentriples into disjoint groups (ASSUMPTION A4).

    Returns ``(groups, elementary)`` where ``elementary`` is the ``(d, M)`` array
    of single-eigentriple reconstructions, which the caller reuses rather than
    recomputing. The groups always partition ``range(d)`` exactly, which is what
    makes the reconstruction identity hold.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown grouping strategy {strategy!r}; known: {list(STRATEGIES)}")

    d = int(s.size)
    if series is None:
        series = elementary_series(U, s, Vt, M)

    if strategy == "singleton":
        return [[t] for t in range(d)], series

    if strategy == "frequency_pairing":
        bins = component_dominant_bins(series, n_fft, fs, f_lo, f_hi)
        return _frequency_pairing(s, bins, sv_rel_tol, freq_tol_bins), series

    L = U.shape[0]
    K = Vt.shape[1]
    return _wcorr_hclust(series, L, K, n_groups_max), series
