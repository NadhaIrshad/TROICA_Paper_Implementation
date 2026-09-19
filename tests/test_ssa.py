"""SSA embedding, SVD, grouping and reconstruction (Blueprint Sections 5.3 and 9)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.decomposition import grouping
from troika.decomposition import ssa as ssa_core

FS = 125.0
N = 4096
M = 1000
L = 400


def _tone(f_hz, n=M, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f_hz * t + phase)


@pytest.fixture(scope="module")
def decomposed():
    """A two-tone window plus noise, decomposed once for the whole module."""
    rng = np.random.default_rng(20150203)
    y = _tone(2.2) + _tone(1.4, amp=3.0) + 0.2 * rng.standard_normal(M)
    U, s, Vt = ssa_core.svd(ssa_core.embed(y, L))
    return y, U, s, Vt


# -------------------------------------------------------------------- embedding


def test_embed_shape_and_hankel_structure():
    """Paper Eq. 2: Y[i, j] = y[i + j], so anti-diagonals are constant."""
    y = np.arange(M, dtype=float)
    Y = ssa_core.embed(y, L)
    assert Y.shape == (L, M - L + 1)
    assert np.array_equal(Y[0, 1:], Y[1, :-1])
    for i in (0, 7, L - 1):
        for j in (0, 5, Y.shape[1] - 1):
            assert Y[i, j] == y[i + j]


def test_embed_rejects_bad_L():
    y = np.arange(100, dtype=float)
    with pytest.raises(ValueError, match="1 <= L <= M"):
        ssa_core.embed(y, 0)
    with pytest.raises(ValueError, match="1 <= L <= M"):
        ssa_core.embed(y, 101)


def test_paper_dimensions():
    """L = 400 and M = 1000 give K = 601 and d = 400 (Blueprint Section 3)."""
    Y = ssa_core.embed(np.zeros(M), L)
    assert Y.shape == (400, 601)
    assert min(Y.shape) == 400


# --------------------------------------------------------------- hankelisation


def test_hankelize_inverts_embed():
    """The M3 acceptance identity: hankelize(embed(y)) == y."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal(M)
    assert np.allclose(ssa_core.hankelize(ssa_core.embed(y, L), M), y, atol=1e-9)


def test_hankelize_averages_anti_diagonals():
    Y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # diagonals: [1], [2,4], [3,5], [6]
    assert np.allclose(ssa_core.hankelize(Y), [1.0, 3.0, 4.0, 6.0])


def test_hankelize_length_mismatch_raises():
    with pytest.raises(ValueError, match="expected"):
        ssa_core.hankelize(np.zeros((3, 4)), M=99)


# --------------------------------------------------------------- elementary


def test_elementary_series_matches_direct_hankelisation(decomposed):
    """The convolution shortcut must equal hankelising each rank-one matrix."""
    _, U, s, Vt = decomposed
    idx = range(12)
    direct = np.stack(
        [ssa_core.hankelize(np.outer(U[:, k] * s[k], Vt[k, :]), M) for k in idx]
    )
    assert np.allclose(ssa_core.elementary_series(U, s, Vt, M, idx), direct, atol=1e-10)


def test_elementary_series_sums_to_the_original(decomposed):
    y, U, s, Vt = decomposed
    assert np.allclose(ssa_core.elementary_series(U, s, Vt, M).sum(axis=0), y, atol=1e-9)


def test_elementary_series_empty_selection(decomposed):
    _, U, s, Vt = decomposed
    assert ssa_core.elementary_series(U, s, Vt, M, []).shape == (0, M)


def test_elementary_series_rejects_wrong_M(decomposed):
    _, U, s, Vt = decomposed
    with pytest.raises(ValueError, match="does not match"):
        ssa_core.elementary_series(U, s, Vt, M + 1)


# ------------------------------------------------------------------- grouping


@pytest.mark.parametrize("strategy", grouping.STRATEGIES)
def test_every_strategy_partitions_the_eigentriples(decomposed, strategy):
    """Groups must be disjoint and cover all d indices, or the sum identity fails."""
    _, U, s, Vt = decomposed
    groups, _ = grouping.group_eigentriples(U, s, Vt, M, strategy, n_fft=N, fs=FS)
    flat = [i for g in groups for i in g]
    assert sorted(flat) == list(range(s.size))
    assert len(flat) == len(set(flat))
    assert all(len(g) > 0 for g in groups)


@pytest.mark.parametrize("strategy", grouping.STRATEGIES)
def test_group_reconstructions_sum_to_the_original(decomposed, strategy):
    """Blueprint Section 9: the sum of all group reconstructions equals y."""
    y, U, s, Vt = decomposed
    groups, _ = grouping.group_eigentriples(U, s, Vt, M, strategy, n_fft=N, fs=FS)
    rec = ssa_core.reconstruct_groups(U, s, Vt, groups, M)
    assert rec.shape == (len(groups), M)
    assert np.allclose(rec.sum(axis=0), y, atol=1e-8)


def test_unknown_strategy_raises(decomposed):
    _, U, s, Vt = decomposed
    with pytest.raises(ValueError, match="unknown grouping strategy"):
        grouping.group_eigentriples(U, s, Vt, M, "nope")


def test_singleton_strategy_gives_one_group_per_triple(decomposed):
    _, U, s, Vt = decomposed
    groups, _ = grouping.group_eigentriples(U, s, Vt, M, "singleton")
    assert groups == [[i] for i in range(s.size)]


def test_pure_sinusoid_concentrates_in_two_eigentriples():
    """Blueprint Section 9: a sinusoid gives a pair with near-equal sigma."""
    y = _tone(1.7)
    U, s, Vt = ssa_core.svd(ssa_core.embed(y, L))
    assert s[:2].sum() / s.sum() > 0.999
    assert s[1] / s[0] > 0.95
    assert s[2] / s[0] < 1e-8


def test_frequency_pairing_merges_the_sinusoid_pair():
    """The default strategy exists to capture exactly that pair (A4)."""
    y = _tone(1.7)
    U, s, Vt = ssa_core.svd(ssa_core.embed(y, L))
    groups, _ = grouping.group_eigentriples(
        U, s, Vt, M, "frequency_pairing", n_fft=N, fs=FS
    )
    assert groups[0] == [0, 1]


def test_frequency_pairing_separates_two_distinct_tones():
    """Two well-separated tones must not land in one group."""
    y = _tone(1.2, amp=3.0) + _tone(2.6, amp=1.0)
    U, s, Vt = ssa_core.svd(ssa_core.embed(y, L))
    groups, elementary = grouping.group_eigentriples(
        U, s, Vt, M, "frequency_pairing", n_fft=N, fs=FS
    )
    components = np.stack([elementary[np.asarray(g)].sum(axis=0) for g in groups])
    bins = grouping.component_dominant_bins(components[:4], N, FS)
    hz = sorted(binmap.bin_to_hz(bins, FS, N))
    assert any(abs(f - 1.2) < 0.05 for f in hz)
    assert any(abs(f - 2.6) < 0.05 for f in hz)
    assert groups[0] == [0, 1] and groups[1] == [2, 3]


def test_wcorr_hclust_respects_a_requested_cluster_count(decomposed):
    _, U, s, Vt = decomposed
    groups, _ = grouping.group_eigentriples(
        U, s, Vt, M, "wcorr_hclust", n_fft=N, fs=FS, n_groups_max=8
    )
    assert len(groups) == 8
    assert sorted(i for g in groups for i in g) == list(range(s.size))


def test_component_dominant_bins_stay_inside_the_band():
    """Components are labelled by their dominant bin within 0.4-5 Hz."""
    series = np.stack([_tone(1.0), _tone(2.5), _tone(4.0)])
    bins = grouping.component_dominant_bins(series, N, FS)
    lo, hi = binmap.band_to_bins(0.4, 5.0, FS, N)
    assert ((bins >= lo) & (bins <= hi)).all()
    assert np.allclose(binmap.bin_to_hz(bins, FS, N), [1.0, 2.5, 4.0], atol=0.05)


# ------------------------------------------------------------------ w-correlation


def test_wcorr_is_symmetric_with_unit_diagonal(decomposed):
    _, U, s, Vt = decomposed
    series = ssa_core.elementary_series(U, s, Vt, M, range(10))
    W = grouping.wcorr(series, L, M - L + 1)
    assert W.shape == (10, 10)
    assert np.allclose(np.diag(W), 1.0)
    assert np.allclose(W, W.T)
    assert (np.abs(W) <= 1.0 + 1e-12).all()


def test_wcorr_pairs_the_sinusoid_components():
    """The two halves of a sinusoid are inseparable, so w-correlation is high."""
    y = _tone(1.7)
    U, s, Vt = ssa_core.svd(ssa_core.embed(y, L))
    series = ssa_core.elementary_series(U, s, Vt, M, range(4))
    W = np.abs(grouping.wcorr(series, L, M - L + 1))
    assert W[0, 1] > 0.9


# ------------------------------------------------------------------- performance


def test_decomposition_is_fast_enough_for_the_runtime_budget(decomposed):
    """Blueprint Section 10 budgets under ~0.5 s per window for the whole pipeline."""
    import time

    y, *_ = decomposed
    start = time.perf_counter()
    U, s, Vt = ssa_core.svd(ssa_core.embed(y, L))
    grouping.group_eigentriples(U, s, Vt, M, "frequency_pairing", n_fft=N, fs=FS)
    assert time.perf_counter() - start < 0.4
