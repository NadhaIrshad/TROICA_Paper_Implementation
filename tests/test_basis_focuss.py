"""Dictionary pruning and FOCUSS (Blueprint Section 5.5, paper Section III-C)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.preprocessing.spectrum import local_maxima, periodogram
from troika.registry import get
from troika.ssr.basis import clear_cache, get_dictionary, kept_bins
from troika.ssr.focuss import focuss, focuss_reference, initial_x
from troika.types import StaticContext, WindowContext

FS = 125.0
N = 4096
MP = 998  # M' = M - 2 after the second-order difference


def _tone(f_hz, n=MP, fs=FS, amp=1.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f_hz * t)


def _normalized(x):
    return (x - x.mean()) / x.std()


def _static():
    return StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)


def _spectrum_from(coeffs, bins):
    half = N // 2
    out = np.zeros(half + 1)
    positive = bins <= half
    np.add.at(out, bins[positive], np.abs(coeffs[positive]) ** 2)
    return out


# -------------------------------------------------------------- Eq. 12 pruning


def test_kept_bins_match_the_worked_example():
    """Blueprint Section 5.5: 0-based 1..229 and 3867..4095, 458 columns."""
    bins = kept_bins(N, FS, 0.4, 5.0)
    positive = bins[bins <= N // 2]
    negative = bins[bins > N // 2]
    assert bins.size == 458
    assert (positive.min(), positive.max()) == (1, 229)
    assert (negative.min(), negative.max()) == (3867, 4095)


def test_kept_bins_are_symmetric_and_exclude_dc():
    """A real input has a conjugate-symmetric spectrum; DC carries no heart rate."""
    bins = kept_bins(N, FS, 0.4, 5.0)
    assert set(N - k for k in bins) == set(bins)
    assert 0 not in bins
    assert np.array_equal(bins, np.sort(bins))


def test_kept_range_covers_the_passband_with_transition_margin():
    """Paper Eq. 12 widens the band by the filter's transition band."""
    bins = kept_bins(N, FS, 0.4, 5.0)
    positive = bins[bins <= N // 2]
    lo_hz = binmap.bin_to_hz(int(positive.min()), FS, N)
    hi_hz = binmap.bin_to_hz(int(positive.max()), FS, N)
    assert lo_hz < 0.4
    assert hi_hz > 5.0
    assert hi_hz == pytest.approx(6.99, abs=0.05)


def test_pruning_keeps_every_heart_rate_bin():
    """40-200 BPM must survive pruning, or the tracker could never find it."""
    bins = set(kept_bins(N, FS, 0.4, 5.0).tolist())
    for bpm in (40, 70, 120, 180, 200):
        assert binmap.bpm_to_bin(bpm, FS, N) in bins


def test_dictionary_shapes_and_gram():
    Phi, G, bins = get_dictionary(MP, N, FS)
    assert Phi.shape == (MP, 458)
    assert Phi.dtype == np.complex128
    assert G.shape == (458, 458)
    assert np.allclose(G, Phi.conj().T @ Phi)


def test_gram_is_hermitian():
    """The Cholesky solve inside FOCUSS depends on this."""
    _, G, _ = get_dictionary(MP, N, FS)
    assert np.allclose(G, G.conj().T)


def test_dictionary_columns_are_dft_atoms():
    """Paper Eq. 9: Phi[m, n] = exp(j 2 pi m n / N)."""
    Phi, _, bins = get_dictionary(MP, N, FS)
    m = np.arange(MP)
    for column, k in ((0, int(bins[0])), (17, int(bins[17]))):
        assert np.allclose(Phi[:, column], np.exp(2j * np.pi * m * k / N))


def test_unpruned_dictionary_has_every_column():
    Phi, _, bins = get_dictionary(64, 128, FS, prune=False)
    assert Phi.shape == (64, 128)
    assert np.array_equal(bins, np.arange(128))


def test_dictionary_is_cached():
    """Blueprint Section 5.5: build the Gram matrix once, not per window."""
    clear_cache()
    first = get_dictionary(MP, N, FS)[0]
    second = get_dictionary(MP, N, FS)[0]
    assert first is second


# ------------------------------------------------------------------- FOCUSS


@pytest.fixture(scope="module")
def dictionary():
    return get_dictionary(MP, N, FS)


def test_fast_and_reference_forms_agree(dictionary):
    """Blueprint Section 9: the n x n push-through form equals the M x M form."""
    Phi, G, _ = dictionary
    z = _normalized(_tone(2.3) + _tone(2.12, amp=3.0))
    fast = focuss(z, Phi, G, 0.8, 0.1, 5)
    slow = focuss_reference(z, Phi, 0.8, 0.1, 5)
    assert np.abs(fast - slow).max() / np.abs(slow).max() < 1e-8


@pytest.mark.parametrize("n_iter", [1, 3, 5])
def test_the_two_forms_agree_at_every_iteration_count(dictionary, n_iter):
    Phi, G, _ = dictionary
    z = _normalized(_tone(1.8) + 0.5 * _tone(3.6))
    fast = focuss(z, Phi, G, 0.8, 0.1, n_iter)
    slow = focuss_reference(z, Phi, 0.8, 0.1, n_iter)
    assert np.abs(fast - slow).max() / np.abs(slow).max() < 1e-8


def test_focuss_finds_a_single_tone(dictionary):
    Phi, G, bins = dictionary
    z = _normalized(_tone(2.0))
    s = _spectrum_from(focuss(z, Phi, G, 0.8, 0.1, 5), bins)
    assert int(np.argmax(s)) == pytest.approx(binmap.hz_to_bin(2.0, FS, N), abs=2)


def test_focuss_sparsifies_across_iterations(dictionary):
    """Paper Section III-C: large coefficients converge within a few iterations."""
    Phi, G, _ = dictionary
    z = _normalized(_tone(2.0) + 0.5 * _tone(3.1))
    _, iterates = focuss(z, Phi, G, 0.8, 0.1, 5, return_iterates=True)
    assert iterates.shape == (6, 458)

    def occupancy(x):
        power = np.abs(x) ** 2
        return int((power > 0.01 * power.max()).sum())

    assert occupancy(iterates[-1]) < occupancy(iterates[1])


def test_focuss_beats_the_periodogram_on_close_tones(dictionary):
    """Paper Fig. 1: SSR separates peaks the Periodogram smears together.

    The tones are 0.08 Hz apart, below the 1/T = 0.125 Hz resolution of an 8 s
    window, and the motion component is three times the heartbeat.
    """
    Phi, G, bins = dictionary
    hr_hz, ma_hz = 2.18, 2.10
    z = _normalized(_tone(hr_hz) + _tone(ma_hz, amp=3.0))

    ssr = _spectrum_from(focuss(z, Phi, G, 0.8, 0.1, 5), bins)
    fft = periodogram(z, N)

    def resolves(spec):
        lo, hi = binmap.band_to_bins(1.8, 2.6, FS, N)
        peaks = local_maxima(spec, lo, hi)
        peaks = peaks[np.argsort(-spec[peaks])][:2]
        hz = binmap.bin_to_hz(peaks, FS, N)
        return any(abs(f - hr_hz) < 0.04 for f in hz) and any(
            abs(f - ma_hz) < 0.04 for f in hz
        )

    assert resolves(ssr), "FOCUSS should separate the two tones"
    assert not resolves(fft), "the Periodogram should merge them"


def test_focuss_is_deterministic(dictionary):
    Phi, G, _ = dictionary
    z = _normalized(_tone(2.0) + 0.3 * _tone(1.3))
    assert np.array_equal(focuss(z, Phi, G), focuss(z, Phi, G))


def test_focuss_does_not_mutate_its_inputs(dictionary):
    Phi, G, _ = dictionary
    z = _normalized(_tone(2.0))
    before_z, before_G = z.copy(), G.copy()
    focuss(z, Phi, G)
    assert np.array_equal(z, before_z) and np.array_equal(G, before_G)


def test_initial_x_modes(dictionary):
    """ASSUMPTION A11."""
    Phi, _, _ = dictionary
    z = _normalized(_tone(2.0))
    assert np.allclose(initial_x(z, Phi, "ones"), 1.0)
    matched = initial_x(z, Phi, "matched_filter")
    assert matched.shape == (458,)
    assert np.isrealobj(matched.real) and matched.real.max() > 0
    with pytest.raises(ValueError, match="unknown x0 mode"):
        initial_x(z, Phi, "nope")


# ------------------------------------------------------------------ plug-ins


def _estimator(name, **params):
    cls = get("spectrum_estimator", name)
    return cls(cls.Params(**params), _static())


def _ctx():
    return WindowContext(idx=0, t_start_s=0.0)


@pytest.mark.parametrize("name", ["focuss", "fft"])
def test_estimators_return_a_one_sided_spectrum(name):
    out = _estimator(name)(_normalized(_tone(2.0)), _ctx())
    assert out.data.shape == (N // 2 + 1,)
    assert (out.data >= 0).all()
    assert np.isfinite(out.data).all()


@pytest.mark.parametrize("name", ["focuss", "fft"])
def test_estimators_peak_at_the_right_frequency(name):
    out = _estimator(name)(_normalized(_tone(2.0)), _ctx())
    assert int(np.argmax(out.data)) == pytest.approx(binmap.hz_to_bin(2.0, FS, N), abs=2)


@pytest.mark.parametrize("name", ["focuss", "fft"])
def test_estimators_fill_their_diag_keys(name):
    """Lab Spec Section 2.3."""
    out = _estimator(name)(_normalized(_tone(2.0)), _ctx())
    assert {"iterates", "kept_bins", "sparsity_per_iter"} <= set(out.diag)


def test_focuss_diag_iterates_shape():
    out = _estimator("focuss", n_iter=5)(_normalized(_tone(2.0)), _ctx())
    assert out.diag["iterates"].shape == (6, 458)
    assert out.diag["sparsity_per_iter"].shape == (6,)


def test_focuss_spectrum_is_zero_outside_the_kept_bins():
    out = _estimator("focuss")(_normalized(_tone(2.0)), _ctx())
    kept = set(int(b) for b in out.diag["kept_bins"] if b <= N // 2)
    outside = [k for k in range(N // 2 + 1) if k not in kept]
    assert np.allclose(out.data[outside], 0.0)


@pytest.mark.parametrize(
    "bad", [{"p": 5.0}, {"p": 0.0}, {"lam": 0.0}, {"n_iter": 0}, {"x0": "nope"}]
)
def test_focuss_params_validate(bad):
    cls = get("spectrum_estimator", "focuss")
    with pytest.raises(ValueError):
        cls.Params(**bad)


def test_paper_parameter_defaults():
    """Paper Section IV-B: p = 0.8, lambda = 0.1, 5 iterations."""
    params = get("spectrum_estimator", "focuss").Params()
    assert (params.p, params.lam, params.n_iter) == (0.8, 0.1, 5)
    assert params.prune_columns is True


def test_focuss_runtime_fits_the_budget():
    """Blueprint Section 10 budgets under ~0.5 s per window for everything."""
    import time

    estimator = _estimator("focuss")
    z = _normalized(_tone(2.0) + _tone(1.3, amp=2.0))
    estimator(z, _ctx())  # warm the dictionary cache
    start = time.perf_counter()
    estimator(z, _ctx())
    assert time.perf_counter() - start < 0.3
