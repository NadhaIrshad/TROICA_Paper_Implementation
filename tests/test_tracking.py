"""Spectral peak tracking (Blueprint Sections 5.6 and 9, paper Section III-D)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.registry import get
from troika.tracking.peaks import search_range, top_peaks
from troika.tracking.selection import select_bin
from troika.tracking.tracker import SpectralPeakTracker
from troika.tracking.verification import limit_jump, predict_trend
from troika.types import StaticContext, WindowContext

FS = 125.0
N = 4096
HALF = N // 2 + 1


def spectrum_with(peaks: dict[int, float], size: int = HALF) -> np.ndarray:
    """A spectrum that is zero except for isolated unit-width peaks."""
    s = np.zeros(size)
    for k, amplitude in peaks.items():
        s[int(k)] = float(amplitude)
    return s


def _static():
    return StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)


def _tracker(**kwargs) -> SpectralPeakTracker:
    return SpectralPeakTracker(FS, N, **kwargs)


# ------------------------------------------------------------- search ranges


def test_r0_and_r1_follow_the_paper():
    """R0 = [k-Ds, k+Ds]; R1 = [2(k-Ds), 2(k+Ds)] in 0-based bins."""
    assert search_range(100, 16, HALF, 1) == (84, 116)
    assert search_range(100, 16, HALF, 2) == (168, 232)


def test_search_ranges_clip_to_the_spectrum():
    assert search_range(5, 16, HALF, 1) == (0, 21)
    assert search_range(HALF - 3, 16, HALF, 2)[1] == HALF - 1


def test_top_peaks_orders_by_amplitude_and_applies_eta():
    s = spectrum_with({10: 1.0, 20: 5.0, 30: 3.0, 40: 0.5})
    assert top_peaks(s, 0, 50, eta=0.0, max_peaks=3).tolist() == [20, 30, 10]
    assert top_peaks(s, 0, 50, eta=2.0, max_peaks=3).tolist() == [20, 30]


def test_top_peaks_caps_at_max_peaks():
    """Paper Section III-D.2: no more than three peaks per range."""
    s = spectrum_with({i: float(i) for i in range(10, 60, 5)})
    assert top_peaks(s, 0, 100, eta=0.0, max_peaks=3).size == 3


def test_top_peaks_on_an_empty_range():
    assert top_peaks(np.zeros(HALF), 50, 40, eta=0.0).size == 0


# ------------------------------------------------------------ Cases 1, 2, 3


def test_case_one_harmonic_pair():
    """Paper Eq. 13: a fundamental with its harmonic present wins."""
    k_prev, k_hr = 100, 104
    s = spectrum_with({k_hr: 1.0, 2 * k_hr: 0.8, 92: 0.9})
    k_b, case, info = select_bin(s, k_prev)
    assert case == 1
    assert k_b == k_hr
    assert (k_hr, 2 * k_hr) in info["pairs"]


def test_case_one_tolerates_an_imperfect_harmonic():
    """ASSUMPTION A7: the pair test allows +/- 2 bins."""
    k_prev, k_hr = 100, 104
    s = spectrum_with({k_hr: 1.0, 2 * k_hr + 2: 0.8})
    k_b, case, _ = select_bin(s, k_prev, harm_tol=2)
    assert (k_b, case) == (k_hr, 1)

    k_b, case, _ = select_bin(s, k_prev, harm_tol=1)
    assert case == 2


def test_case_one_tiebreak_by_power():
    """ASSUMPTION A8: with several pairs, the strongest sum wins."""
    k_prev = 100
    s = spectrum_with({96: 0.4, 192: 0.4, 104: 1.0, 208: 0.9})
    k_b, case, info = select_bin(s, k_prev, pair_tiebreak="max_power")
    assert case == 1 and k_b == 104
    assert len(info["pairs"]) == 2


def test_case_one_tiebreak_by_proximity():
    """The A8 alternative picks the pair nearest the previous estimate."""
    k_prev = 100
    s = spectrum_with({98: 0.4, 196: 0.4, 110: 1.0, 220: 0.9})
    k_b, case, _ = select_bin(s, k_prev, pair_tiebreak="closest_to_prev")
    assert case == 1 and k_b == 98


def test_unknown_tiebreak_raises():
    s = spectrum_with({104: 1.0, 208: 0.8})
    with pytest.raises(ValueError, match="pair_tiebreak"):
        select_bin(s, 100, pair_tiebreak="nope")


def test_case_two_picks_the_candidate_closest_to_the_previous_bin():
    """Paper Eq. 14, with no harmonic pair available."""
    k_prev = 100
    s = spectrum_with({90: 1.0, 103: 0.9})
    k_b, case, _ = select_bin(s, k_prev)
    assert case == 2 and k_b == 103


def test_case_two_includes_halved_harmonic_peaks():
    """Candidates are P0 together with P1 halved (ASSUMPTION A17)."""
    k_prev = 100
    s = spectrum_with({88: 1.0, 202: 0.9})
    k_b, case, info = select_bin(s, k_prev)
    assert case == 2
    assert 101 in info["candidates"].tolist()
    assert k_b == 101


def test_case_two_floor_rounding_option():
    k_prev = 100
    s = spectrum_with({88: 1.0, 203: 0.9})
    assert select_bin(s, k_prev, half_rounding="round")[0] == 102
    assert select_bin(s, k_prev, half_rounding="floor")[0] == 101


def test_case_three_keeps_the_previous_bin():
    """Paper Eq. 15: no peaks anywhere, so the estimate holds."""
    k_b, case, _ = select_bin(np.zeros(HALF), 100)
    assert case == 3 and k_b == 100


def test_eta_is_thirty_percent_of_the_r0_maximum():
    """Paper Section III-D.2, with A9 applying the same value to R1."""
    k_prev = 100
    s = spectrum_with({104: 1.0, 96: 0.2, 208: 0.25})
    _, _, info = select_bin(s, k_prev, eta_frac=0.30)
    assert info["eta"] == pytest.approx(0.30)
    assert 96 not in info["P0"].tolist()  # 0.2 < eta
    assert 208 not in info["P1"].tolist()  # 0.25 < eta, A9


def test_eta_falls_back_to_r1_when_r0_is_empty():
    """A9 fallback, so a visible harmonic is not lost for want of a fundamental."""
    k_prev = 100
    s = spectrum_with({205: 1.0})
    _, case, info = select_bin(s, k_prev)
    assert info["P1"].size == 1
    assert case == 2


def test_peaks_outside_the_search_range_are_ignored():
    """A narrow range is the point: it excludes non-HR peaks (paper Section III-D.3)."""
    k_prev = 100
    s = spectrum_with({300: 10.0, 103: 1.0})
    k_b, case, _ = select_bin(s, k_prev, delta_s=16)
    assert k_b == 103 and case == 2


# --------------------------------------------------------- verification rules


def test_rule_one_clamps_upward_jumps():
    """Paper Eq. 16 with theta = 6 bins (about 11 BPM) and tau = 2."""
    assert limit_jump(120, 100, 6, 2) == (102, True)


def test_rule_one_clamps_downward_jumps():
    assert limit_jump(80, 100, 6, 2) == (98, True)


def test_rule_one_passes_small_changes_through():
    assert limit_jump(105, 100, 6, 2) == (105, False)
    assert limit_jump(95, 100, 6, 2) == (95, False)


def test_rule_one_boundary_is_inclusive():
    """A change of exactly theta already counts as too large."""
    assert limit_jump(106, 100, 6, 2) == (102, True)
    assert limit_jump(94, 100, 6, 2) == (98, True)


def test_trend_needs_enough_history():
    """A third-order fit needs four points; fewer gives no trend."""
    assert predict_trend([100.0], 3, 20, 3.0)[0] == 0
    assert predict_trend([100.0, 101.0, 102.0], 3, 20, 3.0)[0] == 0
    assert predict_trend([], 3, 20, 3.0)[0] == 0


def test_trend_detects_a_rising_heart_rate():
    history = list(np.linspace(100.0, 140.0, 12))
    trend, predicted = predict_trend(history, 3, 20, 3.0)
    assert trend == 1 and predicted > history[-1]


def test_trend_detects_a_falling_heart_rate():
    history = list(np.linspace(160.0, 120.0, 12))
    assert predict_trend(history, 3, 20, 3.0)[0] == -1


def test_trend_is_zero_for_a_flat_history():
    assert predict_trend([120.0] * 12, 3, 20, 3.0)[0] == 0


def test_trend_only_uses_the_recent_history():
    """Paper Eq. 18 fits the previous 20 windows, not the whole recording.

    Stated as the property itself: older values must not reach the fit at all.
    """
    recent = list(np.linspace(100.0, 150.0, 25))
    old = [60.0] * 40
    assert predict_trend(old + recent, 3, 20, 3.0) == predict_trend(recent[-20:], 3, 20, 3.0)


def test_trend_does_not_warn_on_a_degenerate_fit():
    """Blueprint Section 5.6: suppress polyfit's RankWarning."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        predict_trend([100.0, 100.0, 100.0, 100.0], 3, 20, 3.0)


# ------------------------------------------------------------------- tracker


def test_initialisation_takes_the_highest_peak_in_the_band():
    """Paper Section III-D.1, with the A10 band restriction."""
    tracker = _tracker()
    k_true = binmap.bpm_to_bin(120.0, FS, N)
    k, info = tracker.initialize(spectrum_with({k_true: 1.0, 5: 10.0}))
    assert k == k_true
    assert info["case"] is None
    assert tracker.prev_bin == k_true
    assert tracker.bpm(k) == pytest.approx(120.0, abs=1.0)


def test_initialisation_band_excludes_implausible_rates():
    """A 10 BPM peak must not be chosen as the starting heart rate."""
    tracker = _tracker(init_band_bpm=(40.0, 200.0))
    low = binmap.bpm_to_bin(10.0, FS, N)
    inside = binmap.bpm_to_bin(150.0, FS, N)
    k, _ = tracker.initialize(spectrum_with({low: 10.0, inside: 1.0}))
    assert k == inside


def test_ground_truth_initialisation_is_available_for_debugging():
    tracker = _tracker(init_mode="ground_truth")
    k, _ = tracker.initialize(np.zeros(HALF), gt_bpm=123.0)
    assert tracker.bpm(k) == pytest.approx(123.0, abs=1.0)


def test_ground_truth_initialisation_without_ground_truth_raises():
    with pytest.raises(ValueError, match="needs a ground-truth BPM"):
        _tracker(init_mode="ground_truth").initialize(np.zeros(HALF), gt_bpm=None)


def test_unknown_init_mode_raises():
    with pytest.raises(ValueError, match="init_mode"):
        _tracker(init_mode="nope").initialize(np.zeros(HALF))


def test_step_before_initialize_raises():
    with pytest.raises(RuntimeError, match="before initialize"):
        _tracker().step(np.zeros(HALF))


def test_tracker_follows_a_slow_drift():
    """Successive windows overlap heavily, so the estimate should track smoothly."""
    tracker = _tracker()
    start = binmap.bpm_to_bin(100.0, FS, N)
    tracker.initialize(spectrum_with({start: 1.0, 2 * start: 0.8}))
    for step in range(1, 10):
        k = start + step * 2
        k_cur, info = tracker.step(spectrum_with({k: 1.0, 2 * k: 0.8}))
        assert info["case"] == 1
        assert k_cur == k
    assert tracker.prev_bin == start + 18


def test_rule_two_fires_after_three_identical_windows():
    """Paper Eq. 17 with h = 3; the counter includes the current window (A13)."""
    tracker = _tracker()
    k = binmap.bpm_to_bin(100.0, FS, N)
    stuck = spectrum_with({k: 1.0, 2 * k: 0.8})
    tracker.initialize(stuck)

    fired = []
    for _ in range(4):
        _, info = tracker.step(stuck)
        fired.append(info["rule2_fired"])
    assert fired == [False, False, True, False]


def test_rule_two_widens_the_search_range_for_the_next_window():
    """ASSUMPTION A13: the wider Delta_s applies to the window after the stall."""
    tracker = _tracker(delta_s=16, delta_s_wide=20)
    k = binmap.bpm_to_bin(100.0, FS, N)
    stuck = spectrum_with({k: 1.0, 2 * k: 0.8})
    tracker.initialize(stuck)

    used = []
    for _ in range(5):
        _, info = tracker.step(stuck)
        used.append(info["delta_s_used"])
    assert used == [16, 16, 16, 20, 16]


@pytest.mark.parametrize(
    "history,expected_trend",
    [
        (list(np.linspace(100.0, 160.0, 20)), 1),
        (list(np.linspace(160.0, 100.0, 20)), -1),
        ([130.0] * 20, 0),
    ],
)
def test_rule_two_moves_the_estimate_by_twice_the_trend(history, expected_trend):
    """Paper Eq. 17: N_cur = N_prev + 2 * N_Trend.

    The history is seeded directly so the test fixes the arithmetic of Eq. 17
    rather than the behaviour of a cubic fit on a particular ramp.
    """
    tracker = _tracker()
    stuck_at = binmap.bpm_to_bin(130.0, FS, N)
    stuck = spectrum_with({stuck_at: 1.0, 2 * stuck_at: 0.8})

    tracker.state.prev_bin = stuck_at
    tracker.state.bpm_history = list(history)
    tracker.state.same_count = tracker.stall_windows_h - 1

    _, info = tracker.step(stuck)
    assert info["rule2_fired"]
    assert info["trend"] == expected_trend
    assert info["k_cur"] == stuck_at + 2 * expected_trend


def test_verification_can_be_disabled_for_the_ablation():
    """Table I row 4 removes verification, so a large jump goes through."""
    tracker = _tracker(verification_enabled=False)
    start = binmap.bpm_to_bin(100.0, FS, N)
    tracker.initialize(spectrum_with({start: 1.0}))
    far = start + 15
    k_cur, info = tracker.step(spectrum_with({far: 1.0, 2 * far: 0.8}))
    assert k_cur == far
    assert not info["rule1_fired"] and not info["rule2_fired"]


def test_verification_enabled_would_have_clamped_that_jump():
    tracker = _tracker(verification_enabled=True)
    start = binmap.bpm_to_bin(100.0, FS, N)
    tracker.initialize(spectrum_with({start: 1.0}))
    far = start + 15
    k_cur, info = tracker.step(spectrum_with({far: 1.0, 2 * far: 0.8}))
    assert k_cur == start + 2
    assert info["rule1_fired"]


def test_bpm_history_grows_one_entry_per_window():
    tracker = _tracker()
    k = binmap.bpm_to_bin(100.0, FS, N)
    s = spectrum_with({k: 1.0, 2 * k: 0.8})
    tracker.initialize(s)
    for _ in range(5):
        tracker.step(s)
    assert len(tracker.state.bpm_history) == 6


def test_estimates_stay_inside_the_spectrum():
    """Rule 2 near the edge must not index outside the array."""
    tracker = _tracker()
    tracker.initialize(spectrum_with({HALF - 1: 1.0}))
    for _ in range(6):
        k_cur, _ = tracker.step(np.zeros(HALF))
        assert 0 <= k_cur < HALF


# ------------------------------------------------------------------ plug-in


def _plugin(**params):
    cls = get("tracker", "troika")
    return cls(cls.Params(**params), _static())


def test_plugin_initialize_and_step_return_tracker_results():
    tracker = _plugin()
    k = binmap.bpm_to_bin(120.0, FS, N)
    s = spectrum_with({k: 1.0, 2 * k: 0.8})

    first = tracker.initialize(s, WindowContext(idx=0, t_start_s=0.0))
    assert first.data.bin_cur == k
    assert first.data.bpm == pytest.approx(120.0, abs=1.0)
    assert first.data.case is None

    second = tracker.step(s, WindowContext(idx=1, t_start_s=2.0))
    assert second.data.case == 1


def test_plugin_dispatch_uses_initialize_then_step():
    """The base class __call__ routes the first window to initialize."""
    tracker = _plugin()
    k = binmap.bpm_to_bin(120.0, FS, N)
    s = spectrum_with({k: 1.0, 2 * k: 0.8})
    assert tracker(s, WindowContext(idx=0, t_start_s=0.0)).data.case is None
    assert tracker(s, WindowContext(idx=1, t_start_s=2.0)).data.case == 1


def test_plugin_fills_every_diag_key():
    """Lab Spec Section 2.3."""
    tracker = _plugin()
    k = binmap.bpm_to_bin(120.0, FS, N)
    s = spectrum_with({k: 1.0, 2 * k: 0.8})
    tracker.initialize(s, WindowContext(idx=0, t_start_s=0.0))
    diag = tracker.step(s, WindowContext(idx=1, t_start_s=2.0)).diag
    required = {
        "R0", "R1", "eta", "P0", "P1", "pairs", "candidates", "k_b", "case",
        "k_cur", "rule1_fired", "rule2_fired", "same_count", "trend",
        "delta_s_used", "k_prev",
    }
    assert required <= set(diag)


def test_plugin_reports_paper_defaults():
    params = get("tracker", "troika").Params()
    assert (params.delta_s, params.delta_s_wide) == (16, 20)
    assert (params.max_peaks_per_range, params.eta_frac) == (3, 0.30)
    assert params.verification["theta_bins"] == 6
    assert params.verification["tau_bins"] == 2
    assert params.verification["stall_windows_h"] == 3


def test_plugin_flags_ground_truth_initialisation():
    """Lab Spec Section 2.4: this configuration marks a run contaminated."""
    assert not get("tracker", "troika").Params().uses_ground_truth
    assert get("tracker", "troika").Params(init_mode="ground_truth").uses_ground_truth


@pytest.mark.parametrize(
    "bad",
    [
        {"init_mode": "nope"},
        {"init_band_bpm": [200, 40]},
        {"delta_s": 0},
        {"eta_frac": 2.0},
        {"max_peaks_per_range": 0},
        {"pair_tiebreak": "nope"},
        {"half_rounding": "nope"},
        {"verification": {"unknown_key": 1}},
    ],
)
def test_plugin_params_validate(bad):
    with pytest.raises(ValueError):
        get("tracker", "troika").Params(**bad)


def test_partial_verification_dict_keeps_the_other_defaults():
    params = get("tracker", "troika").Params(verification={"enabled": False})
    assert params.verification["enabled"] is False
    assert params.verification["theta_bins"] == 6
