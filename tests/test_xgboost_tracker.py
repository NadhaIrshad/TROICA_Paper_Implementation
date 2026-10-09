"""Learned candidate-ranking tracker (DEVIATIONS.md D9)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.tracking.xgboost_features import (
    FEATURE_NAMES,
    candidate_bins,
    feature_matrix,
    guarded_choice,
    label_candidates,
)
from troika.types import StaticContext, WindowContext

FS = 125.0
N = 4096
HALF = N // 2 + 1


def spectrum_with(peaks: dict[int, float]) -> np.ndarray:
    """A spectrum that is zero except for isolated unit-width peaks."""
    s = np.zeros(HALF)
    for k, amplitude in peaks.items():
        s[int(k)] = float(amplitude)
    return s


def _ctx(prev_bin: int = 50, history: tuple[float, ...] = (90.0, 91.0, 92.0)) -> WindowContext:
    return WindowContext(
        idx=3, t_start_s=6.0, acc=np.zeros((3, 1000)), acc_bins_raw={44},
        prev_bin=prev_bin, bpm_history=list(history),
    )


# ---------------------------------------------------------------- candidates


def test_candidates_are_the_strongest_peaks_in_band_strongest_first():
    s = spectrum_with({30: 1.0, 50: 3.0, 90: 2.0, 10: 9.0, 150: 9.0})
    assert candidate_bins(s, FS, N, 40.0, 200.0, 8).tolist() == [50, 90, 30]
    assert candidate_bins(s, FS, N, 40.0, 200.0, 2).tolist() == [50, 90]


def test_a_flat_spectrum_still_yields_one_candidate():
    assert candidate_bins(np.zeros(HALF), FS, N, 40.0, 200.0, 8).size == 1


def test_feature_matrix_has_one_finite_row_per_candidate():
    s = spectrum_with({50: 3.0, 100: 1.0, 90: 2.0})
    candidates = candidate_bins(s, FS, N, 40.0, 200.0, 8)
    x = feature_matrix(s, _ctx(), candidates, FS, N)
    assert x.shape == (candidates.size, len(FEATURE_NAMES))
    assert np.isfinite(x).all()
    delta = x[:, FEATURE_NAMES.index("delta_previous_bpm")]
    assert delta[0] == pytest.approx(0.0)  # candidate 50 is the previous bin
    assert x[0, FEATURE_NAMES.index("harmonic_to_fundamental")] == pytest.approx(1 / 3)


def test_feature_matrix_works_without_history_or_accelerometer():
    s = spectrum_with({50: 3.0})
    ctx = WindowContext(idx=1, t_start_s=2.0)
    x = feature_matrix(s, ctx, np.array([50]), FS, N)
    assert np.isfinite(x).all()


# -------------------------------------------------------------------- labels


def test_label_marks_the_candidate_nearest_the_truth():
    candidates = np.array([90, 50, 30])
    truth = float(binmap.bin_to_bpm(50, FS, N)) + 1.0
    labels, covered = label_candidates(candidates, truth, FS, N)
    assert labels.tolist() == [0, 1, 0]
    assert covered


def test_label_reports_a_window_with_no_candidate_near_the_truth():
    labels, covered = label_candidates(np.array([90, 50]), 60.0, FS, N)
    assert labels.sum() == 1
    assert not covered


# ---------------------------------------------------------------- jump guard

CANDIDATES = np.array([95, 52, 48])
FAR_WINS = np.array([0.9, 0.4, 0.2])


def test_guard_passes_a_near_winner_through():
    scores = np.array([0.1, 0.8, 0.3])
    assert guarded_choice(CANDIDATES, scores, 50, None, 6, 3) == (52, None, False)


def test_guard_holds_a_far_winner_and_falls_back_to_the_best_near_candidate():
    assert guarded_choice(CANDIDATES, FAR_WINS, 50, None, 6, 3) == (52, (95, 1), True)


def test_guard_keeps_the_previous_bin_when_nothing_is_near():
    choice, pending, held = guarded_choice(np.array([95, 120]), np.array([0.9, 0.1]), 50, None, 6, 3)
    assert (choice, pending, held) == (50, (95, 1), True)


def test_guard_releases_a_far_winner_that_persists():
    prev, pending = 50, None
    picks = []
    for _ in range(3):
        prev, pending, _ = guarded_choice(CANDIDATES, FAR_WINS, prev, pending, 6, 3)
        picks.append(prev)
    assert picks == [52, 52, 95]
    assert pending is None


def test_guard_restarts_the_count_when_the_far_winner_moves():
    _, pending, _ = guarded_choice(CANDIDATES, FAR_WINS, 50, None, 6, 3)
    moved = np.array([140, 52, 48])
    assert guarded_choice(moved, FAR_WINS, 50, pending, 6, 3) == (52, (140, 1), True)


def test_a_near_winner_clears_the_pending_jump():
    scores = np.array([0.1, 0.8, 0.3])
    assert guarded_choice(CANDIDATES, scores, 50, (95, 2), 6, 3) == (52, None, False)


def test_patience_of_one_disables_the_guard():
    assert guarded_choice(CANDIDATES, FAR_WINS, 50, None, 6, 1) == (95, None, False)


# -------------------------------------------------------------------- plug-in


@pytest.fixture
def tracker_factory(tmp_path):
    """Build the plug-in around a tiny model that favours the strongest peak."""
    xgboost = pytest.importorskip("xgboost")
    from troika.plugins.tracker.xgboost import XGBoostTracker, XGBoostTrackerParams

    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURE_NAMES)))
    rank = FEATURE_NAMES.index("candidate_rank")
    x[:, rank] = rng.integers(0, 4, size=200)
    model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, random_state=0)
    model.fit(x, (x[:, rank] == 0).astype(int))
    path = tmp_path / "model.json"
    model.save_model(path)

    def build(**params):
        static = StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)
        return XGBoostTracker(XGBoostTrackerParams(model_path=str(path), **params), static)

    return build


def test_plugin_missing_model_names_the_path(tmp_path):
    pytest.importorskip("xgboost")
    from troika.plugins.tracker.xgboost import XGBoostTracker, XGBoostTrackerParams

    static = StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)
    with pytest.raises(FileNotFoundError, match="nowhere.json"):
        XGBoostTracker(XGBoostTrackerParams(model_path=str(tmp_path / "nowhere.json")), static)


@pytest.mark.parametrize("bad", [{"candidate_count": 0}, {"jump_patience": 0}, {"max_jump_bins": -1}, {"low_bpm": 200.0}])
def test_plugin_params_validate(bad):
    from troika.plugins.tracker.xgboost import XGBoostTrackerParams

    with pytest.raises(ValueError):
        XGBoostTrackerParams(**bad)


def test_plugin_initialises_from_the_highest_peak_and_records_history(tracker_factory):
    tracker = tracker_factory()
    out = tracker.initialize(spectrum_with({50: 3.0, 95: 1.0}), WindowContext(idx=0, t_start_s=0.0))
    assert out.data.bin_cur == 50
    assert tracker.prev_bin == 50
    assert tracker.bpm_history == [pytest.approx(float(binmap.bin_to_bpm(50, FS, N)))]


def test_plugin_holds_a_single_window_jump_but_follows_a_persistent_one(tracker_factory):
    tracker = tracker_factory()
    tracker.initialize(spectrum_with({50: 3.0}), WindowContext(idx=0, t_start_s=0.0))
    far = spectrum_with({95: 3.0, 51: 1.0})  # the toy model prefers the strongest peak
    bins, held = [], []
    for w in range(1, 4):
        ctx = _ctx(prev_bin=tracker.prev_bin, history=tuple(tracker.bpm_history))
        out = tracker.step(far, ctx)
        bins.append(out.data.bin_cur)
        held.append(out.diag["jump_held"])
    assert bins == [51, 51, 95]
    assert held == [True, True, False]


def test_plugin_without_the_guard_jumps_at_once(tracker_factory):
    tracker = tracker_factory(jump_patience=1)
    tracker.initialize(spectrum_with({50: 3.0}), WindowContext(idx=0, t_start_s=0.0))
    out = tracker.step(spectrum_with({95: 3.0, 51: 1.0}), _ctx(prev_bin=50))
    assert out.data.bin_cur == 95
