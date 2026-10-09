"""Hidden-Markov path tracker (DEVIATIONS.md D12)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.plugins.tracker.hmm import HMMTracker, HMMTrackerParams
from troika.tracking.hmm import emission_log, transition_log, viterbi_step
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


def _tracker(**params) -> HMMTracker:
    static = StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)
    return HMMTracker(HMMTrackerParams(**params), static)


def _started_at(k: int, **params) -> HMMTracker:
    tracker = _tracker(**params)
    tracker.initialize(spectrum_with({k: 3.0}), WindowContext(idx=0, t_start_s=0.0))
    return tracker


def _step(tracker: HMMTracker, peaks: dict[int, float], idx: int = 1) -> int:
    return tracker.step(spectrum_with(peaks), WindowContext(idx=idx, t_start_s=2.0 * idx)).data.bin_cur


# ----------------------------------------------------------------------- core


def test_transition_rows_are_distributions_that_favour_staying():
    log_trans = transition_log(40, sigma_bins=3.0, jump_prob=0.01)
    trans = np.exp(log_trans)
    assert trans.sum(axis=1) == pytest.approx(np.ones(40))
    assert (np.argmax(trans, axis=1) == np.arange(40)).all()
    # Far moves keep the uniform share of jump_prob and nothing more.
    assert trans[0, 39] == pytest.approx(0.01 / 40, rel=1e-6)


def test_emission_peaks_at_the_candidates_and_falls_to_the_floor():
    emit = np.exp(emission_log(40, np.array([10, 30]), np.array([1.0, 0.5]), sigma_bins=1.0, floor=0.05))
    assert emit[10] == pytest.approx(1.05)
    assert emit[30] == pytest.approx(0.55)
    assert emit[20] == pytest.approx(0.05)
    assert np.exp(emission_log(40, np.array([], dtype=int), np.array([]), 1.0, 0.05)) == pytest.approx(np.full(40, 0.05))


def test_viterbi_step_is_normalised_and_follows_a_nearby_candidate():
    log_trans = transition_log(40, sigma_bins=3.0, jump_prob=0.01)
    start = emission_log(40, np.array([10]), np.array([1.0]), 1.0, 0.05)
    out = viterbi_step(start, log_trans, emission_log(40, np.array([12]), np.array([1.0]), 1.0, 0.05))
    assert out.max() == 0.0
    assert int(np.argmax(out)) == 12


# -------------------------------------------------------------------- plug-in


@pytest.mark.parametrize(
    "bad",
    [
        {"scorer": "forest"}, {"candidate_count": 0}, {"low_bpm": 200.0},
        {"transition_sigma_bins": 0.0}, {"emission_sigma_bins": 0.0},
        {"jump_prob": 0.0}, {"jump_prob": 1.0}, {"emission_floor": 0.0},
    ],
)
def test_plugin_params_validate(bad):
    with pytest.raises(ValueError):
        HMMTrackerParams(**bad)


def test_plugin_missing_model_names_the_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.pt"):
        _tracker(scorer="mlp", model_path=str(tmp_path / "nowhere.pt"))


def test_plugin_initialises_from_the_highest_peak_and_records_history():
    tracker = _tracker()
    out = tracker.initialize(spectrum_with({50: 3.0, 95: 1.0}), WindowContext(idx=0, t_start_s=0.0))
    assert out.data.bin_cur == 50
    assert tracker.prev_bin == 50
    assert tracker.bpm_history == [pytest.approx(float(binmap.bin_to_bpm(50, FS, N)))]


def test_plugin_follows_a_slowly_moving_peak():
    tracker = _started_at(50)
    assert [_step(tracker, {k: 3.0}, i) for i, k in enumerate((51, 53, 54, 56), start=1)] == [51, 53, 54, 56]


def test_plugin_ignores_a_far_peak_that_appears_for_one_window():
    tracker = _started_at(50)
    for i in range(1, 4):
        assert _step(tracker, {50: 3.0}, i) == 50
    assert _step(tracker, {95: 3.0}, 4) == 50
    assert _step(tracker, {50: 3.0}, 5) == 50


def test_plugin_leaves_a_wrong_first_window_once_the_far_peak_persists():
    tracker = _started_at(50)
    picks = [_step(tracker, {95: 3.0}, i) for i in range(1, 4)]
    assert picks[-1] == 95


def test_plugin_prefers_the_higher_scored_of_two_nearby_candidates():
    tracker = _started_at(50)
    assert _step(tracker, {49: 1.0, 52: 3.0}) == 52


def test_plugin_is_deterministic():
    picks = []
    for _ in range(2):
        tracker = _started_at(50)
        out = tracker.step(spectrum_with({95: 3.0, 51: 1.0}), WindowContext(idx=1, t_start_s=2.0))
        picks.append((out.data.bin_cur, out.diag["log_delta"].tolist()))
    assert picks[0] == picks[1]
