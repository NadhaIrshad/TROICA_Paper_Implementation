"""MLP candidate-ranking tracker (DEVIATIONS.md D11)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.tracking.xgboost_features import FEATURE_NAMES, candidate_bins, feature_matrix
from troika.types import StaticContext, WindowContext

torch = pytest.importorskip("torch")

from troika.tracking.mlp import CandidateMLP, MLPScorer  # noqa: E402

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


def _scorer() -> MLPScorer:
    torch.manual_seed(0)
    aux = np.random.default_rng(0).normal(size=(50, len(FEATURE_NAMES)))
    return MLPScorer.untrained(aux)


# --------------------------------------------------------------------- scorer


def test_mlp_returns_one_logit_per_row():
    assert CandidateMLP(aux_dim=3).eval()(torch.zeros(4, 3)).shape == (4,)


def test_scorer_gives_probabilities_and_survives_a_save_and_load(tmp_path):
    scorer = _scorer()
    s = spectrum_with({50: 3.0, 100: 1.0, 90: 2.0})
    candidates = candidate_bins(s, FS, N, 40.0, 200.0, 8)
    aux = feature_matrix(s, _ctx(), candidates, FS, N)
    scores = scorer.score(aux)
    assert scores.shape == (candidates.size,)
    assert ((scores > 0.0) & (scores < 1.0)).all()

    scorer.save(tmp_path / "model.pt")
    loaded = MLPScorer.load(tmp_path / "model.pt")
    assert loaded.score(aux) == pytest.approx(scores)


def test_standardise_clips_outliers_and_leaves_constant_features_finite():
    aux = np.zeros((10, len(FEATURE_NAMES)))
    aux[:, 0] = np.arange(10.0)
    scorer = MLPScorer.untrained(aux)
    far = np.full((1, len(FEATURE_NAMES)), 1e6)
    z = scorer.standardise(far)
    assert torch.isfinite(z).all()
    assert float(z.max()) == pytest.approx(5.0)


# -------------------------------------------------------------------- plug-in


@pytest.fixture
def tracker_factory(tmp_path):
    """Build the plug-in around an untrained MLP; only the plumbing is tested."""
    from troika.plugins.tracker.mlp import MLPTracker, MLPTrackerParams

    path = tmp_path / "model.pt"
    _scorer().save(path)

    def build(**params):
        static = StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)
        return MLPTracker(MLPTrackerParams(model_path=str(path), **params), static)

    return build


def test_plugin_missing_model_names_the_path(tmp_path):
    from troika.plugins.tracker.mlp import MLPTracker, MLPTrackerParams

    static = StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)
    with pytest.raises(FileNotFoundError, match="nowhere.pt"):
        MLPTracker(MLPTrackerParams(model_path=str(tmp_path / "nowhere.pt")), static)


@pytest.mark.parametrize("bad", [{"candidate_count": 0}, {"jump_patience": 0}, {"max_jump_bins": -1}, {"low_bpm": 200.0}])
def test_plugin_params_validate(bad):
    from troika.plugins.tracker.mlp import MLPTrackerParams

    with pytest.raises(ValueError):
        MLPTrackerParams(**bad)


def test_plugin_initialises_from_the_highest_peak_and_records_history(tracker_factory):
    tracker = tracker_factory()
    out = tracker.initialize(spectrum_with({50: 3.0, 95: 1.0}), WindowContext(idx=0, t_start_s=0.0))
    assert out.data.bin_cur == 50
    assert tracker.prev_bin == 50
    assert tracker.bpm_history == [pytest.approx(float(binmap.bin_to_bpm(50, FS, N)))]


def test_plugin_step_picks_its_top_scoring_candidate_without_the_guard(tracker_factory):
    tracker = tracker_factory(jump_patience=1)
    tracker.initialize(spectrum_with({50: 3.0}), WindowContext(idx=0, t_start_s=0.0))
    out = tracker.step(spectrum_with({95: 3.0, 51: 1.0}), _ctx(prev_bin=50))
    candidates, scores = out.diag["candidates"], out.diag["scores"]
    assert sorted(candidates.tolist()) == [51, 95]
    assert out.data.bin_cur == int(candidates[int(np.argmax(scores))])
    assert out.diag["jump_held"] is False


def test_plugin_is_deterministic(tracker_factory):
    picks = []
    for _ in range(2):
        tracker = tracker_factory()
        tracker.initialize(spectrum_with({50: 3.0}), WindowContext(idx=0, t_start_s=0.0))
        out = tracker.step(spectrum_with({95: 3.0, 51: 1.0}), _ctx(prev_bin=50))
        picks.append((out.data.bin_cur, out.diag["scores"].tolist()))
    assert picks[0] == picks[1]
