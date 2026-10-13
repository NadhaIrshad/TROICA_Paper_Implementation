"""8-bit simulation of the MLP scorer and its HMM tracker (DEVIATIONS.md D15)."""

from __future__ import annotations

import numpy as np
import pytest

from troika.tracking.xgboost_features import FEATURE_NAMES
from troika.types import StaticContext, WindowContext

torch = pytest.importorskip("torch")

from troika.tracking.mlp import MLPScorer  # noqa: E402
from troika.tracking.mlp_int8 import QuantisedMLP, quantise  # noqa: E402

FS = 125.0
N = 4096
HALF = N // 2 + 1
# Raw features on very different scales, as the real ones are.
AUX = np.random.default_rng(0).normal(size=(400, len(FEATURE_NAMES))) * np.logspace(-2, 3, len(FEATURE_NAMES))


def _scorer() -> MLPScorer:
    torch.manual_seed(0)
    return MLPScorer.untrained(AUX)


def spectrum_with(peaks: dict[int, float]) -> np.ndarray:
    s = np.zeros(HALF)
    for k, amplitude in peaks.items():
        s[int(k)] = float(amplitude)
    return s


# ------------------------------------------------------------------ the model


def test_weights_are_int8_and_biases_int32():
    model = quantise(_scorer(), AUX)
    assert model.w1.dtype == np.int8 and model.w2.dtype == np.int8
    assert model.b1.dtype == np.int32 and model.b2.dtype == np.int32
    assert model.w1.shape == (32, len(FEATURE_NAMES)) and model.w2.shape == (1, 32)
    assert model.n_bytes == 32 * len(FEATURE_NAMES) + 32 * 4 + 32 + 4
    assert np.abs(model.w1).max() == 127  # the largest weight uses the whole range


def test_scores_stay_close_to_the_float_model():
    scorer = _scorer()
    quantised = quantise(scorer, AUX).score(AUX)
    assert quantised.shape == (AUX.shape[0],)
    assert ((quantised >= 0.0) & (quantised <= 1.0)).all()
    assert np.abs(quantised - scorer.score(AUX)).max() < 0.05


def test_scores_are_multiples_of_one_256th():
    scores = quantise(_scorer(), AUX).score(AUX[:50])
    assert scores * 256 == pytest.approx(np.rint(scores * 256))


def test_model_survives_a_save_and_load(tmp_path):
    model = quantise(_scorer(), AUX)
    model.save(tmp_path / "model.npz")
    loaded = QuantisedMLP.load(tmp_path / "model.npz")
    assert loaded.w1.dtype == np.int8
    assert np.array_equal(loaded.score(AUX), model.score(AUX))


def test_quantising_needs_calibration_rows():
    with pytest.raises(ValueError, match="calibration_aux"):
        quantise(_scorer(), np.zeros((0, len(FEATURE_NAMES))))


# -------------------------------------------------------------------- plug-in


def test_plugin_params_fix_the_scorer_and_keep_the_parents_checks():
    from troika.plugins.tracker.hmm_int8 import HMMInt8TrackerParams

    assert HMMInt8TrackerParams().scorer == "mlp_int8"
    with pytest.raises(ValueError):
        HMMInt8TrackerParams(scorer="mlp")
    with pytest.raises(ValueError):
        HMMInt8TrackerParams(jump_prob=0.0)


def test_plugin_missing_model_names_the_path(tmp_path):
    from troika.plugins.tracker.hmm_int8 import HMMInt8Tracker, HMMInt8TrackerParams

    static = StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)
    with pytest.raises(FileNotFoundError, match="nowhere.npz"):
        HMMInt8Tracker(HMMInt8TrackerParams(model_path=str(tmp_path / "nowhere.npz")), static)


def test_plugin_tracks_with_the_quantised_scores(tmp_path):
    from troika.plugins.tracker.hmm_int8 import HMMInt8Tracker, HMMInt8TrackerParams

    path = tmp_path / "model.npz"
    quantise(_scorer(), AUX).save(path)
    static = StaticContext(fs=FS, N=N, M=1000, window_s=8.0, step_s=2.0)
    tracker = HMMInt8Tracker(HMMInt8TrackerParams(model_path=str(path)), static)

    tracker.initialize(spectrum_with({50: 3.0}), WindowContext(idx=0, t_start_s=0.0))
    assert tracker.prev_bin == 50
    ctx = WindowContext(
        idx=1, t_start_s=2.0, acc=np.zeros((3, 1000)), acc_bins_raw={44},
        prev_bin=50, bpm_history=[90.0, 91.0, 92.0],
    )
    out = tracker.step(spectrum_with({95: 3.0, 51: 1.0}), ctx)
    assert out.diag["scorer"] == "mlp_int8"
    assert sorted(out.diag["candidates"].tolist()) == [51, 95]
    assert ((out.diag["scores"] >= 0.0) & (out.diag["scores"] <= 1.0)).all()
    assert 40 <= out.data.bpm <= 200
