"""EEMD + spectral subtraction as the first two stages, end to end (DEVIATIONS.md D13, D14)."""

from __future__ import annotations

import numpy as np
import pytest

from test_pipeline_synthetic import synthetic_recording
from troika.config import load_config, repo_root
from troika.pipeline import TroikaEstimator

CONFIG = repo_root() / "configs" / "eemd_ss.yaml"


def test_config_selects_the_two_new_stages_and_leaves_the_rest():
    cfg = load_config(CONFIG)
    default = load_config()
    assert cfg.slot_method("decomposition") == "eemd"
    assert cfg.slot_method("temporal_diff") == "none"
    assert cfg.slot_method("spectrum_estimator") == "spectral_subtraction"
    assert cfg.slot_method("bandpass") == default.slot_method("bandpass")
    assert cfg.slot_method("tracker") == default.slot_method("tracker")
    assert cfg.slot_params("decomposition").ensemble_size == 100
    assert cfg.slot_params("spectrum_estimator").harmonic_weights == [1.0, 0.66, 0.33]
    # The default configuration still runs the paper's stages.
    assert default.slot_method("decomposition") == "ssa"
    assert default.slot_method("spectrum_estimator") == "focuss"


@pytest.fixture(scope="module")
def run_and_recording():
    # A small ensemble keeps the test fast; the heart rate moves from 110 to 130 BPM
    # under a swing artifact five times its size that the accelerometer also carries.
    cfg = load_config(CONFIG, overrides={"decomposition.ensemble_size": 4})
    recording = synthetic_recording(seconds=24.0, hr_start=110.0, hr_end=130.0)
    return TroikaEstimator(cfg).run(recording), recording


def test_pipeline_runs_and_gives_plausible_estimates(run_and_recording):
    run, _ = run_and_recording
    assert np.isfinite(run.bpm_est).all()
    assert (run.bpm_est > 30).all() and (run.bpm_est < 230).all()


def test_pipeline_finds_the_heart_rate_under_the_artifact(run_and_recording):
    run, recording = run_and_recording
    error = np.abs(run.bpm_est - recording.bpm_gt[: run.bpm_est.size])
    assert float(np.median(error)) < 5.0
