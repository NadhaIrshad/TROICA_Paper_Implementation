"""Loading, windowing and ground truth (Blueprint Sections 2, 5.7 and 9)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from troika.config import load_config
from troika.io import ground_truth, loader
from troika.preprocessing import windowing

from _helpers import needs_data

FS = 125.0


# --------------------------------------------------------------------- windows


def test_window_count_formula():
    """W = floor((n - T*fs) / (S*fs)) + 1 (Blueprint Section 2)."""
    assert windowing.n_windows(37500, FS, 8, 2) == 147
    assert windowing.n_windows(1000, FS, 8, 2) == 1
    assert windowing.n_windows(1249, FS, 8, 2) == 1
    assert windowing.n_windows(1250, FS, 8, 2) == 2
    assert windowing.n_windows(999, FS, 8, 2) == 0


def test_first_two_windows_match_the_dataset_readme():
    """Readme: estimate 1 uses samples 1-1000, estimate 2 uses 251-1250."""
    bounds = [(a, b) for _, a, b in windowing.iter_windows(2000, FS, 8, 2)]
    assert bounds[0] == (0, 1000)
    assert bounds[1] == (250, 1250)


def test_windows_overlap_by_six_seconds():
    """T = 8 s, S = 2 s means 75 % overlap (paper Section III)."""
    _, a0, b0 = next(iter(windowing.iter_windows(5000, FS, 8, 2)))
    _, a1, _ = list(windowing.iter_windows(5000, FS, 8, 2))[1]
    assert (b0 - a1) / FS == pytest.approx(6.0)


def test_iter_windows_is_consistent_with_n_windows_and_bounds():
    n = 37937
    items = list(windowing.iter_windows(n, FS, 8, 2))
    assert len(items) == windowing.n_windows(n, FS, 8, 2)
    for w, start, stop in items:
        assert (start, stop) == windowing.window_bounds(w, FS, 8, 2)
        assert stop <= n
    assert items[0][0] == 0 and items[-1][0] == len(items) - 1


def test_window_start_times():
    starts = windowing.window_starts_s(2000, FS, 8, 2)
    assert starts.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_zero_or_negative_step_raises():
    with pytest.raises(ValueError):
        windowing.n_windows(1000, FS, 8, 0)


# ------------------------------------------------------------------ synthetic


def _fake_recording(tmp_path, name="DATA_01_TYPE01.mat", n=2000, rows=6, with_gt=True):
    """Write a synthetic .mat pair in the real dataset's layout."""
    directory = tmp_path / "Training_data"
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    sig = rng.standard_normal((rows, n))
    savemat(directory / name, {"sig": sig})
    if with_gt:
        n_win = windowing.n_windows(n, FS, 8, 2)
        savemat(
            directory / name.replace(".mat", "_BPMtrace.mat"),
            {"BPM0": np.full((n_win, 1), 120.0)},
        )
    return directory / name, sig


def _cfg_for(tmp_path, **overrides):
    cfg = load_config().with_overrides({"data.dir": str(tmp_path), **overrides})
    return cfg


def test_load_recording_splits_rows_correctly(tmp_path):
    """Row 0 ECG, rows 1-2 PPG, rows 3-5 accelerometer (dataset readme)."""
    path, sig = _fake_recording(tmp_path)
    rec = loader.load_recording(path, 1, _cfg_for(tmp_path))
    assert np.array_equal(rec.ecg, sig[0])
    assert np.array_equal(rec.ppg, sig[1])
    assert np.array_equal(rec.ppg_all, sig[1:3])
    assert np.array_equal(rec.acc, sig[3:6])
    assert rec.acc.shape[0] == 3
    assert rec.protocol == "TYPE01"
    assert rec.subject_id == 1


def test_ppg_channel_selects_the_second_channel(tmp_path):
    """ASSUMPTION A15: ppg_channel indexes PPG channels, not rows of sig."""
    path, sig = _fake_recording(tmp_path)
    rec = loader.load_recording(path, 1, _cfg_for(tmp_path, **{"signal.ppg_channel": 1}))
    assert np.array_equal(rec.ppg, sig[2])


def test_out_of_range_ppg_channel_raises(tmp_path):
    path, _ = _fake_recording(tmp_path)
    with pytest.raises(loader.DataError, match="out of range"):
        loader.load_recording(path, 1, _cfg_for(tmp_path, **{"signal.ppg_channel": 5}))


def test_five_row_test_file_has_no_ecg(tmp_path):
    directory = tmp_path / "TestData"
    directory.mkdir(parents=True)
    savemat(directory / "TEST_S01_T01.mat", {"sig": np.zeros((5, 2000))})
    cfg = _cfg_for(tmp_path, **{"data.split": "test"})
    rec = loader.load_recording(directory / "TEST_S01_T01.mat", 1, cfg)
    assert rec.ecg is None
    assert rec.ppg_all.shape[0] == 2 and rec.acc.shape[0] == 3
    assert rec.protocol == "T01"


def test_unexpected_row_count_raises(tmp_path):
    path, _ = _fake_recording(tmp_path, rows=4)
    with pytest.raises(loader.DataError, match="expected 6 rows"):
        loader.load_recording(path, 1, _cfg_for(tmp_path))


def test_missing_ground_truth_gives_none(tmp_path):
    path, _ = _fake_recording(tmp_path, with_gt=False)
    rec = loader.load_recording(path, 1, _cfg_for(tmp_path))
    assert rec.bpm_gt is None


def test_ground_truth_length_mismatch_raises(tmp_path):
    path, _ = _fake_recording(tmp_path, with_gt=False)
    savemat(path.with_name("DATA_01_TYPE01_BPMtrace.mat"), {"BPM0": np.zeros((3, 1))})
    with pytest.raises(ValueError, match="window formula"):
        loader.load_recording(path, 1, _cfg_for(tmp_path))


def test_wrong_subject_count_fails_loudly(tmp_path):
    _fake_recording(tmp_path)
    with pytest.raises(loader.DataError, match="expected 12 training recordings"):
        loader.list_subjects(_cfg_for(tmp_path))


def test_missing_directory_raises(tmp_path):
    with pytest.raises(loader.DataError, match="dataset directory not found"):
        loader.list_subjects(_cfg_for(tmp_path / "nowhere"))


def test_bpmtrace_files_are_not_listed_as_recordings(tmp_path):
    for i in range(1, 13):
        _fake_recording(tmp_path, name=f"DATA_{i:02d}_TYPE02.mat")
    found = loader.list_subjects(_cfg_for(tmp_path))
    assert [s for s, _ in found] == list(range(1, 13))
    assert all("BPMtrace" not in p.name for _, p in found)


# ------------------------------------------------------- ground truth from ECG


def test_gt_from_ecg_recovers_a_synthetic_rate():
    """A beat every 50 samples at 125 Hz is exactly 150 BPM.

    The train starts a quarter-second in so the first window does not open on a
    half-formed beat, which the zero-phase band-pass would smear.
    """
    fs, seconds, period = 125.0, 30.0, 50
    ecg = np.zeros(int(fs * seconds))
    ecg[25::period] = 1.0
    out = ground_truth.gt_from_ecg(ecg, fs, 8, 2)
    assert np.isfinite(out).all()
    assert np.allclose(out, 150.0, atol=1.0)


def test_gt_from_ecg_marks_silent_windows_nan():
    out = ground_truth.gt_from_ecg(np.zeros(2000), 125.0, 8, 2)
    assert np.isnan(out).all()


def test_gt_from_ecg_is_polarity_independent():
    """Several recordings clip on one side, so polarity must not matter."""
    fs, n = 125.0, 3750
    ecg = np.zeros(n)
    ecg[:: int(fs / 2.0)] = 1.0
    assert np.allclose(
        ground_truth.gt_from_ecg(ecg, fs, 8, 2),
        ground_truth.gt_from_ecg(-ecg, fs, 8, 2),
        equal_nan=True,
    )


# ------------------------------------------------------------------ real data


@needs_data
def test_all_twelve_subjects_load(subject_paths, cfg):
    """M1 acceptance: the 12 recordings load (Blueprint Section 10)."""
    assert [s for s, _ in subject_paths] == list(range(1, 13))
    for subject_id, path in subject_paths:
        rec = loader.load_recording(path, subject_id, cfg)
        assert rec.fs == 125
        assert rec.ppg.ndim == 1 and rec.acc.shape[0] == 3
        assert rec.ecg is not None
        assert rec.ppg.shape[0] == rec.acc.shape[1] == rec.ecg.shape[0]
        assert np.isfinite(rec.ppg).all() and np.isfinite(rec.acc).all()
        assert 280 < rec.duration_s < 330


@needs_data
def test_ground_truth_length_matches_the_window_formula(subject_paths, cfg):
    """Confirmed for all 12 training files during dataset inspection."""
    for subject_id, path in subject_paths:
        rec = loader.load_recording(path, subject_id, cfg)
        expected = windowing.n_windows(rec.n_samples, rec.fs, 8, 2)
        assert rec.bpm_gt is not None
        assert rec.bpm_gt.shape == (expected,)
        assert np.isfinite(rec.bpm_gt).all()
        assert 40 < rec.bpm_gt.min() and rec.bpm_gt.max() < 220


@needs_data
def test_subject_one_is_type01_and_the_rest_type02(subject_paths, cfg):
    """DEVIATION D2: the speed schedule differs by recording type."""
    protocols = {
        s: loader.load_recording(p, s, cfg).protocol for s, p in subject_paths
    }
    assert protocols[1] == "TYPE01"
    assert all(v == "TYPE02" for k, v in protocols.items() if k != 1)


@needs_data
def test_shipped_ground_truth_agrees_with_the_ecg(subject5):
    """Blueprint Section 5.7 cross-check: the two sources agree within ~1 BPM."""
    derived = ground_truth.gt_from_ecg(subject5.ecg, subject5.fs, 8, 2)
    agreement = ground_truth.compare_ground_truth(subject5.bpm_gt, derived)
    assert agreement["mean_abs_diff"] < 1.0
    assert agreement["pearson_r"] > 0.99
