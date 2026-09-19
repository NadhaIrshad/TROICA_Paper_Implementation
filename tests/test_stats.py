"""The statistics table (Lab Spec Sections 4.1 and 9, `test_stats`)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from troika import lab
from troika.config import load_config
from troika.evaluation import metrics
from troika.evaluation.report import per_window_frame, summary_dict, write_run
from troika.types import RunResult, WindowResult


def _run(subject_id: int, est, gt, *, contaminated: bool = False) -> RunResult:
    est = np.asarray(est, dtype=float)
    gt = np.asarray(gt, dtype=float)
    windows = [
        WindowResult(
            idx=i,
            t_start_s=2.0 * i,
            bin_cur=int(i),
            bpm_est=float(v),
            case=1 if i else None,
            f_acc=[10, 20],
            n_groups=5,
            n_groups_removed=1,
            timings_ms={"decomposition": 1.0},
        )
        for i, v in enumerate(est)
    ]
    return RunResult(
        subject_id=subject_id,
        bpm_est=est,
        bpm_gt=gt,
        windows=windows,
        contaminated=contaminated,
    )


@pytest.fixture
def cfg():
    return load_config()


@pytest.fixture
def runs():
    """Three short runs whose ground truth varies, so correlation is defined."""
    return {
        1: _run(1, [100.0, 115.0, 125.0], [100.0, 110.0, 130.0]),
        5: _run(5, [110.0, 140.0], [105.0, 145.0]),
        6: _run(6, [90.0, 95.0], [90.0, 100.0]),
    }


# ------------------------------------------------------------------- columns


def test_table_has_every_column_the_spec_lists(runs, cfg):
    stats = lab.evaluate(runs, cfg)
    for column in (
        "Error1_BPM",
        "Error2_pct",
        "bias_BPM",
        "rmse_BPM",
        "median_AE_BPM",
        "max_AE_BPM",
        "within_5_pct",
        "n_windows",
    ):
        assert column in stats.columns, column


def test_bias_and_rmse_hand_checked():
    """bias is the mean signed error; rmse is the root mean square."""
    stats = metrics.per_subject_stats([104.0, 98.0], [100.0, 100.0])
    assert stats["bias_BPM"] == pytest.approx(1.0)
    assert stats["rmse_BPM"] == pytest.approx(np.sqrt((16 + 4) / 2))
    assert stats["Error1_BPM"] == pytest.approx(3.0)
    assert stats["median_AE_BPM"] == pytest.approx(3.0)
    assert stats["max_AE_BPM"] == pytest.approx(4.0)


def test_within_threshold_is_a_percentage():
    stats = metrics.per_subject_stats(
        [100.0, 103.0, 120.0, 90.0], [100.0] * 4, within_bpm=5.0
    )
    assert stats["within_5_pct"] == pytest.approx(50.0)


def test_within_threshold_column_follows_the_config(runs, cfg):
    stats = lab.evaluate(runs, cfg.with_overrides({"evaluation.within_bpm": 2}))
    assert "within_2_pct" in stats.columns


# ------------------------------------------------------------- summary rows


def test_rows_are_subjects_then_summaries(runs, cfg):
    stats = lab.evaluate(runs, cfg)
    assert list(stats.index) == [1, 5, 6, "mean", "std", "Pooled"]


def test_subset_labels_the_mean_row_with_the_subjects(runs, cfg):
    stats = lab.evaluate(runs, cfg, subject_ids=[5, 6])
    assert "mean over subjects 5, 6" in stats.index
    assert [i for i in stats.index if isinstance(i, (int, np.integer))] == [5, 6]


def test_subset_mean_differs_from_the_all_subject_mean(runs, cfg):
    every = lab.evaluate(runs, cfg)
    subset = lab.evaluate(runs, cfg, subject_ids=[5, 6])
    assert every.loc["mean", "Error1_BPM"] != subset.loc[
        "mean over subjects 5, 6", "Error1_BPM"
    ]


def test_pooled_row_counts_every_window(runs, cfg):
    stats = lab.evaluate(runs, cfg)
    assert stats.loc["Pooled", "n_windows"] == sum(r.n_windows for r in runs.values())


def test_pooled_row_carries_agreement_statistics(runs, cfg):
    stats = lab.evaluate(runs, cfg)
    for column in ("BA_mean_BPM", "BA_sd_BPM", "BA_loa_low", "BA_loa_high", "pearson_r"):
        assert column in stats.columns
        assert np.isfinite(stats.loc["Pooled", column])
        assert pd.isna(stats.loc[1, column])


def test_std_row_uses_the_configured_ddof(runs, cfg):
    values = np.array(
        [metrics.error1(r.bpm_est, r.bpm_gt) for r in runs.values()], dtype=float
    )
    sample = lab.evaluate(runs, cfg.with_overrides({"evaluation.std_ddof": 1}))
    population = lab.evaluate(runs, cfg.with_overrides({"evaluation.std_ddof": 0}))
    assert sample.loc["std", "Error1_BPM"] == pytest.approx(values.std(ddof=1))
    assert population.loc["std", "Error1_BPM"] == pytest.approx(values.std(ddof=0))


# ------------------------------------------------------------ paper columns


def test_paper_columns_appear_for_real_subject_numbers(cfg):
    runs = {i: _run(i, [100.0], [100.0]) for i in range(1, 13)}
    stats = lab.evaluate(runs, cfg)
    assert "Error1_paper" in stats.columns
    assert "Error1_delta" in stats.columns
    assert stats.loc[1, "Error1_paper"] == pytest.approx(2.29)
    assert stats.loc[1, "Error1_delta"] == pytest.approx(0.0 - 2.29)


def test_paper_columns_can_be_switched_off(cfg):
    runs = {i: _run(i, [100.0], [100.0]) for i in range(1, 13)}
    stats = lab.evaluate(runs, cfg.with_overrides({"evaluation.compare_with_paper": False}))
    assert "Error1_paper" not in stats.columns


def test_paper_columns_are_skipped_for_unknown_subjects(runs, cfg):
    """Subject 99 is not in the paper, so no comparison can be made."""
    stats = lab.evaluate({99: _run(99, [100.0], [100.0])}, cfg)
    assert "Error1_paper" not in stats.columns


@pytest.mark.parametrize("variant", ["full", "no_ssa", "fft", "no_verification"])
def test_paper_variant_selects_the_right_row(cfg, variant):
    runs = {i: _run(i, [100.0], [100.0]) for i in range(1, 13)}
    stats = lab.evaluate(runs, cfg, paper_variant=variant)
    expected = metrics.paper_reference(variant)
    assert stats.loc[6, "Error1_paper"] == pytest.approx(expected.loc[6, "Error1_paper"])


# --------------------------------------------------------------- leak guard


def test_contaminated_runs_are_refused(cfg):
    runs = {1: _run(1, [100.0], [100.0], contaminated=True)}
    with pytest.raises(lab.ContaminatedRunError, match="ground-truth"):
        lab.evaluate(runs, cfg)


def test_contaminated_runs_report_with_a_banner(cfg, capsys):
    runs = {1: _run(1, [100.0], [100.0], contaminated=True)}
    stats = lab.evaluate(runs, cfg, allow_contaminated=True)
    assert "CONTAMINATED" in capsys.readouterr().out
    assert not stats.empty


def test_one_contaminated_run_taints_the_whole_call(cfg):
    runs = {
        1: _run(1, [100.0], [100.0]),
        2: _run(2, [100.0], [100.0], contaminated=True),
    }
    with pytest.raises(lab.ContaminatedRunError):
        lab.evaluate(runs, cfg)


# ----------------------------------------------------------------- reporting


def test_per_window_frame_has_one_row_per_window(runs):
    frame = per_window_frame(runs)
    assert len(frame) == sum(r.n_windows for r in runs.values())
    for column in ("subject", "window", "bpm_est", "bpm_gt", "error_BPM", "case"):
        assert column in frame.columns


def test_per_window_error_matches_the_estimates(runs):
    frame = per_window_frame(runs)
    subject = frame[frame["subject"] == 1]
    np.testing.assert_allclose(
        subject["error_BPM"].to_numpy(), runs[1].bpm_est - runs[1].bpm_gt
    )


def test_write_run_produces_the_expected_files(runs, cfg, tmp_path):
    stats = lab.evaluate(runs, cfg)
    out = write_run(cfg, runs, stats, out_dir=tmp_path / "run")
    for name in (
        "config_resolved.yaml",
        "config_hash.txt",
        "per_window.csv",
        "per_subject.csv",
        "summary.json",
    ):
        assert (out / name).exists(), name

    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["config_hash"] == cfg.hash()
    assert payload["subjects"] == sorted(runs)
    assert payload["paper"]["Error1_BPM"] == 2.34
    assert payload["contaminated"] is False


def test_summary_dict_reports_contamination(runs, cfg):
    runs = {**runs, 7: _run(7, [100.0], [100.0], contaminated=True)}
    stats = lab.evaluate(runs, cfg, allow_contaminated=True)
    assert summary_dict(cfg, stats, runs)["contaminated"] is True
