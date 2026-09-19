"""Error metrics and the statistics table (Blueprint Sections 5.8 and 9)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from troika.evaluation import metrics
from troika.types import RunResult


def _run(subject_id, est, gt):
    return RunResult(
        subject_id=subject_id,
        bpm_est=np.asarray(est, dtype=float),
        bpm_gt=np.asarray(gt, dtype=float),
    )


def test_error1_hand_computed():
    """Paper Eq. 19: mean absolute error."""
    est = [100.0, 110.0, 90.0]
    gt = [102.0, 106.0, 93.0]
    assert metrics.error1(est, gt) == pytest.approx((2 + 4 + 3) / 3)


def test_error2_hand_computed():
    """Paper Eq. 20: mean absolute error relative to ground truth, as a fraction."""
    est = [110.0, 90.0]
    gt = [100.0, 100.0]
    assert metrics.error2(est, gt) == pytest.approx(0.10)


def test_error1_is_zero_for_perfect_estimates():
    gt = np.linspace(70, 170, 40)
    assert metrics.error1(gt, gt) == 0.0
    assert metrics.error2(gt, gt) == 0.0


def test_metrics_ignore_nan_windows():
    est = np.array([100.0, np.nan, 120.0])
    gt = np.array([100.0, 110.0, 118.0])
    assert metrics.error1(est, gt) == pytest.approx(1.0)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        metrics.error1([1.0, 2.0], [1.0])


def test_bland_altman_algebra():
    """LOA = [mu - 1.96 sigma, mu + 1.96 sigma] over the pooled differences."""
    est = np.array([100.0, 104.0, 98.0, 102.0])
    gt = np.array([100.0, 100.0, 100.0, 100.0])
    out = metrics.bland_altman(est, gt)
    diff = est - gt
    assert out["mean_diff"] == pytest.approx(diff.mean())
    assert out["sd"] == pytest.approx(diff.std())
    assert out["loa_low"] == pytest.approx(diff.mean() - 1.96 * diff.std())
    assert out["loa_high"] == pytest.approx(diff.mean() + 1.96 * diff.std())
    assert out["n"] == 4


def test_bland_altman_sign_convention_is_est_minus_gt():
    """Blueprint Section 5.8: a low estimate gives a negative mean difference."""
    out = metrics.bland_altman([95.0, 96.0], [100.0, 100.0])
    assert out["mean_diff"] < 0


def test_pearson_perfect_and_anticorrelated():
    x = np.linspace(60, 180, 30)
    assert metrics.pearson(x, x) == pytest.approx(1.0)
    assert metrics.pearson(-x, x) == pytest.approx(-1.0)


def test_pearson_constant_series_is_nan():
    assert np.isnan(metrics.pearson(np.ones(10), np.linspace(60, 70, 10)))


def test_paper_reference_table_matches_the_paper():
    """Blueprint Section 8.1 targets, loaded from tests/reference."""
    reference = metrics.paper_reference("full")
    assert len(reference) == 12
    assert reference.loc[1, "Error1_paper"] == pytest.approx(2.29)
    assert reference.loc[10, "Error1_paper"] == pytest.approx(4.70)
    assert reference.loc[6, "Error2_paper"] == pytest.approx(2.25)


@pytest.mark.parametrize("variant", ["full", "no_ssa", "fft", "no_verification"])
def test_paper_reference_has_all_four_table_rows(variant):
    assert len(metrics.paper_reference(variant)) == 12


def test_paper_reference_unknown_variant_raises():
    with pytest.raises(ValueError, match="unknown variant"):
        metrics.paper_reference("nope")


def test_table_one_row_one_reproduces_the_papers_aggregate():
    """Feeding Table I row 1 to the aggregator must give 2.34 +/- 0.82 (ddof=1).

    This is the Blueprint Section 10 acceptance criterion for M1.
    """
    reference = metrics.paper_reference("full")
    runs = []
    for subject, row in reference.iterrows():
        # A two-window run whose mean absolute error is exactly the paper value,
        # with ground truth chosen so the percentage error matches too.
        gt = 100.0 * row["Error1_paper"] / row["Error2_paper"]
        runs.append(_run(subject, [gt + row["Error1_paper"], gt - row["Error1_paper"]], [gt, gt]))

    stats = metrics.summarize(runs, std_ddof=1)
    mean_row = stats.loc["mean"]
    assert mean_row["Error1_BPM"] == pytest.approx(2.34, abs=0.005)
    assert stats.loc["std", "Error1_BPM"] == pytest.approx(0.82, abs=0.01)
    assert mean_row["Error2_pct"] == pytest.approx(1.80, abs=0.01)


def test_ddof_one_is_the_better_match_for_the_papers_std():
    """Which ddof reproduces the paper's 2.34 +/- 0.82.

    Blueprint Section 5.8 states that ddof=1 gives exactly 0.82. Recomputed from
    the published Table I values it gives 0.8265, which displays as 0.83; ddof=0
    gives 0.7913. So ddof=1 is the right choice, four times closer to the
    published figure, but it does not reproduce it exactly. The residual gap is
    consistent with the paper averaging unrounded per-subject errors.
    """
    values = metrics.paper_reference("full")["Error1_paper"].to_numpy()
    assert values.mean() == pytest.approx(2.34, abs=0.005)
    assert values.std(ddof=1) == pytest.approx(0.8265, abs=0.001)
    assert values.std(ddof=0) == pytest.approx(0.7913, abs=0.001)
    assert abs(values.std(ddof=1) - 0.82) < abs(values.std(ddof=0) - 0.82)


def test_per_subject_stats_columns():
    stats = metrics.per_subject_stats([100.0, 110.0], [100.0, 100.0], within_bpm=5.0)
    assert stats["Error1_BPM"] == pytest.approx(5.0)
    assert stats["bias_BPM"] == pytest.approx(5.0)
    assert stats["rmse_BPM"] == pytest.approx(np.sqrt(50.0))
    assert stats["median_AE_BPM"] == pytest.approx(5.0)
    assert stats["max_AE_BPM"] == pytest.approx(10.0)
    assert stats["within_5_pct"] == pytest.approx(50.0)
    assert stats["n_windows"] == 2


def test_summarize_has_per_subject_and_summary_rows():
    runs = [_run(5, [100.0, 101.0], [100.0, 100.0]), _run(6, [98.0, 99.0], [100.0, 100.0])]
    stats = metrics.summarize(runs, std_ddof=1, compare_with_paper=False)
    assert list(stats.index) == [5, 6, "mean", "std", "Pooled"]
    assert stats.loc["Pooled", "n_windows"] == 4
    assert isinstance(stats, pd.DataFrame)


def test_summarize_subset_labels_the_mean_row():
    """Lab Spec Section 4.1: a subset run labels its mean row with the subjects."""
    runs = [_run(i, [100.0], [100.0]) for i in (1, 5, 6)]
    stats = metrics.summarize(runs, subject_ids=[5, 6], compare_with_paper=False)
    assert "mean over subjects 5, 6" in stats.index
    assert list(stats.index)[:2] == [5, 6]


def test_summarize_adds_paper_delta_columns():
    runs = [_run(i, [100.0], [100.0]) for i in range(1, 13)]
    stats = metrics.summarize(runs, compare_with_paper=True)
    assert "Error1_paper" in stats.columns
    assert "Error1_delta" in stats.columns
    # A perfect estimator beats every paper value, so all deltas are negative.
    assert (stats.loc[1:12, "Error1_delta"] < 0).all()


def test_summarize_pooled_matches_direct_pooled_computation():
    runs = [
        _run(1, [100.0, 105.0, 95.0], [100.0, 100.0, 100.0]),
        _run(2, [110.0, 90.0], [100.0, 100.0]),
    ]
    stats = metrics.summarize(runs, compare_with_paper=False)
    est = np.array([100.0, 105.0, 95.0, 110.0, 90.0])
    gt = np.full(5, 100.0)
    assert stats.loc["Pooled", "Error1_BPM"] == pytest.approx(metrics.error1(est, gt))
    assert stats.loc["Pooled", "pearson_r"] == pytest.approx(metrics.pearson(est, gt), nan_ok=True)
    assert stats.loc["Pooled", "BA_sd_BPM"] == pytest.approx(metrics.bland_altman(est, gt)["sd"])


def test_summarize_without_ground_truth_raises():
    run = RunResult(subject_id=1, bpm_est=np.array([100.0]), bpm_gt=None)
    with pytest.raises(ValueError, match="no ground truth"):
        metrics.summarize([run])
