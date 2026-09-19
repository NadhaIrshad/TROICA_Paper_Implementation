"""End-to-end pipeline on synthetic data (Blueprint Section 9, Lab Spec Section 9)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import lab
from troika.config import load_config
from troika.evaluation import metrics
from troika.pipeline import TroikaEstimator
from troika.types import Recording

FS = 125.0


def synthetic_recording(
    seconds: float = 60.0,
    hr_start: float = 80.0,
    hr_end: float = 160.0,
    swing_hz: float = 2.9,
    noise: float = 0.05,
    seed: int = 20150203,
    subject_id: int = 99,
) -> Recording:
    """A heart-rate ramp buried under a rhythmic swing artifact.

    The artifact is five times the pulse amplitude, which is the regime the paper
    describes, and the accelerometer carries it and nothing else, so motion
    removal has something honest to work with.

    The default artifact sits at 2.9 Hz, above the 1.33-2.67 Hz span of the
    heart-rate ramp, so the two never coincide. Artifacts that do cross the heart
    rate are a documented weakness of the method rather than of this
    implementation; see ``test_artifact_crossing_the_heart_rate_is_a_known_weakness``.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS

    hr_hz = np.linspace(hr_start, hr_end, n) / 60.0
    hr_phase = 2 * np.pi * np.cumsum(hr_hz) / FS
    pulse = np.sin(hr_phase) + 0.4 * np.sin(2 * hr_phase)

    swing_phase = 2 * np.pi * swing_hz * t
    swing = np.sin(swing_phase) + 0.5 * np.sin(2 * swing_phase)

    ppg = pulse + 5.0 * swing + noise * rng.standard_normal(n)
    acc = np.stack(
        [
            swing + 0.02 * rng.standard_normal(n),
            0.6 * swing + 0.02 * rng.standard_normal(n),
            0.3 * np.sin(swing_phase + 0.7) + 0.02 * rng.standard_normal(n),
        ]
    )

    from troika.preprocessing.windowing import iter_windows

    gt = np.array(
        [np.mean(hr_hz[a:b]) * 60.0 for _, a, b in iter_windows(n, FS, 8, 2)]
    )
    return Recording(
        subject_id=subject_id,
        fs=FS,
        ppg=ppg,
        acc=acc,
        ecg=None,
        bpm_gt=gt,
        ppg_all=ppg[None, :],
        protocol="TYPE02",
        path=None,
    )


@pytest.fixture(scope="module")
def recording():
    return synthetic_recording()


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def run(cfg, recording):
    return TroikaEstimator(cfg).run(recording)


# ------------------------------------------------------------------ the run


def test_run_produces_one_estimate_per_window(run, recording):
    from troika.preprocessing.windowing import n_windows

    expected = n_windows(recording.n_samples, FS, 8, 2)
    assert run.n_windows == expected
    assert len(run.windows) == expected
    assert run.bpm_gt is not None and run.bpm_gt.shape == (expected,)


def test_estimates_are_finite_and_plausible(run):
    assert np.isfinite(run.bpm_est).all()
    assert (run.bpm_est > 30).all() and (run.bpm_est < 230).all()


def test_mean_absolute_error_is_small(run):
    """Blueprint Section 9 asks for under 5 BPM on a synthetic ramp."""
    assert metrics.error1(run.bpm_est, run.bpm_gt) < 5.0


def test_artifact_crossing_the_heart_rate_is_a_known_weakness():
    """A swing that crosses the heart rate makes the estimate lock onto it.

    Paper Section III-A introduces the +/-Delta exclusion precisely so that a
    cadence coinciding with the heart rate is not removed. The exclusion is
    centred on the *previous estimate*, so once the estimate sits on the cadence
    the cadence is protected from removal, which keeps the estimate there. The
    trap is in the method as published, and it is what makes subjects 6 and 10
    fail here; it is recorded so a future change is measured against it rather
    than discovered again.
    """
    crossing = synthetic_recording(swing_hz=2.2)  # 132 BPM, inside the 80-160 ramp
    clear = synthetic_recording(swing_hz=2.9)  # 174 BPM, above it
    cfg = load_config()
    crossing_error = metrics.error1(
        *_estimates(cfg, crossing)
    )
    clear_error = metrics.error1(*_estimates(cfg, clear))
    assert clear_error < 5.0
    assert crossing_error > clear_error


def _estimates(cfg, rec):
    out = TroikaEstimator(cfg).run(rec)
    return out.bpm_est, out.bpm_gt


def test_estimates_follow_the_ramp(run):
    """The estimate must rise with the ground truth, not sit on the artifact."""
    assert metrics.pearson(run.bpm_est, run.bpm_gt) > 0.9


def test_windows_carry_diagnostics(run):
    for window in run.windows:
        assert window.bin_cur >= 0
        assert window.case in (None, 1, 2, 3)
        assert isinstance(window.f_acc, list)
        assert set(window.timings_ms) >= {"decomposition", "spectrum_estimator", "tracker"}
    assert run.windows[0].case is None
    assert all(w.case is not None for w in run.windows[1:])


def test_motion_removal_actually_removes_something(run):
    """The accelerometer carries the swing, so some components must be dropped."""
    assert sum(w.n_groups_removed for w in run.windows) > 0


def test_run_is_deterministic(cfg, recording):
    a = TroikaEstimator(cfg).run(recording)
    b = TroikaEstimator(cfg).run(recording)
    assert np.array_equal(a.bpm_est, b.bpm_est)


def test_run_does_not_mutate_the_recording(cfg, recording):
    before = recording.ppg.copy()
    TroikaEstimator(cfg).run(recording)
    assert np.array_equal(recording.ppg, before)


def test_estimator_instance_can_be_reused(cfg, recording):
    """run() rebuilds the stages, so a second call is not polluted by the first."""
    estimator = TroikaEstimator(cfg)
    first = estimator.run(recording)
    second = estimator.run(recording)
    assert np.array_equal(first.bpm_est, second.bpm_est)


# -------------------------------------------------------------- ablations


@pytest.mark.parametrize(
    "overrides,label",
    [
        ({"decomposition.method": "none"}, "without SSA"),
        ({"spectrum_estimator.method": "fft"}, "FFT instead of SSR"),
        ({"tracker.params.verification.enabled": False}, "without verification"),
    ],
)
def test_ablation_flags_change_behaviour(cfg, recording, overrides, label):
    """Blueprint Section 9: the ablation switches must actually do something.

    Removing verification is the one case where identical output is the correct
    answer: on a clean recording neither rule ever fires, so switching them off
    cannot change anything. The test asserts the honest condition, that the
    estimates differ unless the rules never fired in the first place.
    """
    baseline = TroikaEstimator(cfg).run(recording)
    ablated = TroikaEstimator(cfg.with_overrides(overrides)).run(recording)
    assert np.isfinite(ablated.bpm_est).all()

    rules_fired = any(w.rule1_fired or w.rule2_fired for w in baseline.windows)
    if rules_fired or "verification" not in str(overrides):
        assert not np.array_equal(baseline.bpm_est, ablated.bpm_est), label


def test_removing_verification_matters_when_the_rules_fire():
    """On a recording where the rules do fire, the ablation must change the result."""
    cfg = load_config()
    crossing = synthetic_recording(swing_hz=2.2)
    baseline = TroikaEstimator(cfg).run(crossing)
    assert any(w.rule1_fired or w.rule2_fired for w in baseline.windows)

    ablated = TroikaEstimator(
        cfg.with_overrides({"tracker.params.verification.enabled": False})
    ).run(crossing)
    assert not np.array_equal(baseline.bpm_est, ablated.bpm_est)


def test_without_decomposition_nothing_is_removed(cfg, recording):
    ablated = TroikaEstimator(cfg.with_overrides({"decomposition.method": "none"})).run(
        recording
    )
    assert sum(w.n_groups_removed for w in ablated.windows) == 0


def test_global_bandpass_mode_runs(cfg, recording):
    """ASSUMPTION A2's alternative filters the whole recording once."""
    out = TroikaEstimator(cfg.with_overrides({"bandpass.mode": "global"})).run(recording)
    assert np.isfinite(out.bpm_est).all()


def test_mismatched_ssr_band_is_rejected(cfg):
    """Eq. 12 prunes to the band-pass band, so the two must agree."""
    bad = cfg.with_overrides({"spectrum_estimator.low_hz": 0.8})
    with pytest.raises(ValueError, match="must match"):
        TroikaEstimator(bad)


# ------------------------------------------------------------------ tracing


def test_trace_none_stores_nothing(cfg, recording):
    assert TroikaEstimator(cfg).run(recording, trace="none").trace is None


def test_light_trace_holds_every_stage(cfg, recording):
    run = TroikaEstimator(cfg).run(recording, trace="light")
    assert run.trace is not None and len(run.trace) == run.n_windows
    window = run.trace.windows[0]
    assert set(window.stages) == {
        "raw",
        "bandpass",
        "decomposition",
        "temporal_diff",
        "spectrum_estimator",
    }
    for record in window.stages.values():
        assert record.signal.dtype == np.float32
        assert record.spectrum.dtype == np.float32


def test_light_trace_spectra_stop_at_seven_hz(cfg, recording):
    run = TroikaEstimator(cfg).run(recording, trace="light")
    from troika import binmap

    expected = binmap.hz_to_bin(7.0, FS, int(cfg.grid.n_fft)) + 1
    assert run.trace.windows[0].stages["bandpass"].spectrum.size == expected


def test_light_trace_stays_inside_the_size_budget(cfg):
    """Lab Spec Section 2.4 targets under 5 MB per subject at the light level."""
    long_recording = synthetic_recording(seconds=300.0)
    run = TroikaEstimator(cfg).run(long_recording, trace="light")
    megabytes = run.trace.nbytes() / 1e6
    assert megabytes < 5.0, f"light trace is {megabytes:.1f} MB"


def test_light_trace_holds_no_full_payload(cfg, recording):
    run = TroikaEstimator(cfg).run(recording, trace="light")
    assert run.trace.full_windows == []
    assert all(not w.is_full for w in run.trace.windows)


def test_full_windows_carry_the_section_two_three_keys(cfg, recording):
    """Lab Spec Section 2.3: paper-default plug-ins fill every listed key."""
    run = TroikaEstimator(cfg).run(recording, trace="light", full_windows=[3])
    window = run.trace[3]
    assert window.is_full
    assert {"bandpass", "decomposition", "temporal_diff", "spectrum_estimator", "tracker"} <= set(
        window.diag
    )
    assert {"filter_response", "description"} <= set(window.diag["bandpass"])
    assert {"components", "removed_mask", "U", "s", "Vt", "groups", "L", "K"} <= set(
        window.diag["decomposition"]
    )
    assert {"pre_normalization", "order"} <= set(window.diag["temporal_diff"])
    assert {"iterates", "kept_bins", "sparsity_per_iter"} <= set(
        window.diag["spectrum_estimator"]
    )
    assert {"R0", "R1", "eta", "P0", "P1", "k_b", "case", "k_cur"} <= set(
        window.diag["tracker"]
    )
    assert {"ppg", "acc", "spectrum"} <= set(window.raw)


def test_trace_lookup_is_by_window_index(cfg, recording):
    run = TroikaEstimator(cfg).run(recording, trace="light", full_windows=[5])
    assert run.trace[5].idx == 5
    with pytest.raises(KeyError, match="not in this trace"):
        run.trace[10_000]


def test_require_reports_a_missing_stage_clearly(cfg, recording):
    run = TroikaEstimator(cfg).run(recording, trace="light")
    with pytest.raises(KeyError, match="full trace"):
        run.trace.windows[0].require("decomposition", ["U"], "plot_ssa_svd")


def test_require_reports_missing_keys_clearly(cfg, recording):
    run = TroikaEstimator(cfg).run(
        recording, trace="light", full_windows=[2]
    )
    with pytest.raises(KeyError, match="generic plot"):
        run.trace[2].require("decomposition", ["not_a_key"], "plot_ssa_svd")


def test_stage_matrix_stacks_the_light_spectra(cfg, recording):
    run = TroikaEstimator(cfg).run(recording, trace="light")
    matrix = run.trace.stage_matrix("spectrum_estimator")
    assert matrix.shape[0] == run.n_windows
    with pytest.raises(KeyError, match="no window in this trace"):
        run.trace.stage_matrix("nonsense")


# --------------------------------------------------------------- leak guard


def test_plugins_do_not_see_ground_truth_by_default(cfg, recording):
    """Lab Spec Section 2.4: ctx.gt_bpm is None unless explicitly allowed."""
    seen = []

    from dataclasses import dataclass

    @dataclass
    class SpyParams:
        pass

    class Spy:
        Params = SpyParams

        def __init__(self, params, static):
            pass

        def __call__(self, x, ctx):
            seen.append(ctx.gt_bpm)
            return x

    from troika import registry

    saved = dict(registry._REGISTRY["temporal_diff"])
    try:
        registry.register("temporal_diff", "spy", override=True)(Spy)
        TroikaEstimator(cfg.with_overrides({"temporal_diff.method": "spy"})).run(recording)
        assert seen and all(value is None for value in seen)
    finally:
        registry._REGISTRY["temporal_diff"].clear()
        registry._REGISTRY["temporal_diff"].update(saved)


def test_default_run_is_not_contaminated(run):
    assert run.contaminated is False


def test_ground_truth_initialisation_marks_the_run_contaminated(cfg, recording):
    contaminated = cfg.with_overrides(
        {"tracker.init_mode": "ground_truth", "run.allow_ground_truth_access": True}
    )
    out = TroikaEstimator(contaminated).run(recording)
    assert out.contaminated is True


def test_evaluate_refuses_a_contaminated_run(cfg, recording):
    contaminated = cfg.with_overrides(
        {"tracker.init_mode": "ground_truth", "run.allow_ground_truth_access": True}
    )
    out = TroikaEstimator(contaminated).run(recording)
    with pytest.raises(lab.ContaminatedRunError, match="ground-truth"):
        lab.evaluate({out.subject_id: out}, cfg)


def test_evaluate_allows_contamination_when_asked(cfg, recording, capsys):
    contaminated = cfg.with_overrides(
        {"tracker.init_mode": "ground_truth", "run.allow_ground_truth_access": True}
    )
    out = TroikaEstimator(contaminated).run(recording)
    stats = lab.evaluate(
        {out.subject_id: out}, cfg, allow_contaminated=True, paper_variant="full"
    )
    assert "CONTAMINATED" in capsys.readouterr().out
    assert not stats.empty


# ----------------------------------------------------------------- lab API


def test_run_subject_and_evaluate(cfg, recording):
    out = lab.run_subject(cfg, recording)
    stats = lab.evaluate({recording.subject_id: out}, cfg, paper_variant="full")
    assert "Error1_BPM" in stats.columns
    assert recording.subject_id in stats.index


def test_run_all_over_several_recordings(cfg):
    subjects = {
        1: synthetic_recording(seconds=30.0, subject_id=1),
        2: synthetic_recording(seconds=30.0, subject_id=2, seed=7),
    }
    runs = lab.run_all(cfg, subjects, n_jobs=1, progress=False)
    assert set(runs) == {1, 2}
    assert all(r.n_windows > 0 for r in runs.values())


def test_run_all_rejects_a_missing_subject(cfg, recording):
    with pytest.raises(KeyError, match="no recording loaded"):
        lab.run_all(cfg, {99: recording}, subject_ids=[1], n_jobs=1)


def test_get_window_trace_tracked_matches_a_normal_run(cfg, recording):
    """Lab Spec Section 2.4: the isolated window must equal the real one."""
    full = TroikaEstimator(cfg).run(recording, trace="light")
    isolated = lab.get_window_trace(recording, cfg, 12, prev_bin="tracked")
    assert isolated.idx == 12
    assert isolated.is_full
    assert isolated.bin_cur == full.trace[12].bin_cur
    assert isolated.bpm_est == pytest.approx(full.trace[12].bpm_est)
    assert not isolated.contaminated


def test_get_window_trace_seeded_is_marked_contaminated(cfg, recording):
    trace = lab.get_window_trace(recording, cfg, 12, prev_bin=55)
    assert trace.contaminated
    assert trace.idx == 12


def test_get_window_trace_ground_truth_seed(cfg, recording):
    trace = lab.get_window_trace(recording, cfg, 12, prev_bin="ground_truth")
    assert trace.contaminated


def test_get_window_trace_rejects_a_bad_seed(cfg, recording):
    with pytest.raises(ValueError, match="prev_bin must be"):
        lab.get_window_trace(recording, cfg, 5, prev_bin="nonsense")
    with pytest.raises(ValueError, match="window index"):
        lab.get_window_trace(recording, cfg, -1)


def test_collect_traces_picks_one_window_per_subject(cfg):
    subjects = {
        1: synthetic_recording(seconds=30.0, subject_id=1),
        2: synthetic_recording(seconds=30.0, subject_id=2, seed=7),
    }
    traces = lab.collect_traces(cfg, subjects, selector=4)
    assert set(traces) == {1, 2}
    assert all(t.idx == 4 and t.is_full for t in traces.values())


def test_collect_traces_time_selector(cfg):
    subjects = {1: synthetic_recording(seconds=30.0, subject_id=1)}
    traces = lab.collect_traces(cfg, subjects, selector="time:10")
    assert traces[1].idx == 5  # 10 s at a 2 s step


def test_unknown_selector_raises(cfg, recording):
    from troika.lab import _resolve_selector

    with pytest.raises(ValueError, match="unknown selector"):
        _resolve_selector("nope", cfg, recording, None)


def test_promote_writes_only_the_differences(cfg, tmp_path):
    changed = cfg.with_overrides({"decomposition.L": 222, "run.name": "promoted"})
    path = tmp_path / "promoted.yaml"
    lab.promote(changed, path)
    reloaded = lab.load_config(path)
    assert reloaded.decomposition.params.L == 222
    assert reloaded.hash() == changed.hash()
    assert "signal" not in path.read_text(encoding="utf-8")
