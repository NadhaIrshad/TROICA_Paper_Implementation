"""Every plot returns a Figure on synthetic data (Lab Spec Section 9).

No real recordings are needed, so these run anywhere. One short synthetic
recording is traced once for the whole module, because tracing is the expensive
part, not the drawing.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import pytest

from troika import lab, viz
from troika.config import load_config
from troika.pipeline import TroikaEstimator

from test_pipeline_synthetic import synthetic_recording

FULL_WINDOW = 6


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def recording():
    return synthetic_recording(seconds=30.0, subject_id=5)


@pytest.fixture(scope="module")
def run(cfg, recording):
    return TroikaEstimator(cfg).run(recording, trace="light", full_windows=[FULL_WINDOW])


@pytest.fixture(scope="module")
def window(run):
    return run.trace[FULL_WINDOW]


@pytest.fixture(scope="module")
def light_window(run):
    return run.trace.windows[0]


@pytest.fixture(scope="module")
def runs(run):
    return {run.subject_id: run}


@pytest.fixture(scope="module")
def stats(runs, cfg):
    return lab.evaluate(runs, cfg, paper_variant="full")


def _is_figure(obj):
    assert isinstance(obj, go.Figure), f"expected a Figure, got {type(obj).__name__}"
    assert obj.data or obj.layout.annotations, "figure has neither traces nor annotations"
    return True


# ------------------------------------------------------------------ group A


@pytest.mark.parametrize("domain", ["both", "time", "freq"])
def test_plot_recording(recording, domain):
    _is_figure(viz.plot_recording(recording, domain=domain))


def test_plot_recording_without_ecg(recording):
    _is_figure(viz.plot_recording(recording, channels=("ppg", "acc", "ecg")))


def test_plot_spectrogram(recording):
    _is_figure(viz.plot_spectrogram(recording))


def test_plot_spectrogram_without_ground_truth(recording):
    from dataclasses import replace

    _is_figure(viz.plot_spectrogram(replace(recording, bpm_gt=None)))


@pytest.mark.parametrize("domain", ["time", "freq", "spectrogram"])
def test_plot_all_subjects_grid(recording, domain):
    subjects = {i: recording for i in range(1, 13)}
    _is_figure(viz.plot_all_subjects(subjects, domain=domain))


def test_plot_all_subjects_overlay(recording):
    _is_figure(viz.plot_all_subjects({1: recording, 2: recording}, layout="overlay"))


# ------------------------------------------------------- groups B, D and G


@pytest.mark.parametrize(
    "stage", ["raw", "bandpass", "decomposition", "temporal_diff", "spectrum_estimator"]
)
def test_plot_stage(window, stage):
    _is_figure(viz.plot_stage(window, stage))


@pytest.mark.parametrize("domain", ["both", "time", "freq"])
def test_plot_stage_domains(window, domain):
    _is_figure(viz.plot_stage(window, "bandpass", domain=domain))


def test_plot_stage_unknown_stage_raises(window):
    with pytest.raises(KeyError, match="does not hold"):
        viz.plot_stage(window, "nonsense")


def test_plot_filter_response(window):
    _is_figure(viz.plot_filter_response(window))


def test_plot_acc_dominant(window):
    _is_figure(viz.plot_acc_dominant(window))


def test_plot_temporal_diff(window):
    _is_figure(viz.plot_temporal_diff(window))


def test_plot_pipeline_overview(window):
    _is_figure(viz.plot_pipeline_overview(window))


def test_plot_decomposition_components(window):
    _is_figure(viz.plot_decomposition_components(window, top=4))


@pytest.mark.parametrize("stage", ["bandpass", "spectrum_estimator"])
def test_plot_stage_spectrogram(run, stage):
    _is_figure(viz.plot_stage_spectrogram(run, stage))


@pytest.mark.parametrize("stitch", ["center", "ola"])
def test_plot_stage_stitched(run, stitch):
    _is_figure(viz.plot_stage_stitched(run, "bandpass", stitch=stitch))


def test_stage_stitched_unknown_mode_raises(run):
    with pytest.raises(ValueError, match="stitch must be"):
        viz.plot_stage_stitched(run, "bandpass", stitch="nope")


# ------------------------------------------------------------------ group C


def test_plot_ssa_embedding(window):
    _is_figure(viz.plot_ssa_embedding(window, zoom=5))


def test_plot_ssa_svd(window):
    _is_figure(viz.plot_ssa_svd(window))


def test_plot_ssa_eigenvectors(window):
    _is_figure(viz.plot_ssa_eigenvectors(window, idx=range(3)))


def test_plot_ssa_grouping(window):
    _is_figure(viz.plot_ssa_grouping(window))


def test_plot_ssa_wcorr(window):
    _is_figure(viz.plot_ssa_wcorr(window, top=12))


def test_plot_ssa_components(window):
    _is_figure(viz.plot_ssa_components(window, top=4))


def test_plot_ssa_reconstruction(window):
    _is_figure(viz.plot_ssa_reconstruction(window))


# ------------------------------------------------------------ groups E and F


def test_plot_ssr_dictionary(window):
    _is_figure(viz.plot_ssr_dictionary(window))


def test_plot_ssr_iterations(window):
    _is_figure(viz.plot_ssr_iterations(window))


def test_plot_ssr_spectrum(window):
    _is_figure(viz.plot_ssr_spectrum(window))


def test_plot_tracking_window(window):
    _is_figure(viz.plot_tracking_window(window))


def test_plot_tracking_run(run):
    _is_figure(viz.plot_tracking_run(run))


def test_plot_verification_timeline(run):
    _is_figure(viz.plot_verification_timeline(run))


# ------------------------------------------------------------ groups H and I


def test_plot_all_subjects_stage(window):
    traces = {i: window for i in range(1, 13)}
    _is_figure(viz.plot_all_subjects_stage(traces))


def test_plot_all_subjects_tracking(runs):
    _is_figure(viz.plot_all_subjects_tracking(runs))


def test_plot_error_bars(stats):
    _is_figure(viz.plot_error_bars(stats))


def test_plot_bland_altman(runs):
    _is_figure(viz.plot_bland_altman(runs))


def test_plot_scatter(runs):
    _is_figure(viz.plot_scatter(runs))


def test_compare_stage(window):
    _is_figure(viz.compare_stage({"a": window, "b": window}, stage="bandpass"))


def test_compare_stage_without_variants_raises():
    with pytest.raises(ValueError, match="no variants"):
        viz.compare_stage({})


def test_compare_runs(runs, stats):
    _is_figure(viz.compare_runs({"default": runs, "variant": runs}, stats))


# ------------------------------------------------------- missing-key errors


def test_missing_diag_raises_a_clear_error(light_window):
    """Lab Spec Section 2.3: name the keys and suggest the generic plot."""
    with pytest.raises(KeyError) as excinfo:
        viz.plot_ssa_svd(light_window)
    assert "full trace" in str(excinfo.value)


def test_missing_key_within_a_stage_names_the_keys(window):
    with pytest.raises(KeyError, match="generic plot"):
        window.require("decomposition", ["not_a_real_key"], "plot_ssa_svd")


def test_tracking_run_without_a_trace_raises(cfg, recording):
    run = TroikaEstimator(cfg).run(recording, trace="none")
    with pytest.raises(KeyError, match="light trace"):
        viz.plot_tracking_run(run)


# ------------------------------------------------------------------- saving


def test_save_writes_html(window, tmp_path):
    path = viz.save(viz.plot_stage(window, "bandpass"), tmp_path / "figure.html")
    assert path.exists() and path.suffix == ".html"
    assert "plotly" in path.read_text(encoding="utf-8").lower()


def test_save_defaults_to_html(window, tmp_path):
    path = viz.save(viz.plot_stage(window, "bandpass"), tmp_path / "figure")
    assert path.suffix == ".html"


def test_setup_notebook_is_callable():
    viz.setup_notebook()


def test_every_catalogue_function_is_exported():
    """Lab Spec Section 5.2 lists the catalogue; all of it must be importable."""
    for name in viz.__all__:
        assert hasattr(viz, name), name
