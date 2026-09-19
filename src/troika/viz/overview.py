"""Pipeline overview and all-subject grids (Lab Spec Section 5.2, groups B and H)."""

from __future__ import annotations

from typing import Callable, Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from troika import binmap
from troika.evaluation import metrics
from troika.trace import WindowTrace
from troika.types import RunResult
from troika.viz.common import (
    FREQ_HOVER,
    add_gt_lines,
    add_protocol_shading,
    freq_axis,
    freq_axis_title,
    hover_frequency,
    line_trace,
    note,
    trim_to_fmax,
)
from troika.viz.theme import COLORS, STAGE_LABELS, apply_layout, stage_color

__all__ = [
    "plot_pipeline_overview",
    "plot_all_subjects_stage",
    "plot_all_subjects_tracking",
    "plot_error_bars",
    "plot_bland_altman",
    "plot_scatter",
]

_OVERVIEW_STAGES = ("raw", "bandpass", "decomposition", "temporal_diff", "spectrum_estimator")


def plot_pipeline_overview(
    trace: WindowTrace,
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    title: str | None = None,
    height: int | None = None,
):
    """Every stage of one window in one tall figure, time left and frequency right.

    The true heart rate is marked at the same place in every frequency panel, so
    the row where it stops being the obvious peak is the row that explains a
    failure.
    """
    static = trace.static
    stages = [s for s in _OVERVIEW_STAGES if s in trace.stages]

    fig = make_subplots(
        rows=len(stages), cols=2,
        subplot_titles=[
            item for s in stages
            for item in (f"{STAGE_LABELS.get(s, s)} (time)", f"{STAGE_LABELS.get(s, s)} (spectrum)")
        ],
        vertical_spacing=max(0.02, 0.24 / len(stages)), horizontal_spacing=0.08,
    )
    for row, stage in enumerate(stages, start=1):
        signal = trace.stages[stage].signal.astype(np.float64)
        t = trace.t_start_s + np.arange(signal.size) / static.fs
        fig.add_trace(
            line_trace(t, signal, STAGE_LABELS.get(stage, stage), stage_color(stage)),
            row=row, col=1,
        )
        spec = trim_to_fmax(
            trace.stages[stage].spectrum.astype(np.float64), static.fs, static.N, fmax_hz
        )
        fig.add_trace(
            go.Scatter(
                x=freq_axis(spec.size, static.fs, static.N, freq_unit), y=spec,
                name=STAGE_LABELS.get(stage, stage), mode="lines",
                line=dict(color=stage_color(stage), width=1.3),
                customdata=hover_frequency(spec.size, static.fs, static.N),
                hovertemplate=FREQ_HOVER, showlegend=False,
            ),
            row=row, col=2,
        )
        add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=row, col=2)

    if trace.bin_cur >= 0:
        estimate = (
            binmap.bin_to_bpm(trace.bin_cur, static.fs, static.N)
            if freq_unit == "bpm"
            else binmap.bin_to_hz(trace.bin_cur, static.fs, static.N)
        )
        fig.add_vline(
            x=float(estimate), line=dict(color=COLORS["tracker"], width=1.6),
            annotation_text="estimate", annotation_font_size=10,
            row=len(stages), col=2,
        )

    fig.update_xaxes(title_text="time (s)", row=len(stages), col=1)
    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=len(stages), col=2)
    subtitle = f"estimate {trace.bpm_est:.1f} BPM"
    if trace.gt_bpm is not None:
        subtitle += f", truth {trace.gt_bpm:.1f} BPM"
    return apply_layout(
        fig,
        title or f"Window {trace.idx} at {trace.t_start_s:.0f} s &middot; {subtitle}",
        height or 190 * len(stages) + 120,
    )


def plot_all_subjects_stage(
    traces: Mapping[int, WindowTrace],
    stage: str = "spectrum_estimator",
    *,
    domain: str = "freq",
    freq_unit: str = "bpm",
    fmax_hz: float = 4.0,
    title: str | None = None,
    height: int | None = None,
):
    """A 4x3 grid of one stage across subjects (Lab Spec Section 5.2 group H).

    Feed it ``lab.collect_traces(cfg, subjects, selector="max_error")`` to see
    every subject's worst window side by side.
    """
    ids = sorted(traces)
    rows, cols = 4, 3
    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=[
            f"S{i}: w{traces[i].idx}, est {traces[i].bpm_est:.0f}"
            + (f" vs {traces[i].gt_bpm:.0f}" if traces[i].gt_bpm else "")
            for i in ids
        ],
        vertical_spacing=0.07, horizontal_spacing=0.05,
    )
    for index, subject_id in enumerate(ids):
        row, col = index // cols + 1, index % cols + 1
        trace = traces[subject_id]
        static = trace.static
        if stage not in trace.stages:
            note(fig, f"no {stage}", row=row, col=col)
            continue
        if domain == "time":
            signal = trace.stages[stage].signal.astype(np.float64)
            t = trace.t_start_s + np.arange(signal.size) / static.fs
            fig.add_trace(
                line_trace(t, signal, f"S{subject_id}", stage_color(stage)), row=row, col=col
            )
        else:
            spec = trim_to_fmax(
                trace.stages[stage].spectrum.astype(np.float64), static.fs, static.N, fmax_hz
            )
            fig.add_trace(
                go.Scatter(
                    x=freq_axis(spec.size, static.fs, static.N, freq_unit), y=spec,
                    mode="lines", line=dict(color=stage_color(stage), width=1.2),
                    showlegend=False,
                ),
                row=row, col=col,
            )
            add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=row, col=col)

    return apply_layout(
        fig, title or f"All subjects: {STAGE_LABELS.get(stage, stage)}", height or 900
    )


def plot_all_subjects_tracking(
    runs: Mapping[int, RunResult], *, title: str | None = None, height: int | None = None
):
    """Estimate against truth over time for every subject, the paper's Fig. 8 twelve times."""
    ids = sorted(runs)
    rows, cols = 4, 3
    titles = []
    for subject_id in ids:
        run = runs[subject_id]
        error = (
            metrics.error1(run.bpm_est, run.bpm_gt) if run.bpm_gt is not None else float("nan")
        )
        titles.append(f"S{subject_id}: Error1 {error:.2f} BPM")

    fig = make_subplots(
        rows=rows, cols=cols, subplot_titles=titles,
        shared_xaxes=True, shared_yaxes=True,
        vertical_spacing=0.07, horizontal_spacing=0.04,
    )
    for index, subject_id in enumerate(ids):
        row, col = index // cols + 1, index % cols + 1
        run = runs[subject_id]
        t = np.array([w.t_start_s for w in run.windows])
        if run.bpm_gt is not None:
            fig.add_trace(
                go.Scatter(
                    x=t[: run.bpm_gt.size], y=run.bpm_gt, name="true HR", mode="lines",
                    line=dict(color=COLORS["ground_truth"], width=1.4, dash="dash"),
                    showlegend=index == 0,
                ),
                row=row, col=col,
            )
        fig.add_trace(
            go.Scatter(
                x=t, y=run.bpm_est, name="estimate", mode="lines",
                line=dict(color=COLORS["tracker"], width=1.4), showlegend=index == 0,
            ),
            row=row, col=col,
        )
        if run.trace is not None:
            add_protocol_shading(
                fig, run.trace.protocol, row=row, col=col,
                duration_s=float(t[-1]) if t.size else None,
            )
        fig.update_yaxes(title_text="BPM" if col == 1 else None, row=row, col=col)
        fig.update_xaxes(title_text="time (s)" if row == rows else None, row=row, col=col)

    return apply_layout(fig, title or "Estimate against truth, all subjects", height or 950)


def plot_error_bars(
    stats: pd.DataFrame, *, title: str | None = None, height: int | None = None
):
    """Error1 and Error2 per subject as bars, with the paper's values as markers."""
    per_subject = stats[[isinstance(i, (int, np.integer)) for i in stats.index]]
    ids = [int(i) for i in per_subject.index]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Error1 (BPM)", "Error2 (%)"], vertical_spacing=0.12,
    )
    for row, (column, paper_column) in enumerate(
        (("Error1_BPM", "Error1_paper"), ("Error2_pct", "Error2_paper")), start=1
    ):
        fig.add_trace(
            go.Bar(
                x=ids, y=per_subject[column], name=column,
                marker_color=COLORS["tracker"], showlegend=row == 1,
            ),
            row=row, col=1,
        )
        if paper_column in per_subject.columns:
            fig.add_trace(
                go.Scatter(
                    x=ids, y=per_subject[paper_column], name="paper", mode="markers",
                    marker=dict(size=10, symbol="diamond", color=COLORS["ground_truth"]),
                    showlegend=row == 1,
                ),
                row=row, col=1,
            )
        mean_row = next((i for i in stats.index if str(i).startswith("mean")), None)
        if mean_row is not None:
            fig.add_hline(
                y=float(stats.loc[mean_row, column]),
                line=dict(color=COLORS["annotation"], width=1, dash="dot"),
                annotation_text="mean", annotation_font_size=10, row=row, col=1,
            )

    fig.update_xaxes(title_text="subject", tickmode="array", tickvals=ids, row=2, col=1)
    return apply_layout(fig, title or "Error per subject", height or 620)


def plot_bland_altman(
    runs: Mapping[int, RunResult], *, title: str | None = None, height: int | None = None
):
    """The paper's Fig. 5, pooled over the selected subjects."""
    est, gt = _pooled(runs)
    stats = metrics.bland_altman(est, gt)
    mean = (est + gt) / 2.0
    diff = est - gt

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=mean, y=diff, name="windows", mode="markers",
            marker=dict(size=4, color=COLORS["tracker"], opacity=0.5),
        )
    )
    for value, label, dash in (
        (stats["mean_diff"], "mean", "solid"),
        (stats["loa_low"], "mean - 1.96 SD", "dash"),
        (stats["loa_high"], "mean + 1.96 SD", "dash"),
    ):
        fig.add_hline(
            y=value, line=dict(color=COLORS["ground_truth"], width=1.2, dash=dash),
            annotation_text=f"{label}: {value:.2f}", annotation_font_size=10,
        )
    fig.update_xaxes(title_text="average of truth and estimate (BPM)")
    fig.update_yaxes(title_text="estimate - truth (BPM)")
    return apply_layout(
        fig,
        title or f"Bland-Altman, {int(stats['n'])} windows, SD {stats['sd']:.2f} BPM",
        height or 480,
    )


def plot_scatter(
    runs: Mapping[int, RunResult], *, title: str | None = None, height: int | None = None
):
    """The paper's Fig. 6: estimates against truth, with the correlation in the title."""
    est, gt = _pooled(runs)
    r = metrics.pearson(est, gt)
    lo = float(min(gt.min(), est.min())) - 5
    hi = float(max(gt.max(), est.max())) + 5

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=gt, y=est, name="windows", mode="markers",
            marker=dict(size=4, color=COLORS["tracker"], opacity=0.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[lo, hi], y=[lo, hi], name="identity", mode="lines",
            line=dict(color=COLORS["ground_truth"], width=1.2, dash="dash"),
        )
    )
    fig.update_xaxes(title_text="ground truth (BPM)", range=[lo, hi])
    fig.update_yaxes(title_text="estimate (BPM)", range=[lo, hi], scaleanchor="x")
    return apply_layout(fig, title or f"Pearson correlation r = {r:.3f}", height or 560)


def _pooled(runs: Mapping[int, RunResult]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every window of every run that has ground truth."""
    est_parts, gt_parts = [], []
    for run in runs.values():
        if run.bpm_gt is None:
            continue
        ok = np.isfinite(run.bpm_est) & np.isfinite(run.bpm_gt)
        est_parts.append(np.asarray(run.bpm_est)[ok])
        gt_parts.append(np.asarray(run.bpm_gt)[ok])
    if not est_parts:
        raise ValueError("no run has ground truth to pool")
    return np.concatenate(est_parts), np.concatenate(gt_parts)


_ = Callable
