"""Tracking views (Lab Spec Section 5.2 group F)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from troika import binmap
from troika.trace import WindowTrace
from troika.types import RunResult
from troika.viz.common import (
    FREQ_HOVER,
    add_gt_lines,
    add_protocol_shading,
    freq_axis,
    freq_axis_title,
    hover_frequency,
    note,
    trim_to_fmax,
)
from troika.viz.theme import COLORS, apply_layout

__all__ = ["plot_tracking_window", "plot_tracking_run", "plot_verification_timeline"]

_CASE_TEXT = {
    1: "Case 1: harmonic pair",
    2: "Case 2: closest candidate",
    3: "Case 3: no peaks, keep previous",
    None: "initialisation",
}


def plot_tracking_window(
    trace: WindowTrace,
    *,
    freq_unit: str = "bpm",
    fmax_hz: float = 6.0,
    title: str | None = None,
    height: int | None = None,
):
    """One window's decision: search ranges, eta, peaks, pairs and the outcome.

    Everything the tracker used is drawn, so a wrong estimate can be explained
    rather than guessed at.
    """
    payload = trace.require(
        "tracker",
        ["R0", "R1", "eta", "P0", "P1", "k_b", "case", "k_cur", "k_prev"],
        "plot_tracking_window",
    )
    static = trace.static
    spectrum = trim_to_fmax(
        trace.stages["spectrum_estimator"].spectrum.astype(np.float64),
        static.fs, static.N, fmax_hz,
    )
    x = freq_axis(spectrum.size, static.fs, static.N, freq_unit)

    def to_x(k: int) -> float:
        return float(
            binmap.bin_to_bpm(int(k), static.fs, static.N)
            if freq_unit == "bpm"
            else binmap.bin_to_hz(int(k), static.fs, static.N)
        )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=spectrum, name="spectrum", mode="lines",
            line=dict(color=COLORS["spectrum_estimator"], width=1.5),
            customdata=hover_frequency(spectrum.size, static.fs, static.N),
            hovertemplate=FREQ_HOVER,
        )
    )

    for label, (lo, hi), color in (
        ("R0 (fundamental)", payload["R0"], "rgba(31,111,180,0.12)"),
        ("R1 (first harmonic)", payload["R1"], "rgba(123,82,171,0.10)"),
    ):
        fig.add_vrect(
            x0=to_x(lo), x1=to_x(hi), fillcolor=color, line_width=0, layer="below",
            annotation_text=label, annotation_position="top left", annotation_font_size=10,
        )

    fig.add_hline(
        y=float(payload["eta"]),
        line=dict(color=COLORS["annotation"], width=1, dash="dot"),
        annotation_text="eta = 30 % of the R0 peak", annotation_font_size=10,
    )

    for name, peaks, symbol in (
        ("P0", np.asarray(payload["P0"], dtype=int), "triangle-up"),
        ("P1", np.asarray(payload["P1"], dtype=int), "triangle-down"),
    ):
        if peaks.size:
            inside = peaks[peaks < spectrum.size]
            if inside.size:
                fig.add_trace(
                    go.Scatter(
                        x=[to_x(k) for k in inside], y=spectrum[inside], name=name,
                        mode="markers", marker=dict(size=10, symbol=symbol, color=COLORS["tracker"]),
                    )
                )

    for a, b in payload.get("pairs", []):
        if a < spectrum.size and b < spectrum.size:
            fig.add_shape(
                type="line", x0=to_x(a), y0=spectrum[a], x1=to_x(b), y1=spectrum[b],
                line=dict(color=COLORS["tracker"], width=1, dash="dot"),
            )

    for k, label, color, dash in (
        (payload["k_prev"], "k_prev", COLORS["annotation"], "dot"),
        (payload["k_b"], "k_b (selected)", COLORS["temporal_diff"], "dash"),
        (payload["k_cur"], "k_cur (after verification)", COLORS["tracker"], "solid"),
    ):
        if k is None:
            continue
        fig.add_vline(
            x=to_x(k), line=dict(color=color, width=1.6, dash=dash),
            annotation_text=label, annotation_position="bottom right", annotation_font_size=10,
        )

    add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit)

    flags = []
    if payload.get("rule1_fired"):
        flags.append("rule 1 clamped the jump")
    if payload.get("rule2_fired"):
        flags.append(f"rule 2 fired (trend {payload.get('trend', 0):+d})")
    subtitle = _CASE_TEXT.get(payload["case"], str(payload["case"]))
    if flags:
        subtitle += " &middot; " + ", ".join(flags)

    fig.update_xaxes(title_text=freq_axis_title(freq_unit))
    fig.update_yaxes(title_text="power")
    return apply_layout(
        fig,
        title or f"Window {trace.idx} at {trace.t_start_s:.0f} s &middot; {subtitle}",
        height or 470,
    )


def plot_tracking_run(
    run: RunResult,
    *,
    freq_unit: str = "bpm",
    fmax_hz: float = 4.0,
    title: str | None = None,
    height: int | None = None,
):
    """Whole recording: the spectrum over time, the estimate, and the error below.

    Markers show where Case 2 or 3 was taken and where a verification rule
    fired, which is where a failure usually starts.
    """
    if run.trace is None:
        raise KeyError(
            "plot_tracking_run needs a light trace; re-run with trace='light'"
        )
    trace = run.trace
    static = trace.static
    matrix = trace.stage_matrix("spectrum_estimator").astype(np.float64)
    peak = matrix.max(axis=1, keepdims=True)
    matrix = matrix / np.where(peak > 0, peak, 1.0)
    matrix = trim_to_fmax(matrix.T, static.fs, static.N, fmax_hz).T

    t = np.array([w.t_start_s for w in trace.windows])
    y = freq_axis(matrix.shape[1], static.fs, static.N, freq_unit)

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.72, 0.28], shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=["estimate over the spectrum", "error (estimate - truth)"],
    )
    fig.add_trace(
        go.Heatmap(
            x=t, y=y, z=10.0 * np.log10(np.maximum(matrix.T, 1e-12)),
            colorscale="Viridis", showscale=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t, y=run.bpm_est if freq_unit == "bpm" else run.bpm_est / 60.0,
            name="estimate", mode="lines", line=dict(color=COLORS["tracker"], width=2),
            customdata=np.stack(
                [np.arange(len(trace.windows)), [w.case if w.case else 0 for w in trace.windows]],
                axis=-1,
            ),
            hovertemplate="window %{customdata[0]}<br>case %{customdata[1]}<br>"
            "%{y:.1f}<extra>estimate</extra>",
        ),
        row=1, col=1,
    )
    if run.bpm_gt is not None:
        fig.add_trace(
            go.Scatter(
                x=t[: run.bpm_gt.size],
                y=run.bpm_gt if freq_unit == "bpm" else run.bpm_gt / 60.0,
                name="true HR", mode="lines",
                line=dict(color="white", width=2, dash="dash"),
            ),
            row=1, col=1,
        )

    for label, predicate, symbol, color in (
        ("case 2", lambda w: w.case == 2, "circle", "#ffd166"),
        ("case 3", lambda w: w.case == 3, "square", "#ef476f"),
        ("rule 1", lambda w: w.rule1_fired, "x", "#ffffff"),
        ("rule 2", lambda w: w.rule2_fired, "star", "#06d6a0"),
    ):
        picked = [i for i, w in enumerate(run.windows) if predicate(w)]
        if picked:
            fig.add_trace(
                go.Scatter(
                    x=t[picked],
                    y=(run.bpm_est if freq_unit == "bpm" else run.bpm_est / 60.0)[picked],
                    name=label, mode="markers",
                    marker=dict(size=7, symbol=symbol, color=color, line=dict(width=0.5)),
                ),
                row=1, col=1,
            )

    if run.bpm_gt is not None:
        error = run.bpm_est - run.bpm_gt
        fig.add_trace(
            go.Scatter(
                x=t, y=error, name="error", mode="lines",
                line=dict(color=COLORS["tracker"], width=1.3), showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line=dict(color=COLORS["annotation"], width=1), row=2, col=1)
        fig.update_yaxes(title_text="BPM", row=2, col=1)
    else:
        note(fig, "no ground truth", row=2, col=1)

    add_protocol_shading(fig, trace.protocol, row=1, col=1, duration_s=float(t[-1]) if t.size else None)
    fig.update_yaxes(title_text=freq_axis_title(freq_unit), row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    return apply_layout(fig, title or f"Subject {run.subject_id}: tracking", height or 680)


def plot_verification_timeline(
    run: RunResult, *, title: str | None = None, height: int | None = None
):
    """Per window: previous, selected and final bin, plus the stall counter and mode."""
    if run.trace is None or not run.trace.full_windows:
        bins = np.array([w.bin_cur for w in run.windows])
        t = np.array([w.t_start_s for w in run.windows])
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
            subplot_titles=["estimated bin", "which rule fired"], vertical_spacing=0.1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=bins, name="k_cur", mode="lines+markers",
                       marker=dict(size=4), line=dict(color=COLORS["tracker"], width=1.4)),
            row=1, col=1,
        )
        for label, values, color in (
            ("rule 1", [w.rule1_fired for w in run.windows], COLORS["temporal_diff"]),
            ("rule 2", [w.rule2_fired for w in run.windows], COLORS["decomposition"]),
        ):
            fig.add_trace(
                go.Scatter(x=t, y=np.asarray(values, dtype=int), name=label, mode="lines",
                           line=dict(color=color, width=1.2)),
                row=2, col=1,
            )
        fig.update_xaxes(title_text="time (s)", row=2, col=1)
        return apply_layout(
            fig, title or f"Subject {run.subject_id}: verification timeline", height or 520
        )

    windows = [w for w in run.trace.windows if "tracker" in w.diag]
    t = np.array([w.t_start_s for w in windows])
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.5, 0.25, 0.25],
        subplot_titles=["bins", "stall counter and trend", "search width"],
        vertical_spacing=0.08,
    )
    for key, label, color in (
        ("k_prev", "k_prev", COLORS["annotation"]),
        ("k_b", "k_b", COLORS["temporal_diff"]),
        ("k_cur", "k_cur", COLORS["tracker"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=t, y=[w.diag["tracker"].get(key) for w in windows], name=label,
                mode="lines", line=dict(color=color, width=1.4),
            ),
            row=1, col=1,
        )
    for key, label, color in (
        ("same_count", "stall counter", COLORS["decomposition"]),
        ("trend", "trend", COLORS["spectrum_estimator"]),
    ):
        fig.add_trace(
            go.Scatter(
                x=t, y=[w.diag["tracker"].get(key, 0) for w in windows], name=label,
                mode="lines", line=dict(color=color, width=1.2),
            ),
            row=2, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=t, y=[w.diag["tracker"].get("delta_s_used") for w in windows],
            name="Delta_s used", mode="lines", line=dict(color=COLORS["bandpass"], width=1.2),
        ),
        row=3, col=1,
    )
    fig.update_xaxes(title_text="time (s)", row=3, col=1)
    return apply_layout(
        fig, title or f"Subject {run.subject_id}: verification timeline", height or 640
    )
