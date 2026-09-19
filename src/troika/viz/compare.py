"""Variant comparison (Lab Spec Section 5.2 group I)."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from troika.trace import WindowTrace
from troika.types import RunResult
from troika.viz.common import (
    add_gt_lines,
    freq_axis,
    freq_axis_title,
    line_trace,
    trim_to_fmax,
)
from troika.viz.theme import COLORS, STAGE_LABELS, apply_layout

__all__ = ["compare_stage", "compare_runs"]

_PALETTE = [
    "#1f6fb4", "#d62728", "#2e9e5b", "#e8871a", "#7b52ab", "#1aa3a3", "#8c8c8c",
]


def compare_stage(
    traces: Mapping[str, WindowTrace],
    stage: str = "bandpass",
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    title: str | None = None,
    height: int | None = None,
):
    """Overlay one stage output from several configurations of the same window.

    The legend carries each variant's timing for that stage, so a variant that
    buys accuracy with runtime shows both at once.
    """
    if not traces:
        raise ValueError("no variants to compare")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"{STAGE_LABELS.get(stage, stage)} (time)",
            f"{STAGE_LABELS.get(stage, stage)} (spectrum)",
        ],
        horizontal_spacing=0.08,
    )
    reference = next(iter(traces.values()))
    for index, (name, trace) in enumerate(traces.items()):
        color = _PALETTE[index % len(_PALETTE)]
        static = trace.static
        if stage not in trace.stages:
            continue
        timing = trace.timings_ms.get(stage)
        label = f"{name} ({timing:.0f} ms)" if timing else name

        signal = trace.stages[stage].signal.astype(np.float64)
        t = trace.t_start_s + np.arange(signal.size) / static.fs
        fig.add_trace(line_trace(t, signal, label, color), row=1, col=1)

        spec = trim_to_fmax(
            trace.stages[stage].spectrum.astype(np.float64), static.fs, static.N, fmax_hz
        )
        fig.add_trace(
            go.Scatter(
                x=freq_axis(spec.size, static.fs, static.N, freq_unit),
                y=spec,
                name=label,
                mode="lines",
                line=dict(color=color, width=1.4),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    add_gt_lines(
        fig, reference.gt_bpm, reference.static.fs, reference.static.N, freq_unit, row=1, col=2
    )
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=1, col=2)
    return apply_layout(
        fig,
        title or f"Window {reference.idx}: {STAGE_LABELS.get(stage, stage)} across variants",
        height or 420,
    )


def compare_runs(
    runs_by_name: Mapping[str, Mapping[int, RunResult]],
    table: pd.DataFrame | None = None,
    *,
    subject: int | None = None,
    title: str | None = None,
    height: int | None = None,
):
    """Estimate traces per variant overlaid, with a statistics table below."""
    if not runs_by_name:
        raise ValueError("no runs to compare")

    first = next(iter(runs_by_name.values()))
    subject = subject if subject is not None else sorted(first)[0]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.6, 0.4],
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
        subplot_titles=[f"Subject {subject}: estimate against truth", "statistics"],
        vertical_spacing=0.12,
    )

    truth_drawn = False
    for index, (name, runs) in enumerate(runs_by_name.items()):
        if subject not in runs:
            continue
        run = runs[subject]
        t = np.array([w.t_start_s for w in run.windows])
        if not truth_drawn and run.bpm_gt is not None:
            fig.add_trace(
                go.Scatter(
                    x=t[: run.bpm_gt.size],
                    y=run.bpm_gt,
                    name="true HR",
                    mode="lines",
                    line=dict(color=COLORS["ground_truth"], width=1.6, dash="dash"),
                ),
                row=1,
                col=1,
            )
            truth_drawn = True
        fig.add_trace(
            go.Scatter(
                x=t,
                y=run.bpm_est,
                name=name,
                mode="lines",
                line=dict(color=_PALETTE[index % len(_PALETTE)], width=1.5),
            ),
            row=1,
            col=1,
        )

    if table is not None and not table.empty:
        display = table.reset_index()
        columns = [
            c
            for c in display.columns
            if c in ("variant", "subject", "Error1_BPM", "Error2_pct", "bias_BPM", "n_windows")
        ]
        display = display[columns].round(2)
        fig.add_trace(
            go.Table(
                header=dict(
                    values=list(display.columns),
                    fill_color="rgba(0,0,0,0.05)",
                    align="left",
                    font=dict(size=11),
                ),
                cells=dict(
                    values=[display[c] for c in display.columns],
                    align="left",
                    font=dict(size=10),
                    height=20,
                ),
            ),
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="BPM", row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    return apply_layout(fig, title or "Variant comparison", height or 760)
