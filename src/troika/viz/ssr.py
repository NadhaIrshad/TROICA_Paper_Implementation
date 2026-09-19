"""Sparse-reconstruction internals (Lab Spec Section 5.2 group E)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from troika import binmap
from troika.trace import WindowTrace
from troika.viz.common import (
    FREQ_HOVER,
    add_bin_markers,
    add_gt_lines,
    freq_axis,
    freq_axis_title,
    hover_frequency,
    trim_to_fmax,
)
from troika.viz.theme import COLORS, apply_layout

__all__ = ["plot_ssr_dictionary", "plot_ssr_iterations", "plot_ssr_spectrum"]


def plot_ssr_dictionary(
    trace: WindowTrace,
    *,
    show_atoms: int = 3,
    title: str | None = None,
    height: int | None = None,
):
    """Which of the N grid columns survive the Eq. 12 pruning.

    The kept set is symmetric under ``k -> N - k`` and excludes DC, which the
    mask makes obvious at a glance.
    """
    payload = trace.require("spectrum_estimator", ["kept_bins"], "plot_ssr_dictionary")
    kept = np.asarray(payload["kept_bins"], dtype=np.int64)
    static = trace.static

    mask = np.zeros(static.N, dtype=float)
    mask[kept] = 1.0

    rows = 2 if show_atoms else 1
    fig = make_subplots(
        rows=rows, cols=1,
        subplot_titles=(
            [f"kept columns: {kept.size} of {static.N}", "dictionary atoms (real part)"]
            if show_atoms
            else [f"kept columns: {kept.size} of {static.N}"]
        ),
        vertical_spacing=0.15,
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(static.N), y=mask, name="kept", mode="lines",
            line=dict(color=COLORS["spectrum_estimator"], width=1), fill="tozeroy",
        ),
        row=1, col=1,
    )
    for corner, label in ((0.4, "0.4 Hz"), (5.0, "5 Hz")):
        k = int(binmap.hz_to_bin(corner, static.fs, static.N))
        for x in (k, static.N - k):
            fig.add_vline(
                x=x, line=dict(color=COLORS["annotation"], width=1, dash="dot"),
                annotation_text=label, annotation_font_size=9, row=1, col=1,
            )

    if show_atoms:
        m = np.arange(min(static.M, 400))
        for k in kept[: int(show_atoms)]:
            atom = np.cos(2 * np.pi * m * int(k) / static.N)
            fig.add_trace(
                go.Scatter(
                    x=m, y=atom, mode="lines", name=f"bin {int(k)}",
                    line=dict(width=1),
                ),
                row=2, col=1,
            )
        fig.update_xaxes(title_text="sample", row=2, col=1)

    fig.update_yaxes(title_text="kept", range=[-0.1, 1.2], row=1, col=1)
    fig.update_xaxes(title_text="grid index k", row=1, col=1)
    return apply_layout(
        fig, title or f"Window {trace.idx}: pruned SSR dictionary (Eq. 12)", height or 480
    )


def plot_ssr_iterations(
    trace: WindowTrace,
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    overlay: bool = True,
    title: str | None = None,
    height: int | None = None,
):
    """Spectrum after each FOCUSS iteration, with a slider, plus sparsity per iteration.

    Paper Section IV-B stops at five iterations because the large coefficients
    converge quickly; this shows whether that holds for the window at hand.
    """
    payload = trace.require(
        "spectrum_estimator", ["iterates", "kept_bins", "sparsity_per_iter"],
        "plot_ssr_iterations",
    )
    iterates = np.asarray(payload["iterates"])
    kept = np.asarray(payload["kept_bins"], dtype=np.int64)
    sparsity = np.asarray(payload["sparsity_per_iter"], dtype=np.int64)
    static = trace.static

    half = static.N // 2
    positive = kept <= half
    x_bins = kept[positive]
    order = np.argsort(x_bins)
    x_bins = x_bins[order]
    x = (
        binmap.bin_to_bpm(x_bins, static.fs, static.N)
        if freq_unit == "bpm"
        else binmap.bin_to_hz(x_bins, static.fs, static.N)
    )
    limit = (fmax_hz * 60.0 if freq_unit == "bpm" else fmax_hz) if fmax_hz else np.inf
    keep = x <= limit

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.7, 0.3],
        subplot_titles=["spectrum per iteration", "coefficients above 1 % of the peak"],
        vertical_spacing=0.16,
    )
    n_iter = iterates.shape[0]
    for i in range(n_iter):
        power = (np.abs(iterates[i][positive]) ** 2)[order]
        fig.add_trace(
            go.Scatter(
                x=x[keep], y=power[keep], name=f"iteration {i}", mode="lines",
                line=dict(width=1.6 if i == n_iter - 1 else 1.0),
                opacity=1.0 if i == n_iter - 1 else (0.25 if overlay else 1.0),
                visible=True if overlay else (i == n_iter - 1),
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=np.arange(sparsity.size), y=sparsity, name="sparsity", mode="lines+markers",
            line=dict(color=COLORS["spectrum_estimator"], width=1.6), showlegend=False,
        ),
        row=2, col=1,
    )

    steps = []
    for i in range(n_iter):
        visible = [j <= i for j in range(n_iter)] + [True]
        steps.append(dict(method="update", args=[{"visible": visible}], label=str(i)))
    fig.update_layout(
        sliders=[dict(active=n_iter - 1, currentvalue=dict(prefix="up to iteration "), steps=steps)]
    )

    add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=1, col=1)
    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=1, col=1)
    fig.update_xaxes(title_text="iteration", row=2, col=1)
    return apply_layout(fig, title or f"Window {trace.idx}: FOCUSS iterations", height or 620)


def plot_ssr_spectrum(
    trace: WindowTrace,
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    log_y: bool = False,
    title: str | None = None,
    height: int | None = None,
):
    """The sparse spectrum against the periodogram of the same input.

    This is the Fig. 1 comparison on real data: what the sparse estimate
    resolves that the periodogram smears.
    """
    static = trace.static
    estimate = trim_to_fmax(
        trace.stages["spectrum_estimator"].spectrum.astype(np.float64),
        static.fs, static.N, fmax_hz,
    )
    differenced = trace.stages["temporal_diff"].signal.astype(np.float64)

    from troika.preprocessing.spectrum import periodogram

    reference = trim_to_fmax(periodogram(differenced, static.N), static.fs, static.N, fmax_hz)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = freq_axis(estimate.size, static.fs, static.N, freq_unit)
    fig.add_trace(
        go.Scatter(
            x=x[: reference.size], y=reference[: x.size], name="periodogram", mode="lines",
            line=dict(color=COLORS["raw"], width=1.2),
            customdata=hover_frequency(min(x.size, reference.size), static.fs, static.N),
            hovertemplate=FREQ_HOVER,
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=estimate, name="sparse estimate", mode="lines",
            line=dict(color=COLORS["spectrum_estimator"], width=1.8),
            customdata=hover_frequency(estimate.size, static.fs, static.N),
            hovertemplate=FREQ_HOVER,
        ),
        secondary_y=False,
    )
    if trace.ctx is not None:
        add_bin_markers(fig, trace.ctx.acc_bins, static.fs, static.N, freq_unit)
    add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit)

    fig.update_yaxes(title_text="sparse power", type="log" if log_y else "linear", secondary_y=False)
    fig.update_yaxes(title_text="periodogram power", secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text=freq_axis_title(freq_unit))
    return apply_layout(
        fig, title or f"Window {trace.idx}: sparse spectrum against the periodogram", height or 430
    )
