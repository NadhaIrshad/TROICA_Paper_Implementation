"""Per-window stage plots and whole-recording stage views.

Lab Spec Section 5.2 groups B (per-window stages), D (generic decomposition) and
G (whole-recording stage views).
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from troika import binmap
from troika.trace import RunTrace, WindowTrace
from troika.types import RunResult
from troika.viz.common import (
    FREQ_HOVER,
    add_bin_markers,
    add_excluded_band,
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
    "plot_stage",
    "plot_filter_response",
    "plot_acc_dominant",
    "plot_temporal_diff",
    "plot_decomposition_components",
    "plot_stage_spectrogram",
    "plot_stage_stitched",
]

#: What each stage's input is, for the light-versus-bold overlay of plot_stage.
_STAGE_INPUT = {
    "bandpass": "raw",
    "decomposition": "bandpass",
    "temporal_diff": "decomposition",
    "spectrum_estimator": "temporal_diff",
}


def _static(trace: WindowTrace):
    if trace.static is None:
        raise ValueError("this trace carries no StaticContext; re-run the pipeline")
    return trace.static


def _signal(trace: WindowTrace, stage: str) -> NDArray[np.float64]:
    if stage not in trace.stages:
        raise KeyError(
            f"stage {stage!r} is not in this trace; it holds {sorted(trace.stages)}"
        )
    return trace.stages[stage].signal.astype(np.float64)


def _spectrum(trace: WindowTrace, stage: str) -> NDArray[np.float64]:
    return trace.stages[stage].spectrum.astype(np.float64)


def plot_stage(
    trace: WindowTrace,
    stage: str,
    domain: str = "both",
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    show_gt: bool = True,
    log_y: bool = False,
    title: str | None = None,
    height: int | None = None,
):
    """Generic per-stage view: input light, output bold, time left and frequency right.

    Needs a light trace at least. ``stage`` is one of ``raw``, ``bandpass``,
    ``decomposition``, ``temporal_diff``, ``spectrum_estimator``.
    """
    static = _static(trace)
    if stage not in trace.stages:
        raise KeyError(
            f"plot_stage got stage {stage!r}, which this trace does not hold; "
            f"available: {sorted(trace.stages)}"
        )

    n_cols = 2 if domain == "both" else 1
    fig = make_subplots(
        rows=1,
        cols=n_cols,
        subplot_titles=(
            [f"{STAGE_LABELS.get(stage, stage)} (time)", f"{STAGE_LABELS.get(stage, stage)} (spectrum)"]
            if domain == "both"
            else None
        ),
        horizontal_spacing=0.08,
    )

    previous = _STAGE_INPUT.get(stage)
    layers = []
    if previous and previous in trace.stages:
        layers.append((previous, 0.45, 1.0))
    layers.append((stage, 1.0, 1.8))

    for name, opacity, width in layers:
        signal = _signal(trace, name)
        t = trace.t_start_s + np.arange(signal.size) / static.fs
        if domain in ("both", "time"):
            fig.add_trace(
                line_trace(
                    t,
                    signal,
                    STAGE_LABELS.get(name, name),
                    stage_color(name),
                    width=width,
                    opacity=opacity,
                ),
                row=1,
                col=1,
            )
        if domain in ("both", "freq"):
            spec = trim_to_fmax(_spectrum(trace, name), static.fs, static.N, fmax_hz)
            x = freq_axis(spec.size, static.fs, static.N, freq_unit)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=spec,
                    name=STAGE_LABELS.get(name, name),
                    mode="lines",
                    line=dict(color=stage_color(name), width=width),
                    opacity=opacity,
                    customdata=hover_frequency(spec.size, static.fs, static.N),
                    hovertemplate=FREQ_HOVER,
                    showlegend=domain != "both",
                ),
                row=1,
                col=n_cols,
            )

    if domain in ("both", "freq"):
        if show_gt:
            add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=1, col=n_cols)
        fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=1, col=n_cols)
        if log_y:
            fig.update_yaxes(type="log", row=1, col=n_cols)
    if domain in ("both", "time"):
        fig.update_xaxes(title_text="time (s)", row=1, col=1)

    return apply_layout(
        fig,
        title or f"Window {trace.idx} at {trace.t_start_s:.0f} s: {STAGE_LABELS.get(stage, stage)}",
        height or 380,
    )


def plot_filter_response(trace_or_diag, *, title: str | None = None, height: int | None = None):
    """Magnitude and phase of the band-pass, with the 0.4 and 5 Hz corners marked.

    Needs ``diag['filter_response']`` from the bandpass stage, so a full trace.
    """
    if isinstance(trace_or_diag, WindowTrace):
        payload = trace_or_diag.require(
            "bandpass", ["filter_response"], "plot_filter_response"
        )
        response = payload["filter_response"]
        description = payload.get("description", "")
    else:
        response = trace_or_diag.get("filter_response", trace_or_diag)
        description = trace_or_diag.get("description", "") if isinstance(trace_or_diag, dict) else ""

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, subplot_titles=["magnitude", "phase"],
        vertical_spacing=0.12,
    )
    fig.add_trace(
        go.Scatter(
            x=response["f_hz"],
            y=response["mag_db"],
            name="magnitude",
            mode="lines",
            line=dict(color=COLORS["bandpass"], width=1.6),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=response["f_hz"],
            y=response["phase_rad"],
            name="phase",
            mode="lines",
            line=dict(color=COLORS["raw"], width=1.4),
        ),
        row=2,
        col=1,
    )
    for corner in (0.4, 5.0):
        fig.add_vline(
            x=corner,
            line=dict(color=COLORS["annotation"], width=1, dash="dot"),
            annotation_text=f"{corner} Hz",
            annotation_font_size=10,
        )
    fig.add_hline(y=-3.0, line=dict(color=COLORS["annotation"], width=1, dash="dot"), row=1, col=1)
    fig.update_yaxes(title_text="dB (zero-phase, squared)", range=[-80, 10], row=1, col=1)
    fig.update_yaxes(title_text="radians", row=2, col=1)
    fig.update_xaxes(title_text="frequency (Hz)", range=[0, 12], row=2, col=1)
    return apply_layout(fig, title or f"Band-pass response: {description}", height or 480)


def plot_acc_dominant(
    trace: WindowTrace,
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    delta_bins: int = 10,
    n_harmonics: int = 2,
    title: str | None = None,
    height: int | None = None,
):
    """The three accelerometer spectra, the 50 % line, and F_acc before and after.

    Needs a full trace, because the raw accelerometer window is only stored then.
    """
    static = _static(trace)
    if "acc" not in trace.raw:
        raise KeyError(
            "plot_acc_dominant needs the raw accelerometer window, which only a "
            f"full trace holds; re-run with lab.get_window_trace(rec, cfg, w={trace.idx})"
        )
    if trace.ctx is None:
        raise KeyError("plot_acc_dominant needs the window context")

    from troika.preprocessing.spectrum import periodogram

    acc = trace.ctx.acc if trace.ctx.acc is not None else trace.raw["acc"]
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=[f"ACC {axis}" for axis in "xyz"], vertical_spacing=0.08,
    )
    for row, axis in enumerate("xyz", start=1):
        spec = trim_to_fmax(periodogram(acc[row - 1], static.N), static.fs, static.N, fmax_hz)
        x = freq_axis(spec.size, static.fs, static.N, freq_unit)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=spec,
                name=f"ACC {axis}",
                mode="lines",
                line=dict(color=COLORS["acc"], width=1.4),
                customdata=hover_frequency(spec.size, static.fs, static.N),
                hovertemplate=FREQ_HOVER,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        ceiling = float(spec.max()) if spec.size else 0.0
        fig.add_hline(
            y=0.5 * ceiling,
            line=dict(color=COLORS["annotation"], width=1, dash="dot"),
            annotation_text="50 % of axis max",
            annotation_font_size=9,
            row=row,
            col=1,
        )
        add_excluded_band(
            fig, trace.ctx.prev_bin, delta_bins, static.fs, static.N, freq_unit,
            n_harmonics=n_harmonics, row=row, col=1,
        )
        add_bin_markers(
            fig, trace.ctx.acc_bins, static.fs, static.N, freq_unit,
            color=COLORS["kept"], name="kept (F_acc refined)", row=row, col=1,
        )
        dropped = set(trace.ctx.acc_bins_raw) - set(trace.ctx.acc_bins)
        add_bin_markers(
            fig, dropped, static.fs, static.N, freq_unit,
            color=COLORS["removed"], name="excluded by +/-Delta", row=row, col=1,
        )
        add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=row, col=1)

    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=3, col=1)
    return apply_layout(
        fig,
        title or f"Window {trace.idx}: accelerometer dominant frequencies",
        height or 640,
    )


def plot_temporal_diff(
    trace: WindowTrace,
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    title: str | None = None,
    height: int | None = None,
):
    """Signal and spectrum before and after the difference, plus normalisation.

    Shows what paper Section III-B claims: the harmonics become more prominent
    and the aperiodic content is suppressed.
    """
    static = _static(trace)
    payload = trace.require("temporal_diff", ["pre_normalization", "order"], "plot_temporal_diff")

    before = _signal(trace, "decomposition")
    after = _signal(trace, "temporal_diff")
    pre_norm = np.asarray(payload["pre_normalization"], dtype=np.float64)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "before difference (time)", "before difference (spectrum)",
            f"after order-{payload['order']} difference (time)", "after difference (spectrum)",
        ],
        vertical_spacing=0.14, horizontal_spacing=0.08,
    )
    t_before = trace.t_start_s + np.arange(before.size) / static.fs
    t_after = trace.t_start_s + np.arange(after.size) / static.fs

    fig.add_trace(line_trace(t_before, before, "cleansed PPG", COLORS["decomposition"]), row=1, col=1)
    fig.add_trace(
        line_trace(t_after, pre_norm, "differenced", COLORS["temporal_diff"], opacity=0.5),
        row=2, col=1,
    )
    fig.add_trace(
        line_trace(t_after, after, "differenced, normalised", COLORS["temporal_diff"], width=1.6),
        row=2, col=1,
    )

    for row, stage in ((1, "decomposition"), (2, "temporal_diff")):
        spec = trim_to_fmax(_spectrum(trace, stage), static.fs, static.N, fmax_hz)
        x = freq_axis(spec.size, static.fs, static.N, freq_unit)
        fig.add_trace(
            go.Scatter(
                x=x, y=spec, name=STAGE_LABELS[stage], mode="lines",
                line=dict(color=stage_color(stage), width=1.4),
                customdata=hover_frequency(spec.size, static.fs, static.N),
                hovertemplate=FREQ_HOVER, showlegend=False,
            ),
            row=row, col=2,
        )
        add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=row, col=2)

    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=2, col=2)
    return apply_layout(
        fig, title or f"Window {trace.idx}: temporal difference", height or 620
    )


def plot_decomposition_components(
    trace: WindowTrace,
    *,
    top: int = 12,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    title: str | None = None,
    height: int | None = None,
):
    """Every component of any decomposition, kept green and removed red.

    Reads only the generic ``diag`` keys, so a user-written plug-in that fills
    them, such as an EMD variant, gets this plot with no extra code
    (Lab Spec Section 5.2 group D).
    """
    static = _static(trace)
    payload = trace.require(
        "decomposition",
        ["components", "component_labels", "component_dominant_bins", "removed_mask"],
        "plot_decomposition_components",
    )
    components = np.asarray(payload["components"], dtype=np.float64)
    labels = list(payload["component_labels"])
    removed = np.asarray(payload["removed_mask"], dtype=bool)
    dominant = np.asarray(payload["component_dominant_bins"], dtype=np.int64)

    from troika.preprocessing.spectrum import periodogram

    energy = (components**2).sum(axis=1)
    order = np.argsort(-energy)[: int(top)]

    fig = make_subplots(
        rows=len(order), cols=2,
        subplot_titles=[
            item
            for i in order
            for item in (
                f"{labels[i]}{' [removed]' if removed[i] else ''}",
                f"{binmap.bin_to_bpm(int(dominant[i]), static.fs, static.N):.0f} BPM",
            )
        ],
        vertical_spacing=max(0.01, 0.35 / max(len(order), 1)),
        horizontal_spacing=0.08,
    )
    t = trace.t_start_s + np.arange(components.shape[1]) / static.fs
    for row, index in enumerate(order, start=1):
        color = COLORS["removed"] if removed[index] else COLORS["kept"]
        fig.add_trace(
            line_trace(t, components[index], labels[index], color, width=1.1),
            row=row, col=1,
        )
        spec = trim_to_fmax(periodogram(components[index], static.N), static.fs, static.N, fmax_hz)
        fig.add_trace(
            go.Scatter(
                x=freq_axis(spec.size, static.fs, static.N, freq_unit),
                y=spec, mode="lines", line=dict(color=color, width=1.1), showlegend=False,
            ),
            row=row, col=2,
        )
        if trace.ctx is not None:
            add_bin_markers(
                fig, trace.ctx.acc_bins, static.fs, static.N, freq_unit, row=row, col=2
            )
        add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=row, col=2)

    fig.update_xaxes(title_text="time (s)", row=len(order), col=1)
    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=len(order), col=2)
    kept = int((~removed).sum())
    return apply_layout(
        fig,
        title
        or f"Window {trace.idx}: {len(removed)} components, {int(removed.sum())} removed, {kept} kept",
        height or max(420, 110 * len(order)),
    )


# ------------------------------------------------- whole-recording (group G)


def plot_stage_spectrogram(
    run: RunResult,
    stage: str = "spectrum_estimator",
    *,
    normalize: str | None = "window",
    log: bool = True,
    freq_unit: str = "bpm",
    fmax_hz: float = 4.0,
    show_gt: bool = True,
    title: str | None = None,
    height: int | None = None,
):
    """Per-window stage spectra as a heatmap, with estimate and truth over it.

    Needs a light trace. ``normalize`` is ``window``, ``global`` or ``None``.
    """
    trace = _run_trace(run, "plot_stage_spectrogram")
    matrix = trace.stage_matrix(stage).astype(np.float64)
    static = trace.static

    if normalize == "window":
        peak = matrix.max(axis=1, keepdims=True)
        matrix = matrix / np.where(peak > 0, peak, 1.0)
    elif normalize == "global":
        matrix = matrix / max(matrix.max(), 1e-30)
    elif normalize is not None:
        raise ValueError(f"normalize must be 'window', 'global' or None, got {normalize!r}")

    matrix = trim_to_fmax(matrix.T, static.fs, static.N, fmax_hz).T
    z = 10.0 * np.log10(np.maximum(matrix, 1e-12)) if log else matrix
    x = np.array([w.t_start_s for w in trace.windows])
    y = freq_axis(matrix.shape[1], static.fs, static.N, freq_unit)

    fig = go.Figure(
        go.Heatmap(
            x=x, y=y, z=z.T, colorscale="Viridis",
            colorbar=dict(title="dB" if log else "power"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=run.bpm_est if freq_unit == "bpm" else run.bpm_est / 60.0,
            name="estimate", mode="lines",
            line=dict(color=COLORS["tracker"], width=2),
        )
    )
    if show_gt and run.bpm_gt is not None:
        fig.add_trace(
            go.Scatter(
                x=x[: run.bpm_gt.size],
                y=run.bpm_gt if freq_unit == "bpm" else run.bpm_gt / 60.0,
                name="true HR", mode="lines",
                line=dict(color="white", width=2, dash="dash"),
            )
        )
    add_protocol_shading(fig, trace.protocol, duration_s=float(x[-1]) if x.size else None)
    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text=freq_axis_title(freq_unit))
    return apply_layout(
        fig,
        title or f"Subject {run.subject_id}: {STAGE_LABELS.get(stage, stage)} over time",
        height or 460,
    )


def plot_stage_stitched(
    run: RunResult,
    stage: str = "bandpass",
    stitch: str = "center",
    *,
    title: str | None = None,
    height: int | None = None,
):
    """Stitch a stage's per-window output back into one time series.

    ``center`` keeps the central ``step_s`` slice of each window; ``ola`` uses a
    Hann-weighted overlap-add, which is exact at 75 % overlap. Per-window
    filtering leaves seams either way, which the subtitle says.
    """
    trace = _run_trace(run, "plot_stage_stitched")
    static = trace.static
    windows = [w for w in trace.windows if stage in w.stages]
    if not windows:
        raise KeyError(f"no window holds stage {stage!r}")

    step = int(round(static.step_s * static.fs))
    length = int(windows[-1].t_start_s * static.fs) + windows[-1].stages[stage].signal.size
    out = np.zeros(length)
    weight = np.zeros(length)

    for window in windows:
        signal = window.stages[stage].signal.astype(np.float64)
        start = int(round(window.t_start_s * static.fs))
        stop = min(start + signal.size, length)
        n = stop - start
        if stitch == "center":
            middle = (signal.size - step) // 2
            out[start + middle : start + middle + step] = signal[middle : middle + step]
            weight[start + middle : start + middle + step] = 1.0
        elif stitch == "ola":
            window_fn = np.hanning(signal.size)
            out[start:stop] += signal[:n] * window_fn[:n]
            weight[start:stop] += window_fn[:n]
        else:
            raise ValueError(f"stitch must be 'center' or 'ola', got {stitch!r}")

    stitched = np.divide(out, weight, out=np.zeros_like(out), where=weight > 0)
    t = np.arange(length) / static.fs

    fig = go.Figure()
    fig.add_trace(line_trace(t, stitched, STAGE_LABELS.get(stage, stage), stage_color(stage)))
    add_protocol_shading(fig, trace.protocol, duration_s=float(t[-1]) if t.size else None)
    note(fig, "per-window filtering leaves seams at window boundaries")
    fig.update_xaxes(title_text="time (s)")
    return apply_layout(
        fig,
        title or f"Subject {run.subject_id}: {STAGE_LABELS.get(stage, stage)}, stitched ({stitch})",
        height or 360,
    )


def _run_trace(run: RunResult, plot: str) -> RunTrace:
    if run.trace is None:
        raise KeyError(
            f"{plot} needs a light trace; re-run with "
            f"lab.run_subject(cfg, rec, trace='light')"
        )
    return run.trace
