"""SSA internals (Lab Spec Section 5.2 group C). All need a full trace."""

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
    line_trace,
    trim_to_fmax,
)
from troika.viz.theme import COLORS, apply_layout

__all__ = [
    "plot_ssa_embedding",
    "plot_ssa_svd",
    "plot_ssa_eigenvectors",
    "plot_ssa_grouping",
    "plot_ssa_wcorr",
    "plot_ssa_components",
    "plot_ssa_reconstruction",
]


def _ssa(trace: WindowTrace, keys: list[str], plot: str) -> dict:
    return trace.require("decomposition", keys, plot)


def plot_ssa_embedding(
    trace: WindowTrace, zoom: int = 8, *, title: str | None = None, height: int | None = None
):
    """The L x K trajectory matrix, with a numbered corner showing its structure.

    Paper Eq. 2: ``Y[i, j] = y[i + j]``, so every anti-diagonal is constant. The
    inset makes that visible, which is the whole point of the embedding step.
    """
    payload = _ssa(trace, ["trajectory", "L", "K"], "plot_ssa_embedding")
    Y = np.asarray(payload["trajectory"], dtype=np.float64)

    fig = make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        subplot_titles=[
            f"trajectory matrix ({payload['L']} x {payload['K']})",
            f"top-left {zoom}x{zoom} block (constant anti-diagonals)",
        ],
        horizontal_spacing=0.1,
    )
    fig.add_trace(
        go.Heatmap(z=Y, colorscale="RdBu", zmid=0, colorbar=dict(title="value")),
        row=1, col=1,
    )
    block = Y[:zoom, :zoom]
    fig.add_trace(
        go.Heatmap(
            z=block, colorscale="RdBu", zmid=0, showscale=False,
            text=np.round(block, 2), texttemplate="%{text}", textfont=dict(size=8),
        ),
        row=1, col=2,
    )
    fig.update_yaxes(autorange="reversed", title_text="i (row)", row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_xaxes(title_text="j (column)", row=1, col=1)
    return apply_layout(fig, title or f"Window {trace.idx}: SSA embedding", height or 440)


def plot_ssa_svd(trace: WindowTrace, *, title: str | None = None, height: int | None = None):
    """Singular values on a log axis and the cumulative energy they carry.

    Near-equal adjacent pairs are marked, because a sinusoid produces exactly
    such a pair and that is what the default grouping strategy looks for.
    """
    payload = _ssa(trace, ["s"], "plot_ssa_svd")
    s = np.asarray(payload["s"], dtype=np.float64)
    index = np.arange(s.size)

    scale = np.maximum(s[:-1], 1e-300)
    pairs = np.flatnonzero(np.abs(s[:-1] - s[1:]) / scale < 0.1)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=index, y=s, name="singular values", mode="lines+markers",
            marker=dict(size=3), line=dict(color=COLORS["decomposition"], width=1.2),
        ),
        secondary_y=False,
    )
    if pairs.size:
        fig.add_trace(
            go.Scatter(
                x=pairs, y=s[pairs], name="near-equal pair", mode="markers",
                marker=dict(size=8, color=COLORS["tracker"], symbol="circle-open"),
            ),
            secondary_y=False,
        )
    total = s.sum()
    fig.add_trace(
        go.Scatter(
            x=index, y=np.cumsum(s) / (total if total > 0 else 1.0),
            name="cumulative energy", mode="lines",
            line=dict(color=COLORS["annotation"], width=1.2, dash="dot"),
        ),
        secondary_y=True,
    )
    fig.update_yaxes(type="log", title_text="sigma_i", secondary_y=False)
    fig.update_yaxes(title_text="cumulative fraction", range=[0, 1.02], secondary_y=True)
    fig.update_xaxes(title_text="eigentriple index")
    return apply_layout(fig, title or f"Window {trace.idx}: SSA singular values", height or 420)


def plot_ssa_eigenvectors(
    trace: WindowTrace,
    idx: range | list[int] = range(12),
    *,
    fmax_hz: float = 6.0,
    freq_unit: str = "hz",
    title: str | None = None,
    height: int | None = None,
):
    """Left singular vectors with their spectra, and the pair scatter.

    A sinusoid makes ``u_i`` against ``u_{i+1}`` trace a circle, which is the
    clearest visual test that two eigentriples belong together.
    """
    payload = _ssa(trace, ["U", "s"], "plot_ssa_eigenvectors")
    U = np.asarray(payload["U"], dtype=np.float64)
    static = trace.static
    chosen = [i for i in list(idx) if i < U.shape[1]]
    if not chosen:
        raise ValueError("no eigenvector indices inside the decomposition")

    from troika.preprocessing.spectrum import periodogram

    fig = make_subplots(
        rows=len(chosen), cols=3,
        subplot_titles=[
            item for i in chosen
            for item in (f"u_{i}", f"spectrum of u_{i}", f"u_{i} vs u_{i + 1}")
        ],
        vertical_spacing=max(0.01, 0.3 / len(chosen)), horizontal_spacing=0.06,
    )
    for row, i in enumerate(chosen, start=1):
        fig.add_trace(
            line_trace(np.arange(U.shape[0]), U[:, i], f"u_{i}", COLORS["decomposition"]),
            row=row, col=1,
        )
        spec = trim_to_fmax(periodogram(U[:, i], static.N), static.fs, static.N, fmax_hz)
        fig.add_trace(
            go.Scatter(
                x=freq_axis(spec.size, static.fs, static.N, freq_unit), y=spec,
                mode="lines", line=dict(color=COLORS["decomposition"], width=1.1),
                showlegend=False,
            ),
            row=row, col=2,
        )
        if i + 1 < U.shape[1]:
            fig.add_trace(
                go.Scatter(
                    x=U[:, i], y=U[:, i + 1], mode="lines",
                    line=dict(color=COLORS["temporal_diff"], width=1.0), showlegend=False,
                ),
                row=row, col=3,
            )
    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=len(chosen), col=2)
    return apply_layout(
        fig, title or f"Window {trace.idx}: SSA eigenvectors", height or 130 * len(chosen) + 120
    )


def plot_ssa_grouping(trace: WindowTrace, *, title: str | None = None, height: int | None = None):
    """Singular values coloured by group, with a table describing each group."""
    payload = _ssa(
        trace,
        ["s", "groups", "group_dominant_bins", "group_singular_range", "removed_mask"],
        "plot_ssa_grouping",
    )
    s = np.asarray(payload["s"], dtype=np.float64)
    groups = payload["groups"]
    dominant = np.asarray(payload["group_dominant_bins"], dtype=np.int64)
    removed = np.asarray(payload["removed_mask"], dtype=bool)
    static = trace.static

    fig = make_subplots(
        rows=2, cols=1, row_heights=[0.45, 0.55],
        specs=[[{"type": "scatter"}], [{"type": "table"}]],
        subplot_titles=["singular values by group", "groups"], vertical_spacing=0.12,
    )
    for g, members in enumerate(groups):
        members = np.asarray(members, dtype=int)
        fig.add_trace(
            go.Scatter(
                x=members, y=s[members], mode="markers",
                marker=dict(
                    size=6,
                    color=COLORS["removed"] if removed[g] else COLORS["kept"],
                    symbol="x" if removed[g] else "circle",
                ),
                name=f"group {g}", showlegend=False,
                hovertemplate=(
                    f"group {g}<br>members %{{x}}<br>sigma %{{y:.3g}}<br>"
                    f"{binmap.bin_to_bpm(int(dominant[g]), static.fs, static.N):.0f} BPM"
                    f"<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    order = np.argsort(-np.array([s[np.asarray(m, dtype=int)].max() for m in groups]))[:25]
    fig.add_trace(
        go.Table(
            header=dict(
                values=["group", "members", "sigma range", "dominant bin", "Hz", "BPM", "removed"],
                fill_color="rgba(0,0,0,0.05)", align="left", font=dict(size=11),
            ),
            cells=dict(
                values=[
                    [int(g) for g in order],
                    [str(groups[g])[:28] for g in order],
                    [f"{payload['group_singular_range'][g][0]:.3g} .. "
                     f"{payload['group_singular_range'][g][1]:.3g}" for g in order],
                    [int(dominant[g]) for g in order],
                    [f"{binmap.bin_to_hz(int(dominant[g]), static.fs, static.N):.3f}" for g in order],
                    [f"{binmap.bin_to_bpm(int(dominant[g]), static.fs, static.N):.1f}" for g in order],
                    ["yes" if removed[g] else "no" for g in order],
                ],
                align="left", font=dict(size=10), height=20,
            ),
        ),
        row=2, col=1,
    )
    fig.update_yaxes(type="log", title_text="sigma_i", row=1, col=1)
    fig.update_xaxes(title_text="eigentriple index", row=1, col=1)
    return apply_layout(
        fig,
        title or f"Window {trace.idx}: grouping ({payload.get('grouping_strategy', '')})",
        height or 780,
    )


def plot_ssa_wcorr(
    trace: WindowTrace, *, top: int = 40, title: str | None = None, height: int | None = None
):
    """Weighted-correlation matrix of the elementary reconstructions.

    Computed on demand from the stored eigentriples, which Lab Spec Section 11
    allows as a cheap derived view.
    """
    payload = _ssa(trace, ["U", "s", "Vt", "L", "K", "groups"], "plot_ssa_wcorr")
    from troika.decomposition.ssa import elementary_series, wcorr

    U = np.asarray(payload["U"])
    s = np.asarray(payload["s"])
    Vt = np.asarray(payload["Vt"])
    M = int(payload["L"]) + int(payload["K"]) - 1
    n = min(int(top), s.size)

    series = elementary_series(U, s, Vt, M, range(n))
    W = np.abs(wcorr(series, int(payload["L"]), int(payload["K"])))

    fig = go.Figure(
        go.Heatmap(z=W, colorscale="Greys", zmin=0, zmax=1, colorbar=dict(title="|w-corr|"))
    )
    boundary = 0
    for members in payload["groups"]:
        boundary += len(members)
        if boundary >= n:
            break
        fig.add_hline(y=boundary - 0.5, line=dict(color=COLORS["tracker"], width=1))
        fig.add_vline(x=boundary - 0.5, line=dict(color=COLORS["tracker"], width=1))
    fig.update_yaxes(autorange="reversed", title_text="eigentriple")
    fig.update_xaxes(title_text="eigentriple")
    return apply_layout(
        fig, title or f"Window {trace.idx}: w-correlation (first {n})", height or 560
    )


def plot_ssa_components(
    trace: WindowTrace,
    top: int = 12,
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    title: str | None = None,
    height: int | None = None,
):
    """Each group's series and spectrum, kept green and removed red.

    The SSA-flavoured sibling of ``plot_decomposition_components``.
    """
    from troika.viz.stages import plot_decomposition_components

    return plot_decomposition_components(
        trace, top=top, freq_unit=freq_unit, fmax_hz=fmax_hz, title=title, height=height
    )


def plot_ssa_reconstruction(
    trace: WindowTrace,
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    title: str | None = None,
    height: int | None = None,
):
    """Original against cleansed and removed, with the reconstruction residual.

    The residual is ``sum of all groups - original`` and must be about zero;
    seeing it is the quickest check that the decomposition is intact.
    """
    payload = _ssa(trace, ["components", "removed_mask"], "plot_ssa_reconstruction")
    components = np.asarray(payload["components"], dtype=np.float64)
    removed = np.asarray(payload["removed_mask"], dtype=bool)
    static = trace.static

    original = components.sum(axis=0)
    cleansed = components[~removed].sum(axis=0) if (~removed).any() else np.zeros_like(original)
    dropped = components[removed].sum(axis=0) if removed.any() else np.zeros_like(original)
    band_passed = trace.stages["bandpass"].signal.astype(np.float64)
    residual = float(np.abs(original - band_passed).max())

    from troika.preprocessing.spectrum import periodogram

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            f"time domain (max |sum of groups - input| = {residual:.2e})",
            "spectrum",
        ],
        vertical_spacing=0.14,
    )
    t = trace.t_start_s + np.arange(original.size) / static.fs
    for name, y, color, width in (
        ("band-passed input", band_passed, COLORS["bandpass"], 1.0),
        ("cleansed (kept)", cleansed, COLORS["kept"], 1.6),
        ("removed sum", dropped, COLORS["removed"], 1.0),
    ):
        fig.add_trace(line_trace(t, y, name, color, width=width), row=1, col=1)
        spec = trim_to_fmax(periodogram(y, static.N), static.fs, static.N, fmax_hz)
        fig.add_trace(
            go.Scatter(
                x=freq_axis(spec.size, static.fs, static.N, freq_unit), y=spec,
                name=name, mode="lines", line=dict(color=color, width=width),
                customdata=hover_frequency(spec.size, static.fs, static.N),
                hovertemplate=FREQ_HOVER, showlegend=False,
            ),
            row=2, col=1,
        )
    if trace.ctx is not None:
        add_bin_markers(fig, trace.ctx.acc_bins, static.fs, static.N, freq_unit, row=2, col=1)
    add_gt_lines(fig, trace.gt_bpm, static.fs, static.N, freq_unit, row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_xaxes(title_text=freq_axis_title(freq_unit), row=2, col=1)
    return apply_layout(fig, title or f"Window {trace.idx}: SSA reconstruction", height or 620)
