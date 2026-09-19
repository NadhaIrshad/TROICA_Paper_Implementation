"""Shared helpers for the plotting layer (Lab Spec Section 5.1).

Frequency axes, ground-truth markers, decimation, protocol shading and the
"missing data" annotation all live here so every plot behaves the same way.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from troika import binmap
from troika.viz.theme import COLORS, SPEED_SEGMENTS

__all__ = [
    "freq_axis",
    "freq_axis_title",
    "hover_frequency",
    "add_gt_lines",
    "add_bin_markers",
    "add_excluded_band",
    "add_protocol_shading",
    "decimate",
    "note",
    "trim_to_fmax",
]


def freq_axis(
    n_bins: int, fs: float, n_fft: int, unit: str = "hz"
) -> NDArray[np.float64]:
    """X values for a spectrum of ``n_bins`` bins, in Hz or BPM."""
    bins = np.arange(n_bins)
    if unit == "hz":
        return binmap.bin_to_hz(bins, fs, n_fft)
    if unit == "bpm":
        return binmap.bin_to_bpm(bins, fs, n_fft)
    raise ValueError(f"freq_unit must be 'hz' or 'bpm', got {unit!r}")


def freq_axis_title(unit: str = "hz") -> str:
    """Axis label matching :func:`freq_axis`."""
    return "frequency (Hz)" if unit == "hz" else "heart rate (BPM)"


def hover_frequency(n_bins: int, fs: float, n_fft: int) -> NDArray[np.float64]:
    """Custom data giving Hz, BPM and bin index for every bin.

    Lab Spec Section 5.1: the hover always shows all three, whichever unit the
    axis is in.
    """
    bins = np.arange(n_bins)
    return np.stack(
        [binmap.bin_to_hz(bins, fs, n_fft), binmap.bin_to_bpm(bins, fs, n_fft), bins],
        axis=-1,
    )


#: Hover template matching :func:`hover_frequency`.
FREQ_HOVER = (
    "%{customdata[0]:.3f} Hz &middot; %{customdata[1]:.1f} BPM &middot; "
    "bin %{customdata[2]:.0f}<br>power %{y:.3g}<extra>%{fullData.name}</extra>"
)


def trim_to_fmax(
    spectrum: NDArray[np.float64], fs: float, n_fft: int, fmax_hz: float | None
) -> NDArray[np.float64]:
    """Cut a spectrum at ``fmax_hz``; ``None`` keeps everything."""
    if fmax_hz is None:
        return spectrum
    hi = min(int(binmap.hz_to_bin(fmax_hz, fs, n_fft)), spectrum.size - 1)
    return spectrum[: hi + 1]


def add_gt_lines(
    fig,
    gt_bpm: float | None,
    fs: float,
    n_fft: int,
    unit: str = "hz",
    *,
    row: int | None = None,
    col: int | None = None,
    harmonics: int = 2,
) -> None:
    """Mark the true heart rate, and its harmonics more faintly.

    Degrades to doing nothing when there is no ground truth, which is the
    graceful path the spec asks for.
    """
    if gt_bpm is None or not np.isfinite(gt_bpm):
        return
    for harmonic in range(1, int(harmonics) + 1):
        bpm = float(gt_bpm) * harmonic
        x = bpm if unit == "bpm" else bpm / 60.0
        fig.add_vline(
            x=x,
            line=dict(color=COLORS["ground_truth"], width=1.5 if harmonic == 1 else 1,
                      dash="dash"),
            opacity=1.0 if harmonic == 1 else 0.35,
            annotation_text="true HR" if harmonic == 1 else f"{harmonic}x",
            annotation_position="top right",
            annotation_font_size=10,
            row=row,
            col=col,
        )


def add_bin_markers(
    fig,
    bins: Iterable[int],
    fs: float,
    n_fft: int,
    unit: str = "hz",
    *,
    color: str | None = None,
    name: str = "acc dominant",
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Draw a vertical line per bin, e.g. the accelerometer's dominant bins."""
    bins = sorted(set(int(b) for b in bins))
    if not bins:
        return
    color = color or COLORS["acc"]
    for i, b in enumerate(bins):
        x = binmap.bin_to_bpm(b, fs, n_fft) if unit == "bpm" else binmap.bin_to_hz(b, fs, n_fft)
        fig.add_vline(
            x=float(x),
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.8,
            annotation_text=name if i == 0 else None,
            annotation_position="bottom right",
            annotation_font_size=10,
            row=row,
            col=col,
        )


def add_excluded_band(
    fig,
    prev_bin: int | None,
    delta: int,
    fs: float,
    n_fft: int,
    unit: str = "hz",
    *,
    n_harmonics: int = 2,
    row: int | None = None,
    col: int | None = None,
) -> None:
    """Shade the +/-Delta zone kept out of ``F_acc`` (paper Section III-A)."""
    if prev_bin is None:
        return
    for harmonic in range(1, int(n_harmonics) + 1):
        centre = int(prev_bin) * harmonic
        lo, hi = centre - int(delta), centre + int(delta)
        if lo > n_fft // 2:
            break
        to_x = binmap.bin_to_bpm if unit == "bpm" else binmap.bin_to_hz
        fig.add_vrect(
            x0=float(to_x(max(lo, 0), fs, n_fft)),
            x1=float(to_x(hi, fs, n_fft)),
            fillcolor=COLORS["excluded"],
            line_width=0,
            layer="below",
            row=row,
            col=col,
        )


def add_protocol_shading(
    fig,
    protocol: str | None,
    *,
    row: int | None = None,
    col: int | None = None,
    duration_s: float | None = None,
) -> None:
    """Shade the nominal speed segments of a recording (DEVIATION D2).

    The boundaries are nominal, so the labels say so.
    """
    segments = SPEED_SEGMENTS.get(str(protocol or "").upper())
    if not segments:
        return
    for i, (start, stop, label) in enumerate(segments):
        if duration_s is not None and start > duration_s:
            break
        fig.add_vrect(
            x0=start,
            x1=min(stop, duration_s) if duration_s else stop,
            fillcolor="rgba(0,0,0,0.05)" if i % 2 else "rgba(0,0,0,0.0)",
            line_width=0,
            layer="below",
            annotation_text=f"{label} (nominal)" if i % 2 == 0 else label,
            annotation_position="top left",
            annotation_font_size=9,
            row=row,
            col=col,
        )


def decimate(
    x: NDArray[np.float64], y: NDArray[np.float64], max_points: int = 4000
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Min/max envelope decimation for long time series (Lab Spec Section 5.1).

    Keeps the visual extent of the signal, unlike plain subsampling, which would
    hide the peaks that matter in a PPG trace.
    """
    n = int(y.size)
    if n <= max_points:
        return x, y
    buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    xs: list[float] = []
    ys: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi <= lo:
            continue
        chunk = y[lo:hi]
        lo_i = int(lo + np.argmin(chunk))
        hi_i = int(lo + np.argmax(chunk))
        for idx in sorted((lo_i, hi_i)):
            xs.append(float(x[idx]))
            ys.append(float(y[idx]))
    return np.asarray(xs), np.asarray(ys)


def line_trace(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    name: str,
    color: str,
    *,
    width: float = 1.4,
    dash: str | None = None,
    opacity: float = 1.0,
    max_points: int = 4000,
    customdata: NDArray[np.float64] | None = None,
    hovertemplate: str | None = None,
) -> go.Scattergl | go.Scatter:
    """A decimated line, using WebGL above 10k points (Lab Spec Section 5.1)."""
    x_d, y_d = (x, y) if customdata is not None else decimate(np.asarray(x), np.asarray(y), max_points)
    cls = go.Scattergl if y_d.size > 10_000 else go.Scatter
    return cls(
        x=x_d,
        y=y_d,
        name=name,
        mode="lines",
        line=dict(color=color, width=width, dash=dash),
        opacity=opacity,
        customdata=customdata,
        hovertemplate=hovertemplate,
    )


def note(fig, text: str, *, row: int | None = None, col: int | None = None) -> None:
    """Annotate a panel with a message instead of raising (Lab Spec Section 5.1)."""
    fig.add_annotation(
        text=text,
        showarrow=False,
        xref="paper" if row is None else "x domain",
        yref="paper" if row is None else "y domain",
        x=0.5,
        y=0.5,
        font=dict(color=COLORS["annotation"], size=12),
        row=row,
        col=col,
    )


_ = Sequence, Any  # used in type hints of the modules importing from here
