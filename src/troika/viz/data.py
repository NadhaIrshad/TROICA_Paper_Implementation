"""Whole-recording data plots (Lab Spec Section 5.2 group A)."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from plotly.subplots import make_subplots

from troika import binmap
from troika.types import Recording
from troika.viz.common import (
    FREQ_HOVER,
    add_protocol_shading,
    freq_axis,
    freq_axis_title,
    hover_frequency,
    line_trace,
    note,
)
from troika.viz.theme import COLORS, apply_layout

__all__ = ["plot_recording", "plot_spectrogram", "plot_all_subjects"]

_CHANNEL_COLORS = {"ppg": COLORS["bandpass"], "acc": COLORS["acc"], "ecg": COLORS["ground_truth"]}


def _welch(x: NDArray[np.float64], fs: float, nperseg: int) -> tuple:
    from scipy.signal import welch

    nperseg = int(min(nperseg, x.size))
    return welch(x, fs=fs, nperseg=nperseg)


def plot_recording(
    rec: Recording,
    channels: Sequence[str] = ("ppg", "acc", "ecg"),
    domain: str = "both",
    *,
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    nperseg: int = 4096,
    protocol_shading: bool = True,
    title: str | None = None,
    height: int | None = None,
):
    """One subject: each channel in time on the left and its spectrum on the right.

    Requires only a :class:`~troika.types.Recording`; no trace is needed. The
    spectrum is a Welch estimate over the whole recording, so it shows which
    rhythms dominate the session rather than any single window.
    """
    channels = [c for c in channels if c != "ecg" or rec.ecg is not None]
    if not channels:
        raise ValueError("no channels to plot")

    n_rows = len(channels)
    n_cols = 2 if domain == "both" else 1
    titles: list[str] = []
    for channel in channels:
        if domain in ("both", "time"):
            titles.append(f"{channel.upper()} (time)")
        if domain in ("both", "freq"):
            titles.append(f"{channel.upper()} (spectrum)")

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=titles,
        shared_xaxes=False,
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    t = np.arange(rec.n_samples) / rec.fs
    for row, channel in enumerate(channels, start=1):
        signals = _channel_signals(rec, channel)
        for name, y in signals:
            color = _CHANNEL_COLORS.get(channel, COLORS["raw"])
            if domain in ("both", "time"):
                fig.add_trace(
                    line_trace(t, y, name, color, opacity=0.9), row=row, col=1
                )
            if domain in ("both", "freq"):
                freqs, power = _welch(y, rec.fs, nperseg)
                mask = freqs <= fmax_hz
                x = freqs[mask] * (60.0 if freq_unit == "bpm" else 1.0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=power[mask],
                        name=name,
                        mode="lines",
                        line=dict(color=color, width=1.3),
                        showlegend=False,
                    ),
                    row=row,
                    col=n_cols,
                )
        if protocol_shading and domain in ("both", "time"):
            add_protocol_shading(fig, rec.protocol, row=row, col=1, duration_s=rec.duration_s)

    for row in range(1, n_rows + 1):
        if domain in ("both", "time"):
            fig.update_xaxes(title_text="time (s)" if row == n_rows else None, row=row, col=1)
        if domain in ("both", "freq"):
            fig.update_xaxes(
                title_text=freq_axis_title(freq_unit) if row == n_rows else None,
                row=row,
                col=n_cols,
            )

    return apply_layout(
        fig,
        title or f"Subject {rec.subject_id}" + (f" ({rec.protocol})" if rec.protocol else ""),
        height or 260 * n_rows,
    )


def _channel_signals(rec: Recording, channel: str) -> list[tuple[str, NDArray[np.float64]]]:
    if channel == "ppg":
        if rec.ppg_all is not None and rec.ppg_all.shape[0] > 1:
            return [(f"PPG {i + 1}", rec.ppg_all[i]) for i in range(rec.ppg_all.shape[0])]
        return [("PPG", rec.ppg)]
    if channel == "acc":
        return [(f"ACC {axis}", rec.acc[i]) for i, axis in enumerate("xyz")]
    if channel == "ecg":
        return [("ECG", rec.ecg)] if rec.ecg is not None else []
    raise ValueError(f"unknown channel {channel!r}; use 'ppg', 'acc' or 'ecg'")


def plot_spectrogram(
    rec: Recording,
    channel: str = "ppg",
    *,
    nperseg: int = 1000,
    overlap: int = 750,
    freq_unit: str = "bpm",
    fmax_hz: float = 4.0,
    log: bool = True,
    show_gt: bool = True,
    title: str | None = None,
    height: int | None = None,
):
    """Time-frequency heatmap of one channel, with the true heart rate over it.

    This is the quickest way to see the motion-artifact ridge and the heart-rate
    ridge crossing, which is the situation the whole framework exists for.
    """
    from scipy.signal import spectrogram as _spectrogram

    signals = _channel_signals(rec, channel)
    y = signals[0][1]
    freqs, times, power = _spectrogram(
        y, fs=rec.fs, nperseg=int(nperseg), noverlap=int(overlap), nfft=4096
    )
    mask = freqs <= fmax_hz
    z = power[mask]
    if log:
        z = 10.0 * np.log10(np.maximum(z, 1e-20))
    axis = freqs[mask] * (60.0 if freq_unit == "bpm" else 1.0)

    fig = go.Figure(
        go.Heatmap(
            x=times,
            y=axis,
            z=z,
            colorscale="Viridis",
            colorbar=dict(title="dB" if log else "power"),
        )
    )

    if show_gt and rec.bpm_gt is not None:
        gt_t = np.arange(rec.bpm_gt.size) * 2.0 + 4.0  # window centres at T=8, S=2
        gt_y = rec.bpm_gt if freq_unit == "bpm" else rec.bpm_gt / 60.0
        fig.add_trace(
            go.Scatter(
                x=gt_t,
                y=gt_y,
                name="true HR",
                mode="lines",
                line=dict(color="white", width=2, dash="dash"),
            )
        )
    elif show_gt:
        note(fig, "no ground truth for this recording")

    fig.update_xaxes(title_text="time (s)")
    fig.update_yaxes(title_text=freq_axis_title(freq_unit))
    return apply_layout(
        fig,
        title or f"Subject {rec.subject_id}: {channel.upper()} spectrogram",
        height or 420,
    )


def plot_all_subjects(
    recs: Mapping[int, Recording],
    channel: str = "ppg",
    domain: str = "time",
    *,
    layout: str = "grid",
    freq_unit: str = "hz",
    fmax_hz: float = 6.0,
    nperseg: int = 4096,
    title: str | None = None,
    height: int | None = None,
):
    """All subjects at once: a 4x3 grid, or one overlay with a legend per subject."""
    ids = sorted(recs)
    if layout == "overlay":
        fig = go.Figure()
        for subject_id in ids:
            rec = recs[subject_id]
            y = _channel_signals(rec, channel)[0][1]
            if domain == "time":
                t = np.arange(rec.n_samples) / rec.fs
                fig.add_trace(line_trace(t, y, f"S{subject_id}", COLORS["bandpass"], opacity=0.7))
            else:
                freqs, power = _welch(y, rec.fs, nperseg)
                mask = freqs <= fmax_hz
                fig.add_trace(
                    go.Scatter(
                        x=freqs[mask] * (60.0 if freq_unit == "bpm" else 1.0),
                        y=power[mask],
                        name=f"S{subject_id}",
                        mode="lines",
                    )
                )
        fig.update_xaxes(
            title_text="time (s)" if domain == "time" else freq_axis_title(freq_unit)
        )
        return apply_layout(fig, title or f"All subjects: {channel.upper()}", height or 500)

    rows, cols = 4, 3
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"S{i}" for i in ids],
        shared_xaxes=True,
        shared_yaxes=(domain != "spectrogram"),
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
    )
    for index, subject_id in enumerate(ids):
        row, col = index // cols + 1, index % cols + 1
        rec = recs[subject_id]
        y = _channel_signals(rec, channel)[0][1]
        if domain == "time":
            t = np.arange(rec.n_samples) / rec.fs
            fig.add_trace(
                line_trace(t, y, f"S{subject_id}", COLORS["bandpass"], max_points=1500),
                row=row,
                col=col,
            )
        elif domain == "freq":
            freqs, power = _welch(y, rec.fs, nperseg)
            mask = freqs <= fmax_hz
            fig.add_trace(
                go.Scatter(
                    x=freqs[mask] * (60.0 if freq_unit == "bpm" else 1.0),
                    y=power[mask],
                    name=f"S{subject_id}",
                    mode="lines",
                    line=dict(color=COLORS["bandpass"], width=1.2),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
        elif domain == "spectrogram":
            from scipy.signal import spectrogram as _spectrogram

            freqs, times, power = _spectrogram(y, fs=rec.fs, nperseg=500, noverlap=375, nfft=2048)
            mask = freqs <= fmax_hz
            fig.add_trace(
                go.Heatmap(
                    x=times,
                    y=freqs[mask] * (60.0 if freq_unit == "bpm" else 1.0),
                    z=10.0 * np.log10(np.maximum(power[mask], 1e-20)),
                    colorscale="Viridis",
                    showscale=False,
                ),
                row=row,
                col=col,
            )
        else:
            raise ValueError(f"unknown domain {domain!r}; use time, freq or spectrogram")

    return apply_layout(
        fig, title or f"All subjects: {channel.upper()} ({domain})", height or 900
    )


_ = (binmap, freq_axis, hover_frequency, FREQ_HOVER)  # used by sibling modules
