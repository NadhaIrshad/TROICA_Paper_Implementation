"""Illustrative figures in the style of the paper (Blueprint Section 8.4).

    python experiments/make_figures.py

Figure 1: a synthetic heart rate and hand swing at close frequencies, resolved
by sparse reconstruction and merged by the periodogram.
Figures 2 and 3: real windows where the heart-rate peak is buried or absent.
Figure 9: raw PPG, its periodogram, the PPG after SSA, and the sparse spectrum,
with the true heart rate marked from the ECG.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from troika import binmap, lab, viz
from troika.evaluation.report import make_run_dir
from troika.preprocessing.spectrum import local_maxima, periodogram
from troika.ssr.basis import get_dictionary
from troika.ssr.focuss import focuss
from troika.viz.theme import COLORS, apply_layout

FS = 125.0
N = 4096


def figure_one(hr_hz: float = 2.18, ma_hz: float = 2.10, ma_gain: float = 3.0):
    """Paper Fig. 1: leakage merges two close peaks that sparse reconstruction separates."""
    m = 998
    t = np.arange(m) / FS
    signal = np.sin(2 * np.pi * hr_hz * t) + ma_gain * np.sin(2 * np.pi * ma_hz * t)
    signal = (signal - signal.mean()) / signal.std()

    Phi, G, bins = get_dictionary(m, N, FS)
    coefficients = focuss(signal, Phi, G, 0.8, 0.1, 5)
    half = N // 2
    sparse = np.zeros(half + 1)
    positive = bins <= half
    np.add.at(sparse, bins[positive], np.abs(coefficients[positive]) ** 2)
    fft = periodogram(signal, N)

    lo, hi = binmap.band_to_bins(0.0, 4.5, FS, N)
    x = binmap.bin_to_hz(np.arange(lo, hi + 1), FS, N)

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=[
            f"(a) band-passed signal: heartbeat at {hr_hz} Hz under a "
            f"{ma_gain:g}x hand swing at {ma_hz} Hz",
            "(b) periodogram: the two peaks are not separated",
            "(c) sparse reconstruction: the two peaks are separated",
        ],
        vertical_spacing=0.11,
    )
    fig.add_trace(
        go.Scatter(x=t[:800], y=signal[:800], mode="lines",
                   line=dict(color=COLORS["bandpass"], width=1.2), showlegend=False),
        row=1, col=1,
    )
    for row, (spec, color) in enumerate(((fft, COLORS["raw"]), (sparse, COLORS["spectrum_estimator"])), start=2):
        fig.add_trace(
            go.Scatter(x=x, y=spec[lo : hi + 1], mode="lines",
                       line=dict(color=color, width=1.4), showlegend=False),
            row=row, col=1,
        )
        fig.add_vline(
            x=hr_hz, line=dict(color=COLORS["ground_truth"], width=1.4, dash="dash"),
            annotation_text="heart rate", annotation_font_size=10, row=row, col=1,
        )
        peaks = local_maxima(spec, lo, hi)
        peaks = peaks[np.argsort(-spec[peaks])][:2]
        fig.add_trace(
            go.Scatter(
                x=binmap.bin_to_hz(peaks, FS, N), y=spec[peaks], mode="markers",
                marker=dict(size=11, symbol="circle-open", color=COLORS["tracker"],
                            line=dict(width=2)),
                showlegend=False,
            ),
            row=row, col=1,
        )
    fig.update_xaxes(title_text="time (s)", row=1, col=1)
    fig.update_xaxes(title_text="frequency (Hz)", row=3, col=1)
    return apply_layout(fig, "Fig. 1 style: leakage against sparse reconstruction", 760)


def buried_peak_figure(rec, cfg, window: int, label: str):
    """Paper Figs. 2 and 3: a real window whose heart-rate peak is buried or absent."""
    trace = lab.get_window_trace(rec, cfg, window)
    return viz.plot_pipeline_overview(
        trace,
        title=f"{label}: subject {rec.subject_id}, window {window} "
        f"(estimate {trace.bpm_est:.0f} BPM, truth {trace.gt_bpm:.0f} BPM)",
    )


def figure_nine(rec, cfg, window: int):
    """Paper Fig. 9: what SSA does to a window with strong motion artifact."""
    trace = lab.get_window_trace(rec, cfg, window)
    return viz.plot_ssa_reconstruction(
        trace,
        title=f"Fig. 9 style: subject {rec.subject_id}, window {window}, "
        f"PPG before and after SSA",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="make_figures", description="Illustrative figures in the paper's style."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = lab.load_config(args.config)
    if args.overrides:
        cfg = cfg.with_overrides(args.overrides)

    out = make_run_dir(cfg.with_overrides({"run.name": "figures"}))
    figures = {"fig1_leakage_vs_ssr": figure_one()}

    try:
        subjects = lab.load_subjects(cfg, [args.subject])
        rec = subjects[args.subject]
        run = lab.run_subject(cfg, rec, trace="light")
        error = np.abs(run.bpm_est - run.bpm_gt)
        worst = int(np.nanargmax(error))
        median = int(np.argsort(error)[len(error) // 2])

        figures["fig2_buried_peak"] = buried_peak_figure(rec, cfg, worst, "Fig. 2 style, buried peak")
        figures["fig3_clean_window"] = buried_peak_figure(rec, cfg, median, "Fig. 3 style, typical window")
        figures["fig9_ssa_benefit"] = figure_nine(rec, cfg, worst)
        figures["fig8_tracking"] = viz.plot_tracking_run(run)
    except Exception as exc:
        print(f"skipping the real-data figures: {exc}")

    for name, figure in figures.items():
        viz.save(figure, out / f"{name}.html")
    print(f"wrote {out} with {len(figures)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
