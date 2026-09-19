"""One subject in detail, with debug figures (Blueprint Section 8).

    python experiments/run_subject.py --subject 5
    python experiments/run_subject.py --subject 6 --windows 20 40 --open

Writes the per-window diagnostics and a set of interactive HTML figures, which
is the fastest way to see where a subject goes wrong without a notebook.
"""

from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from troika import lab, viz
from troika.evaluation import metrics
from troika.evaluation.report import make_run_dir, per_window_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_subject", description="Run one subject and write debug figures."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subject", type=int, default=5)
    parser.add_argument(
        "--windows", nargs="*", type=int, default=None,
        help="windows to trace fully; default is the worst and the median window",
    )
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--open", action="store_true", help="open the figures in a browser")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = lab.load_config(args.config)
    if args.overrides:
        cfg = cfg.with_overrides(args.overrides)

    subjects = lab.load_subjects(cfg, [args.subject])
    rec = subjects[args.subject]
    run = lab.run_subject(cfg, rec, trace="light")

    error = np.abs(run.bpm_est - run.bpm_gt) if run.bpm_gt is not None else None
    if args.windows is not None:
        windows = list(args.windows)
    elif error is not None:
        order = np.argsort(error)
        windows = [int(np.nanargmax(error)), int(order[len(order) // 2])]
    else:
        windows = [len(run.windows) // 2]

    print(f"subject {args.subject}: {run.n_windows} windows")
    if run.bpm_gt is not None:
        print(f"  Error1 {metrics.error1(run.bpm_est, run.bpm_gt):.2f} BPM")
        print(f"  Error2 {metrics.error2(run.bpm_est, run.bpm_gt) * 100:.2f} %")
        print(f"  Pearson r {metrics.pearson(run.bpm_est, run.bpm_gt):.4f}")
    cases = pd.Series([w.case for w in run.windows]).value_counts(dropna=False)
    print(f"  cases: {cases.to_dict()}")
    print(
        f"  rule 1 fired {sum(w.rule1_fired for w in run.windows)} times, "
        f"rule 2 {sum(w.rule2_fired for w in run.windows)} times"
    )

    out = make_run_dir(cfg.with_overrides({"run.name": f"subject{args.subject}"}))
    per_window_frame({args.subject: run}).to_csv(out / "per_window.csv", index=False)
    cfg.to_yaml(out / "config_resolved.yaml")

    figures = {
        "recording": viz.plot_recording(rec),
        "spectrogram": viz.plot_spectrogram(rec),
        "tracking_run": viz.plot_tracking_run(run),
        "verification": viz.plot_verification_timeline(run),
        "stage_spectrogram": viz.plot_stage_spectrogram(run),
    }
    for window in windows:
        trace = lab.get_window_trace(rec, cfg, window)
        figures[f"w{window}_overview"] = viz.plot_pipeline_overview(trace)
        figures[f"w{window}_tracking"] = viz.plot_tracking_window(trace)
        figures[f"w{window}_acc"] = viz.plot_acc_dominant(trace)
        figures[f"w{window}_ssa"] = viz.plot_ssa_reconstruction(trace)
        figures[f"w{window}_ssr"] = viz.plot_ssr_spectrum(trace)

    for name, figure in figures.items():
        path = viz.save(figure, out / f"{name}.html")
        if args.open and name.endswith("overview"):
            webbrowser.open(path.as_uri())

    print(f"\nwrote {out} with {len(figures)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
