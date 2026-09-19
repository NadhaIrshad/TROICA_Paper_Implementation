"""Main experiment: run TROIKA over the dataset and write a result folder.

Thin command line over ``troika.lab`` (Lab Spec Section 4), so modes 1 and 2 of
Section 1 are the same code path and only the config differs.

    python experiments/run_all.py --config configs/default.yaml
    python experiments/run_all.py --config configs/default.yaml --subjects 5 6
    python experiments/run_all.py --set decomposition.L=300 --set bandpass.method=fir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from troika import lab
from troika.evaluation.report import write_run


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the main experiment."""
    parser = argparse.ArgumentParser(
        prog="run_all",
        description="Estimate heart rate with TROIKA and report the paper's metrics.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="config file merged over configs/default.yaml (default: %(default)s)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="subject numbers to run; default is whatever run.subjects says",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, e.g. --set decomposition.L=300 (repeatable)",
    )
    parser.add_argument(
        "--trace",
        choices=["none", "light"],
        default=None,
        help="trace level; light is needed for whole-recording plots",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None, help="parallel workers across subjects"
    )
    parser.add_argument(
        "--paper-variant",
        default="full",
        choices=["full", "no_ssa", "fft", "no_verification"],
        help="which row of the paper's tables to compare against",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="print the results without writing a folder"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the experiment and print the statistics table."""
    args = build_parser().parse_args(argv)

    cfg = lab.load_config(args.config)
    if args.overrides:
        cfg = cfg.with_overrides(args.overrides)
    if args.trace is not None:
        cfg = cfg.with_overrides({"run.trace.level": args.trace})

    print(f"config: {args.config}  hash {cfg.hash()}")
    changed = cfg.diff_from_default()
    if changed:
        print(f"differs from default: {changed}")

    subjects = lab.load_subjects(cfg, args.subjects)
    print(f"loaded {len(subjects)} subject(s): {sorted(subjects)}")

    runs = lab.run_all(
        cfg, subjects, subject_ids=args.subjects, trace=args.trace, n_jobs=args.n_jobs
    )
    stats = lab.evaluate(
        runs, cfg, subject_ids=args.subjects, paper_variant=args.paper_variant
    )

    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print()
        print(stats.to_string(float_format=lambda v: f"{v:.2f}"))

    if not args.no_write:
        out = write_run(cfg, runs, stats)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
