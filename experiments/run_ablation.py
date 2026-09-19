"""Ablations: Tables I and II rows 2-4, and Fig. 7 (Blueprint Section 8.2).

Runs the full framework and the paper's three ablations over the same subjects
and prints one table per variant next to the paper's values.

    python experiments/run_ablation.py
    python experiments/run_ablation.py --subjects 5 6

Qualitatively, each ablation should produce some catastrophic failure while the
full pipeline does not. Reproducing that pattern matters more than matching
every cell, since the paper's own cells come from a heuristic parameter set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from troika import lab
from troika.evaluation import metrics
from troika.evaluation.report import make_run_dir

#: Ablation name, config overrides, and which paper row it corresponds to.
VARIANTS: dict[str, tuple[dict, str]] = {
    "full (SSA + FOCUSS + Vrf)": ({}, "full"),
    "without SSA (FOCUSS + Vrf)": ({"decomposition.method": "none"}, "no_ssa"),
    "FFT instead of SSR (SSA + FFT + Vrf)": (
        {"spectrum_estimator.method": "fft"},
        "fft",
    ),
    "without verification (SSA + FOCUSS)": (
        {"tracker.params.verification.enabled": False},
        "no_verification",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_ablation", description="Reproduce the paper's ablation tables."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--no-write", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    base = lab.load_config(args.config)
    if args.overrides:
        base = base.with_overrides(args.overrides)
    subjects = lab.load_subjects(base, args.subjects)
    ids = sorted(subjects)
    print(f"ablations over subjects {ids}\n")

    error1 = pd.DataFrame(index=list(VARIANTS), columns=ids, dtype=float)
    error2 = pd.DataFrame(index=list(VARIANTS), columns=ids, dtype=float)
    paper1 = pd.DataFrame(index=list(VARIANTS), columns=ids, dtype=float)

    for label, (overrides, paper_variant) in VARIANTS.items():
        cfg = base.with_overrides(overrides) if overrides else base
        runs = lab.run_all(cfg, subjects, subject_ids=ids, n_jobs=args.n_jobs)
        reference = metrics.paper_reference(paper_variant)
        for subject_id, run in runs.items():
            error1.loc[label, subject_id] = metrics.error1(run.bpm_est, run.bpm_gt)
            error2.loc[label, subject_id] = metrics.error2(run.bpm_est, run.bpm_gt) * 100.0
            if subject_id in reference.index:
                paper1.loc[label, subject_id] = reference.loc[subject_id, "Error1_paper"]
        failures = int((error1.loc[label] > 10).sum())
        print(
            f"{label:40} mean {error1.loc[label].mean():6.2f} BPM, "
            f"{failures} subject(s) above 10 BPM"
        )

    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print("\nError1 (BPM), this implementation")
        print(error1.to_string(float_format=lambda v: f"{v:.2f}"))
        print("\nError1 (BPM), paper Table I")
        print(paper1.to_string(float_format=lambda v: f"{v:.2f}"))
        print("\nError2 (%), this implementation")
        print(error2.to_string(float_format=lambda v: f"{v:.2f}"))

    print("\nCatastrophic failures (Error1 above 10 BPM), ours against the paper:")
    for label in VARIANTS:
        ours = sorted(int(s) for s in error1.columns if error1.loc[label, s] > 10)
        theirs = sorted(int(s) for s in paper1.columns if paper1.loc[label, s] > 10)
        print(f"  {label:40} ours {ours or 'none'}   paper {theirs or 'none'}")

    if not args.no_write:
        out = make_run_dir(base.with_overrides({"run.name": "ablation"}))
        error1.to_csv(out / "error1.csv")
        error2.to_csv(out / "error2.csv")
        paper1.to_csv(out / "error1_paper.csv")
        base.to_yaml(out / "config_resolved.yaml")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
