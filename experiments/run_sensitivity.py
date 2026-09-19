"""Sensitivity to parameter values: the paper's Fig. 10 (Blueprint Section 8.3).

On subject 5, one parameter is changed at a time from the defaults L = 400,
Delta = 10, tau = 2 and Delta_s = 16. The paper's bar chart runs 0 to 2.5 BPM,
so the claim under test is that the average absolute error stays near 2 BPM for
every value.

    python experiments/run_sensitivity.py
    python experiments/run_sensitivity.py --subjects 5 6
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from troika import lab
from troika.evaluation import metrics
from troika.evaluation.report import make_run_dir

#: Paper Section IV-D: the four parameters and the values it tried.
SWEEP: dict[str, tuple[str, list]] = {
    "L": ("decomposition.L", [100, 200, 300, 400]),
    "Delta": ("acc_dominant.exclude_delta_bins", [5, 10, 12, 15]),
    "tau": ("tracker.params.verification.tau_bins", [0, 1, 2, 3, 4]),
    "Delta_s": ("tracker.delta_s", [12, 14, 16, 20]),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_sensitivity", description="Reproduce the paper's Fig. 10."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--subjects", nargs="+", type=int, default=[5],
        help="the paper uses subject 5 only (default: %(default)s)",
    )
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

    def score(cfg) -> float:
        runs = lab.run_all(cfg, subjects, subject_ids=ids, n_jobs=args.n_jobs, progress=False)
        return float(
            np.mean([metrics.error1(r.bpm_est, r.bpm_gt) for r in runs.values()])
        )

    rows: list[dict] = []
    baseline = score(base)
    rows.append({"parameter": "paper defaults", "value": "", "Error1_BPM": baseline})
    print(f"{'setting':28} {'Error1 (BPM)':>13}")
    print(f"{'paper defaults':28} {baseline:13.2f}")

    for name, (key, values) in SWEEP.items():
        for value in values:
            cfg = base.with_overrides({key: value})
            error = score(cfg)
            rows.append({"parameter": name, "value": value, "Error1_BPM": error})
            print(f"{f'{name} = {value}':28} {error:13.2f}")

    table = pd.DataFrame(rows)
    spread = table["Error1_BPM"].max() - table["Error1_BPM"].min()
    print(
        f"\nsubjects {ids}: error ranges {table['Error1_BPM'].min():.2f} to "
        f"{table['Error1_BPM'].max():.2f} BPM, a spread of {spread:.2f}"
    )
    print(
        "The paper claims near-flat sensitivity on subject 5; a large spread here "
        "means one of these parameters matters more in this implementation."
    )

    if not args.no_write:
        out = make_run_dir(base.with_overrides({"run.name": "sensitivity"}))
        table.to_csv(out / "sensitivity.csv", index=False)
        base.to_yaml(out / "config_resolved.yaml")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
