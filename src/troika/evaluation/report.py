"""Writing a run to disk (Blueprint Section 8, Lab Spec Section 3.1).

Every experiment writes one folder under ``results/``:

===========================  =============================================
``config_resolved.yaml``     the fully resolved config that produced the run
``config_hash.txt``          its stable hash
``per_window.csv``           one row per window per subject, with diagnostics
``per_subject.csv``          the statistics table, including summary rows
``summary.json``             headline numbers plus the comparison to the paper
===========================  =============================================
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from troika.config import Config, repo_root
from troika.types import RunResult

__all__ = ["make_run_dir", "per_window_frame", "write_run", "summary_dict"]


def make_run_dir(cfg: Config, *, timestamp: str | None = None) -> Path:
    """Create and return ``results/<run name>_<timestamp>/``."""
    root = Path(cfg.results.dir)
    if not root.is_absolute():
        root = repo_root() / root
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    path = root / f"{cfg.run.name}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def per_window_frame(runs: Mapping[int, RunResult] | list[RunResult]) -> pd.DataFrame:
    """One row per window per subject, with the per-window diagnostics."""
    items = list(runs.values()) if isinstance(runs, Mapping) else list(runs)
    rows: list[dict[str, object]] = []
    for run in sorted(items, key=lambda r: r.subject_id):
        gt = run.bpm_gt
        for i, window in enumerate(run.windows):
            truth = float(gt[i]) if gt is not None and i < len(gt) else np.nan
            rows.append(
                {
                    "subject": run.subject_id,
                    "window": window.idx,
                    "t_start_s": window.t_start_s,
                    "bpm_est": window.bpm_est,
                    "bpm_gt": truth,
                    "error_BPM": window.bpm_est - truth,
                    "bin_cur": window.bin_cur,
                    "case": window.case,
                    "rule1_fired": window.rule1_fired,
                    "rule2_fired": window.rule2_fired,
                    "n_groups": window.n_groups,
                    "n_groups_removed": window.n_groups_removed,
                    "n_acc_bins": len(window.f_acc),
                    "f_acc": " ".join(str(b) for b in window.f_acc),
                    **{f"ms_{k}": v for k, v in window.timings_ms.items()},
                }
            )
    return pd.DataFrame(rows)


def summary_dict(cfg: Config, stats: pd.DataFrame, runs: Mapping[int, RunResult]) -> dict:
    """Headline numbers of a run, next to the paper's values."""
    mean_row = next((i for i in stats.index if str(i).startswith("mean")), None)
    pooled = stats.loc["Pooled"] if "Pooled" in stats.index else None

    def _value(row, column):
        if row is None or column not in stats.columns:
            return None
        value = row[column]
        return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)

    return {
        "run_name": cfg.run.name,
        "config_hash": cfg.hash(),
        "config_source": cfg.source,
        "subjects": sorted(int(s) for s in runs),
        "contaminated": any(r.contaminated for r in runs.values()),
        "n_windows_total": int(sum(r.n_windows for r in runs.values())),
        "Error1_BPM_mean": _value(stats.loc[mean_row] if mean_row else None, "Error1_BPM"),
        "Error1_BPM_std": _value(stats.loc["std"] if "std" in stats.index else None, "Error1_BPM"),
        "Error2_pct_mean": _value(stats.loc[mean_row] if mean_row else None, "Error2_pct"),
        "pooled_pearson_r": _value(pooled, "pearson_r"),
        "pooled_BA_mean_BPM": _value(pooled, "BA_mean_BPM"),
        "pooled_BA_sd_BPM": _value(pooled, "BA_sd_BPM"),
        "pooled_BA_loa": [_value(pooled, "BA_loa_low"), _value(pooled, "BA_loa_high")],
        "paper": {
            "Error1_BPM": 2.34,
            "Error1_BPM_std": 0.82,
            "Error2_pct": 1.80,
            "pearson_r": 0.992,
            "BA_sd_BPM": 3.07,
            "BA_loa": [-7.26, 4.79],
        },
    }


def write_run(
    cfg: Config,
    runs: Mapping[int, RunResult],
    stats: pd.DataFrame,
    *,
    out_dir: Path | None = None,
) -> Path:
    """Write a complete result folder and return its path."""
    path = out_dir or make_run_dir(cfg)
    path.mkdir(parents=True, exist_ok=True)

    cfg.to_yaml(path / "config_resolved.yaml")
    (path / "config_hash.txt").write_text(cfg.hash() + "\n", encoding="utf-8")
    per_window_frame(runs).to_csv(path / "per_window.csv", index=False)
    stats.to_csv(path / "per_subject.csv")
    (path / "summary.json").write_text(
        json.dumps(summary_dict(cfg, stats, runs), indent=2) + "\n", encoding="utf-8"
    )
    return path
