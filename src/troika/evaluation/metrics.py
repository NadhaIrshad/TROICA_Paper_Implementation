"""Error metrics and the per-subject statistics table.

Blueprint Section 5.8 and Lab Spec Section 4.1. Paper Eqs. (19) and (20) plus
the Bland-Altman analysis and Pearson correlation of Section IV-C.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from troika.config import repo_root
from troika.types import RunResult

__all__ = [
    "error1",
    "error2",
    "bland_altman",
    "pearson",
    "per_subject_stats",
    "summarize",
    "paper_reference",
]


def _pair(est: ArrayLike, gt: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align two BPM series and drop windows where either side is missing."""
    e = np.asarray(est, dtype=np.float64).ravel()
    g = np.asarray(gt, dtype=np.float64).ravel()
    if e.shape != g.shape:
        raise ValueError(f"estimate/ground-truth length mismatch: {e.shape} vs {g.shape}")
    ok = np.isfinite(e) & np.isfinite(g)
    return e[ok], g[ok]


def error1(est: ArrayLike, gt: ArrayLike) -> float:
    """Average absolute error in BPM (paper Eq. 19)."""
    e, g = _pair(est, gt)
    if e.size == 0:
        return float("nan")
    return float(np.mean(np.abs(e - g)))


def error2(est: ArrayLike, gt: ArrayLike) -> float:
    """Average absolute error as a fraction of ground truth (paper Eq. 20).

    Returns a fraction; multiply by 100 for the percentage the paper reports.
    """
    e, g = _pair(est, gt)
    if e.size == 0:
        return float("nan")
    return float(np.mean(np.abs(e - g) / g))


def bland_altman(est: ArrayLike, gt: ArrayLike) -> dict[str, float]:
    """Bland-Altman agreement statistics (paper Section IV-C).

    The difference is ``est - gt``. The paper does not state the sign convention,
    but its mean difference of about -1.24 BPM implies estimates are slightly low
    under this convention (Blueprint Section 5.8). The limits of agreement are
    ``[mu - 1.96 sigma, mu + 1.96 sigma]``.
    """
    e, g = _pair(est, gt)
    if e.size < 2:
        nan = float("nan")
        return {"mean_diff": nan, "sd": nan, "loa_low": nan, "loa_high": nan, "n": float(e.size)}
    diff = e - g
    mu = float(np.mean(diff))
    # Population std, matching the paper's sigma = 3.07 over the pooled windows.
    sd = float(np.std(diff))
    return {
        "mean_diff": mu,
        "sd": sd,
        "loa_low": mu - 1.96 * sd,
        "loa_high": mu + 1.96 * sd,
        "n": float(e.size),
    }


def pearson(est: ArrayLike, gt: ArrayLike) -> float:
    """Pearson correlation between estimates and ground truth (paper: 0.992)."""
    e, g = _pair(est, gt)
    if e.size < 2 or np.std(e) == 0 or np.std(g) == 0:
        return float("nan")
    return float(np.corrcoef(e, g)[0, 1])


def per_subject_stats(
    est: ArrayLike, gt: ArrayLike, *, within_bpm: float = 5.0
) -> dict[str, float]:
    """Every per-subject column of the Lab Spec Section 4.1 table."""
    e, g = _pair(est, gt)
    if e.size == 0:
        return {
            "Error1_BPM": float("nan"),
            "Error2_pct": float("nan"),
            "bias_BPM": float("nan"),
            "rmse_BPM": float("nan"),
            "median_AE_BPM": float("nan"),
            "max_AE_BPM": float("nan"),
            f"within_{within_bpm:g}_pct": float("nan"),
            "n_windows": 0.0,
        }
    diff = e - g
    abs_err = np.abs(diff)
    return {
        "Error1_BPM": float(abs_err.mean()),
        "Error2_pct": float(np.mean(abs_err / g) * 100.0),
        "bias_BPM": float(diff.mean()),
        "rmse_BPM": float(np.sqrt(np.mean(diff**2))),
        "median_AE_BPM": float(np.median(abs_err)),
        "max_AE_BPM": float(abs_err.max()),
        f"within_{within_bpm:g}_pct": float(np.mean(abs_err <= within_bpm) * 100.0),
        "n_windows": float(e.size),
    }


def paper_reference(variant: str = "full") -> pd.DataFrame:
    """Table I and II values from the paper, indexed by subject.

    ``variant`` is one of ``full``, ``no_ssa``, ``fft``, ``no_verification``
    (Blueprint Sections 8.1 and 8.2).
    """
    path = repo_root() / "tests" / "reference" / "paper_tables.csv"
    table = pd.read_csv(path)
    known = sorted(table["variant"].unique())
    if variant not in known:
        raise ValueError(f"unknown variant {variant!r}; known variants are {known}")
    sub = table[table["variant"] == variant].set_index("subject")
    return sub[["Error1_BPM", "Error2_pct"]].rename(
        columns={"Error1_BPM": "Error1_paper", "Error2_pct": "Error2_paper"}
    )


def summarize(
    results: Iterable[RunResult] | Mapping[int, RunResult],
    *,
    std_ddof: int = 1,
    within_bpm: float = 5.0,
    compare_with_paper: bool = True,
    paper_variant: str = "full",
    subject_ids: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Per-subject statistics with ``mean +/- std`` and ``Pooled`` summary rows.

    Lab Spec Section 4.1. The mean row uses the sample standard deviation
    (``ddof = 1``), which is what reproduces the paper's 2.34 +/- 0.82 from the
    12 per-subject values of Table I. The pooled row aggregates every window of
    the selected subjects and carries the Bland-Altman and Pearson statistics.
    """
    runs = list(results.values()) if isinstance(results, Mapping) else list(results)
    if subject_ids is not None:
        wanted = set(int(s) for s in subject_ids)
        runs = [r for r in runs if int(r.subject_id) in wanted]
    runs.sort(key=lambda r: r.subject_id)
    if not runs:
        raise ValueError("no runs to summarise")

    within_col = f"within_{within_bpm:g}_pct"
    rows: list[dict[str, object]] = []
    pooled_est: list[NDArray[np.float64]] = []
    pooled_gt: list[NDArray[np.float64]] = []

    for run in runs:
        if run.bpm_gt is None:
            raise ValueError(f"subject {run.subject_id} has no ground truth to score against")
        stats = per_subject_stats(run.bpm_est, run.bpm_gt, within_bpm=within_bpm)
        rows.append({"subject": run.subject_id, **stats})
        e, g = _pair(run.bpm_est, run.bpm_gt)
        pooled_est.append(e)
        pooled_gt.append(g)

    table = pd.DataFrame(rows).set_index("subject")
    table["n_windows"] = table["n_windows"].astype(int)

    if compare_with_paper:
        try:
            reference = paper_reference(paper_variant)
        except (FileNotFoundError, ValueError):
            reference = None
        if reference is not None and table.index.isin(reference.index).all():
            table = table.join(reference)
            table["Error1_delta"] = table["Error1_BPM"] - table["Error1_paper"]
            table["Error2_delta"] = table["Error2_pct"] - table["Error2_paper"]

    numeric = table.select_dtypes(include=[np.number])
    ids = ", ".join(str(r.subject_id) for r in runs)
    mean_label = f"mean over subjects {ids}" if subject_ids is not None else "mean"

    summary = pd.DataFrame(index=[mean_label, "std", "Pooled"], columns=table.columns, dtype=object)
    summary.loc[mean_label, numeric.columns] = numeric.mean()
    summary.loc["std", numeric.columns] = numeric.std(ddof=std_ddof)

    est_all = np.concatenate(pooled_est)
    gt_all = np.concatenate(pooled_gt)
    pooled = per_subject_stats(est_all, gt_all, within_bpm=within_bpm)
    ba = bland_altman(est_all, gt_all)
    for key, value in pooled.items():
        if key in summary.columns:
            summary.loc["Pooled", key] = value
    summary.loc["Pooled", "n_windows"] = int(est_all.size)

    for name, value in (
        ("BA_mean_BPM", ba["mean_diff"]),
        ("BA_sd_BPM", ba["sd"]),
        ("BA_loa_low", ba["loa_low"]),
        ("BA_loa_high", ba["loa_high"]),
        ("pearson_r", pearson(est_all, gt_all)),
    ):
        table[name] = np.nan
        summary[name] = np.nan
        summary.loc["Pooled", name] = value

    out = pd.concat([table, summary])
    out.index.name = "subject"
    return out
