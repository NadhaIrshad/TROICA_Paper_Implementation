"""Export selected XGBoost-TROIKA windows in the paper's stage order.

Each exported ``stages.npz`` follows:
raw -> bandpass -> SSA -> temporal difference -> SSR -> XGBoost tracking.
Only explicitly requested windows are fully traced, avoiding the very large
SSA diagnostic payload of recording every window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from troika import lab
from troika.evaluation.report import make_run_dir
from troika.tracking.xgboost_features import FEATURE_NAMES, feature_matrix


def _arrays(trace):
    """Return the paper-ordered numerical arrays for one full window trace."""
    stages = trace.stages
    tracker = trace.diag["tracker"]
    candidates = np.asarray(tracker.get("candidates", []), dtype=np.int64)
    features = feature_matrix(
        trace.raw["spectrum"], trace.ctx, candidates,
        trace.static.fs, trace.static.N,
    )
    return {
        "raw_ppg": trace.raw["ppg"],
        "raw_acc": trace.raw["acc"],
        "bandpass_ppg": stages["bandpass"].signal,
        "bandpass_periodogram": stages["bandpass"].spectrum,
        "ssa_cleansed_ppg": stages["decomposition"].signal,
        "ssa_periodogram": stages["decomposition"].spectrum,
        "temporal_difference": stages["temporal_diff"].signal,
        "temporal_difference_periodogram": stages["temporal_diff"].spectrum,
        "ssr_spectrum": trace.raw["spectrum"],
        "candidate_bins": candidates,
        "candidate_scores": np.asarray(tracker.get("scores", []), dtype=float),
        "candidate_features": features,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export paper-ordered TROIKA stage outputs")
    parser.add_argument("--config", default="configs/xgboost_tracker.yaml")
    parser.add_argument("--subjects", nargs="+", type=int, default=[1])
    parser.add_argument("--windows", nargs="+", type=int, default=[20, 40])
    parser.add_argument("--all-windows", action="store_true", help="export every window; can be slow and large")
    args = parser.parse_args()

    cfg = lab.load_config(args.config)
    subjects = lab.load_subjects(cfg, args.subjects)
    out = make_run_dir(cfg.with_overrides({"run.name": f"{cfg.run.name}_stage_export"}))
    cfg.to_yaml(out / "config_resolved.yaml")
    rows = []

    for subject_id in args.subjects:
        rec = subjects[subject_id]
        if args.all_windows:
            from troika.preprocessing.windowing import n_windows
            windows = list(range(n_windows(rec.n_samples, cfg.signal.fs, cfg.signal.window_s, cfg.signal.step_s)))
        else:
            windows = list(args.windows)
        for window in windows:
            trace = lab.get_window_trace(rec, cfg, window)
            folder = out / "stages" / f"subject_{subject_id:02d}" / f"window_{window:03d}"
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "stages.npz", **_arrays(trace))
            metadata = {
                "subject_id": subject_id,
                "window": window,
                "t_start_s": trace.t_start_s,
                "ground_truth_bpm": trace.gt_bpm,
                "estimated_bpm": trace.bpm_est,
                "estimated_bin": trace.bin_cur,
                "feature_names": list(FEATURE_NAMES),
                "stage_order": ["raw", "bandpass", "ssa", "temporal_difference", "ssr", "xgboost_tracking"],
            }
            (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            rows.append(metadata)
            print(f"exported subject {subject_id:02d}, window {window:03d}: {folder}")

    pd.DataFrame(rows).to_csv(out / "export_manifest.csv", index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
