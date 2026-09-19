"""One-at-a-time assumption sweep (scratch driver for M8)."""
from __future__ import annotations
import sys, time
import numpy as np
from joblib import Parallel, delayed
from troika.config import load_config
from troika.io import loader
from troika.pipeline import TroikaEstimator
from troika.evaluation import metrics

BASE = load_config()
SUBS = dict(loader.list_subjects(BASE))

def _one(cfg, sid):
    rec = loader.load_recording(SUBS[sid], sid, cfg)
    run = TroikaEstimator(cfg).run(rec)
    return sid, metrics.error1(run.bpm_est, run.bpm_gt)

def evaluate(label, overrides, n_jobs=6):
    cfg = BASE.with_overrides(overrides) if overrides else BASE
    t0 = time.perf_counter()
    res = dict(Parallel(n_jobs=n_jobs)(delayed(_one)(cfg, s) for s in range(1, 13)))
    v = np.array([res[s] for s in range(1, 13)])
    print(f"{label:34} mean={v.mean():6.2f} std={v.std(ddof=1):5.2f} "
          f"fail>10={int((v > 10).sum())} worst=S{int(np.argmax(v)) + 1}:{v.max():6.1f} "
          f"[{time.perf_counter() - t0:.0f}s]", flush=True)
    print("   " + " ".join(f"{res[s]:5.1f}" for s in range(1, 13)), flush=True)
    return v

if __name__ == "__main__":
    print("subjects:            " + " ".join(f"S{s:<4d}" for s in range(1, 13)), flush=True)
    paper = metrics.paper_reference("full")["Error1_paper"]
    print("paper Table I row 1: " + " ".join(f"{paper[s]:5.2f}" for s in range(1, 13)), flush=True)
    print(flush=True)
    P = {"acc_dominant.threshold_domain": "power"}
    evaluate("power, n_harm=2 (baseline)", {**P, "acc_dominant.n_harmonics": 2})
    evaluate("power, n_harm=1", {**P, "acc_dominant.n_harmonics": 1})
    evaluate("power, n_harm=1, tol=0", {**P, "acc_dominant.n_harmonics": 1,
                                        "decomposition.match_tol_bins": 0})
    evaluate("power, n_harm=1, tol=5", {**P, "acc_dominant.n_harmonics": 1,
                                        "decomposition.match_tol_bins": 5})
    evaluate("power, n_harm=1, rel_thr=0.3", {**P, "acc_dominant.n_harmonics": 1,
                                              "acc_dominant.rel_threshold": 0.3})
    evaluate("power, n_harm=1, delta=5", {**P, "acc_dominant.n_harmonics": 1,
                                          "acc_dominant.exclude_delta_bins": 5})
    evaluate("power, n_harm=1, A3 no norm", {**P, "acc_dominant.n_harmonics": 1,
                                             "temporal_diff.normalize_after": False})
    evaluate("power, n_harm=1, A15 ch2", {**P, "acc_dominant.n_harmonics": 1,
                                          "signal.ppg_channel": 1})
    evaluate("power, n_harm=1, A4 wcorr", {**P, "acc_dominant.n_harmonics": 1,
                                           "decomposition.grouping": "wcorr_hclust"})
    evaluate("power, n_harm=1, A8 closest", {**P, "acc_dominant.n_harmonics": 1,
                                             "tracker.pair_tiebreak": "closest_to_prev"})
