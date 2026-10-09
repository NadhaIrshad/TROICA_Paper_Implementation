"""Evaluate the HMM tracker, leave-one-subject-out, on models already trained.

The HMM (DEVIATIONS.md D12) has no weights of its own, so nothing is fitted
here. With a learned scorer, each subject is scored by the model that was
trained without it, as saved by that scorer's ``--loso`` run, once behind the
scorer's own jump guard and once behind the HMM. ``--scorer spectrum`` uses the
candidates' SSR power and needs no model.

Examples:
    python experiments/eval_hmm_tracker.py
    python experiments/eval_hmm_tracker.py --scorer efficientnet1d
    python experiments/eval_hmm_tracker.py --scorer mlp --set tracker.jump_prob=0.001
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from troika import lab
from troika.evaluation.report import make_run_dir

#: Where each scorer's ``--loso`` run saves its fold models, and their suffix.
MODEL_DIRS = {
    "xgboost": ("models/loso", ".json"),
    "efficientnet1d": ("models/loso_efficientnet1d", ".pt"),
    "mlp": ("models/loso_mlp", ".pt"),
}
PREFIX = {"spectrum": "spectrum", "xgboost": "xgb", "efficientnet1d": "effnet", "mlp": "mlp"}


def error1(run_cfg, rec) -> float:
    run = lab.run_subject(run_cfg, rec)
    return float(np.mean(np.abs(run.bpm_est - run.bpm_gt)))


def variants(args, held_out: int) -> dict[str, dict]:
    """Column name to config overrides for one held-out subject."""
    prefix = PREFIX[args.scorer]
    if args.scorer == "spectrum":
        return {f"{prefix}_hmm": {"tracker.scorer": "spectrum"}}
    directory, suffix = MODEL_DIRS[args.scorer]
    model_dir = Path(args.models_dir or directory)
    out: dict[str, dict] = {}
    for label, iteration in (("", 0), ("_rollouts", args.rollout_iters)):
        path = str(model_dir / f"held_out_{held_out:02d}_iter{iteration}{suffix}")
        out[f"{prefix}_guard{label}"] = {"tracker.method": args.scorer, "tracker.model_path": path}
        out[f"{prefix}_hmm{label}"] = {"tracker.scorer": args.scorer, "tracker.model_path": path}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Leave-one-subject-out Error1 of the HMM tracker")
    parser.add_argument("--config", default="configs/hmm_tracker.yaml")
    parser.add_argument("--scorer", choices=tuple(PREFIX), default="spectrum")
    parser.add_argument("--models-dir", default=None, help="folder of held_out_XX_iterN models; default is where the scorer's --loso run saves them")
    parser.add_argument("--rollout-iters", type=int, default=2, help="roll-out iteration of the final models, as in the training run")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8, help="subject runs in parallel")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides for the HMM, e.g. tracker.jump_prob=0.001")
    args = parser.parse_args()

    cfg = lab.load_config(args.config).with_overrides(args.set)
    subjects = lab.load_subjects(cfg, args.subjects)
    ids = sorted(subjects)

    columns = {h: variants(args, h) for h in ids}
    missing = sorted({o["tracker.model_path"] for v in columns.values() for o in v.values() if "tracker.model_path" in o and not Path(o["tracker.model_path"]).is_file()})
    if missing:
        raise SystemExit(f"run the {args.scorer} training script with --loso first; missing models: {missing}")

    jobs = [(h, "troika", {"tracker.method": "troika"}) for h in ids]
    jobs += [(h, name, overrides) for h in ids for name, overrides in columns[h].items()]
    print(f"{len(jobs)} subject runs ...", flush=True)
    with Parallel(n_jobs=args.jobs) as pool:
        out = pool(delayed(error1)(cfg.with_overrides({**overrides, "run.n_jobs": 1}), subjects[h]) for h, _, overrides in jobs)

    table = pd.DataFrame(index=ids, columns=["troika", *columns[ids[0]]], dtype=float)
    for (h, name, _), value in zip(jobs, out):
        table.loc[h, name] = value
    table.index.name = "held_out_subject"
    table.loc["mean"] = table.loc[ids].mean()
    table.loc["std"] = table.loc[ids].std(ddof=1)

    out_dir = make_run_dir(cfg.with_overrides({"run.name": f"hmm_{args.scorer}_loso"}))
    table.to_csv(out_dir / "loso_error1.csv")
    if args.scorer == "spectrum":
        print("\nError1 (BPM); the spectrum scorer is not trained, so no subject needs holding out")
    else:
        print(f"\nLeave-one-subject-out Error1 (BPM); every {args.scorer} model is scored on a subject it never saw")
    print(table.round(2).to_string())
    print(f"wrote {out_dir / 'loso_error1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
