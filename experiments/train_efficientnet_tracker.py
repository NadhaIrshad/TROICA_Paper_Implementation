"""Train the TinyEfficientNet1D final-stage tracker from SSR spectra produced by TROIKA.

Same protocol as ``train_xgboost_tracker.py``, with the network in place of the
XGBoost classifier. Training rows come from two sources, aggregated:

* iteration 0: the paper tracker's trajectory;
* iterations 1..n: roll-outs of the EfficientNet1D tracker itself, relabelled
  against the ECG, so the model sees the states its own mistakes lead to.

Each row is one SSR peak candidate: a spectrum crop centred on it for the
convolutions, and the XGBoost feature row as ``aux``.

Examples:
    python experiments/train_efficientnet_tracker.py
    python experiments/train_efficientnet_tracker.py --loso
    python experiments/train_efficientnet_tracker.py --loso --subjects 5 7 9 --rollout-iters 1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from troika import lab
from troika.evaluation.report import make_run_dir
from troika.tracking.xgboost_features import (
    FEATURE_NAMES,
    candidate_bins,
    feature_matrix,
    label_candidates,
)


def collect(run_cfg, feature_cfg, rec, half_width: int) -> dict:
    """Run one subject and return its labelled candidate rows and its Error1.

    Truth is used only here, after the run, to label the rows; it is never
    passed into the pipeline. Rows are built from the light trace, whose SSR
    spectrum is float32 and cut at 7 Hz; the HR band, its second harmonic and
    the crop around any candidate all sit below that, and the cut part is
    zero-padded back.
    """
    from troika.tracking.efficientnet1d import candidate_crops

    run = lab.run_subject(run_cfg, rec, trace="light")
    assert run.trace is not None and run.bpm_gt is not None
    fs, n_fft = feature_cfg.signal.fs, feature_cfg.grid.n_fft
    params = feature_cfg.tracker.params
    crops, rows, labels, covered = [], [], [], []
    for trace in run.trace.windows[1:]:  # window 0 is fixed max-peak initialisation
        spectrum = np.zeros(n_fft // 2 + 1)
        light = trace.stages["spectrum_estimator"].spectrum
        spectrum[: light.size] = light
        candidates = candidate_bins(spectrum, fs, n_fft, params.low_bpm, params.high_bpm, params.candidate_count)
        y, ok = label_candidates(candidates, float(trace.gt_bpm), fs, n_fft)
        covered.append(ok)
        if ok:
            crops.append(candidate_crops(spectrum, candidates, half_width))
            rows.append(feature_matrix(spectrum, trace.ctx, candidates, fs, n_fft))
            labels.append(y)
    return {
        "crops": np.vstack(crops) if crops else np.zeros((0, 2 * half_width + 1)),
        "x": np.vstack(rows) if rows else np.zeros((0, len(FEATURE_NAMES))),
        "y": np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64),
        "covered": int(np.sum(covered)),
        "windows": len(covered),
        "error1": float(np.mean(np.abs(run.bpm_est - run.bpm_gt))),
    }


def error1(run_cfg, rec) -> float:
    run = lab.run_subject(run_cfg, rec)
    return float(np.mean(np.abs(run.bpm_est - run.bpm_gt)))


def fit_and_save(parts: list[dict], path: Path, args) -> None:
    try:
        import torch
        import torch.nn.functional as F

        from troika.tracking.efficientnet1d import CandidateScorer
    except ImportError as exc:
        raise SystemExit("Install PyTorch first: pip install -e '.[dl]'") from exc
    crops = np.vstack([p["crops"] for p in parts])
    aux = np.vstack([p["x"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    if not len(y):
        raise RuntimeError("no labelled candidate rows; check the dataset and candidate band")

    torch.manual_seed(args.seed)  # weight initialisation
    shuffle = torch.Generator().manual_seed(args.seed)
    scorer = CandidateScorer.untrained(args.half_width, aux)
    x = torch.from_numpy(crops)
    a = scorer.standardise(aux)
    t = torch.from_numpy(y.astype(np.float64))
    optimiser = torch.optim.AdamW(scorer.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scorer.model.train()
    for _ in range(args.epochs):
        for batch in torch.randperm(len(t), generator=shuffle).split(args.batch_size):
            optimiser.zero_grad()
            loss = F.binary_cross_entropy_with_logits(scorer.model(x[batch], a[batch]), t[batch])
            loss.backward()
            optimiser.step()
    scorer.model.eval()
    scorer.save(path)


def train_folds(cfg, subjects, folds: dict[str, list[int]], model_dir: Path, args, pool) -> tuple[dict, dict]:
    """Fit one model per fold; ``folds`` maps a fold name to its training subjects.

    Returns ``(paths, baseline)``: ``paths[name][i]`` is the model after ``i``
    roll-out iterations, and ``baseline[s]`` is the paper tracker's result on
    subject ``s``, which is the same for every fold and so computed once.
    """
    baseline_cfg = cfg.with_overrides({"tracker.method": "troika", "run.n_jobs": 1})
    ids = sorted({s for train in folds.values() for s in train})
    print(f"paper-tracker rows for {len(ids)} subjects ...", flush=True)
    baseline = dict(zip(ids, pool(delayed(collect)(baseline_cfg, cfg, subjects[s], args.half_width) for s in ids)))

    parts = {name: [baseline[s] for s in train] for name, train in folds.items()}
    paths: dict[str, list[Path]] = {name: [] for name in folds}
    for iteration in range(args.rollout_iters + 1):
        print(f"fitting {len(folds)} model(s), iteration {iteration} ...", flush=True)
        for name in folds:
            path = model_dir / f"{name}_iter{iteration}.pt"
            fit_and_save(parts[name], path, args)
            paths[name].append(path)
        if iteration == args.rollout_iters:
            break
        jobs = [(name, s) for name, train in folds.items() for s in train]
        print(f"roll-out {iteration + 1}/{args.rollout_iters}: {len(jobs)} subject runs ...", flush=True)
        out = pool(
            delayed(collect)(
                cfg.with_overrides({"tracker.model_path": str(paths[name][-1]), "run.n_jobs": 1}),
                cfg, subjects[s], args.half_width,
            )
            for name, s in jobs
        )
        for (name, _), part in zip(jobs, out):
            parts[name].append(part)
    return paths, baseline


def run_loso(cfg, subjects, ids, args, pool) -> None:
    folds = {f"held_out_{h:02d}": [s for s in ids if s != h] for h in ids}
    paths, baseline = train_folds(cfg, subjects, folds, Path("models") / "loso_efficientnet1d", args, pool)

    variants = {
        "effnet_plain": lambda p: {"tracker.model_path": str(p[0]), "tracker.jump_patience": 1},
        "effnet_guard": lambda p: {"tracker.model_path": str(p[0])},
        "effnet_guard_rollouts": lambda p: {"tracker.model_path": str(p[-1])},
    }
    jobs = [(h, v) for h in ids for v in variants]
    print(f"held-out evaluation: {len(jobs)} subject runs ...", flush=True)
    out = pool(
        delayed(error1)(
            cfg.with_overrides({**variants[v](paths[f"held_out_{h:02d}"]), "run.n_jobs": 1}),
            subjects[h],
        )
        for h, v in jobs
    )
    table = pd.DataFrame({"troika": {h: baseline[h]["error1"] for h in ids}})
    for (h, v), value in zip(jobs, out):
        table.loc[h, v] = value
    table.index.name = "held_out_subject"
    table.loc["mean"] = table.loc[ids].mean()
    table.loc["std"] = table.loc[ids].std(ddof=1)

    out_dir = make_run_dir(cfg.with_overrides({"run.name": "efficientnet1d_loso"}))
    table.to_csv(out_dir / "loso_error1.csv")
    print("\nLeave-one-subject-out Error1 (BPM); every EfficientNet1D number is on a subject the model never saw")
    print(table.round(2).to_string())
    print(f"wrote {out_dir / 'loso_error1.csv'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train TROIKA's TinyEfficientNet1D final-stage tracker")
    parser.add_argument("--config", default="configs/efficientnet1d_tracker.yaml")
    parser.add_argument("--model", default="models/efficientnet1d_tracker.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--rollout-iters", type=int, default=2, help="retraining rounds on the tracker's own roll-outs; 0 = paper-tracker rows only")
    parser.add_argument("--jobs", type=int, default=8, help="subject runs in parallel")
    parser.add_argument("--loso", action="store_true", help="fit one model per held-out subject and report their Error1")
    parser.add_argument("--half-width", type=int, default=32, help="spectrum bins either side of a candidate that the network reads")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = lab.load_config(args.config)
    subjects = lab.load_subjects(cfg, args.subjects)
    ids = sorted(subjects)
    with Parallel(n_jobs=args.jobs) as pool:
        if args.loso:
            run_loso(cfg, subjects, ids, args, pool)
            return 0
        path = Path(args.model)
        paths, baseline = train_folds(cfg, subjects, {path.stem: ids}, path.parent / "rollouts", args, pool)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(paths[path.stem][-1].read_bytes())
    covered = sum(b["covered"] for b in baseline.values())
    windows = sum(b["windows"] for b in baseline.values())
    print(f"saved {path} after {args.rollout_iters} roll-out iteration(s)")
    print(f"candidate coverage within 5 BPM on the paper tracker's spectra: {covered / windows:.1%}")
    print("aux features:", ", ".join(FEATURE_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
