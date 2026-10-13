"""EEMD + spectral subtraction + MLP scorer + HMM tracker, leave-one-subject-out.

The four parts are DEVIATIONS.md D13 (EEMD), D14 (spectral subtraction), D11
(MLP) and D12 (HMM). The MLP models of ``train_mlp_tracker.py`` were fitted to
FOCUSS spectra and do not apply here, so this script fits its own, to rows from
the paper tracker running on the new spectra (the first round of
``train_mlp_tracker.py``; there are no roll-out rounds). Each subject is then
scored by the model trained without it, behind the jump guard and behind the
HMM.

Nothing is shared with the FOCUSS runs: the models go to
``models/loso_eemd_ss_mlp/`` and the table to
``results/eemd_ss_mlp_hmm_loso_<timestamp>/``.

``--config`` with ``--name`` runs the same experiment on another pair of
stages, with its models and table under that name instead.

Examples:
    python experiments/train_eemd_ss_mlp_hmm.py
    python experiments/train_eemd_ss_mlp_hmm.py --subjects 5 7 9 --set decomposition.ensemble_size=20
    python experiments/train_eemd_ss_mlp_hmm.py --eval-only
    python experiments/train_eemd_ss_mlp_hmm.py --config configs/ssa_ss.yaml --name ssa_ss
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

def collect(run_cfg, feature_cfg, rec) -> dict:
    """Run one subject and return its labelled candidate rows and its Error1.

    Truth is used only here, after the run, to label the rows; it is never
    passed into the pipeline. Rows are built from the light trace, whose
    spectrum is float32 and cut at 7 Hz; the HR band and its second harmonic
    sit below that, and the cut part is zero-padded back.
    """
    run = lab.run_subject(run_cfg, rec, trace="light")
    assert run.trace is not None and run.bpm_gt is not None
    fs, n_fft = feature_cfg.signal.fs, feature_cfg.grid.n_fft
    params = feature_cfg.tracker.params
    rows, labels, covered = [], [], []
    for trace in run.trace.windows[1:]:  # window 0 is fixed max-peak initialisation
        spectrum = np.zeros(n_fft // 2 + 1)
        light = trace.stages["spectrum_estimator"].spectrum
        spectrum[: light.size] = light
        candidates = candidate_bins(spectrum, fs, n_fft, params.low_bpm, params.high_bpm, params.candidate_count)
        y, ok = label_candidates(candidates, float(trace.gt_bpm), fs, n_fft)
        covered.append(ok)
        if ok:
            rows.append(feature_matrix(spectrum, trace.ctx, candidates, fs, n_fft))
            labels.append(y)
    return {
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

        from troika.tracking.mlp import MLPScorer
    except ImportError as exc:
        raise SystemExit("Install PyTorch first: pip install -e '.[dl]'") from exc
    aux = np.vstack([p["x"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    if not len(y):
        raise RuntimeError("no labelled candidate rows; check the dataset and candidate band")

    # Only the fit runs on args.device; the weights are initialised and saved on
    # the CPU, where the tracker scores candidates inside the subject workers.
    device = torch.device(args.device)
    torch.manual_seed(args.seed)  # weight initialisation
    shuffle = torch.Generator().manual_seed(args.seed)
    scorer = MLPScorer.untrained(aux)
    a = scorer.standardise(aux).to(device)
    t = torch.from_numpy(y.astype(np.float64)).to(device)
    scorer.model.to(device)
    optimiser = torch.optim.AdamW(scorer.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scorer.model.train()
    for _ in range(args.epochs):
        for batch in torch.randperm(len(t), generator=shuffle).split(args.batch_size):
            batch = batch.to(device)
            optimiser.zero_grad()
            loss = F.binary_cross_entropy_with_logits(scorer.model(a[batch]), t[batch])
            loss.backward()
            optimiser.step()
    scorer.model.eval().cpu()
    scorer.save(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="EEMD + spectral subtraction + MLP + HMM, leave-one-subject-out")
    parser.add_argument("--config", default="configs/eemd_ss.yaml", help="the two new stages; the tracker is set by this script")
    parser.add_argument("--name", default="eemd_ss", help="names the output: models/loso_<name>_mlp/ and results/<name>_mlp_hmm_loso_<timestamp>/")
    parser.add_argument("--candidate-count", type=int, default=8, help="strongest peaks scored per window, in training and in tracking; 88 covers every peak of the band")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8, help="subject runs in parallel")
    parser.add_argument("--eval-only", action="store_true", help="skip the fitting and evaluate the models a previous run saved")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides for every run, e.g. decomposition.ensemble_size=50")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu", help="where the MLP is fitted, e.g. cpu, cuda or cuda:1")
    args = parser.parse_args()

    # `stages` keeps the paper tracker of the config; the other two swap the tracker only.
    stages = lab.load_config(args.config).with_overrides([*args.set, "run.n_jobs=1"])
    count = {"tracker.candidate_count": args.candidate_count}
    mlp_cfg = stages.with_overrides({"tracker.method": "mlp", **count})
    hmm_cfg = stages.with_overrides({"tracker.method": "hmm", "tracker.scorer": "mlp", **count})
    subjects = lab.load_subjects(stages, args.subjects)
    ids = sorted(subjects)
    model_dir = Path("models") / f"loso_{args.name}_mlp"
    paths = {h: model_dir / f"held_out_{h:02d}.pt" for h in ids}

    with Parallel(n_jobs=args.jobs) as pool:
        if args.eval_only:
            missing = [str(p) for p in paths.values() if not p.exists()]
            if missing:
                raise SystemExit(f"--eval-only needs the models of a finished training run; missing: {missing}")
            print(f"reusing {len(paths)} saved models; paper-tracker runs for {len(ids)} subjects ...", flush=True)
            troika = dict(zip(ids, pool(delayed(error1)(stages, subjects[s]) for s in ids)))
        else:
            print(f"paper-tracker rows for {len(ids)} subjects ...", flush=True)
            rows = dict(zip(ids, pool(delayed(collect)(stages, mlp_cfg, subjects[s]) for s in ids)))
            troika = {s: rows[s]["error1"] for s in ids}
            covered = sum(r["covered"] for r in rows.values())
            windows = sum(r["windows"] for r in rows.values())
            print(f"candidate coverage within 5 BPM of the truth: {covered / windows:.1%}", flush=True)
            print(f"fitting {len(ids)} model(s) ...", flush=True)
            for h in ids:
                fit_and_save([rows[s] for s in ids if s != h], paths[h], args)

        variants = {
            "mlp_guard": lambda p: mlp_cfg.with_overrides({"tracker.model_path": str(p)}),
            "mlp_hmm": lambda p: hmm_cfg.with_overrides({"tracker.model_path": str(p)}),
        }
        jobs = [(h, v) for h in ids for v in variants]
        print(f"held-out evaluation: {len(jobs)} subject runs ...", flush=True)
        out = pool(delayed(error1)(variants[v](paths[h]), subjects[h]) for h, v in jobs)

    table = pd.DataFrame({"troika": troika})
    for (h, v), value in zip(jobs, out):
        table.loc[h, v] = value
    table.index.name = "held_out_subject"
    table.loc["mean"] = table.loc[ids].mean()
    table.loc["std"] = table.loc[ids].std(ddof=1)

    out_dir = make_run_dir(stages.with_overrides({"run.name": f"{args.name}_mlp_hmm_loso"}))
    table.to_csv(out_dir / "loso_error1.csv")
    print(f"\nLeave-one-subject-out Error1 (BPM) on the spectra of {args.config};")
    print("troika is the paper's tracker, and every MLP number is on a subject the model never saw")
    print(table.round(2).to_string())
    print(f"wrote {out_dir / 'loso_error1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
