"""Quantise the MLP scorer to 8 bits, measure what that costs, and export it for Edge Impulse.

Works on the fold models a ``train_eemd_ss_mlp_hmm.py`` run saved under
``--name`` (DEVIATIONS.md D15). Three steps:

1. Each fold's MLP is quantised (``troika.tracking.mlp_int8``), with the tensor
   ranges taken from that fold's training subjects only.
2. Every subject is tracked by the HMM twice, with the float model and with the
   8-bit one that never saw it, and the two Error1 columns are written side by
   side.
3. One model is fitted to all subjects, which is the one a device would carry,
   and written to ``edge_impulse/<name>/``:

   * ``mlp.onnx``: float32, input the 13 standardised features, output the score;
   * ``representative_features.npy``: standardised training rows, which Edge
     Impulse needs to quantise the model itself;
   * ``scaler.json``: the feature order, means, deviations and clip that the
     device applies in floating point before the model;
   * ``mlp_int8.npz``: this repository's own 8-bit version of the same model.

Examples:
    python experiments/export_edge_impulse.py
    python experiments/export_edge_impulse.py --config configs/ssa_ss.yaml --name ssa_ss
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from troika import lab
from troika.evaluation.report import make_run_dir
from troika.tracking.mlp_int8 import AUX_CLIP, quantise
from troika.tracking.xgboost_features import (
    FEATURE_NAMES,
    candidate_bins,
    feature_matrix,
    label_candidates,
)


def collect(run_cfg, feature_cfg, rec) -> dict:
    """Run one subject and return its labelled candidate rows.

    Truth is used only here, after the run, to label the rows; it is never
    passed into the pipeline. As in the training script, rows come from the
    light trace, whose spectrum is float32 and cut at 7 Hz.
    """
    run = lab.run_subject(run_cfg, rec, trace="light")
    assert run.trace is not None and run.bpm_gt is not None
    fs, n_fft = feature_cfg.signal.fs, feature_cfg.grid.n_fft
    params = feature_cfg.tracker.params
    rows, labels = [], []
    for trace in run.trace.windows[1:]:  # window 0 is fixed max-peak initialisation
        spectrum = np.zeros(n_fft // 2 + 1)
        light = trace.stages["spectrum_estimator"].spectrum
        spectrum[: light.size] = light
        candidates = candidate_bins(spectrum, fs, n_fft, params.low_bpm, params.high_bpm, params.candidate_count)
        y, ok = label_candidates(candidates, float(trace.gt_bpm), fs, n_fft)
        if ok:
            rows.append(feature_matrix(spectrum, trace.ctx, candidates, fs, n_fft))
            labels.append(y)
    return {
        "x": np.vstack(rows) if rows else np.zeros((0, len(FEATURE_NAMES))),
        "y": np.concatenate(labels) if labels else np.zeros(0, dtype=np.int64),
    }


def error1(run_cfg, rec) -> float:
    run = lab.run_subject(run_cfg, rec)
    return float(np.mean(np.abs(run.bpm_est - run.bpm_gt)))


def export_onnx(scorer, path: Path) -> None:
    """Write the float32 network, standardised features in and score out, as ONNX."""
    import torch

    class Probability(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, z):
            return torch.sigmoid(self.model(z)).unsqueeze(-1)

    net = Probability(copy.deepcopy(scorer.model).float()).eval()
    example = torch.zeros(1, scorer.aux_mean.numel(), dtype=torch.float32)
    torch.onnx.export(net, example, str(path), input_names=["features"], output_names=["score"], opset_version=13)


def main() -> int:
    parser = argparse.ArgumentParser(description="8-bit MLP: accuracy check and Edge Impulse export")
    parser.add_argument("--config", default="configs/none_ss.yaml", help="the stages the fold models were trained on")
    parser.add_argument("--name", default="none_ss", help="the --name of the training run: reads models/loso_<name>_mlp/")
    parser.add_argument("--out", default="edge_impulse", help="export folder; files go to <out>/<name>/")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8, help="subject runs in parallel")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, as given to the training run")
    # The all-subjects model is fitted exactly as the fold models were.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from train_eemd_ss_mlp_hmm import fit_and_save
    from troika.tracking.mlp import MLPScorer

    stages = lab.load_config(args.config).with_overrides([*args.set, "run.n_jobs=1"])
    mlp_cfg = stages.with_overrides({"tracker.method": "mlp"})
    subjects = lab.load_subjects(stages, args.subjects)
    ids = sorted(subjects)
    float_dir = Path("models") / f"loso_{args.name}_mlp"
    int8_dir = Path("models") / f"loso_{args.name}_mlp_int8"
    float_paths = {h: float_dir / f"held_out_{h:02d}.pt" for h in ids}
    int8_paths = {h: int8_dir / f"held_out_{h:02d}.npz" for h in ids}
    missing = [str(p) for p in float_paths.values() if not p.exists()]
    if missing:
        raise SystemExit(f"run train_eemd_ss_mlp_hmm.py --name {args.name} first; missing models: {missing}")

    with Parallel(n_jobs=args.jobs) as pool:
        print(f"paper-tracker rows for {len(ids)} subjects ...", flush=True)
        rows = dict(zip(ids, pool(delayed(collect)(stages, mlp_cfg, subjects[s]) for s in ids)))

        # --- 1. quantise each fold on its own training subjects ----------------
        gaps = {}
        for h in ids:
            scorer = MLPScorer.load(float_paths[h])
            model = quantise(scorer, np.vstack([rows[s]["x"] for s in ids if s != h]))
            model.save(int8_paths[h])
            gaps[h] = np.abs(model.score(rows[h]["x"]) - scorer.score(rows[h]["x"]))

        # --- 2. track every subject with both versions -------------------------
        variants = {
            "mlp_hmm": lambda h: {"tracker.method": "hmm", "tracker.scorer": "mlp", "tracker.model_path": str(float_paths[h])},
            "mlp_hmm_int8": lambda h: {"tracker.method": "hmm_int8", "tracker.model_path": str(int8_paths[h])},
        }
        jobs = [(h, v) for h in ids for v in variants]
        print(f"held-out evaluation: {len(jobs)} subject runs ...", flush=True)
        out = pool(delayed(error1)(stages.with_overrides(variants[v](h)), subjects[h]) for h, v in jobs)

    table = pd.DataFrame(index=ids, columns=list(variants), dtype=float)
    for (h, v), value in zip(jobs, out):
        table.loc[h, v] = value
    table["score_gap_mean"] = [float(gaps[h].mean()) for h in ids]
    table["score_gap_max"] = [float(gaps[h].max()) for h in ids]
    table.index.name = "held_out_subject"
    table.loc["mean"] = table.loc[ids].mean()
    table.loc["std"] = table.loc[ids].std(ddof=1)
    out_dir = make_run_dir(stages.with_overrides({"run.name": f"{args.name}_mlp_int8_loso"}))
    table.to_csv(out_dir / "loso_error1.csv")
    print("\nError1 (BPM) behind the HMM, float MLP against its 8-bit version, each on a subject the model never saw;")
    print("score_gap is the absolute difference between the two versions' candidate scores (scores run from 0 to 1)")
    print(table.round(3).to_string())
    print(f"wrote {out_dir / 'loso_error1.csv'}", flush=True)

    # --- 3. the model a device would carry: all subjects ----------------------
    export_dir = Path(args.out) / args.name
    export_dir.mkdir(parents=True, exist_ok=True)
    everyone = [rows[s] for s in ids]
    fit_and_save(everyone, export_dir / "mlp_all_subjects.pt", args)
    scorer = MLPScorer.load(export_dir / "mlp_all_subjects.pt")
    aux = np.vstack([r["x"] for r in everyone])
    export_onnx(scorer, export_dir / "mlp.onnx")
    np.save(export_dir / "representative_features.npy", scorer.standardise(aux).numpy().astype(np.float32))
    deployed = quantise(scorer, aux)
    deployed.save(export_dir / "mlp_int8.npz")
    scaler = {
        "feature_names": list(FEATURE_NAMES),
        "mean": scorer.aux_mean.tolist(),
        "std": scorer.aux_std.tolist(),
        "clip": AUX_CLIP,
        "note": "model input = clip((feature - mean) / std, -clip, clip), computed in floating point",
    }
    (export_dir / "scaler.json").write_text(json.dumps(scaler, indent=2))

    n_weights = sum(p.numel() for p in scorer.model.parameters())
    print(f"\nall-subjects model: {n_weights} weights, {4 * n_weights} bytes as float32, {deployed.n_bytes} bytes as int8 with int32 biases")
    print(f"wrote {export_dir}/: mlp.onnx, representative_features.npy, scaler.json, mlp_int8.npz, mlp_all_subjects.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
