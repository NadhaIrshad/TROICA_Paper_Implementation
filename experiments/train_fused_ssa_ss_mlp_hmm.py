"""Both PPG channels fused, then SSA + spectral subtraction + MLP + HMM, leave-one-subject-out.

The fusion is the CCAF attention block of ``Channel Fusion/main.py``
(``troika.preprocessing.ccaf``, DEVIATIONS.md D16) and nothing else from that
file. Its output replaces PPG channel 1 at the start of the pipeline; the
stages of ``--config`` and the MLP and HMM trackers are the ones of
``train_eemd_ss_mlp_hmm.py``.

CCAF is trained against the ECG heart rate, so one is fitted per held-out
subject, on the other subjects only. Everything downstream of it is then
specific to that fold: the fused signals, the paper-tracker runs that give the
MLP its training rows, and the MLP. ``--fusion mean``, ``ch1`` and ``ch2`` send
a plain average or a single channel through the same 25 Hz path instead, which
shows what the attention adds over that path.

Output: fused windows and CCAF weights in ``models/loso_<name>_fusion/``, MLPs
in ``models/loso_<name>_mlp/``, the table in
``results/<name>_mlp_hmm_loso_<timestamp>/``.

Examples:
    python experiments/train_fused_ssa_ss_mlp_hmm.py
    python experiments/train_fused_ssa_ss_mlp_hmm.py --subjects 5 7 9 10
    python experiments/train_fused_ssa_ss_mlp_hmm.py --fusion ch1 --name fused_ch1_ssa_ss
    python experiments/train_fused_ssa_ss_mlp_hmm.py --fusion mean --name fused_mean_ssa_ss
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fused PPG channels + SSA + spectral subtraction + MLP + HMM, leave-one-subject-out")
    parser.add_argument("--config", default="configs/ssa_ss.yaml", help="the stages after the fusion; the band-pass and the tracker are set by this script")
    parser.add_argument("--name", default="fused_ssa_ss", help="names the output folders under models/ and results/")
    parser.add_argument("--fusion", choices=("ccaf", "mean", "ch1", "ch2"), default="ccaf", help="ccaf = attention; mean = plain average; ch1 / ch2 = one channel through the same path")
    parser.add_argument("--candidate-count", type=int, default=8, help="strongest peaks scored per window, in training and in tracking")
    parser.add_argument("--subjects", nargs="+", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=8, help="subject runs in parallel")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides for every run")
    # CCAF, defaults as in Channel Fusion/main.py
    parser.add_argument("--ccaf-epochs", type=int, default=40)
    parser.add_argument("--ccaf-qat-epochs", type=int, default=10)
    parser.add_argument("--ccaf-lr", type=float, default=3e-3)
    parser.add_argument("--ccaf-seed", type=int, default=0)
    # MLP, defaults as in train_eemd_ss_mlp_hmm.py
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu", help="where the MLP is fitted, e.g. cpu, cuda or cuda:1")
    args = parser.parse_args()

    import torch

    from train_eemd_ss_mlp_hmm import fit_and_save
    from troika.preprocessing.ccaf import ccaf_windows, fuse, snr_np, train_ccaf, zscore

    base = lab.load_config(args.config).with_overrides([*args.set, "run.n_jobs=1"])
    subjects = lab.load_subjects(base, args.subjects)
    ids = sorted(subjects)
    if len(ids) < 2:
        raise SystemExit("leave-one-subject-out needs at least two subjects")

    # --- 1. the block's input: both channels, 25 Hz, per-window z-scored ----------
    windows = {s: ccaf_windows(subjects[s].ppg_all, base.signal.fs, base.signal.window_s, base.signal.step_s) for s in ids}
    truth = {s: np.asarray(subjects[s].bpm_gt, dtype=np.float64) for s in ids}
    for s in ids:
        if len(windows[s]) != len(truth[s]):
            raise SystemExit(f"subject {s}: {len(windows[s])} windows but {len(truth[s])} ground-truth values")

    # --- 2. fuse; with CCAF, once per held-out subject -------------------------
    fusion_dir = Path("models") / f"loso_{args.name}_fusion"
    per_fold = args.fusion == "ccaf"

    def fused_path(h: int, s: int) -> Path:
        fold = f"held_out_{h:02d}" if per_fold else "all_folds"
        return fusion_dir / fold / f"subject_{s:02d}.npy"

    snr_fused = {}
    if per_fold:
        for n, h in enumerate(ids, start=1):
            train = [s for s in ids if s != h]
            print(f"CCAF {n}/{len(ids)}: training without subject {h} ...", flush=True)
            model = train_ccaf(
                np.concatenate([windows[s] for s in train]),
                np.concatenate([truth[s] for s in train]),
                fs=int(round(base.signal.fs / 5)),
                epochs=args.ccaf_epochs, qat_epochs=args.ccaf_qat_epochs, lr=args.ccaf_lr, seed=args.ccaf_seed,
            )
            fused_path(h, h).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), fused_path(h, h).parent / "ccaf.pt")
            for s in ids:
                fused = fuse(model, windows[s])
                np.save(fused_path(h, s), fused)
                if s == h:
                    snr_fused[h] = float(np.median(snr_np(fused, truth[h])))
    else:
        pick = {
            "ch1": lambda x: x[:, 0].copy(),
            "ch2": lambda x: x[:, 1].copy(),
            "mean": lambda x: zscore(x.mean(1)),
        }[args.fusion]
        fused_path(ids[0], ids[0]).parent.mkdir(parents=True, exist_ok=True)
        for s in ids:
            fused = pick(windows[s]).astype(np.float32)
            np.save(fused_path(s, s), fused)
            snr_fused[s] = float(np.median(snr_np(fused, truth[s])))

    # --- 3. configs: the fused window replaces PPG channel 1 at the band-pass ----
    count = {"tracker.candidate_count": args.candidate_count}

    def stages(h: int, s: int):
        return base.with_overrides({"bandpass.method": "fused", "bandpass.windows_path": str(fused_path(h, s))})

    def mlp_cfg(h: int, s: int):
        return stages(h, s).with_overrides({"tracker.method": "mlp", **count})

    def hmm_cfg(h: int, s: int):
        return stages(h, s).with_overrides({"tracker.method": "hmm", "tracker.scorer": "mlp", **count})

    model_dir = Path("models") / f"loso_{args.name}_mlp"
    paths = {h: model_dir / f"held_out_{h:02d}.pt" for h in ids}

    with Parallel(n_jobs=args.jobs) as pool:
        # --- 4. training rows from the paper's tracker on the fused signal -------
        pairs = [(h, s) for h in ids for s in ids if s != h] if per_fold else [(s, s) for s in ids]
        print(f"paper-tracker rows: {len(pairs)} subject runs ...", flush=True)
        out = pool(delayed(collect)(stages(h, s), mlp_cfg(h, s), subjects[s]) for h, s in pairs)
        rows = dict(zip(pairs, out))
        covered = sum(r["covered"] for r in rows.values())
        total = sum(r["windows"] for r in rows.values())
        print(f"candidate coverage within 5 BPM of the truth: {covered / total:.1%}", flush=True)

        print(f"fitting {len(ids)} model(s) ...", flush=True)
        for h in ids:
            fit_and_save([rows[(h, s) if per_fold else (s, s)] for s in ids if s != h], paths[h], args)

        # --- 5. each subject by the fusion and the MLP that never saw it ---------
        variants = {
            "troika": lambda h: stages(h, h),
            "mlp_guard": lambda h: mlp_cfg(h, h).with_overrides({"tracker.model_path": str(paths[h])}),
            "mlp_hmm": lambda h: hmm_cfg(h, h).with_overrides({"tracker.model_path": str(paths[h])}),
        }
        jobs = [(h, v) for h in ids for v in variants]
        print(f"held-out evaluation: {len(jobs)} subject runs ...", flush=True)
        out = pool(delayed(error1)(variants[v](h), subjects[h]) for h, v in jobs)

    table = pd.DataFrame(index=ids, columns=list(variants), dtype=float)
    for (h, v), value in zip(jobs, out):
        table.loc[h, v] = value
    # Median spectral SNR at the true heart rate, as Channel Fusion/main.py measures it.
    table["snr_ch1_db"] = [float(np.median(snr_np(windows[s][:, 0], truth[s]))) for s in ids]
    table["snr_ch2_db"] = [float(np.median(snr_np(windows[s][:, 1], truth[s]))) for s in ids]
    table["snr_fused_db"] = [snr_fused[s] for s in ids]
    table.index.name = "held_out_subject"
    table.loc["mean"] = table.loc[ids].mean()
    table.loc["std"] = table.loc[ids].std(ddof=1)

    out_dir = make_run_dir(base.with_overrides({"run.name": f"{args.name}_mlp_hmm_loso"}))
    table.to_csv(out_dir / "loso_error1.csv")
    print(f"\nLeave-one-subject-out Error1 (BPM), PPG fusion '{args.fusion}', then the stages of {args.config};")
    print("troika is the paper's tracker; every fused signal and MLP number is on a subject neither model saw;")
    print("snr columns: median dB at the true heart rate, for each channel and for the fused signal")
    print(table.round(2).to_string())
    print(f"wrote {out_dir / 'loso_error1.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
