"""High-level API for notebooks and the command line (Lab Spec Section 4).

Everything a notebook needs in one place: load a config and the subjects, run
one subject or all twelve, score them, pull a full trace for one window, compare
variants, and promote a winning configuration back into a config file.

``experiments/run_all.py`` is a thin command-line wrapper over
:func:`run_all`, :func:`evaluate` and the reporting layer, so the command line
and the notebooks run exactly the same code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from troika import registry
from troika.config import Config, load_config as _load_config, resolve_path
from troika.evaluation import metrics
from troika.io import loader
from troika.pipeline import TroikaEstimator
from troika.trace import RunTrace, WindowTrace
from troika.types import Recording, RunResult

__all__ = [
    "load_config",
    "load_subjects",
    "run_subject",
    "run_all",
    "evaluate",
    "get_window_trace",
    "compare_variants",
    "compare_runs",
    "collect_traces",
    "promote",
    "ContaminatedRunError",
]


class ContaminatedRunError(RuntimeError):
    """Raised when statistics are requested from a run that saw ground truth."""


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a config, optionally applying dotted-path overrides.

    ``lab.load_config("configs/default.yaml")`` is the notebook entry point.
    """
    cfg = _load_config(path)
    return cfg.with_overrides(overrides) if overrides else cfg


def load_subjects(
    cfg: Config, subject_ids: Sequence[int] | None = None
) -> dict[int, Recording]:
    """Load every subject of the configured split, keyed by subject number."""
    if subject_ids is None:
        subject_ids = _configured_subjects(cfg)
    return loader.load_all(cfg, subject_ids)


def _configured_subjects(cfg: Config) -> list[int] | None:
    """Subject list from ``run.subjects``; ``None`` means all of them."""
    configured = cfg.run.subjects
    if isinstance(configured, str) and configured.lower() == "all":
        return None
    return [int(s) for s in configured]


def _progress(items, description: str, enable: bool = True):
    """Wrap an iterable in tqdm when it is installed and output is a terminal."""
    if not enable:
        return items
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=description)


# ------------------------------------------------------------------- running


def run_subject(
    cfg: Config,
    rec: Recording,
    *,
    trace: str | None = None,
    full_windows: Sequence[int] | None = None,
) -> RunResult:
    """Run the pipeline over one recording.

    ``trace`` is ``none``, ``light`` or ``full``; ``full_windows`` additionally
    records those window indices at the full level, which is how the stage plots
    get their payload without storing a gigabyte per subject.
    """
    return TroikaEstimator(cfg).run(rec, trace=trace, full_windows=full_windows)


def run_all(
    cfg: Config,
    subjects: Mapping[int, Recording],
    subject_ids: Sequence[int] | None = None,
    *,
    trace: str | None = None,
    n_jobs: int | None = None,
    progress: bool = True,
) -> dict[int, RunResult]:
    """Run every selected subject, in parallel across subjects.

    Windows of one recording stay sequential, because the pipeline is stateful
    across them; only subjects are distributed (Blueprint Section 4).
    """
    if subject_ids is None:
        subject_ids = _configured_subjects(cfg)
    chosen = sorted(subjects) if subject_ids is None else [int(s) for s in subject_ids]
    missing = [s for s in chosen if s not in subjects]
    if missing:
        raise KeyError(f"no recording loaded for subject(s) {missing}")

    jobs = int(cfg.run.n_jobs if n_jobs is None else n_jobs)

    # A plug-in defined in a notebook cell cannot be imported by a worker
    # process, so such a run stays here rather than failing in the workers.
    notebook_defined = [
        f"{slot}/{cfg.slot_method(slot)}"
        for slot in registry.SLOTS
        if not registry.is_builtin(slot, cfg.slot_method(slot))
    ]
    if jobs > 1 and notebook_defined:
        print(
            f"running on one core: {', '.join(notebook_defined)} "
            f"{'is' if len(notebook_defined) == 1 else 'are'} defined outside the "
            f"package and cannot be sent to worker processes. Move it to a file "
            f"with `python -m troika.plugins.new` to run in parallel."
        )
        jobs = 1

    if jobs > 1 and len(chosen) > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=min(jobs, len(chosen)))(
            delayed(run_subject)(cfg, subjects[s], trace=trace) for s in chosen
        )
        return {s: r for s, r in zip(chosen, results)}

    return {
        s: run_subject(cfg, subjects[s], trace=trace)
        for s in _progress(chosen, "subjects", progress)
    }


# ---------------------------------------------------------------- evaluation


def evaluate(
    runs: Mapping[int, RunResult] | Iterable[RunResult],
    cfg: Config | None = None,
    *,
    subject_ids: Sequence[int] | None = None,
    allow_contaminated: bool = False,
    paper_variant: str = "full",
) -> pd.DataFrame:
    """Per-subject statistics with summary rows (Lab Spec Section 4.1).

    Refuses to report a run that used ground-truth-derived state unless
    ``allow_contaminated`` is set, and then prints a banner (Lab Spec 2.4).
    """
    items = list(runs.values()) if isinstance(runs, Mapping) else list(runs)
    contaminated = [r.subject_id for r in items if r.contaminated]
    if contaminated:
        if not allow_contaminated:
            raise ContaminatedRunError(
                f"subject(s) {sorted(contaminated)} used ground-truth-derived state, "
                f"so these numbers are not a valid result. Pass "
                f"allow_contaminated=True to see them anyway."
            )
        print("CONTAMINATED: uses ground truth")

    std_ddof = 1
    within = 5.0
    compare = True
    if cfg is not None:
        std_ddof = int(cfg.evaluation.std_ddof)
        within = float(cfg.evaluation.within_bpm)
        compare = bool(cfg.evaluation.compare_with_paper)

    return metrics.summarize(
        items,
        std_ddof=std_ddof,
        within_bpm=within,
        compare_with_paper=compare,
        paper_variant=paper_variant,
        subject_ids=subject_ids,
    )


# ------------------------------------------------------------------- tracing


def get_window_trace(
    rec: Recording,
    cfg: Config,
    w: int,
    prev_bin: str | int = "tracked",
) -> WindowTrace:
    """Full trace of a single window (Lab Spec Section 2.4, isolated-window mode).

    ``prev_bin``:

    ``"tracked"``
        Run windows ``0 .. w-1`` at the light level and window ``w`` at the full
        level, so the state feeding window ``w`` is the honest one. The default.
    ``"ground_truth"``
        Seed the previous bin from the ECG ground truth of window ``w-1``.
    an int
        Seed the previous bin directly.

    The last two mark the trace contaminated, because the window's state did not
    come from the pipeline alone.
    """
    w = int(w)
    if w < 0:
        raise ValueError(f"window index must be >= 0, got {w}")

    if prev_bin == "tracked":
        run = TroikaEstimator(cfg).run(
            rec, trace="light", full_windows=[w], stop_after=w + 1
        )
        return run.trace[w]

    if prev_bin == "ground_truth":
        if rec.bpm_gt is None:
            raise ValueError(f"subject {rec.subject_id} has no ground truth to seed from")
        if w == 0:
            raise ValueError("window 0 has no previous window to seed from")
        from troika import binmap

        seed = int(binmap.bpm_to_bin(float(rec.bpm_gt[w - 1]), cfg.signal.fs, cfg.grid.n_fft))
    elif isinstance(prev_bin, (int, np.integer)):
        seed = int(prev_bin)
    else:
        raise ValueError(
            f"prev_bin must be 'tracked', 'ground_truth' or an int, got {prev_bin!r}"
        )

    # Run only the requested window, with the tracker's previous bin seeded.
    sliced = _single_window_recording(rec, cfg, w)
    run = TroikaEstimator(cfg).run(
        sliced, trace="full", full_windows=[0], prev_bin_override=seed
    )
    trace = run.trace.windows[0]
    trace.idx = w
    trace.t_start_s = w * float(cfg.signal.step_s)
    trace.contaminated = True
    return trace


def _single_window_recording(rec: Recording, cfg: Config, w: int) -> Recording:
    """A one-window slice of a recording, for isolated-window exploration."""
    from troika.preprocessing.windowing import window_bounds

    start, stop = window_bounds(w, cfg.signal.fs, cfg.signal.window_s, cfg.signal.step_s)
    if stop > rec.n_samples:
        raise ValueError(f"window {w} runs past the end of subject {rec.subject_id}")
    return Recording(
        subject_id=rec.subject_id,
        fs=rec.fs,
        ppg=rec.ppg[start:stop],
        acc=rec.acc[:, start:stop],
        ecg=None if rec.ecg is None else rec.ecg[start:stop],
        bpm_gt=None if rec.bpm_gt is None else rec.bpm_gt[w : w + 1],
        ppg_all=None if rec.ppg_all is None else rec.ppg_all[:, start:stop],
        protocol=rec.protocol,
        path=rec.path,
    )


def collect_traces(
    cfg: Config,
    subjects: Mapping[int, Recording],
    selector: str | int = "max_error",
    *,
    runs: Mapping[int, RunResult] | None = None,
) -> dict[int, WindowTrace]:
    """One full window trace per subject, chosen by ``selector``.

    ``selector`` is a window index, ``"time:120"`` for a moment in seconds,
    ``"max_error"`` or ``"median_error"``. The error selectors need a scored run,
    which is computed when ``runs`` is not supplied.
    """
    if runs is None and isinstance(selector, str) and selector.endswith("_error"):
        runs = run_all(cfg, subjects, progress=False)

    out: dict[int, WindowTrace] = {}
    for subject_id, rec in sorted(subjects.items()):
        w = _resolve_selector(selector, cfg, rec, None if runs is None else runs.get(subject_id))
        out[subject_id] = get_window_trace(rec, cfg, w)
    return out


def _resolve_selector(
    selector: str | int, cfg: Config, rec: Recording, run: RunResult | None
) -> int:
    """Turn a selector into a window index for one subject."""
    if isinstance(selector, (int, np.integer)):
        return int(selector)
    if isinstance(selector, str) and selector.startswith("time:"):
        seconds = float(selector.split(":", 1)[1])
        return max(0, int(round(seconds / float(cfg.signal.step_s))))
    if selector in ("max_error", "median_error"):
        if run is None or run.bpm_gt is None:
            raise ValueError(f"selector {selector!r} needs a scored run with ground truth")
        error = np.abs(np.asarray(run.bpm_est) - np.asarray(run.bpm_gt))
        if selector == "max_error":
            return int(np.nanargmax(error))
        order = np.argsort(error)
        return int(order[len(order) // 2])
    raise ValueError(
        f"unknown selector {selector!r}; use an int, 'time:<seconds>', "
        f"'max_error' or 'median_error'"
    )


# ----------------------------------------------------------------- comparison


def compare_variants(
    rec: Recording,
    variants: Mapping[str, Config],
    window: int,
    stage: str = "bandpass",
    *,
    prev_bin: str | int = "tracked",
    draw: bool = True,
):
    """Run one window under several configs and overlay one stage's output.

    Returns the Plotly figure when ``draw`` is set, else the traces keyed by
    variant name, so the comparison can be inspected without Plotly installed.
    """
    traces = {
        name: get_window_trace(rec, cfg, window, prev_bin=prev_bin)
        for name, cfg in variants.items()
    }
    if not draw:
        return traces
    from troika.viz.compare import compare_stage

    return compare_stage(traces, stage=stage)


def compare_runs(
    runs_by_name: Mapping[str, Mapping[int, RunResult]],
    *,
    cfg: Config | None = None,
    draw: bool = True,
):
    """Compare whole runs: estimate traces overlaid, plus a statistics table."""
    table = pd.concat(
        {
            name: evaluate(runs, cfg, allow_contaminated=True)
            for name, runs in runs_by_name.items()
        },
        names=["variant"],
    )
    if not draw:
        return table
    from troika.viz.compare import compare_runs as _draw

    return _draw(runs_by_name, table)


# ------------------------------------------------------------------ promotion


def promote(cfg: Config, path: str | Path = "configs/my_experiment.yaml") -> str:
    """Write only what differs from the default into ``path``, and print it.

    Lab Spec Section 6 step 6: the way a variant found in a notebook becomes the
    configuration that ``experiments/run_all.py`` reproduces.
    """
    target = Path(path)
    if not target.is_absolute() and not target.parent.is_dir():
        # A notebook runs from notebooks/, so "configs/..." must still land in
        # the repository's configs/ rather than creating notebooks/configs/.
        target = resolve_path(target.parent) / target.name
    text = cfg.to_yaml(target, diff_only=True)
    print(f"wrote {target}:\n{text}")
    return text


_ = (RunTrace, Callable)  # re-exported names used in type hints elsewhere
