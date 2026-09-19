# Conventions for this repository

Blueprint Section 12, plus Lab Spec Section 11. Read
`docs/TROIKA_Implementation_Blueprint_v2.md` for the algorithms and
`docs/TROIKA_Exploration_Lab_Spec.md` for the slot architecture. Where they
disagree, the Lab Spec wins (its Section 12 lists the conflicts).

## Fidelity to the paper

- Do not silently change a `[PAPER]` parameter. Any deviation is either a config
  option with a `# ASSUMPTION Ax` comment, or an entry in `DEVIATIONS.md` with a
  `# DEVIATION` comment at the site.
- Every assumption A1-A19 from Blueprint Section 11 is a config key, never a
  hard-coded constant.
- When a real-data result is off, change **one** assumption at a time, re-run the
  Subject 5 and 6 check plus the full run, and record the effect in
  `docs/assumption_log.md`.

## Code

- 0-based frequency bins everywhere. Every bin, Hz and BPM conversion goes
  through `troika.binmap`. Never write raw `fs / N` arithmetic elsewhere.
- `float64` and `complex128`. Seed any synthetic-data code.
- No global state except the cached SSR dictionary in `troika/ssr/basis.py`. The
  tracker is the only stateful object in the pipeline.
- `pathlib` for paths; the code runs on Windows, macOS and Linux.
- Type hints and docstrings on every public function. Each docstring cites the
  blueprint section or paper equation it implements.
- The tested core lives in `preprocessing/`, `decomposition/`, `ssr/` and
  `tracking/`. Plug-ins under `plugins/` only adapt those functions and fill
  `diag`; they do not reimplement them.

## Pipeline

- Windows of one recording are processed sequentially: the previous estimate
  feeds both the motion-artifact removal and the tracker. Parallelise across
  subjects only.
- No plug-in may read ground truth. `ctx.gt_bpm` is `None` unless
  `run.allow_ground_truth_access` is set, and any run that uses ground-truth
  state is marked `contaminated`.
- Log per-window diagnostics (`WindowResult`) and write them to CSV.

## Visualisation

- Plotly only, no matplotlib in `viz/`.
- Every plot function returns a `plotly.graph_objects.Figure` and never calls
  `.show()`.
- Plots read from traces; they do not recompute pipeline results, beyond cheap
  derived views such as the periodogram of a stored signal.
- A plot whose required `diag` keys are missing raises a clear error naming the
  keys and suggesting the generic alternative. Missing data degrades to an
  annotation, not an exception.

## Tests and git

- Every pipeline stage is unit-testable in isolation with synthetic data. Write
  the test with the code, not after.
- `pytest` must pass at the end of every milestone.
- Never commit `datasets/`, `data/`, `results/` or `.venv/`.
- Commit at the end of every milestone, e.g. `M3: SSA decomposition`.
