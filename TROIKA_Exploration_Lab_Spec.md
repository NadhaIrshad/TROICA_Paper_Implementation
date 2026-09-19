# TROIKA Exploration Lab Spec

Addendum to `TROIKA_Implementation_Blueprint_v2.md`. It adds: step-by-step interactive visualization, a plug-in system so any pipeline stage can be swapped and compared, a notebook workflow in VS Code, and a path to promote a winning variant into the experiment config.

**Precedence.** The Blueprint defines the algorithms and the paper-faithful core. This file defines the layer around it. Where the two disagree (listed in Section 12), **this file wins**. Do not rewrite the tested low-level functions from Blueprint Section 7.2 (`ssa.py`, `focuss.py`, `selection.py`, ...). The new plug-ins wrap them.

Decisions already made with the user:
- Work happens in **Jupyter-style notebooks in VS Code**, cell by cell.
- Build **only the plug-in system**, with the paper-default methods (plus the alternatives the Blueprint already lists: FIR bandpass, FFT spectrum, SSA grouping strategies). The user will write extra alternatives (e.g. EMD instead of SSA) themselves, so authoring them must be easy.
- Plots are **interactive Plotly** (zoom, hover, toggle traces).

---

## 1. The four ways of working

| # | Mode | Command / entry point | Result |
|---|---|---|---|
| 1 | Baseline | `python experiments/run_all.py --config configs/default.yaml [--subjects 5 6]` | Error stats per subject and for all 12 with the paper's default parameters |
| 2 | Experiment | `python experiments/run_all.py --config configs/my_experiment.yaml [--subjects ...]` | Same statistics for the user's parameters |
| 3 | Explore | Notebooks in `notebooks/` (Section 8) | Plot every internal step, for one subject, one window, or all 12 |
| 4 | Try variants, then promote | `lab.compare_variants(...)`, `lab.compare_runs(...)`, `lab.promote(cfg, path)` | Change a stage (bandpass type, EMD vs SSA, ...), see the effect on plots and errors, write the winning setting into `configs/my_experiment.yaml`, and run mode 2 for final numbers |

Modes 1 and 2 are the same code path (`lab.run_all` + `lab.evaluate`), only the config differs. Modes 3 and 4 are notebooks calling the same functions.

---

## 2. Architecture: swappable stages ("slots")

The pipeline order is fixed. Five stages are **slots** whose implementation is chosen by name in the config.

```
raw ─► [bandpass] ─► acc_dominant (helper, not a slot) ─► [decomposition] ─► [temporal_diff]
                                                             ─► [spectrum_estimator] ─► [tracker] ─► BPM
```

| Slot | Paper-default method | Other methods to register now | Input → output |
|---|---|---|---|
| `bandpass` | `butter` | `fir` | `(4, M)` stacked [PPG; ACC x,y,z] → `(4, M)` |
| `decomposition` | `ssa` | `none` (identity) | PPG `(M,)` + context → cleansed PPG `(M,)` |
| `temporal_diff` | `diff` | `none` | `(M,)` → `(M′,)` |
| `spectrum_estimator` | `focuss` | `fft` (paper's ablation) | `(M′,)` → power spectrum `(N/2+1,)` |
| `tracker` | `troika` | none | spectrum + context → bin/BPM (stateful) |

Nothing else is built now. Inside `ssa`, the grouping strategies from the Blueprint (`frequency_pairing`, `wcorr_hclust`, `singleton`) remain an internal parameter.

### 2.1 Plug-in contract

`src/troika/plugins/base.py` defines one abstract base class per slot (`Bandpass`, `Decomposition`, `TemporalDiff`, `SpectrumEstimator`, `Tracker`). Each documents its input/output shapes and its `diag` schema (Section 2.3).

```python
@dataclass
class StageOutput:
    data: np.ndarray | TrackerResult      # the main output of the stage
    diag: dict[str, Any] = field(default_factory=dict)   # internals for plots; may be empty

@dataclass
class StaticContext:          # fixed for a run
    fs: float; N: int; M: int; window_s: float; step_s: float

@dataclass
class WindowContext:          # changes per window
    idx: int; t_start_s: float
    acc: np.ndarray | None                 # (3, M) band-passed accelerometer
    acc_bins_raw: set[int]                 # F_acc before refinement
    acc_bins: set[int]                     # refined F̃_acc
    prev_bin: int | None
    bpm_history: list[float]
    gt_bpm: float | None                   # None unless the leak guard allows it (2.4)

class Decomposition(ABC):                  # same pattern for the other slots
    Params: ClassVar[type]                 # dataclass with defaults; validates config
    def __init__(self, params, static: StaticContext): ...
    @abstractmethod
    def __call__(self, x: np.ndarray, ctx: WindowContext) -> StageOutput: ...
```

`tracker` additionally has `initialize(spectrum, ctx)` / `step(spectrum, ctx)` returning a `TrackerResult` (bin, bpm, case, rule flags), and owns its state.

### 2.2 Registry

`src/troika/registry.py`:
```python
register(slot: str, name: str, *, override: bool = False)   # class decorator
register_function(slot, name, params: dict, *, override=False)   # quick function form
get(slot, name) -> class;  available(slot) -> list[str]
```
- Built-in plug-ins live in `src/troika/plugins/<slot>/<name>.py` and are auto-imported by `troika.plugins.__init__` (scan the folder, no manual list).
- **Registering inside a notebook must work** (no package edit needed) and `override=True` lets the user re-run the cell after editing. Function form:
  ```python
  @register_function("decomposition", "my_variant", params={"n_modes": 6})
  def my_variant(x, ctx, *, n_modes):
      ...
      return StageOutput(data=cleansed, diag={"components": comps, "removed_mask": mask})
  ```
  (A bare `np.ndarray` return is also accepted and wrapped with an empty `diag`.)
- Unknown method names raise `KeyError` listing `available(slot)`.

### 2.3 `diag` schemas (what plots may rely on)

Plots read `diag` by key and **must fail with a clear message** ("plot_ssa_svd needs diag keys ['U','s','Vt'] from decomposition 'emd'; use plot_decomposition_components instead") rather than a stack trace. Keys are optional unless marked required.

| Slot | Keys |
|---|---|
| `bandpass` | `filter_response`: `{f_hz, mag_db, phase_rad}`; `description`: str |
| `decomposition` (generic, works for any method, including a future EMD) | `components (g, M)`, `component_labels list[str]`, `component_dominant_bins (g,)`, `removed_mask (g,) bool` |
| `decomposition` (SSA-specific) | `trajectory (L, K)`, `U (L, d)`, `s (d,)`, `Vt (d, K)`, `groups list[list[int]]`, `group_dominant_bins`, `group_singular_range`, `L`, `K`, `grouping_strategy` |
| `temporal_diff` | `pre_normalization (M′,)`, `order` |
| `spectrum_estimator` | `iterates (n_iter+1, n_cols) complex` (FOCUSS), `kept_bins (n_cols,)`, `sparsity_per_iter` |
| `tracker` | `R0`, `R1` (bin ranges), `eta`, `P0`, `P1`, `pairs`, `candidates`, `k_b`, `case`, `k_cur`, `rule1_fired`, `rule2_fired`, `same_count`, `trend`, `delta_s_used`, `k_prev` |

Paper-default plug-ins must fill **every** listed key for their slot.

### 2.4 Tracing and the ground-truth leak guard

**Trace levels** (a run at `full` for every window would be about 1 GB per subject because of SSA matrices):
- `none`: metrics only.
- `light` (per window): estimated bin/BPM, case, rule flags, refined `F̃_acc`, number of groups removed, stage timings, and for each of the stages `bandpass`, `decomposition`, `temporal_diff`, `spectrum_estimator`: the time-domain output as `float32` and its spectrum truncated to 0–7 Hz as `float32`. Target < 5 MB per subject.
- `full` (chosen windows only): everything in Section 2.3 plus raw window signals. `lab.run_subject(cfg, rec, trace="light", full_windows=[40, 90])` runs sequentially and stores `full` only for those windows.

**Isolated-window mode.** Stage plots for window `w` need `prev_bin`, which normally comes from tracking windows `0…w−1`. `lab.get_window_trace(rec, cfg, w, prev_bin="tracked" | "ground_truth" | int)`:
- `tracked` runs windows `0…w−1` at `light` level, then `w` at `full` (default, honest).
- `ground_truth` / int seeds `prev_bin` directly for quick exploration and marks the trace `contaminated=True`.

**Leak guard.** `ctx.gt_bpm` is `None` for plug-ins unless `run.allow_ground_truth_access: true`. Any run or trace using GT-derived state (isolated mode with `ground_truth`, `tracker.params.init_mode: ground_truth`) sets `contaminated=True`; `lab.evaluate` refuses to report statistics from contaminated runs unless `allow_contaminated=True`, and then prints a banner "CONTAMINATED: uses ground truth".

---

## 3. Configuration layout (supersedes Blueprint Section 6)

Every slot has `method` + `params`. Params are validated against the chosen plug-in's `Params` dataclass (unknown key, wrong type, or out-of-range value → error naming the key).

`configs/default.yaml` (paper values, never edited):

```yaml
run:
  name: default
  subjects: all                    # all | [1, 5, 6]
  n_jobs: 4                        # parallel across subjects only
  allow_ground_truth_access: false
  trace: {level: none}             # none | light

signal:  {fs: 125, window_s: 8, step_s: 2, ppg_channel: 0}
grid:    {n_fft: 4096}             # shared frequency grid N

bandpass:
  method: butter                   # butter | fir
  params: {low_hz: 0.4, high_hz: 5.0, order: 4, mode: per_window}   # mode: per_window | global

acc_dominant:                      # helper feeding WindowContext, not a slot
  rel_threshold: 0.5
  exclude_delta_bins: 10
  n_harmonics: 2

decomposition:
  method: ssa                      # ssa | none
  params: {L: 400, grouping: frequency_pairing, sv_rel_tol: 0.1, freq_tol_bins: 2, match_tol_bins: 2}

temporal_diff:
  method: diff                     # diff | none
  params: {order: 2, normalize_after: true}

spectrum_estimator:
  method: focuss                   # focuss | fft
  params: {p: 0.8, lam: 0.1, n_iter: 5, x0: ones, prune_columns: true}

tracker:
  method: troika
  params:
    init_mode: max_peak            # max_peak | ground_truth (debug; marks run contaminated)
    init_band_bpm: [40, 200]
    delta_s: 16
    delta_s_wide: 20
    max_peaks_per_range: 3
    eta_frac: 0.30
    harm_tol_bins: 2
    verification: {enabled: true, theta_bins: 6, tau_bins: 2, stall_windows_h: 3,
                   trend_history_windows: 20, trend_poly_order: 3, trend_threshold_bpm: 3}

evaluation: {std_ddof: 1, compare_with_paper: true, within_bpm: 5}
```

Ablation files change one line each: `decomposition.method: none`, `spectrum_estimator.method: fft`, `tracker.params.verification.enabled: false`. `bandpass.params.mode` is A2, `n_harmonics` is A6, etc. (Blueprint Section 11 IDs still apply; the `params` names stay the same as the old flat names).

`configs/my_experiment.yaml`: full listing, editable, merged over `default.yaml` (partial files are fine).

### 3.1 Config API (`troika.config`)

```python
cfg = load_config("configs/my_experiment.yaml")           # default merged with file
cfg2 = cfg.with_overrides({"bandpass.method": "fir", "decomposition.L": 300})
cfg.diff_from_default() -> dict                            # nested dict of changes only
cfg.to_yaml(path)                                          # full resolved config
cfg.hash() -> str                                          # stable, for cache keys and result folders
```
Rules:
- **Shorthand:** `decomposition.L` resolves to `decomposition.params.L` when unambiguous.
- **Changing `method` resets `params` to the new plug-in's defaults** (then applies any explicitly given params). Otherwise stale params of the old method would fail validation.
- The same override syntax is used by `--set` on the CLI: `--set bandpass.method=fir --set decomposition.L=300`.
- Each result folder stores `config_resolved.yaml` and the config hash.

---

## 4. Programmatic API for notebooks (`troika.lab`)

```python
from troika import lab, viz
cfg      = lab.load_config("configs/default.yaml")            # or my_experiment.yaml
subjects = lab.load_subjects(cfg)                             # dict {1..12: Recording}

run  = lab.run_subject(cfg, subjects[5], trace="light", full_windows=[40, 90])
runs = lab.run_all(cfg, subjects, subject_ids=None, trace="light")   # dict {id: RunResult}; tqdm; parallel

stats = lab.evaluate(runs)                                    # DataFrame, all subjects
stats = lab.evaluate(runs, subject_ids=[5, 6])                # subset
tr    = lab.get_window_trace(subjects[5], cfg, w=40, prev_bin="tracked")

lab.compare_variants(subjects[5], {"butter": cfg, "fir": cfg.with_overrides({"bandpass.method": "fir"})},
                     window=40, stage="bandpass")             # overlaid stage outputs, Figure
lab.compare_runs({"A": runs_a, "B": runs_b})                  # est-vs-gt overlay + stats table, side by side
lab.collect_traces(cfg, subjects, selector="max_error")       # full trace for one chosen window per subject
lab.promote(cfg, "configs/my_experiment.yaml")                # write diff-only YAML, print it
```

`experiments/run_all.py` is a thin CLI over `lab.run_all` + `lab.evaluate` + `evaluation.report`. Optional: `cache_dir=".cache"` keyed by `(config hash, subject id, data file mtime)`; low priority.

### 4.1 Statistics table (`lab.evaluate`)

Per subject (row per subject, then summary rows):

| Column | Definition |
|---|---|
| `Error1_BPM` | mean absolute error (paper Eq. 19) |
| `Error2_pct` | mean absolute percentage error (paper Eq. 20) |
| `bias_BPM` | mean signed error (est − gt) |
| `rmse_BPM`, `median_AE_BPM`, `max_AE_BPM` | extras, not in the paper |
| `within_5_pct` | % of windows with \|error\| ≤ `within_bpm` |
| `n_windows` | count |

Summary rows: **Mean ± std over the selected subjects** (ddof from config), and **Pooled** (all windows of the selected subjects): Bland–Altman mean and limits of agreement, σ, Pearson r. When `compare_with_paper` is true and the selected subjects are among the paper's 12, add `Error1_paper`, `Error2_paper` columns from `tests/reference/paper_tables.csv` and a delta column. A subset run labels its mean row "mean over subjects 5, 6".

---

## 5. Visualization (`troika.viz`, Plotly)

### 5.1 Conventions (all plots)

- Every function **returns a `plotly.graph_objects.Figure`** and never calls `.show()`; the notebook displays the last expression. `viz.save(fig, path)` writes interactive HTML (PNG through `kaleido` if installed). `viz.setup_notebook()` sets the renderer suitable for VS Code notebooks and enables `%autoreload`-friendly imports.
- Common keyword arguments: `domain="both"|"time"|"freq"`, `freq_unit="hz"|"bpm"` (primary axis switch; hover always shows Hz, BPM, and bin `k`), `fmax_hz=6.0`, `show_gt=True`, `title=None`, `height=None`.
- **Domain layout:** `domain="both"` gives time on the left column and frequency on the right, one row per signal.
- **Time axis** in seconds (window plots start at the window's `t_start`; whole-recording plots use the recording clock). **Frequency plots** use the same `N = 4096` grid as the pipeline; power on linear axis with a `log_y` toggle.
- **Ground truth:** when a trace or run has it, mark the true HR (and 2× harmonic, lighter) as vertical dashed lines in every frequency plot. For whole-recording plots draw it as a line over time.
- **Colour roles** (`viz/theme.py`, one place to change): raw = grey, band-passed = blue, decomposition = green, temporal difference = orange, spectrum estimator = purple, tracker = red, ground truth = black dashed, accelerometer dominant bins = teal lines, excluded ±Δ zone = light-red band, removed components = red, kept = green. Template `plotly_white`.
- **Large data:** whole-recording time plots decimate with min/max envelope to ≤ `max_points=4000` per trace and switch to `Scattergl` above 10k points. Legends are click-to-toggle.
- Missing data (no ECG, no GT, wrong `diag` keys) degrades gracefully with a message annotation, not an exception, except the diag-key case above, which raises a clear error.
- Protocol shading: speed segments from the paper (boundaries at 30, 90, 150, 210, 270 s: 1–2, 6–8, 12–15, 6–8, 12–15, 1–2 km/h) drawn as labelled background bands on whole-recording time axes (`protocol_shading=True`). These are nominal times; label them "nominal".

### 5.2 Catalogue

**A. Data (need `Recording`)**

| Function | Shows |
|---|---|
| `plot_recording(rec, channels=("ppg","acc","ecg"), domain="both")` | One subject: each channel in time (left) and spectrum (right). Spectrum of the whole recording via Welch (state `nperseg`); protocol shading |
| `plot_spectrogram(rec, channel="ppg", nperseg=..., overlap=...)` | Time–frequency heatmap (BPM axis) with GT HR line over it. Shows MA versus heart-rate ridges |
| `plot_all_subjects(recs, channel="ppg", domain="time"\|"freq"\|"spectrogram", layout="grid"\|"overlay")` | All 12: 4×3 grid with shared-axes toggle, or one overlay with a legend entry per subject |

**B. Per-window stage plots (need `WindowTrace`, full level)**

| Function | Shows |
|---|---|
| `plot_stage(trace, stage, domain="both")` | Generic dispatcher for `raw`, `bandpass`, `decomposition`, `temporal_diff`, `spectrum`, `tracking`: input (light) vs output (bold), time and frequency |
| `plot_filter_response(trace_or_diag)` | Magnitude (dB) and phase of the band-pass with 0.4 and 5 Hz markers, to compare `butter` vs `fir` |
| `plot_acc_dominant(trace)` | Three ACC spectra, 50 % threshold, picked peaks, raw `F_acc` vs refined `F̃_acc`, ±Δ exclusion band around previous HR fundamental and harmonic |
| `plot_temporal_diff(trace)` | Signal before/after 2nd-order difference and after normalization; spectra before/after (shows harmonic emphasis and MA suppression) |
| `plot_pipeline_overview(trace)` | One tall figure: raw → bandpass → decomposition → diff → SSR spectrum → tracking, time left / frequency right, GT marker aligned across all rows |

**C. SSA internals (`diag` SSA keys)**

| Function | Shows |
|---|---|
| `plot_ssa_embedding(trace, zoom=8)` | Heatmap of the L×K trajectory matrix; inset of the top-left `zoom×zoom` block with numbers so the Hankel (constant anti-diagonal) structure is visible |
| `plot_ssa_svd(trace)` | Singular values σ_i (log y) and cumulative energy; markers where near-equal pairs occur |
| `plot_ssa_eigenvectors(trace, idx=range(12))` | Grid of left singular vectors `u_i` (time) with their spectra; pair scatter `u_i` vs `u_{i+1}` (a circle indicates a sinusoid) |
| `plot_ssa_grouping(trace)` | σ_i coloured by group; table of groups (members, σ range, dominant bin/Hz/BPM, matched ACC bin, removed yes/no) |
| `plot_ssa_wcorr(trace)` | Weighted-correlation matrix of rank-one reconstructions (computed on demand) with group boundaries |
| `plot_ssa_components(trace, top=12)` | Each group's reconstructed series (time) and spectrum; kept green, removed red; ACC dominant bins and the component's own dominant bin marked |
| `plot_ssa_reconstruction(trace)` | Original vs cleansed vs removed sum in time and frequency; residual `‖sum of all groups − original‖` (should be ~0) |

**D. Generic decomposition (works for any plug-in filling the generic `diag` keys)**

| Function | Shows |
|---|---|
| `plot_decomposition_components(trace)` | Components, their spectra, dominant bins, removed mask, ACC bins. This is what an EMD plug-in would use out of the box |

**E. SSR internals**

| Function | Shows |
|---|---|
| `plot_ssr_dictionary(trace)` | Kept-column mask across the 0…N grid (Eq. 12 pruning) with band edges; optionally a few dictionary atoms (real part) |
| `plot_ssr_iterations(trace)` | Spectrum after iteration 0…n_iter with a Plotly slider and overlay mode; sparsity (# coefficients above 1 % of max) per iteration |
| `plot_ssr_spectrum(trace)` | Final spectrum vs periodogram of the same input, GT HR marker, ACC dominant frequencies; shows what SSR resolves that FFT smears (Fig. 1 style) |

**F. Tracking**

| Function | Shows |
|---|---|
| `plot_tracking_window(trace)` | Spectrum with R0/R1 shaded, η line, P0/P1 markers, harmonic pair links, `k_prev`, `k_b` (with case label), `k_cur` after verification, GT bin, rule annotations |
| `plot_tracking_run(run)` | Whole recording: SSR-spectrum time–frequency heatmap with estimated HR, GT HR, markers where case 2/3 or rule 1/2 fired, and an error panel (est − gt) below; hover shows window index and case |
| `plot_verification_timeline(run)` | Per window: `k_prev`, `k_b`, `k_cur`, stall counter, trend, Δs mode |

**G. Whole-recording stage views (need `light` trace)**

| Function | Shows |
|---|---|
| `plot_stage_spectrogram(run, stage)` | Heatmap of per-window stage-output spectra (BPM axis; `normalize="window"\|"global"\|None`, `log=True`) with GT and estimate overlaid |
| `plot_stage_stitched(run, stage, stitch="center"\|"ola")` | Time-domain stage output over the whole recording. `center` keeps the central `step_s` slice of each window; `ola` uses Hann-weighted overlap-add (constant-overlap-add holds at 75 % overlap). Warn that per-window filtering leaves seams |

**H. All 12 subjects**

| Function | Shows |
|---|---|
| `plot_all_subjects_stage(runs_or_traces, stage, selector=...)` | 4×3 grid of any stage plot; `selector`: window index, `"time:120"` (seconds), `"max_error"`, `"median_error"` (needs `lab.collect_traces`) |
| `plot_all_subjects_tracking(runs)` | 4×3 grid of est vs gt over time (Fig. 8 for every subject), subject titles show Error1 |
| `plot_error_bars(stats)` | Error1 and Error2 per subject as bars, paper values as markers, mean line |
| `plot_bland_altman(runs)`, `plot_scatter(runs)` | Paper Figs. 5 and 6, pooled over selected subjects |

**I. Comparison**

| Function | Shows |
|---|---|
| `compare_variants(...)` (in `lab`, drawn by `viz.compare`) | Same window, several configs: overlaid stage output in time and frequency, one legend entry per variant, stage timing in the legend |
| `compare_runs(...)` | Est-vs-gt traces per variant overlaid, plus a stats table (Error1/Error2 per subject and mean) below |

**J. Explorer (optional, last)**

`viz.explore(subjects, cfg)` builds an `ipywidgets` panel: dropdowns for subject, stage/plot, and a window slider; re-renders the selected plot. Must work in VS Code notebooks.

---

## 6. Plug-in authoring workflow (for the user's own alternatives)

1. **Quick trial in a notebook** with `@register_function` or a class decorated with `@register(...)` (Section 2.2). Use `%load_ext autoreload` and `override=True`.
2. Try it: `cfg2 = cfg.with_overrides({"decomposition.method": "my_variant", "decomposition.n_modes": 8})`, then `lab.compare_variants(...)`, `viz.plot_stage(...)`, `lab.run_all(cfg2, ...)` and `lab.evaluate`.
3. **Fill the `diag` schema** for the slot so the generic plots work (decomposition: `components`, `component_labels`, `removed_mask`, `component_dominant_bins`). Extra plot-specific keys are optional.
4. **Contract check:** `troika.testing.check_plugin("decomposition", "my_variant")` runs the plug-in on synthetic input and verifies: output length/shape, finite values, no mutation of the input, deterministic on repeated calls, `diag` keys have the right shapes, and it does not read `ctx.gt_bpm`. Returns a readable report.
5. **Promote to a file:** `python -m troika.plugins.new decomposition my_variant` scaffolds `src/troika/plugins/decomposition/my_variant.py` (Params dataclass, class skeleton, docstring listing the required `diag` keys) and `tests/plugins/test_my_variant.py` calling `check_plugin`. Paste the notebook code in.
6. **Promote the config:** `lab.promote(cfg2, "configs/my_experiment.yaml")` writes only the differences from default (with the new method's full `params`), then `python experiments/run_all.py --config configs/my_experiment.yaml` gives the final numbers.

Example: an EMD variant of `decomposition` takes the band-passed PPG window and context (which includes the ACC dominant bins), splits it into modes, drops the modes whose dominant bin matches `ctx.acc_bins`, and returns the sum of the kept modes. Every mode is reported in `diag["components"]`, so `plot_decomposition_components` and `plot_stage` work without extra code. (The user writes it; do not implement it.)

---

## 7. Files added or changed

```
src/troika/
├── registry.py                      # NEW: register, register_function, get, available
├── trace.py                         # NEW: StageOutput, WindowContext, WindowTrace, light/full records
├── lab.py                           # NEW: high-level API for notebooks (Section 4)
├── testing.py                       # NEW: check_plugin
├── config.py                        # CHANGED: slot layout, merge, overrides, diff, hash, promote
├── pipeline.py                      # CHANGED: build slots from config, sequential windows, tracing, leak guard
├── plugins/                         # NEW: thin adapters over the tested core functions
│   ├── base.py
│   ├── new.py                       # scaffold generator
│   ├── bandpass/{butter.py, fir.py}
│   ├── decomposition/{ssa.py, none.py}       # ssa wraps decomposition/ssa.py, grouping.py, motion_removal.py
│   ├── temporal_diff/{diff.py, none.py}
│   ├── spectrum/{focuss.py, fft.py}
│   └── tracker/troika.py            # wraps tracking/*
└── viz/
    ├── __init__.py                  # setup_notebook, save, re-exports
    ├── theme.py
    ├── data.py                      # A
    ├── stages.py                    # B, D, G
    ├── ssa.py                       # C
    ├── ssr.py                       # E
    ├── tracking.py                  # F
    ├── overview.py                  # pipeline overview, all-12 grids (H)
    ├── compare.py                   # I
    └── explore.py                   # J
notebooks/                           # percent-format .py sources (Section 8)
tests/                               # Section 9
configs/{default.yaml, my_experiment.yaml, ablations/*.yaml}   # new slot layout
experiments/run_all.py               # CHANGED: thin CLI over lab
pyproject.toml                       # extras: lab = plotly, nbformat, ipykernel, ipywidgets, jupytext, tqdm; optional kaleido
```
Keep the Blueprint's `decomposition/`, `ssr/`, `tracking/`, `preprocessing/` packages as the tested implementation layer. The plug-ins only adapt them and fill `diag`.

---

## 8. Notebooks (VS Code, cell by cell)

**Format.** Author each notebook as a Python file with `# %%` cell markers (VS Code runs these cell by cell in the Interactive Window and they diff cleanly in git). Configure `jupytext` so `.ipynb` versions can be generated with one command (`make notebooks`). Every notebook starts with:

```python
# %% setup
%load_ext autoreload
%autoreload 2
from troika import lab, viz
viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg)
```

**VS Code notes** (put in README): Python and Jupyter extensions, select the project's virtual environment as kernel, `pip install -e ".[lab]"`, click "Run Cell" above any `# %%`.

| Notebook | Cells |
|---|---|
| `00_data.py` | Load 12 subjects; `plot_recording` for one subject (time and frequency); `plot_all_subjects` in time, freq, spectrogram; protocol shading |
| `01_bandpass_and_acc.py` | Trace one window; `plot_stage(trace, "bandpass")`; `plot_filter_response`; `plot_acc_dominant`; **variant cell**: `fir` vs `butter` with `compare_variants` |
| `02_ssa_internals.py` | Embedding, SVD, eigenvectors, grouping, w-correlation, components, reconstruction, with `L` and `grouping` changed in-cell to see the effect |
| `03_diff_and_ssr.py` | `plot_temporal_diff`, `plot_ssr_dictionary`, `plot_ssr_iterations`, `plot_ssr_spectrum`; **variant cell**: `fft` vs `focuss`, `lam` and `n_iter` sweeps |
| `04_tracking.py` | `plot_tracking_window` for chosen windows, `plot_tracking_run` and `plot_verification_timeline` for a whole recording |
| `05_overview_all12.py` | `plot_pipeline_overview` for one subject; `run_all` on 12; `plot_all_subjects_stage` with `selector="max_error"`; `plot_all_subjects_tracking`; `plot_error_bars`; Bland–Altman and scatter |
| `06_try_variants.py` | Register a notebook plug-in (template stub), `compare_variants` and `compare_runs` between default and modified configs, `evaluate` on one subject and on all 12, `promote` to `configs/my_experiment.yaml` |

Each notebook must run top to bottom on the real data. Notebooks that need a full trace use `lab.get_window_trace` with `prev_bin="tracked"` unless the cell is explicitly labelled as a contaminated exploration cell.

---

## 9. Tests for the new layer

| Test | Verifies |
|---|---|
| `test_registry.py` | Register/get/override, notebook-style function registration, unknown method error lists options |
| `test_config_slots.py` | Merge with partial file, unknown-key error, method switch resets params, shorthand override, `diff_from_default` → `to_yaml` → reload round trip equals the original config, stable hash |
| `test_regression_slots.py` | **Refactor guard:** the slot-based pipeline reproduces the pre-refactor `bpm_est` on a fixed synthetic recording (golden file captured before the refactor) |
| `test_trace.py` | `light` trace size under budget on a 300 s synthetic recording; `full` trace contains every Section 2.3 key for paper plug-ins; isolated-window `tracked` equals the corresponding window of a normal run |
| `test_leak_guard.py` | Plug-ins see `gt_bpm is None` by default; GT-seeded runs are `contaminated`; `evaluate` refuses them |
| `test_stats.py` | Subset vs all-12 rows; paper-delta columns; bias/RMSE hand-checked |
| `test_viz_smoke.py` | Every function in Section 5.2 returns a `Figure` on a synthetic recording (no real data needed); missing-key case raises the clear error; `save` writes HTML |
| `test_plugin_contract.py` | `check_plugin` passes for all built-ins and fails for deliberately broken ones (wrong length, NaNs, input mutation, reading GT); scaffold output imports |

---

## 10. Milestones

| ID | Work | Done when |
|---|---|---|
| L0 | Slot refactor: registry, plug-in base classes, config layout, paper-default plug-ins as adapters | `test_regression_slots.py` passes; `run_all.py` output unchanged |
| L1 | Tracing (light/full), isolated-window mode, leak guard | `test_trace.py`, `test_leak_guard.py` pass |
| L2 | `lab` API and statistics table; `run_all.py` becomes a thin wrapper; `--set`, `--subjects` | Per-subject and all-12 stats for default and experiment configs |
| L3 | `viz` foundation: theme, save, data plots (A), all-12 data grids, bandpass/acc/diff plots (B) | Notebooks 00 and 01 run |
| L4 | SSA internals (C) and generic decomposition plot (D) | Notebook 02 runs; reconstruction residual ≈ 0 shown |
| L5 | SSR (E), tracking (F), pipeline overview, whole-recording views (G), all-12 stage plots (H) | Notebooks 03 to 05 run |
| L6 | Variant comparison (I), `promote`, scaffold generator, `check_plugin` | Notebook 06 runs; a dummy plug-in goes from notebook to file to config without errors |
| L7 | `viz.explore` widget, README section for VS Code, `make notebooks` | Explorer works in a VS Code notebook |

---

## 11. Conventions for Claude Code

- Add new behaviour without changing the numerical results of the paper-default path. Guard with the regression test before and after L0.
- Plots never compute pipeline results themselves except cheap derived views (periodogram of a stored signal, w-correlation). Everything else comes from traces.
- Plotly figures only; no matplotlib in `viz`.
- Every public function has a docstring stating required trace level and `diag` keys.
- Keep plot code separate from data code so figures can be unit-tested with synthetic traces.
- No plug-in may read ground truth; only the evaluation layer and the leak-guarded exploration paths may.

---

## 12. Where this file supersedes the Blueprint

1. Config layout: flat sections (`ssa`, `ssr`, `tracking`, `temporal_diff`, `bandpass`) become slot sections with `method` + `params`; `ssa.enabled: false` becomes `decomposition.method: none`; `ssr.method` becomes `spectrum_estimator.method`; `spectrum.n_fft` becomes `grid.n_fft`. Blueprint Section 6 and 6.1 examples (`--set ssa.L=300`) become `--set decomposition.L=300`.
2. `pipeline.py` builds stages from the registry instead of calling modules directly; `TroikaEstimator.run` gains trace and isolated-window options.
3. New `run:` fields (`allow_ground_truth_access`, `trace`) and `evaluation.within_bpm`.
4. Ablation YAMLs use the new keys.

---

## 13. Assumptions made in this spec (change if wrong)

- Stage plots are per window; whole-recording views are built from light traces (spectrograms and stitched signals).
- The tracker is one slot with no alternatives; its internals are inspected, not swapped.
- `bandpass` runs on the stacked `(4, M)` array so PPG and ACC share one filter.
- All-12 grids are 4 rows × 3 columns.
- Extra statistics (bias, RMSE, median, max, within-5-BPM) are added beyond the paper's Error1/Error2; Error1 is the "mean error".
- Percent-format `.py` notebooks are the source of truth; `.ipynb` files are generated.
