# TROIKA Implementation Blueprint

Heart-rate monitoring from wrist-type PPG during intensive exercise
Paper: Zhang, Pi, Liu, *"TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type PPG Signals During Intensive Physical Exercise"*, IEEE TBME, 2014 (DOI 10.1109/TBME.2014.2359372)

This document is the spec for a from-scratch implementation. It is meant to be handed to Claude Code in VS Code. It contains: (1) a full analysis of the paper, (2) the exact algorithm per stage, (3) the repository layout with module/function contracts, (4) the experiments needed to reproduce the paper's tables and figures, (5) a test plan and build order, and (6) a register of everything the paper leaves unspecified.

Tags used throughout:
- **[PAPER]** = stated explicitly in the paper.
- **[ASSUMPTION Ax]** = the paper is silent or ambiguous; I chose a default. Each is listed in Section 11 and must be a config option, not hard-coded.

Language/stack assumed: Python 3.10+, NumPy, SciPy, pandas, matplotlib, PyYAML, pytest. (The original code is MATLAB; nothing here depends on that.)

---

## 1. Paper at a glance

**Problem.** Estimate heart rate (HR) from a single-channel wrist PPG signal while the wearer runs (up to 15 km/h). Hand movement produces motion artifacts (MA) that are much stronger than the pulse component, and MA spectra overlap with the HR band.

**Why standard approaches fail (from the paper).**
- ICA needs multiple sensors, and its independence assumption fails for MA-contaminated PPG.
- Adaptive noise cancellation depends on a good reference signal, which is hard to build during exercise.
- Accelerometer-only methods are weak because hand-motion in 3-D is not the same as skin–sensor gap change.
- Periodogram (FFT) has high variance and leakage: a strong MA peak smears a nearby weak HR peak (Fig. 1).
- MUSIC-type line-spectrum methods need a model order, which is unknowable here.

**The TROIKA idea.** Three stages, each fixing a specific failure:

| Stage | Name | Purpose |
|---|---|---|
| 1 | Signal decomposiTion (SSA) | Remove MA components (identified via accelerometer spectrum) and sparsify the PPG spectrum |
| 2 | sparse signal RecOnstructIon (FOCUSS) | High-resolution spectrum with low variance, no leakage, no model order |
| 3 | spectral peaK trAcking (+ verification) | Pick the HR peak using the previous estimate and harmonic structure; handle "peak missing / buried" cases |

Plus two auxiliary operations: band-pass filtering (0.4–5 Hz) up front and a 2nd-order temporal difference between stage 1 and stage 2.

**Headline results to reproduce (12 subjects, 5-min treadmill runs, peak 15 km/h).**

| Metric | Paper value |
|---|---|
| Mean absolute error (Error1), mean ± std across subjects | 2.34 ± 0.82 BPM |
| Mean absolute error percentage (Error2) | 1.80 % |
| Bland–Altman limits of agreement (pooled windows) | [−7.26, 4.79] BPM, σ = 3.07 BPM |
| Pearson r (pooled windows) | 0.992 |
| Ablation | Removing SSA, replacing FOCUSS with FFT, or removing verification each causes tracking failure on some subjects (esp. Subject 6) |

**Caveats worth knowing before you start.** 12 male subjects aged 18–35, one dataset, no held-out evaluation, and the hyperparameters were set heuristically (possibly looking at the same data). Treat the numbers as a target, not a guarantee. Also, several algorithmic details are underspecified (Section 11), so expect small deviations from the paper's exact numbers.

---

## 2. Data and protocol

**[PAPER]** 12 male subjects; per subject one recording containing:
- 1 PPG channel (wrist, green LED),
- 3-axis accelerometer (wrist),
- ECG (chest, wet electrodes), used **only** for ground truth.

All signals sampled at **fs = 125 Hz**. Treadmill protocol (total 5 min = 300 s):

| Segment | Speed | Duration |
|---|---|---|
| 1 | 1–2 km/h | 0.5 min |
| 2 | 6–8 km/h | 1 min |
| 3 | 12–15 km/h | 1 min |
| 4 | 6–8 km/h | 1 min |
| 5 | 12–15 km/h | 1 min |
| 6 | 1–2 km/h | 0.5 min |

Subjects were told to deliberately pull clothes, wipe sweat, and push treadmill buttons with the wristband hand, besides free swinging. That is where the hardest MA comes from.

**Data location [PAPER]:** the author's homepage (`https://sites.google.com/site/researchbyzhang/`). Download manually; never commit raw data.

**File layout: verify before coding.** The widely circulated version of this data (the IEEE Signal Processing Cup 2015 training set, 12 subjects) is, from my memory and *not* from the paper, one `.mat` per recording containing a 6×n array `sig` with rows `[ECG, PPG ch1, PPG ch2, ACC x, ACC y, ACC z]` plus a per-window ground-truth BPM file. The loader must first print keys/shapes of a file and be adapted accordingly. The paper uses **one** PPG channel; if the files have two, use channel 1 **[ASSUMPTION A15]**.

**Window count.** With T = 8 s, S = 2 s, n samples: `W = floor((n − T·fs) / (S·fs)) + 1`, about 147 windows for a 300 s recording.

---

## 3. Notation and index conventions (read this twice)

The paper uses **1-based** frequency-bin indices (MATLAB). Use **0-based** in code and convert once, centrally, in `binmap.py`.

| Symbol | Meaning | Default |
|---|---|---|
| fs | sampling rate | 125 Hz |
| T, S | window length / step | 8 s, 2 s |
| M | samples per window (T·fs) | 1000 |
| M′ | samples after 2nd-order difference (M − 2) | **998** |
| L, K | SSA window length / K = M − L + 1 | 400, 601 |
| N | number of frequency grid points (FFT / SSR grid) | 4096 |
| k | 0-based frequency bin, f = k·fs/N | |
| Δf | bin width = fs/N | 0.0305 Hz = **1.831 BPM** |
| BPM(k) | 60·k·fs/N | |

Convert paper formulas to 0-based with `k = N_f − 1`:

| Paper (1-based) | Code (0-based) |
|---|---|
| f = (N_f − 1)/N · fs | f = k · fs / N |
| Harmonic range R1 = [2(N_prev−Δs−1)+1, …, 2(N_prev+Δs−1)+1] | [2(k_prev − Δs), …, 2(k_prev + Δs)] |
| Case 2 candidate (N¹ − 1)/2 + 1 | round(k¹ / 2) |
| Ncur = Nprev + 2·NTrend | same (differences are offset-invariant) |

Sanity numbers that must hold in tests:
- θ = 6 bins ≈ 11 BPM, τ = 2 bins ≈ 3.7 BPM, Δs = 16 bins ≈ 29 BPM, Δ = 10 bins ≈ 0.305 Hz **[PAPER states 0.3 Hz]**.
- Remark 3 of the paper says one grid step is "about 1 BPM". With N = 4096, fs = 125 it is actually 1.83 BPM. Use the real number. Quantization alone contributes error, so the final BPM is on a 1.83-BPM grid.

---

## 4. End-to-end workflow

```
raw PPG (1 ch)  ─┐                      raw ACC (3 ch)
                 │                            │
          sliding window T=8 s, step S=2 s (per recording)
                 │                            │
        band-pass 0.4–5 Hz            band-pass 0.4–5 Hz
                 │                            │
                 │                  periodogram (4096-pt) per axis
                 │                  dominant bins: peaks > 50 % of axis max
                 │                  F_acc = union over 3 axes
                 │                            │
                 │        remove bins within ±Δ of previous HR (fund. + harmonic)
                 │                            │  → F̃_acc
                 ▼                            ▼
        ┌──────────────── SSA ─────────────────┐
        │ embed (L×K) → SVD → group → diag-avg │
        │ drop groups whose dominant bin ∈ F̃_acc │
        │ sum remaining groups → cleansed PPG  │
        └──────────────────┬───────────────────┘
                           ▼
              2nd-order temporal difference   (M → M′ = 998)
                           ▼
        FOCUSS sparse spectrum (p=0.8, λ=0.1, 5 iterations,
        pruned DFT dictionary, N=4096)  →  s[k] = |x_k|²
                           ▼
        Spectral peak tracking (uses previous estimate k_prev)
          • search R0 (fundamental) and R1 (2nd harmonic)
          • Case 1 harmonic pair / Case 2 closest / Case 3 keep previous
          • Verification rule 1: limit jump;  rule 2: anti-stall + trend
                           ▼
              k_cur → BPM = 60·k_cur·fs/N   ──►  k_prev for next window
                           │
        ground truth (ECG) per window ──► Error1, Error2, Bland–Altman, Pearson
```

The pipeline is **stateful across windows** (previous estimate feeds both SSA and tracking), so windows of one recording must be processed sequentially. Parallelize across subjects only.

---

## 5. Stage-by-stage specification

### 5.1 Windowing and band-pass filtering

- **[PAPER]** Window T = 8 s, step S = 2 s (75 % overlap). Choose T large for resolution and S small so successive estimates stay close.
- **[PAPER]** Within each window, band-pass **both** the PPG and the three acceleration channels to 0.4–5 Hz. This removes out-of-band MA and partly sparsifies the spectrum.
- **[ASSUMPTION A1]** Filter design unspecified: default 4th-order Butterworth, zero-phase (`sosfiltfilt`). Alternative: FIR (`firwin`) + `filtfilt`.
- **[ASSUMPTION A2]** Paper filters per window (edge effects at both ends). Default `per_window` to follow the paper; provide `global` (filter the whole recording once) for comparison.

### 5.2 Acceleration dominant frequencies (F_acc)

For each of the 3 acceleration axes in the current window:
1. Periodogram with 4096-point FFT: `P[k] = |FFT(x, 4096)[k]|² / len(x)`, k = 0…N/2.
2. Restrict to the 0.4–5 Hz bins **[ASSUMPTION A19]** (the signal is band-passed anyway).
3. Find local maxima; keep those with amplitude **> 50 % of that axis's maximum**. **[PAPER]**
4. `F_acc` = union of the kept bin indices across the three axes. **[PAPER]**

Refinement using the previous HR estimate **[PAPER]**:
- `Np` = bins of the heartbeat fundamental and harmonic from the previous window: `{k_prev, 2·k_prev}`. **[ASSUMPTION A6]** (paper says "fundamental and harmonic"; default first harmonic only; make `n_harmonics` configurable.)
- Remove from `F_acc` every bin within `±Δ` (Δ = 10) of any element of `Np`. Result: `F̃_acc`.
- For the very first window there is no previous estimate: use `F̃_acc = F_acc`.

Rationale: if the wearer's cadence happens to coincide with HR (common when running), removing that component would also remove the heartbeat.

### 5.3 SSA decomposition (Embedding → SVD → Grouping → Reconstruction)

Applied to the band-passed PPG window `y ∈ R^M`, M = 1000. **[PAPER]** L = 400 (rule of thumb: L close to M/2; L < M/2).

**Embedding.** Build the L×K Hankel trajectory matrix (K = M − L + 1 = 601):
`Y[i, j] = y[i + j]`, i = 0…L−1, j = 0…K−1.

**SVD.** `Y = Σ_{i=1}^{d} σ_i u_i v_iᵀ`, d = min(L, K) = 400. Use `np.linalg.svd(Y, full_matrices=False)`.

**Grouping.** Partition eigentriple indices {1…d} into disjoint groups; each group `Y_I = Σ_{t∈I} σ_t u_t v_tᵀ`.
- **[PAPER]** Grouping is "automatically finished by clustering singular values as described in [23, p. 66]" (Golyandina et al. book). The paper says only that eigentriples in a group should share a common trait, e.g. same frequency or harmonic relation. The exact algorithm is **not given** **[ASSUMPTION A4]**.
- Implement grouping as a swappable strategy in `grouping.py`:
  1. `frequency_pairing` (**default**). For each eigentriple, compute the dominant frequency bin of its rank-one reconstruction (or of `u_i` zero-padded to 4096). Merge consecutive eigentriples when their singular values are close (relative gap < `sv_rel_tol`, default 0.1) **and** their dominant bins differ by ≤ `freq_tol` bins (default 2). A sinusoid produces a pair of near-equal singular values with the same frequency, which is exactly what this captures. Leftovers stay singletons.
  2. `wcorr_hclust`: hierarchical clustering on the weighted-correlation matrix of the rank-one reconstructions (the classical SSA approach).
  3. `singleton`: each eigentriple is its own group (baseline / debugging).
- Evaluate strategies on Subjects 5 and 6 and keep the one that best matches Figs. 7–9 and Table I.

**Reconstruction (diagonal averaging / Hankelization).** For a group matrix `Y_I` (L×K), the reconstructed series has length M:
`ỹ[n] = mean of Y_I[i, j] over all (i, j) with i + j = n`.
Efficient implementation: precompute `idx = (i + j).ravel()` and `counts = np.bincount(idx)` once per (L, K); then `ỹ = np.bincount(idx, weights=Y_I.ravel()) / counts`. Identity check: `Σ_p ỹ_p == y` (up to ~1e-9).

**MA component removal [PAPER].**
1. For each reconstructed group series `ỹ_p`, compute its dominant frequency bin `d_p` (argmax of the 4096-point periodogram within 0.4–5 Hz).
2. If `d_p ∈ F̃_acc`, mark group p as MA and drop it.
3. `y_recon = Σ_{p ∉ I_noise} ỹ_p`. This is the cleansed PPG.
- **[ASSUMPTION A5]** "∈ F̃_acc" is implemented as `min_a |d_p − a| ≤ match_tol` bins; default `match_tol = 2` (bin-exact matching is fragile). Ablate with 0.
- **Fallbacks:** if every group is removed, use the band-passed PPG (log a warning). If none removed, the result is the band-passed PPG (identity).

### 5.4 Temporal difference

**[PAPER]** Compute the **second-order difference** of the cleansed PPG: `z = np.diff(y_recon, n=2)`, length M′ = 998. Rationale: differencing preserves the fundamental and harmonics of the (quasi-)periodic heartbeat while suppressing the aperiodic MA and random spectral fluctuations (except rhythmic swing).

**[ASSUMPTION A3]** Normalize `z` (zero mean, unit variance) before SSR so that λ = 0.1 has a consistent meaning across subjects. The paper never states this but plots normalized signals in Fig. 9. Make it a config flag and tune λ if you disable it.

### 5.5 SSR with FOCUSS

**Model [PAPER].** `y = Φx + v`, Φ ∈ C^{M′×N}, `Φ[m, n] = exp(j·2π·m·n / N)`, m = 0…M′−1, n = 0…N−1. Estimated spectrum: `s[k] = |x̂_k|²`.

**Column pruning [PAPER, Eq. 12].** Only bins inside the band-pass region (plus transition margin) can be non-zero, so drop the other columns of Φ. The paper gives (1-based):

```
I_Φ = [ f_lo·N/fs + 1 − Δf1 ,  f_hi·N/fs + 1 + Δf2 ]  ∪  [ N − f_hi·N/fs + 1 − Δf2 ,  N − f_lo·N/fs + 1 + Δf1 ]
with f_lo = 0.4, f_hi = 5, Δf1 = f_lo·N/fs − 1, Δf2 = 2·N/fs
```

Worked out for N = 4096, fs = 125: positive side covers 1-based bins ≈ 2 … 230 (≈ 0.03 – 7 Hz), i.e. 0-based bins **1 … 229**; the negative side is its mirror, 0-based **3867 … 4095**. About 458 columns instead of 4096. Implementation notes:
- **[ASSUMPTION A12]** Round lower bounds up and upper bounds down, then convert to 0-based.
- Unit test: the kept set is symmetric under `k → N − k`, and excludes k = 0.
- Keep both positive and negative frequency columns (real input → complex-conjugate-symmetric spectrum).

**Algorithm: Regularized M-FOCUSS [PAPER, ref. 28], single measurement vector.** Parameters **[PAPER]**: p = 0.8, λ = 0.1, **5 iterations** (no need to converge), N = 4096.

Iteration k = 1…5, starting from `x₀ = 1` (all ones) **[ASSUMPTION A11]**:
```
W_k   = diag( |x_{k-1}|^(1 − p/2) )                       # real diagonal
q_k   = (Φ W_k)ᴴ ( Φ W_k (Φ W_k)ᴴ + λ I_M′ )⁻¹ y
x_k   = W_k q_k
```
Equivalent, and much cheaper since n_cols ≈ 458 < M′ = 998 (push-through identity):
```
G     = Φᴴ Φ                       # n×n, precompute ONCE per (M′, N, kept columns) and cache
b     = Φᴴ y                       # once per window
A_k   = (w wᵀ) ∘ G + λ I_n         # w = |x_{k-1}|^(1 − p/2);  ∘ = elementwise
q_k   = solve(A_k, w ∘ b)
x_k   = w ∘ q_k
```
Add a unit test that the two forms agree to ~1e-8.

Return `s = |x|²` mapped back onto a length-(N/2 + 1) array (positive-frequency bins 0…N/2; zeros for pruned bins). Peak tracking only looks at positive bins.

**Alternative estimator for ablation [PAPER].** Replace SSR with the Periodogram: `s[k] = |FFT(z, N)[k]|²`, same output shape. Everything else unchanged.

**Why so much preprocessing before SSR (Remarks 1–3 of the paper).** SSR assumes a sparse spectrum; with N ≫ M′ the columns of Φ are highly correlated, so even a few strong MA components hurt. Larger N reduces off-grid error but raises column correlation, hence the N = 4096 compromise.

### 5.6 Spectral peak tracking

Stateful object; state = `{k_prev, bpm_history, same_count, lost}`.

**(a) Initialization [PAPER].** The wearer holds the hand still for the first few seconds (the first protocol segment is 1–2 km/h). For the first window, take the **highest spectral peak** of `s` as the HR bin: `k_prev = argmax s`. **[ASSUMPTION A10]** Restrict the argmax to a plausible band, default 40–200 BPM. Add a debug option `init_mode: ground_truth` (uses ECG GT for window 0) to isolate initialization failures from tracking failures. Never use it for reported results.

**(b) Peak selection [PAPER].** Let `Δs = 16` (or 20 in "lost" mode, see rule 2).
- Search range for fundamental: `R0 = [k_prev − Δs, k_prev + Δs]`.
- Search range for first harmonic: `R1 = [2(k_prev − Δs), 2(k_prev + Δs)]`. Clip both to valid bins.
- In each range take the local maxima (use `scipy.signal.find_peaks` on `s`; **[ASSUMPTION A16]** plateaus handled by find_peaks), keep **at most 3 highest** whose amplitude ≥ η, where **η = 30 % of the highest peak in R0**. **[PAPER]**
  - **[ASSUMPTION A9]** Use that same absolute η for R1. If R0 has no peak, fall back to 30 % of R1's own maximum.
- Let `P0 = {k⁰_i}` (≤3 peaks in R0), `P1 = {k¹_j}` (≤3 peaks in R1).

Three cases **[PAPER]**; `k_b` = selected bin:
- **Case 1**: there is a pair `(k⁰_i, k¹_j)` in harmonic relation, i.e. `|k¹_j − 2·k⁰_i| ≤ harm_tol` → `k_b = k⁰_i`. **[ASSUMPTION A7]** `harm_tol = 2` bins. **[ASSUMPTION A8]** If several pairs qualify, choose the pair with the largest `s[k⁰] + s[k¹]`.
- **Case 2**: peaks exist but no harmonic pair → candidates `C = P0 ∪ { round(k¹_j / 2) : k¹_j ∈ P1 }` **[ASSUMPTION A17 on rounding]**, and `k_b = argmin_{c∈C} |c − k_prev|`.
- **Case 3**: no peaks in either range → `k_b = k_prev` (HR peaks tend to stay put between overlapping windows).

**(c) Verification [PAPER].**

*Rule 1, limit the jump.* HR rarely changes > 10 BPM between windows. With θ = 6 bins, τ = 2 bins:
```
if   k_b − k_prev ≥  θ:  k_cur = k_prev + τ
elif k_b − k_prev ≤ −θ:  k_cur = k_prev − τ
else:                     k_cur = k_b
```

*Rule 2, anti-stall / trend recovery.* If `k_b == k_prev` for **h = 3** successive windows (the counter includes the current window):
```
k_cur = k_prev + 2 · NTrend                      # NTrend ∈ {−1, 0, +1}
```
and the search range is broadened to `Δs = 20` while this condition holds (otherwise `Δs = 16`). **[ASSUMPTION A13]** The broadening applies to the *next* window's search, since the current window's search has already run. NTrend:
```
BPM_prev    = last estimated BPM
BPM_predict = 3rd-order polynomial fit on the last (up to) 20 estimated BPM values, extrapolated one step ahead
NTrend = +1 if BPM_predict − BPM_prev ≥  3
         −1 if BPM_predict − BPM_prev ≤ −3
          0 otherwise
```
If fewer than 4 history points exist, NTrend = 0. Suppress polyfit's RankWarning.

Output: `BPM = 60 · k_cur · fs / N`; then `k_prev ← k_cur`, append BPM to history.

### 5.7 Ground truth [PAPER, Sec. IV-C]

Per window: count cardiac cycles `H` in the window's duration `D` seconds, `HR = 60·H/D`. No ECG HR-estimation algorithm was used, to avoid its errors.
- **[ASSUMPTION A14]** If a per-window ground-truth BPM file ships with the data, use it. Otherwise detect R-peaks (e.g., band-pass 5–20 Hz + `scipy.signal.find_peaks` with a refractory distance ≈ 0.3 s) and compute `HR = 60 · (n_peaks − 1) / (t_last − t_first)` inside each window. Cross-check the two sources when both exist (expect agreement within ~1 BPM).

### 5.8 Metrics [PAPER, Eqs. 19–20 and Bland–Altman]

Per subject, over its W windows:
- **Error1** = mean(|est − gt|)
- **Error2** = mean(|est − gt| / gt) (report in %)

Across subjects: mean ± **sample** std (ddof = 1) of the 12 per-subject values. I verified against Table I row 1 that ddof = 1 gives 2.34 ± 0.82.

Pooled over **all windows of all subjects**:
- Bland–Altman: `diff = est − gt` (**sign convention not stated; the paper's mean is ≈ −1.24, so estimates are slightly low**), x-axis = (est + gt)/2, `μ = mean(diff)`, `σ = std(diff)`, LOA = `[μ − 1.96σ, μ + 1.96σ]`.
- Pearson correlation between est and gt.

---

## 6. Default configuration (`configs/default.yaml`)

```yaml
signal:
  fs: 125
  window_s: 8
  step_s: 2
  ppg_channel: 0            # A15

bandpass:
  low_hz: 0.4
  high_hz: 5.0
  design: butter            # A1: butter | fir
  order: 4
  mode: per_window          # A2: per_window | global

spectrum:
  n_fft: 4096               # N (FFT and SSR grid)

acc_dominant:
  rel_threshold: 0.5        # PAPER: 50 % of axis max
  exclude_delta_bins: 10    # PAPER: Δ
  n_harmonics: 2            # A6: fundamental + 1st harmonic

ssa:
  enabled: true
  L: 400                    # PAPER
  grouping: frequency_pairing   # A4
  sv_rel_tol: 0.1
  freq_tol_bins: 2
  match_tol_bins: 2         # A5

temporal_diff:
  enabled: true
  order: 2                  # PAPER
  normalize_after: true     # A3

ssr:
  method: focuss            # focuss | fft
  p: 0.8                    # PAPER
  lam: 0.1                  # PAPER
  n_iter: 5                 # PAPER
  x0: ones                  # A11
  prune_columns: true

tracking:
  init_mode: max_peak       # max_peak | ground_truth (debug only)
  init_band_bpm: [40, 200]  # A10
  delta_s: 16               # PAPER
  delta_s_wide: 20          # PAPER
  max_peaks_per_range: 3    # PAPER
  eta_frac: 0.30            # PAPER
  harm_tol_bins: 2          # A7
  verification:
    enabled: true
    theta_bins: 6           # PAPER
    tau_bins: 2             # PAPER
    stall_windows_h: 3      # PAPER
    trend_history_windows: 20   # PAPER
    trend_poly_order: 3     # PAPER
    trend_threshold_bpm: 3  # PAPER

evaluation:
  std_ddof: 1
```

If `fs` changes, scale `n_fft` proportionally (paper: fs = 25 Hz → N ≈ 4096/5) so that Δs, Δ, θ, τ keep the same physical meaning.

---

## 7. Repository structure

```
troika-hr/
├── CLAUDE.md                        # conventions for Claude Code (Section 12)
├── README.md
├── pyproject.toml                   # numpy, scipy, pandas, matplotlib, pyyaml, pytest, joblib
├── configs/
│   ├── default.yaml
│   └── ablations/
│       ├── no_ssa.yaml              # ssa.enabled: false
│       ├── fft_instead_of_ssr.yaml  # ssr.method: fft
│       └── no_verification.yaml     # tracking.verification.enabled: false
├── data/                            # gitignored
│   └── raw/
├── docs/
│   └── TROIKA_Implementation_Blueprint.md
├── src/troika/
│   ├── __init__.py
│   ├── config.py                    # dataclasses + YAML loader + validation
│   ├── types.py                     # Recording, WindowResult, RunResult
│   ├── binmap.py                    # bin <-> Hz <-> BPM (single source of truth)
│   ├── io/
│   │   ├── loader.py
│   │   └── ground_truth.py
│   ├── preprocessing/
│   │   ├── windowing.py
│   │   ├── bandpass.py
│   │   ├── spectrum.py
│   │   └── temporal_diff.py
│   ├── decomposition/
│   │   ├── ssa.py                   # embed, svd, hankelize
│   │   ├── grouping.py              # frequency_pairing | wcorr_hclust | singleton
│   │   └── motion_removal.py        # F_acc logic + component removal
│   ├── ssr/
│   │   ├── basis.py                 # pruned DFT dictionary, cached Gram matrix
│   │   ├── focuss.py
│   │   └── estimators.py            # SSRSpectrum, FFTSpectrum
│   ├── tracking/
│   │   ├── peaks.py                 # local maxima / top-k helpers
│   │   ├── selection.py             # Cases 1-3
│   │   ├── verification.py          # rule 1, rule 2, trend predictor
│   │   └── tracker.py               # stateful SpectralPeakTracker
│   ├── pipeline.py                  # TroikaEstimator
│   └── evaluation/
│       ├── metrics.py
│       ├── plots.py
│       └── report.py                # tables I/II style CSV + markdown
├── experiments/
│   ├── run_subject.py               # one subject, debug plots
│   ├── run_all.py                   # main results (Tables I/II row 1, Figs. 5, 6, 8)
│   ├── run_ablation.py              # Tables I/II rows 2-4, Fig. 7
│   ├── run_sensitivity.py           # Fig. 10
│   └── make_figures.py              # Figs. 1-3, 9 style illustrations
├── tests/
│   ├── reference/paper_tables.csv   # Table I/II numbers from Section 8
│   ├── test_binmap.py
│   ├── test_bandpass_windowing.py
│   ├── test_ssa.py
│   ├── test_motion_removal.py
│   ├── test_basis_focuss.py
│   ├── test_tracking.py
│   ├── test_metrics.py
│   └── test_pipeline_synthetic.py
├── notebooks/                       # exploration only
└── results/                         # gitignored: per-run folder with csv/json/png
```

### 7.1 Data types (`types.py`)

```python
@dataclass(frozen=True)
class Recording:
    subject_id: int
    fs: float
    ppg: np.ndarray                 # (n,)
    acc: np.ndarray                 # (3, n)
    ecg: np.ndarray | None          # (n,)
    bpm_gt: np.ndarray | None       # (W,) if provided with the dataset

@dataclass
class WindowResult:
    idx: int
    t_start_s: float
    bin_cur: int
    bpm_est: float
    case: int | None                # 1, 2, 3, or None for the init window
    rule1_fired: bool
    rule2_fired: bool
    f_acc: list[int]                # refined F̃_acc
    n_groups_removed: int
    spectrum: np.ndarray | None     # optional, for debugging plots

@dataclass
class RunResult:
    subject_id: int
    bpm_est: np.ndarray
    bpm_gt: np.ndarray
    windows: list[WindowResult]
```

### 7.2 Module contracts

**`binmap.py`**
```python
def bin_to_hz(k, fs, N) -> float
def bin_to_bpm(k, fs, N) -> float
def bpm_to_bin(bpm, fs, N) -> int          # nearest bin
def band_to_bins(f_lo, f_hi, fs, N) -> tuple[int, int]
```

**`io/loader.py`**
```python
def load_recording(path: str, subject_id: int, cfg) -> Recording
def list_subjects(data_dir: str) -> list[tuple[int, str]]   # (subject_id, path), 12 entries expected
```

**`io/ground_truth.py`**
```python
def gt_from_file(path, n_windows) -> np.ndarray
def gt_from_ecg(ecg, fs, window_s, step_s) -> np.ndarray     # A14
```

**`preprocessing/windowing.py`**
```python
def n_windows(n_samples, fs, window_s, step_s) -> int
def iter_windows(n_samples, fs, window_s, step_s) -> Iterator[tuple[int, int, int]]  # (w, start, stop)
```

**`preprocessing/bandpass.py`**
```python
def make_bandpass(fs, low, high, design, order) -> sos_or_ba
def apply_bandpass(x, filt) -> np.ndarray                   # zero-phase, last axis
```

**`preprocessing/spectrum.py`**
```python
def periodogram(x, n_fft) -> np.ndarray                     # length n_fft//2 + 1, |FFT|^2 / len(x)
def local_maxima(s, lo, hi) -> np.ndarray                   # bin indices in [lo, hi]
def dominant_bins(s, lo, hi, rel_threshold) -> np.ndarray   # peaks > rel_threshold * max(s[lo:hi])
```

**`preprocessing/temporal_diff.py`**
```python
def temporal_difference(x, order=2, normalize=True) -> np.ndarray   # returns len(x) - order samples
```

**`decomposition/ssa.py`**
```python
def embed(y, L) -> np.ndarray                               # (L, K)
def hankelize(Y, M) -> np.ndarray                           # diagonal averaging, length M
def ssa(y, L) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # U, s, Vt
def group_matrices(U, s, Vt, groups) -> list[np.ndarray]    # each (L, K)
def reconstruct_groups(U, s, Vt, groups, M) -> np.ndarray   # (g, M); sums to y
```

**`decomposition/grouping.py`**
```python
def group_eigentriples(U, s, Vt, M, strategy, **kw) -> list[list[int]]
```

**`decomposition/motion_removal.py`**
```python
def acc_dominant_bins(acc_win, cfg) -> set[int]                        # 5.2 steps 1-4
def refine_acc_bins(f_acc, prev_bin, delta, n_harmonics) -> set[int]   # F̃_acc
def remove_motion_components(ppg_win, f_acc_refined, cfg) -> tuple[np.ndarray, dict]
    # returns cleansed PPG and info: {n_groups, n_removed, dominant_bins}
```

**`ssr/basis.py`**
```python
def kept_bins(N, fs, f_lo, f_hi) -> np.ndarray                          # Eq. 12, 0-based, symmetric
@lru_cache
def get_dictionary(M, N, fs, f_lo, f_hi) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # Phi (M, n_cols), G = Phi^H Phi, kept bin indices
```

**`ssr/focuss.py`**
```python
def focuss(y, Phi, G, p, lam, n_iter, x0=None) -> np.ndarray             # fast n×n form
def focuss_reference(y, Phi, p, lam, n_iter, x0=None) -> np.ndarray      # M×M form for tests
```

**`ssr/estimators.py`**
```python
class SpectrumEstimator(Protocol):
    def __call__(self, z: np.ndarray) -> np.ndarray: ...                 # length N//2 + 1

class SSRSpectrum(SpectrumEstimator): ...    # FOCUSS-based
class FFTSpectrum(SpectrumEstimator): ...    # ablation
```

**`tracking/selection.py`**
```python
def select_bin(s, k_prev, delta_s, eta_frac, max_peaks, harm_tol) -> tuple[int, int, dict]
    # returns (k_b, case in {1,2,3}, info with P0, P1, pairs)
```

**`tracking/verification.py`**
```python
def limit_jump(k_b, k_prev, theta, tau) -> tuple[int, bool]
def predict_trend(bpm_history, order, max_hist, thresh) -> int             # -1, 0, +1
```

**`tracking/tracker.py`**
```python
class SpectralPeakTracker:
    def __init__(self, cfg, fs, N): ...
    def initialize(self, s) -> WindowResult
    def step(self, s) -> WindowResult
    @property
    def prev_bin(self) -> int | None
```

**`pipeline.py`**
```python
class TroikaEstimator:
    def __init__(self, cfg): ...
    def run(self, rec: Recording) -> RunResult
```
Per-window pseudocode:
```python
for w, a, b in iter_windows(n, fs, T, S):
    ppg, acc = rec.ppg[a:b], rec.acc[:, a:b]
    ppg_f, acc_f = bandpass(ppg), bandpass(acc)
    f_acc = acc_dominant_bins(acc_f)
    if tracker.prev_bin is not None:
        f_acc = refine_acc_bins(f_acc, tracker.prev_bin, cfg.delta, cfg.n_harmonics)
    ppg_c = remove_motion_components(ppg_f, f_acc)[0] if cfg.ssa.enabled else ppg_f
    z = temporal_difference(ppg_c, order=2, normalize=True)
    s = estimator(z)
    res = tracker.initialize(s) if w == 0 else tracker.step(s)
```

**`evaluation/metrics.py`**
```python
def error1(est, gt) -> float
def error2(est, gt) -> float                      # fraction; multiply by 100 for %
def bland_altman(est, gt) -> dict                 # mean_diff, sd, loa_low, loa_high
def pearson(est, gt) -> float
def summarize(results: list[RunResult]) -> dict   # per-subject table + pooled stats
```

---

## 8. Experiments and reference targets

All experiments write to `results/<run_id>/` (`per_window.csv`, `per_subject.csv`, `summary.json`, PNGs).

### 8.1 Main result (`run_all.py`): Tables I/II row 1, Figs. 5, 6, 8

Targets (do not expect exact equality, see Section 11):

| | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Error1 (BPM) | 2.29 | 2.19 | 2.00 | 2.15 | 2.01 | 2.76 | 1.67 | 1.93 | 1.86 | 4.70 | 1.72 | 2.84 |
| Error2 (%) | 1.90 | 1.87 | 1.66 | 1.82 | 1.49 | 2.25 | 1.26 | 1.62 | 1.59 | 2.93 | 1.15 | 1.99 |

Aggregates: Error1 = 2.34 ± 0.82 BPM, Error2 = 1.80 %, LOA [−7.26, 4.79], σ = 3.07, r = 0.992.
Suggested pass criteria for "faithful reproduction": mean Error1 ≤ ~3.0 BPM, r ≥ 0.98, no subject with Error1 > 10 BPM.
Figures: Bland–Altman (Fig. 5), est-vs-gt scatter with r in the title (Fig. 6), Subject 5 est-vs-gt trace with speed-segment annotations (Fig. 8).

### 8.2 Ablation (`run_ablation.py`): Tables I/II rows 2–4, Fig. 7

Error1 (BPM); catastrophic failures (> 10 BPM) in bold:

| Variant | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 | S10 | S11 | S12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Without SSA (FOCUSS + Vrf) | 4.56 | 4.63 | 3.97 | 2.81 | 2.06 | **55.2** | 1.84 | 1.75 | 1.84 | 5.86 | 4.92 | 8.76 |
| FFT instead of SSR (SSA + FFT + Vrf) | **62.73** | 5.55 | 1.99 | 3.25 | 1.11 | **55.84** | 1.17 | 1.68 | 0.45 | **12.26** | 1.84 | 2.54 |
| Without verification (SSA + FOCUSS) | 3.45 | **14.00** | **24.07** | 2.62 | 2.10 | **54.79** | 2.97 | 1.77 | 1.92 | **51.89** | 2.69 | **60.14** |

Error2 (%) for the same rows (Table II):
- No SSA: 3.22, 3.66, 3.01, 2.39, 1.48, 39.11, 1.39, 1.45, 1.53, 3.59, 3.28, 6.08
- FFT: 42.79, 5.13, 1.80, 2.93, 0.88, 39.50, 0.89, 1.57, 0.37, 7.44, 1.23, 1.73
- No verification: 2.69, 12.91, 19.36, 2.30, 1.50, 38.87, 2.14, 1.48, 1.53, 32.82, 1.79, 41.00

Interpretation to check: qualitatively, each ablation should produce *some* catastrophic failures while the full pipeline does not. Subject 6 fails in all three ablations (Fig. 7 shows the estimate stuck near 70–90 BPM while true HR climbs to ~150+). Replicating **that pattern** matters more than matching every cell.
Note: the paper doesn't say whether the temporal difference is kept in ablations. Keep it on. **[ASSUMPTION A18]**

### 8.3 Sensitivity (`run_sensitivity.py`): Fig. 10

On **Subject 5**, vary one parameter at a time from the default (L = 400, Δ = 10, τ = 2, Δs = 16):
- L ∈ {100, 200, 300, 400}
- Δ ∈ {5, 10, 12, 15}
- τ ∈ {0, 1, 2, 3, 4}
- Δs ∈ {12, 14, 16, 20}

Expected: average absolute error stays ≈ 2 BPM (paper's bar chart y-axis 0–2.5) for all values. Plot as a bar chart with the same labels.

### 8.4 Illustrative figures (`make_figures.py`)

- Fig. 1-style: synthetic HR + hand-swing component with close frequencies → Periodogram vs FOCUSS spectra.
- Figs. 2–3-style: real windows where the HR peak is buried / absent.
- Fig. 9-style: raw PPG, its periodogram, PPG after SSA, and SSR spectrum, with the true HR bin marked (from ECG).

### 8.5 Optional extras (not in the paper; clearly separate from reproduction)

- Sampling-rate study (paper claims performance is nearly unchanged at 25 Hz with retuned parameters).
- Parabolic interpolation around the final peak to beat the 1.83-BPM grid.
- Runtime profiling per stage.

---

## 9. Test plan

| Test file | What it verifies |
|---|---|
| `test_binmap.py` | Round trip bin↔BPM; Δf = 0.0305 Hz; θ = 6 bins ≈ 11 BPM; Δ = 10 bins ≈ 0.3 Hz |
| `test_bandpass_windowing.py` | Window count formula (n = 37500 → 147); passband gain ≈ 1 at 1–3 Hz, ≪ 1 at 0.1 and 10 Hz |
| `test_ssa.py` | `hankelize(embed(y)) == y`; Σ of all group reconstructions == y (atol 1e-8) for every grouping strategy; a pure sinusoid concentrates in 2 eigentriples with near-equal σ |
| `test_motion_removal.py` | Synthetic PPG = HR sinusoid + MA sinusoid; acc containing the MA frequency → MA group removed, HR retained; exclusion of ±Δ around the previous HR prevents removing a cadence-overlapping HR; all-removed fallback |
| `test_basis_focuss.py` | Kept-bin symmetry `k → N−k`, DC excluded, ≈ 458 columns; `focuss` (n×n form) == `focuss_reference` (M×M form); two close sinusoids (Fig. 1 setup) are resolved by FOCUSS but merged by FFT |
| `test_tracking.py` | Scripted spectra for Case 1 / 2 / 3; rule 1 clamps jumps to ±τ; rule 2 fires after 3 identical windows, uses trend, widens Δs; trend predictor with < 4 points returns 0; harmonic tolerance and tie-breaks |
| `test_metrics.py` | Hand-computed Error1/Error2; Bland–Altman LOA algebra; feeding Table I row 1 to the aggregator returns 2.34 ± 0.82 (ddof = 1) |
| `test_pipeline_synthetic.py` | 60 s synthetic run (HR ramp 80→160 BPM + swing MA + noise): mean abs error < 5 BPM, no exceptions; ablation flags change behaviour |
| Smoke test on real data | Skipped when `data/raw` is missing |

---

## 10. Build order (milestones for Claude Code)

Each milestone ends with passing tests and a short note in `README.md`.

1. **M0, scaffold.** `pyproject.toml`, package layout, `config.py` (dataclasses + YAML), `types.py`, `binmap.py`, CI-style `pytest` run.
2. **M1, data + metrics.** Loader (inspect real files first, then code), windowing, ground truth, `metrics.py`, `report.py`. *Done when* Table I row 1 fed to `summarize` reproduces 2.34 ± 0.82 and the 12 recordings load.
3. **M2, preprocessing.** Band-pass, periodogram/peaks, temporal difference.
4. **M3, SSA.** embed/SVD/hankelize + all three grouping strategies. *Done when* perfect-reconstruction tests pass.
5. **M4, motion removal.** `F_acc`, refinement, component removal on synthetic data, then on Subject 5 (Fig. 9 reproduction).
6. **M5, SSR.** Dictionary pruning/caching, FOCUSS (both forms), FFT estimator. *Done when* the Fig. 1 synthetic example works.
7. **M6, tracking.** Selection, verification, tracker, all unit tests.
8. **M7, pipeline.** `TroikaEstimator`, ablation flags, per-window logging. First end-to-end on Subject 5, then all subjects.
9. **M8, experiments.** `run_all`, `run_ablation`, `run_sensitivity`, figures. Compare with Section 8 targets; iterate on assumptions A1–A19 using the ablation switches, **one change at a time, logged**.
10. **M9, write-up.** Results table vs paper, list of which assumptions moved the numbers.

Runtime budget: SVD of 400×601 ≈ tens of ms, FOCUSS n×n solves ≈ tens of ms, so under ~0.5 s/window and roughly 10 minutes for all 12 subjects on a single core. Use `joblib` across subjects only.

---

## 11. Assumption register and paper ambiguities

| ID | Topic | What the paper says | Default chosen | Options to test |
|---|---|---|---|---|
| A1 | Band-pass design | 0.4–5 Hz, "practical filter" with transition band | Butterworth-4, zero-phase | FIR, other orders |
| A2 | Filtering scope | Per window | per_window | global (removes edge effects) |
| A3 | Normalization before SSR | Not stated (Fig. 9 shows normalized) | z-score after differencing | none (then retune λ); unit-norm; Φ/√M scaling |
| A4 | SSA grouping | "Clustering singular values, ref. [23, p. 66]" | `frequency_pairing` | `wcorr_hclust`, `singleton` |
| A5 | Match test "dominant bin ∈ F̃_acc" | Membership | ±2 bins | 0, 1, 3, 5 |
| A6 | Harmonics in Np | "Fundamental and harmonic" | {k, 2k} | {k}, {k, 2k, 3k} |
| A7 | Harmonic-pair tolerance | "Harmonic relation" | ±2 bins | 1, 3 |
| A8 | Multiple harmonic pairs | Not stated | largest s[k⁰]+s[k¹] | closest to k_prev |
| A9 | η on R1 | "In each search range… η = 30 % of highest peak in R0" | same absolute η | separate η for R1 |
| A10 | Init band | "Highest peak" | 40–200 BPM | unrestricted |
| A11 | FOCUSS x₀ | Not stated | all ones | min-norm of Φᴴy; \|Φᴴy\| |
| A12 | Eq. 12 rounding | Fractional bounds | ceil low / floor high | keep wider |
| A13 | Rule 2 semantics | "Nb = Nprev for h successive windows; broaden Δs" | counter includes current window; Δs wide applies to next window | apply only when Case 3 fired |
| A14 | Ground truth | ECG cycle count / duration | dataset file if present, else R-peak intervals | count/8 |
| A15 | PPG channel | Single channel | channel 1 | channel 2 |
| A16 | Peak definition | Not stated | `find_peaks` local maxima | with prominence |
| A17 | Case 2 rounding of k¹/2 | (N¹−1)/2+1, 1-based | round(k¹/2) | floor |
| A18 | Temporal diff in ablations | Not stated | kept | dropped |
| A19 | F_acc band restriction | Not stated | 0.4–5 Hz | full spectrum |

**Other inconsistencies to be aware of**
- Remark 3: "1 BPM per grid step" is really 1.83 BPM for N = 4096, fs = 125.
- Section III-A says the number of samples M = 1000 but the SSR input has M′ = 998 after the second-order difference. Build Φ for M′.
- Eq. 12 is garbled in the PDF text extraction; the reading in Section 5.5 is consistent with the stated Δf1, Δf2 and gives a sensible ≈ 0.03–7 Hz kept range. Confirm against the PDF image if in doubt.
- The paper states "single-channel PPG" while distributed data may hold two; only one is used.
- Parameters (Δs = 16, θ = 6, etc.) were chosen by heuristics on this same 12-subject dataset; Fig. 10 shows insensitivity only for L, Δ, τ, Δs and only on Subject 5.
- The paper's comparisons to prior work (Pearson 0.75/0.78 etc.) come from different datasets and protocols and are not like-for-like.

---

## 12. Conventions for Claude Code (drop into `CLAUDE.md`)

- Follow this blueprint; do not silently change **[PAPER]** parameters. Any deviation must be a config option with a comment `# ASSUMPTION Ax` or `# DEVIATION: reason`.
- 0-based bins everywhere; all conversions go through `binmap.py`. Never write raw `fs/N` arithmetic elsewhere.
- No global state except the cached dictionary in `ssr/basis.py`; the tracker is the only stateful object in the pipeline.
- Every pipeline stage must be unit-testable in isolation with synthetic data; write the test before or with the code.
- Log per-window diagnostics (`WindowResult`) and write them to CSV; debugging tracking failures without them is painful.
- Use `float64` / `complex128`. Set seeds in any synthetic-data code.
- Do not commit data or `results/`.
- When a real-data result is off, change **one** assumption at a time, re-run the Subject 5/6 check plus the full run, and record the effect in `docs/assumption_log.md`.
- Type hints and docstrings on every public function; each docstring cites the section of this blueprint / the paper equation it implements.
