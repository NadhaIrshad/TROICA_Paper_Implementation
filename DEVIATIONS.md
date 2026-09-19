# Deviations from the two specifications

Every entry here is something this implementation does that neither
`docs/TROIKA_Implementation_Blueprint_v2.md` nor
`docs/TROIKA_Exploration_Lab_Spec.md` states. Choices where the *paper* is
silent are not deviations; they are assumptions A1-A19 and live in the config
with a `# ASSUMPTION Ax` comment.

Assumption tuning is logged separately in `docs/assumption_log.md`.

---

## D1. Data and results paths in the config

**What.** `configs/default.yaml` gains two sections the specs do not define:

```yaml
data:
  dir: datasets/IEEE_SPC_2015
  split: training
results:
  dir: results
```

**Why.** Both specs describe the loader and the result folders but never say
where either lives, so nothing could run. `data.dir` is relative to the
repository root; the loader appends `Training_data` or `TestData` depending on
`data.split`.

**Effect on results.** None.

---

## D2. Speed protocol differs per recording type

**What.** The visualisation layer shades speed segments per recording, driven by
`Recording.protocol`, instead of the single schedule both specs assume.

**Why.** The paper (Section IV-A) describes one protocol: 1-2, 6-8, 12-15, 6-8,
12-15, 1-2 km/h. The dataset readme gives two, and the file names say which
applies:

| Type | Files | Schedule |
|---|---|---|
| `TYPE01` | subject 1 only | rest 30 s, 8, 15, 8, 15 km/h, rest 30 s |
| `TYPE02` | subjects 2-12 | rest 30 s, 6, 12, 6, 12 km/h, rest 30 s |

The paper's "1-2 km/h" opening segment is the readme's "rest", and its ranges
are the union of the two schedules. Lab Spec Section 5.1 asks for boundaries at
30, 90, 150, 210, 270 s, which both schedules share, so only the labels differ.

**Effect on results.** None; shading is cosmetic and is labelled "nominal".

---

## D3. Recording lengths are not 300 s

**What.** Code never assumes 147 windows or a 300 s duration. The window count
comes from the formula alone.

**Why.** Blueprint Section 2 says "about 147 windows for a 300 s recording". The
actual recordings run 288 to 326 s, giving 140 to 160 windows. The formula
`W = floor((n - T*fs) / (S*fs)) + 1` matches `len(BPM0)` exactly for all 12
training and all 10 test files, so the formula is right and the constant is not.

**Effect on results.** None.

---

## D4. `Recording` carries more than the Blueprint's fields

**What.** `Recording` adds `ppg_all`, `protocol` and `path` to the Blueprint
Section 7.1 definition.

**Why.** `ppg_all` lets a notebook compare PPG channel 1 and channel 2 (A15)
without reloading; `protocol` drives D2; `path` records provenance in result
folders.

**Effect on results.** None; the pipeline reads only `ppg` and `acc`.

---

## D5. Extra config keys for assumptions the specs left implicit

**What.** `default.yaml` adds four keys that the spec text describes but its YAML
listing omits:

| Key | Assumption |
|---|---|
| `acc_dominant.restrict_to_band` | A19, restrict the accelerometer peak search to 0.4-5 Hz |
| `decomposition.params.n_groups_max` | A4, cluster count for `wcorr_hclust` |
| `tracker.params.pair_tiebreak` | A8, which harmonic pair wins when several qualify |
| `tracker.params.half_rounding` | A17, how `k1 / 2` is rounded in Case 2 |

**Why.** The rule is that every `[ASSUMPTION Ax]` must be a config option rather
than a hard-coded constant. These four were named in Blueprint Section 11 but
missing from the Section 6 and Lab Spec Section 3 YAML listings.

**Effect on results.** None at the defaults, which are the values the Blueprint
prescribes.

---

## D6. Two new assumptions, A20 and A21

**What.** The assumption register in Blueprint Section 11 runs A1 to A19. Two
more were needed, and both are config keys with a `# ASSUMPTION` comment like
the rest:

| ID | Key | Question the paper leaves open |
|---|---|---|
| A20 | `tracker.params.init_spectrum` | Which spectrum the first window's highest peak is taken from |
| A21 | `acc_dominant.threshold_domain` | Whether the 50 % dominant-peak rule is on amplitude or on periodogram power |

**Why.** Both are genuine ambiguities in the paper that change the result by
tens of BPM, and neither spec flagged them.

A20: paper Section III-D.1 says the first estimate is "the highest spectral peak
in a PPG spectrum". The blueprint read that as the spectrum estimator's output,
which sits after the second-order difference. Differencing weights power by
roughly f to the fourth, so for a subject whose rate starts near 69 BPM the true
peak falls to rank 8 and initialisation fails. Taking the peak from a
pre-difference PPG spectrum, which is what the paper's words say, takes the mean
error over 12 subjects from 45.2 to 13.3 BPM.

A21: paper Section III-A thresholds at 50 % of the maximum *amplitude*, but the
spectrum it thresholds is a periodogram, which holds power. The two readings
differ by a factor of four in the threshold. The default is `power`, which
measures better; see `docs/assumption_log.md`.

**Effect on results.** Large, and measured. Both are documented in the
assumption log with the numbers.

---

## D7. `WindowContext.init_spectrum`

**What.** `WindowContext` gains an `init_spectrum` field, set only for the first
window.

**Why.** A20 needs the tracker to initialise from a pre-difference spectrum,
but the tracker slot receives only the spectrum estimator's output. The pipeline
computes the alternative and passes it through the context.

**Effect on results.** None beyond A20 itself. Plug-ins that ignore the field
behave exactly as before.

---

## D8. Extra experiment scripts

**What.** `experiments/` also holds `_sweep.py`, which runs the one-at-a-time
assumption sweep behind `docs/assumption_log.md`.

**Why.** Blueprint Section 12 requires changing one assumption at a time and
recording the effect; a script makes that reproducible rather than a note.

**Effect on results.** None.
