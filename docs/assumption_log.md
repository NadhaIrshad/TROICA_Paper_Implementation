# Assumption log

One entry per assumption that was examined or changed, with what was measured
and what it cost. Blueprint Section 12 requires changing one assumption at a
time and recording the effect here.

Assumptions A1-A19 are defined in Blueprint Section 11 and are config keys. A20
and A21 are new, added because the paper is ambiguous in two places the
blueprint did not flag; both are config keys like the rest and are also listed
in `DEVIATIONS.md`.

Unless stated otherwise, every number is Error1 in BPM, averaged over the 12
training subjects, from `experiments/_sweep.py`.

---

## Summary of the one-at-a-time sweep

Baseline is the paper's values with the blueprint's defaults for everything the
paper leaves open. Every row changes one thing.

| Variant | Mean | Subjects above 10 BPM | Worst |
|---|---|---|---|
| **A6 = 1, A5 = 0 (adopted default)** | **8.25** | **1** | S10 75.3 |
| A6 = 1 alone | 9.07 | 2 | S10 75.1 |
| A4 `wcorr_hclust` (with A6 = 1) | 8.80 | 2 | S10 75.4 |
| A8 `closest_to_prev` (with A6 = 1) | 9.12 | 2 | S10 75.7 |
| A3 no normalisation (with A6 = 1) | 9.17 | 2 | S10 77.0 |
| Delta = 5 instead of 10 (with A6 = 1) | 9.19 | 2 | S10 75.8 |
| A6 = 2 (blueprint default) | 13.34 | 3 | S10 75.7 |
| A15 PPG channel 2 (with A6 = 1) | 17.28 | 5 | S2 113.3 |
| A5 = 5 bins (with A6 = 1) | 25.93 | 7 | S2 114.3 |
| Acc threshold 0.3 instead of 0.5 | 26.08 | 5 | S2 113.2 |
| A21 amplitude reading | 31.51 | 7 | S2 113.3 |

Two changes were adopted, and both are choices the blueprint already listed as
options rather than departures from the paper.

---

## A6, how many heartbeat harmonics the exclusion protects. **Changed to 1.**

**Paper:** "Denote by N_p the location indexes of fundamental and harmonic
frequencies of the heartbeat estimated in the previous time window. We exclude
{N_p - Delta, ..., N_p + Delta} from F_acc." **Blueprint default:** 2, meaning
the fundamental and the first harmonic. **Adopted:** 1, the fundamental only.

**Why it matters.** The exclusion is centred on the *previous estimate*. Once
the estimate sits on the running cadence rather than the heart rate, the
exclusion protects the cadence from removal, which keeps the estimate there. It
is self-reinforcing.

Measured directly on subject 6, windows 17 onward: the refined `F_acc` is
**empty** for twelve consecutive windows. Every accelerometer dominant bin fell
inside the excluded zone, so nothing was removed, the cadence stayed in the PPG,
and the estimate held near 78-88 BPM while the true rate climbed from 87 to 121.

Protecting only the fundamental halves the protected bandwidth and breaks the
trap:

| | Subject 6 | Mean over 12 |
|---|---|---|
| `n_harmonics: 2` | 54.4 | 13.34 |
| `n_harmonics: 1` | 2.4 | 9.07 |

No other subject is materially affected. The paper's wording supports either
reading; the data does not.

---

## A5, how a component's dominant bin is matched to `F_acc`. **Changed to 0.**

**Paper:** membership, `d_p` in `F_acc`. **Blueprint default:** a tolerance of 2
bins, on the grounds that exact matching is fragile. **Adopted:** 0, exact.

Measured with A6 = 1 already applied:

| Tolerance | Mean | Subject 1 | Subject 12 | Above 10 BPM |
|---|---|---|---|---|
| 0 bins | **8.25** | 3.5 | 7.5 | 1 |
| 2 bins | 9.07 | 8.7 | 11.9 | 2 |
| 5 bins | 25.93 | 27.4 | 19.5 | 7 |

The tolerance is not a robustness knob, it is a bluntness knob: each extra bin
widens the deletion zone around every accelerometer peak, and at 5 bins the
method deletes heart-rate components wholesale. Exact matching, the paper's
literal reading, measures best.

---

## A21, whether the 50 % threshold is on amplitude or power. **New. Kept as power.**

**Paper Section III-A:** "the dominant frequencies are the ones corresponding to
the spectral peaks with **amplitude** larger than 50 % of the maximum amplitude
in a given spectrum." The spectrum in question is a periodogram, whose values
are **power**. Thresholding amplitude at 50 % is thresholding power at 25 %, so
the two readings are not the same and the paper's own words are ambiguous.

The amplitude reading keeps roughly four times as many peaks. On synthetic data
it helps, because it catches artifact harmonics that sit below the power
threshold. On the real recordings it is much worse: 31.51 against 13.34, with
seven subjects above 10 BPM instead of three, because the enlarged `F_acc`
deletes heart-rate components along with the motion ones.

**Decision:** default to `power`. The option is `acc_dominant.threshold_domain`.

A related observation, recorded rather than worked around: the rule is relative
to each axis's own maximum, so an axis with no rhythmic content still yields
"dominant" bins drawn from its own leakage floor.

---

## A20, which spectrum the first window is initialised from. **New. Default `decomposition`.**

**Paper Section III-D.1:** "HR can be estimated by choosing the highest spectral
peak in a PPG spectrum during this stage." **Blueprint reading:** the highest
peak of `s`, the spectrum estimator's output. That output sits *after* the
second-order difference.

Differencing weights power by roughly f to the fourth. Measured on subject 6,
window 0, whose true rate is 68.7 BPM:

| Stage | Top peaks (BPM) | True rate's rank |
|---|---|---|
| band-passed | 70, 139, 198 | 1 |
| after decomposition | 70, 139, 79 | 1 |
| after temporal difference | 207, 200, 139 | 8 |
| spectrum estimator | 207, 198, 139 | 6 |

The heart rate is the clear top peak until the difference promotes content near
200 BPM by a factor of 69 in power. Initialising from the differenced spectrum
therefore fails systematically for subjects whose rate starts low.

| `init_spectrum` | Mean | Above 10 BPM |
|---|---|---|
| `estimator` (blueprint reading) | 45.21 | 8 |
| `bandpass` | 13.34 | 3 |
| `decomposition` (adopted) | 13.34 | 3 |

`bandpass` and `decomposition` agree to within 0.1 BPM. `decomposition` is the
default because it has the motion components already removed, which is the
better-posed spectrum even though it does not change the outcome here.

The paper's phrase "a PPG spectrum" supports this reading directly; the
differenced signal is not a PPG signal.

---

## A1, band-pass design. *Design note, unchanged.*

**Default:** 4th-order Butterworth, zero-phase. **Alternative:** Hamming FIR.

The FIR alternative cannot match the Butterworth inside an 8 s window. The
zero-phase pass needs more than three times the filter length, so a 1000-sample
window caps the design at about 333 taps, and a Hamming FIR of that length has a
transition band of roughly 1.2 Hz. A corner at 0.4 Hz is therefore impossible:
the FIR eats into the bottom of the heart-rate band.

Amplitude gain on pure tones, 8 s window, edges discarded:

| Frequency | Butterworth order 4 | FIR 301 taps |
|---|---|---|
| 0.1 Hz | 0.004 | 0.039 |
| 0.3 Hz | 0.080 | 0.141 |
| 0.5 Hz | 0.893 | 0.394 |
| 0.7 Hz | 1.004 | 0.705 |
| 1.0 Hz | 1.005 | 0.973 |
| 1.5-3 Hz | 1.00 | 1.00 |
| 4.5 Hz | 0.734 | 0.922 |
| 6.0 Hz | 0.158 | 0.000 |

0.5 Hz is 30 BPM and 0.7 Hz is 42 BPM, so the FIR attenuates the very bottom of
the plausible range while the Butterworth does not. Above 1 Hz the two agree and
the FIR has the sharper upper edge.

Converting a few-hundred-tap FIR to second-order sections destroys precision
(gains of order 1e21 at 331 taps), so the FIR path keeps its taps in
transfer-function form and uses `filtfilt`.

---

## A15, which PPG channel. *Unchanged, channel 1.*

Channel 2 measures far worse: 17.28 against 9.07, five subjects above 10 BPM
instead of two. The two channels are 2 cm apart and their correlation ranges
from -0.21 on subject 1 to 0.62 on subject 5, so they are not interchangeable.

---

## A3, A4, A8, and Delta. *Unchanged; all within noise.*

Each was measured with A6 = 1 applied and moved the mean by less than 0.2 BPM:
no normalisation 9.17, `wcorr_hclust` grouping 8.80, `closest_to_prev` tie-break
9.12, Delta = 5 giving 9.19, against 9.07 for the defaults. None changes which
subjects fail. The paper's claim that the framework is insensitive to these
holds here.

---

## Blueprint corrections found while implementing

None of these changes the algorithm.

**Aggregate standard deviation.** Blueprint Section 5.8 says it verified that
`ddof = 1` on the 12 Table I values gives 2.34 +/- 0.82. Recomputed, `ddof = 1`
gives 2.3433 +/- **0.8265**, which rounds to 0.83; `ddof = 0` gives 0.7913. So
`ddof = 1` is the right choice and is four times closer to the published figure,
but it does not reproduce it exactly. The gap is consistent with the paper
averaging unrounded per-subject errors. `evaluation.std_ddof` stays 1.

**Window count.** Blueprint Section 2 says "about 147 windows for a 300 s
recording". The recordings run 288 to 326 s, giving 140 to 160 windows. The
formula itself is exact: it matches `len(BPM0)` for all 22 recordings. Recorded
as DEVIATION D3.

**Speed protocol.** The paper's single schedule does not match the dataset,
which has two. Recorded as DEVIATION D2.

**Equation 12 confirmed.** The blueprint warned that Eq. 12 is garbled in the
PDF text extraction. Its reading is right: with `Delta_f1 = 0.4 N / fs - 1` and
`Delta_f2 = 2 N / fs` the kept set is 0-based 1..229 and 3867..4095, 458 columns
covering 0.03 to 6.99 Hz, symmetric under `k -> N - k` and excluding DC, exactly
as the blueprint predicted.
