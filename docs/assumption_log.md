# Assumption log

One entry per assumption that was examined or changed, with what was measured
and what it cost. Blueprint Section 12 requires changing one assumption at a
time and recording the effect here.

Assumptions A1-A19 are defined in Blueprint Section 11 and are config keys.
Entries that are only design notes, not tuning results, are marked *note*.

---

## A1, band-pass design. *note*

**Default:** 4th-order Butterworth, zero-phase. **Alternative:** Hamming FIR.

The FIR alternative cannot match the Butterworth inside an 8 s window. The
zero-phase pass needs more than three times the filter length, so a
1000-sample window caps the design at about 333 taps, and a Hamming FIR of that
length has a transition band of roughly 1.2 Hz. Placing a corner at 0.4 Hz is
therefore impossible: the FIR eats into the bottom of the heart-rate band.

Amplitude gain measured on pure tones, 8 s window, edges discarded:

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
the plausible heart-rate range while the Butterworth does not. Above 1 Hz the
two agree, and the FIR has the sharper upper edge.

**Decision.** Keep Butterworth as the A1 default. The FIR default tap count is
301, near the window's ceiling, so the comparison is as favourable to the FIR as
the window allows. Anyone switching to `fir` should know the low edge is soft.

Converting a few-hundred-tap FIR to second-order sections destroys precision
(gains of order 1e21 at 331 taps), so the FIR path keeps its taps in
transfer-function form and uses `filtfilt` rather than `sosfiltfilt`.

---

## Blueprint corrections found while implementing

These are places where the blueprint's stated number or claim did not survive
being recomputed. None of them changes the algorithm.

**Aggregate standard deviation.** Blueprint Section 5.8 says it verified that
`ddof = 1` on the 12 Table I values gives 2.34 +/- 0.82. Recomputed, `ddof = 1`
gives 2.3433 +/- **0.8265**, which rounds to 0.83; `ddof = 0` gives 0.7913. So
`ddof = 1` is the right choice and is four times closer to the published figure,
but it does not reproduce it exactly. The residual gap is consistent with the
paper averaging unrounded per-subject errors. `evaluation.std_ddof` stays 1.

**Window count.** Blueprint Section 2 says "about 147 windows for a 300 s
recording". The recordings run 288 to 326 s, giving 140 to 160 windows. The
formula itself is exact: it matches `len(BPM0)` for all 22 recordings. Recorded
as DEVIATION D3.

**Speed protocol.** The paper's single schedule does not match the dataset,
which has two. Recorded as DEVIATION D2.
