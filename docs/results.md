# Results against the paper

What this implementation reproduces, what it does not, and why.

Written for: someone who has read the paper and wants to know how far this code
gets and where to look when it disagrees.

Reproduce with:

```bash
python experiments/run_all.py --config configs/default.yaml
python experiments/run_ablation.py
python experiments/run_sensitivity.py
```

---

## Headline

| | This implementation | Paper |
|---|---|---|
| Error1, mean over 12 subjects | 8.25 BPM | 2.34 BPM |
| Error1, mean over 11 subjects (excluding subject 10) | **2.16 +/- 2.02 BPM** | 2.34 +/- 0.82 BPM |
| Error2, mean over 11 subjects | 1.87 % | 1.80 % |
| Pearson r, pooled over 11 subjects | 0.976 | 0.992 |
| Bland-Altman limits, pooled over 11 subjects | [-11.05, 9.60], sigma 5.27 | [-7.26, 4.79], sigma 3.07 |

One subject dominates every aggregate. Eleven of twelve reproduce the paper, and
nine of those beat its per-subject value. Subject 10 fails outright.

---

## Per subject

Error1 in BPM. Bold marks where this implementation is worse than the paper by
more than 1 BPM.

| Subject | This work | Paper | Difference |
|---|---|---|---|
| 1 | **3.50** | 2.29 | +1.21 |
| 2 | 3.09 | 2.19 | +0.90 |
| 3 | 1.13 | 2.00 | -0.87 |
| 4 | 1.46 | 2.15 | -0.69 |
| 5 | 0.87 | 2.01 | -1.14 |
| 6 | 2.31 | 2.76 | -0.45 |
| 7 | 0.72 | 1.67 | -0.95 |
| 8 | 0.78 | 1.93 | -1.15 |
| 9 | 0.62 | 1.86 | -1.24 |
| 10 | **75.28** | 4.70 | +70.58 |
| 11 | 1.78 | 1.72 | +0.06 |
| 12 | **7.50** | 2.84 | +4.66 |

Against the blueprint's suggested pass criteria (mean Error1 at or below about
3.0 BPM, r at or above 0.98, no subject above 10 BPM): the first two are met on
eleven subjects, the third is not, because of subject 10.

---

## Why subject 10 fails

Its first window is misread, and the error never recovers.

Subject 10 has the highest resting rate in the set, 123.5 BPM in window 0. The
band-passed PPG of that window peaks at 70 BPM, with the true rate at only 10 %
of that peak's power. The accelerometer already shows a running cadence of
141-165 BPM in the same window, so the paper's premise for initialisation, that
"wearers are required to reduce hand motions as much as possible for several
seconds", does not hold for this recording.

The failure is not purely initialisation. Seeding the first window from the ECG,
which `tracker.params.init_mode: ground_truth` does as a debugging aid, more than
halves the error but does not fix it:

| Subject 10 | Error1 |
|---|---|
| honest initialisation | 75.28 |
| ground-truth initialisation (contaminated, debug only) | 33.16 |

So roughly half of subject 10's error is a bad start and half is the tracker
failing to hold on afterwards. The paper's own worst subject is also subject 10,
at 4.70 BPM, so it was the hardest recording there too, by a much smaller margin.

---

## The failure mode that mattered most

The exclusion zone that protects the heart rate can protect the motion artifact
instead, and then it cannot let go.

Paper Section III-A removes from `F_acc` every bin within plus or minus Delta of
the heartbeat frequency **estimated in the previous window**, so that a cadence
coinciding with the heart rate is not deleted along with it. The zone is centred
on the estimate, not on the truth. Once the estimate sits on the cadence, the
cadence is protected from removal, which keeps the estimate there.

Measured on subject 6 with the blueprint's defaults: from window 17 onward the
refined `F_acc` is **empty** for twelve consecutive windows. Every accelerometer
peak fell inside the excluded zone, nothing was removed, and the estimate held
near 78-88 BPM while the true rate climbed from 87 to 121.

Protecting only the fundamental rather than the fundamental and its harmonic
halves the protected bandwidth and breaks the trap: subject 6 goes from 54.4 to
2.4 BPM. The paper's wording ("fundamental and harmonic frequencies") supports
either reading. This is ASSUMPTION A6 and it is a config key.

The same trap is reproducible on synthetic data, and there is a test for it:
`test_artifact_crossing_the_heart_rate_is_a_known_weakness`.

---

## What had to be decided that the paper does not state

Two ambiguities changed the result by tens of BPM and were not in the
blueprint's register. Both are now config keys, A20 and A21, documented in
`docs/assumption_log.md` and `DEVIATIONS.md`.

**A20, which spectrum initialises the first window.** The paper says "the highest
spectral peak in a PPG spectrum". The blueprint read that as the spectrum
estimator's output, which sits after the second-order difference. Differencing
weights power by roughly the fourth power of frequency, so for a subject whose
rate starts near 69 BPM the true peak drops to rank 8 and the estimate starts
near 200 BPM. Taking the peak from a pre-difference PPG spectrum takes the mean
over 12 subjects from 45.2 to 13.3 BPM.

**A21, amplitude or power.** The paper keeps accelerometer peaks above 50 % of
the maximum *amplitude*, but the spectrum it thresholds is a periodogram, which
holds power. The readings differ by a factor of four in the threshold. Power
measures far better here, 13.3 against 31.5.

Two further assumptions were tuned, both within the options the blueprint
already listed: A6 from 2 harmonics to 1, and A5 from a 2-bin match tolerance to
exact matching. Together they take the mean from 13.34 to 8.25 BPM and reduce
the catastrophic failures from three subjects to one.

Everything else was left at the paper's or the blueprint's value. A3, A4, A8 and
Delta each moved the mean by less than 0.2 BPM, which supports the paper's claim
that the framework is insensitive to them.

---

## Ablations

Paper Section IV-D claims that removing any one of the three parts makes the
framework lose tracking on some recordings, while the complete framework does
not. Error1 in BPM, and the count of subjects above 10 BPM:

| Variant | Ours, all 12 | Ours, no S10 | Paper, all 12 | Failures ours | Failures paper |
|---|---|---|---|---|---|
| full (SSA + FOCUSS + Vrf) | 8.25 | **2.16** | 2.34 | 1 (S10) | none |
| without SSA | 15.90 | 10.49 | 8.18 | 3 (S2, S6, S10) | 1 (S6) |
| FFT instead of SSR | 3.59 | 2.60 | 12.53 | 2 (S10, S12) | 3 (S1, S6, S10) |
| without verification | 3.57 | 2.79 | 18.53 | 1 (S10) | 5 (S2, S3, S6, S10, S12) |

**The decomposition claim reproduces, strongly.** Removing SSA is the one
ablation that is clearly worse here, and it fails on subject 6 exactly as the
paper reports, going from 2.31 to 54.15 BPM. It also breaks subject 2, which the
paper keeps. This is the part of the framework this implementation can confirm
is indispensable.

**The other two claims reproduce only in direction, not in magnitude.** Removing
the sparse reconstruction or the verification costs about 0.5 BPM here, against
the paper's 10 and 16 BPM. Both stay ordered correctly, full better than
ablated, but neither produces the catastrophic failures the paper reports.

The likely reason is that the paper's ablations inherit its parameter choices,
and this implementation's parameters differ in the two places the paper is
ambiguous (A20 and A21) and in the two that were tuned (A5 and A6). A pipeline
whose decomposition removes motion more aggressively leaves the estimator and
the tracker less to rescue, so removing them costs less. That is a plausible
account, not a demonstrated one.

---

## What is not reproduced, and why it is hard to

**The exact numbers.** The paper's parameters were, in its own words, "assigned
by heuristics", on the same twelve recordings used to report the results, with no
held-out set. Several algorithmic details are underspecified, and two of them
(A20, A21) change the answer by more than the entire reported error. Matching
2.34 +/- 0.82 exactly would mean guessing the authors' choices, not
reimplementing a specification.

**The standard deviation across subjects.** 2.02 BPM over eleven subjects
against the paper's 0.82 over twelve. The paper's per-subject errors sit in a
remarkably tight band, 1.67 to 4.70, and this implementation's spread is wider
even where its mean is lower.

**Subject 10.** Discussed above.

---

## Runtime

About 200 ms per window on one core: roughly 100 ms for the decomposition and
75 ms for the sparse reconstruction. A 300 s recording takes about 30 s, and all
twelve take about 100 s across six workers. That is inside the blueprint's
budget of roughly 10 minutes for the full set on a single core.

Two optimisations were needed to get there, both exact rather than approximate:

* Elementary SSA series are computed as a convolution rather than by
  hankelising each rank-one matrix, since the anti-diagonal sums of
  `sigma u v^T` are `sigma (u * v)`. That is 40 times faster and agrees to 6e-14.
* The FOCUSS solve uses Cholesky, because `(w w^T) . G + lambda I` is Hermitian
  positive definite. 127 ms to 75 ms per window, agreeing to 1e-11.
