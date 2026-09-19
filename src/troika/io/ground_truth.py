"""Per-window ground-truth heart rate (Blueprint Section 5.7, ASSUMPTION A14).

Paper Section IV-C: in each time window the number of cardiac cycles ``H`` and
the duration ``D`` in seconds were counted from the simultaneously recorded ECG,
and the ground truth is ``60 H / D``. No ECG heart-rate estimation algorithm was
used, to avoid its errors.

This dataset ships that computation as ``BPM0``, one value per window, so
:func:`gt_from_file` is the default path. :func:`gt_from_ecg` reimplements the
count for cross-checking; on this data the two agree to under 1 BPM on 11 of the
12 subjects, and disagree on subject 11 only because its ECG is clipped.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat
from scipy.signal import butter, find_peaks, sosfiltfilt

__all__ = ["gt_from_file", "gt_from_ecg", "compare_ground_truth"]


def gt_from_file(path: str | Path, n_windows: int | None = None) -> NDArray[np.float64]:
    """Read a per-window ground-truth BPM file (``BPM0``).

    Raises ``ValueError`` when the trace length disagrees with ``n_windows``;
    every file in this dataset matches the window formula exactly, so a mismatch
    means the window geometry is wrong.
    """
    path = Path(path)
    mat = loadmat(path)
    keys = [k for k in mat if not k.startswith("__")]
    if "BPM0" in mat:
        values = mat["BPM0"]
    elif len(keys) == 1:
        values = mat[keys[0]]
    else:
        raise ValueError(f"{path.name}: expected a 'BPM0' variable, found {keys}")

    bpm = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(bpm).all():
        raise ValueError(f"{path.name}: ground truth contains NaN or Inf")
    if n_windows is not None and bpm.size != int(n_windows):
        raise ValueError(
            f"{path.name}: ground truth has {bpm.size} values but the window "
            f"formula gives {n_windows} windows"
        )
    return bpm


def _rpeaks(ecg: NDArray[np.float64], fs: float) -> NDArray[np.int64]:
    """Detect R-peaks: band-pass 5-20 Hz, then peaks with a 0.3 s refractory gap.

    The polarity of the band-passed ECG is chosen by whichever sign yields the
    more regular peak train, because several recordings in this dataset clip at
    the ADC limit on one side.
    """
    nyq = 0.5 * float(fs)
    sos = butter(4, [5.0 / nyq, min(20.0 / nyq, 0.99)], btype="band", output="sos")
    filtered = sosfiltfilt(sos, np.asarray(ecg, dtype=np.float64))
    scale = np.std(filtered)
    if scale <= 0:
        return np.zeros(0, dtype=np.int64)
    filtered = filtered / scale

    best: NDArray[np.int64] = np.zeros(0, dtype=np.int64)
    best_score = np.inf
    for sign in (1.0, -1.0):
        peaks, _ = find_peaks(
            sign * filtered, distance=int(round(0.3 * fs)), height=0.5
        )
        if peaks.size < 3:
            continue
        intervals = np.diff(peaks)
        # Lower coefficient of variation means a more plausible R-R train.
        score = float(np.std(intervals) / max(np.mean(intervals), 1e-9))
        if score < best_score:
            best_score, best = score, peaks.astype(np.int64)
    return best


def gt_from_ecg(
    ecg: NDArray[np.float64],
    fs: float,
    window_s: float,
    step_s: float,
) -> NDArray[np.float64]:
    """Per-window ground truth from ECG R-peak intervals (ASSUMPTION A14).

    ``HR = 60 * (n_peaks - 1) / (t_last - t_first)`` inside each window, which is
    the paper's cycles-per-duration count with the duration measured between the
    first and last detected beat. Windows with fewer than three detected peaks
    yield ``NaN``.
    """
    from troika.preprocessing.windowing import iter_windows

    ecg = np.asarray(ecg, dtype=np.float64)
    peaks = _rpeaks(ecg, fs)
    out: list[float] = []
    for _, start, stop in iter_windows(ecg.size, fs, window_s, step_s):
        inside = peaks[(peaks >= start) & (peaks < stop)]
        if inside.size < 3:
            out.append(np.nan)
            continue
        duration = (inside[-1] - inside[0]) / float(fs)
        out.append(60.0 * (inside.size - 1) / duration if duration > 0 else np.nan)
    return np.asarray(out, dtype=np.float64)


def compare_ground_truth(
    from_file: NDArray[np.float64], from_ecg: NDArray[np.float64]
) -> dict[str, float]:
    """Agreement between the shipped trace and the ECG-derived one.

    Returns mean and max absolute difference, the correlation, and how many
    windows the ECG path could not score. Blueprint Section 5.7 expects
    agreement within about 1 BPM where the ECG is clean.
    """
    a = np.asarray(from_file, dtype=np.float64)
    b = np.asarray(from_ecg, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    ok = np.isfinite(b)
    if ok.sum() < 2:
        return {
            "mean_abs_diff": float("nan"),
            "max_abs_diff": float("nan"),
            "pearson_r": float("nan"),
            "n_undetected": float((~ok).sum()),
        }
    diff = np.abs(a[ok] - b[ok])
    return {
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff": float(diff.max()),
        "pearson_r": float(np.corrcoef(a[ok], b[ok])[0, 1]),
        "n_undetected": float((~ok).sum()),
    }
