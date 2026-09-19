"""Loading IEEE Signal Processing Cup 2015 recordings (Blueprint Section 2).

The 12-subject training set is the dataset of the paper. Layout, confirmed
against the dataset readme and by inspecting all 22 files:

* ``Training_data/DATA_xx_TYPExx.mat`` holds a ``6 x n`` float64 array ``sig``:
  row 0 ECG, rows 1-2 the two PPG channels, rows 3-5 accelerometer x, y, z.
  Per-window ground truth is in ``DATA_xx_TYPExx_BPMtrace.mat`` as ``BPM0``.
* ``TestData/TEST_Sxx_Txx.mat`` holds a ``5 x n`` array: rows 0-1 PPG, rows 2-4
  accelerometer. No ECG. Ground truth is in ``TrueBPM/True_Sxx_Txx.mat``.

All signals are sampled at 125 Hz. The paper uses a single PPG channel; this
dataset ships two, and the choice is ASSUMPTION A15.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from troika.config import Config, repo_root
from troika.io.ground_truth import gt_from_file
from troika.types import Recording

__all__ = [
    "list_subjects",
    "load_recording",
    "load_all",
    "resolve_data_dir",
    "DataError",
]

#: Training recordings: DATA_<subject>_TYPE<nn>.mat, excluding the BPM traces.
_TRAIN_RE = re.compile(r"^DATA_(\d+)_(TYPE\d+)$", re.IGNORECASE)
#: Test recordings: TEST_S<subject>_T<nn>.mat.
_TEST_RE = re.compile(r"^TEST_S(\d+)_(T\d+)$", re.IGNORECASE)

#: Expected subject count per split, used to fail loudly on a wrong data.dir.
_EXPECTED = {"training": 12, "test": 10}


class DataError(RuntimeError):
    """Raised when the dataset is missing, incomplete or shaped unexpectedly."""


def resolve_data_dir(cfg: Config) -> Path:
    """Directory holding the recordings of the configured split.

    ``data.dir`` is taken relative to the repository root when it is not
    absolute, so notebooks and the CLI agree regardless of the working
    directory (DEVIATION D1).
    """
    root = Path(cfg.data.dir)
    if not root.is_absolute():
        root = repo_root() / root
    split = str(cfg.data.split).lower()
    sub = {"training": "Training_data", "test": "TestData"}.get(split)
    if sub is None:
        raise DataError(f"data.split must be 'training' or 'test', got {cfg.data.split!r}")
    path = root / sub
    if not path.is_dir():
        raise DataError(
            f"dataset directory not found: {path}\n"
            f"Set data.dir in the config to the folder holding {sub}/."
        )
    return path


def list_subjects(cfg: Config) -> list[tuple[int, Path]]:
    """Sorted ``(subject_id, path)`` pairs for the configured split.

    Raises :class:`DataError` when the count is not the expected 12 (training)
    or 10 (test), so a wrong ``data.dir`` fails immediately rather than
    producing a partial run.
    """
    directory = resolve_data_dir(cfg)
    split = str(cfg.data.split).lower()
    pattern = _TRAIN_RE if split == "training" else _TEST_RE

    found: list[tuple[int, Path]] = []
    for path in sorted(directory.glob("*.mat")):
        if path.stem.upper().endswith("_BPMTRACE"):
            continue
        match = pattern.match(path.stem)
        if match:
            found.append((int(match.group(1)), path))

    found.sort(key=lambda item: (item[0], item[1].name))
    expected = _EXPECTED[split]
    if len(found) != expected:
        names = [p.name for _, p in found]
        raise DataError(
            f"expected {expected} {split} recordings in {directory}, found "
            f"{len(found)}: {names}"
        )
    return found


def _read_sig(path: Path) -> NDArray[np.float64]:
    """Read the ``sig`` array from a ``.mat`` recording."""
    mat = loadmat(path)
    if "sig" not in mat:
        keys = [k for k in mat if not k.startswith("__")]
        raise DataError(f"{path.name}: no 'sig' variable; found {keys}")
    sig = np.asarray(mat["sig"], dtype=np.float64)
    if sig.ndim != 2:
        raise DataError(f"{path.name}: 'sig' must be 2-D, got shape {sig.shape}")
    return np.ascontiguousarray(sig)


def _split_rows(sig: NDArray[np.float64], path: Path) -> tuple:
    """Split ``sig`` into (ecg, ppg_all, acc) according to its row count."""
    n_rows = sig.shape[0]
    if n_rows == 6:  # training: ECG, PPG1, PPG2, ACC x, y, z
        return sig[0], sig[1:3], sig[3:6]
    if n_rows == 5:  # test: PPG1, PPG2, ACC x, y, z
        return None, sig[0:2], sig[2:5]
    raise DataError(
        f"{path.name}: expected 6 rows (training) or 5 rows (test), got {n_rows}"
    )


def _protocol(path: Path) -> str | None:
    """``TYPE01``/``TYPE02`` or ``T01``/``T02`` from the file name (DEVIATION D2)."""
    for pattern in (_TRAIN_RE, _TEST_RE):
        match = pattern.match(path.stem)
        if match:
            return match.group(2).upper()
    return None


def _ground_truth_path(path: Path) -> Path | None:
    """Locate the per-window ground-truth file that belongs to a recording."""
    train = path.with_name(f"{path.stem}_BPMtrace.mat")
    if train.exists():
        return train
    match = _TEST_RE.match(path.stem)
    if match:
        true = path.parent.parent / "TrueBPM" / f"True_S{match.group(1)}_{match.group(2)}.mat"
        if true.exists():
            return true
    return None


def load_recording(path: str | Path, subject_id: int, cfg: Config) -> Recording:
    """Load one recording.

    ``cfg.signal.ppg_channel`` indexes the PPG channels, not the rows of ``sig``:
    0 selects PPG channel 1 (ASSUMPTION A15). Ground truth is attached when the
    matching ``BPMtrace`` or ``TrueBPM`` file exists, and is ``None`` otherwise.
    """
    path = Path(path)
    sig = _read_sig(path)
    ecg, ppg_all, acc = _split_rows(sig, path)

    channel = int(cfg.signal.ppg_channel)  # ASSUMPTION A15
    if not 0 <= channel < ppg_all.shape[0]:
        raise DataError(
            f"{path.name}: signal.ppg_channel={channel} out of range; "
            f"the file has {ppg_all.shape[0]} PPG channels"
        )

    if not np.isfinite(sig).all():
        raise DataError(f"{path.name}: signal contains NaN or Inf")

    fs = float(cfg.signal.fs)
    gt_path = _ground_truth_path(path)
    bpm_gt = None
    if gt_path is not None:
        from troika.preprocessing.windowing import n_windows

        expected = n_windows(sig.shape[1], fs, cfg.signal.window_s, cfg.signal.step_s)
        bpm_gt = gt_from_file(gt_path, expected)

    return Recording(
        subject_id=int(subject_id),
        fs=fs,
        ppg=np.ascontiguousarray(ppg_all[channel]),
        acc=np.ascontiguousarray(acc),
        ecg=None if ecg is None else np.ascontiguousarray(ecg),
        bpm_gt=bpm_gt,
        ppg_all=np.ascontiguousarray(ppg_all),
        protocol=_protocol(path),
        path=str(path),
    )


def load_all(cfg: Config, subject_ids: list[int] | None = None) -> dict[int, Recording]:
    """Load every subject of the configured split, or just ``subject_ids``."""
    wanted = None if subject_ids is None else set(int(s) for s in subject_ids)
    out: dict[int, Recording] = {}
    for subject_id, path in list_subjects(cfg):
        if wanted is not None and subject_id not in wanted:
            continue
        out[subject_id] = load_recording(path, subject_id, cfg)
    if wanted is not None:
        missing = sorted(wanted - set(out))
        if missing:
            raise DataError(f"no recording for subject(s) {missing}")
    return out
