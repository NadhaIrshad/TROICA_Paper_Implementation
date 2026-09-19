"""Core data containers (Blueprint Section 7.1, extended by Lab Spec Section 2).

These types carry no algorithm logic; they are the contract between the loader,
the pipeline, the tracing layer and the evaluation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Recording",
    "TrackerResult",
    "WindowResult",
    "RunResult",
    "StageOutput",
    "StaticContext",
    "WindowContext",
]


@dataclass(frozen=True)
class Recording:
    """One subject's recording.

    Attributes
    ----------
    subject_id:
        1-based subject number parsed from the file name.
    fs:
        Sampling rate in Hz (125 for this dataset).
    ppg:
        Selected PPG channel, shape ``(n,)``.
    acc:
        Three-axis accelerometer, shape ``(3, n)``, order x, y, z.
    ecg:
        Chest ECG, shape ``(n,)``, or ``None`` for the competition test set.
        Used only to derive or cross-check ground truth, never by the pipeline.
    bpm_gt:
        Per-window ground-truth BPM, shape ``(W,)``, or ``None``.
    ppg_all:
        Every available PPG channel, shape ``(n_ppg, n)``, so a notebook can
        compare channel 1 and channel 2 without reloading (A15).
    protocol:
        ``"TYPE01"`` or ``"TYPE02"`` for training recordings, ``"T01"``/``"T02"``
        for test recordings, else ``None``. Drives the speed-segment shading,
        which differs per type (see DEVIATIONS.md).
    path:
        Source file, for provenance in result folders.
    """

    subject_id: int
    fs: float
    ppg: NDArray[np.float64]
    acc: NDArray[np.float64]
    ecg: NDArray[np.float64] | None = None
    bpm_gt: NDArray[np.float64] | None = None
    ppg_all: NDArray[np.float64] | None = None
    protocol: str | None = None
    path: str | None = None

    @property
    def n_samples(self) -> int:
        """Number of samples in the recording."""
        return int(self.ppg.shape[-1])

    @property
    def duration_s(self) -> float:
        """Recording duration in seconds."""
        return self.n_samples / float(self.fs)


@dataclass
class TrackerResult:
    """What a tracker plug-in returns for one window (Lab Spec Section 2.1)."""

    bin_cur: int
    bpm: float
    case: int | None = None
    rule1_fired: bool = False
    rule2_fired: bool = False


@dataclass
class WindowResult:
    """Per-window diagnostics (Blueprint Section 7.1).

    ``case`` is ``None`` for the initialisation window, which runs no peak
    selection.
    """

    idx: int
    t_start_s: float
    bin_cur: int
    bpm_est: float
    case: int | None = None
    rule1_fired: bool = False
    rule2_fired: bool = False
    f_acc: list[int] = field(default_factory=list)
    n_groups: int = 0
    n_groups_removed: int = 0
    timings_ms: dict[str, float] = field(default_factory=dict)
    spectrum: NDArray[np.float64] | None = None


@dataclass
class RunResult:
    """Result of running the pipeline over one recording."""

    subject_id: int
    bpm_est: NDArray[np.float64]
    bpm_gt: NDArray[np.float64] | None
    windows: list[WindowResult] = field(default_factory=list)
    config_hash: str | None = None
    contaminated: bool = False
    trace: Any | None = None

    @property
    def n_windows(self) -> int:
        """Number of windows processed."""
        return int(self.bpm_est.shape[0])


@dataclass
class StageOutput:
    """Return value of a slot plug-in (Lab Spec Section 2.1).

    ``diag`` carries internals for the visualisation layer. Its schema per slot
    is Lab Spec Section 2.3; paper-default plug-ins fill every listed key.
    """

    data: Any
    diag: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StaticContext:
    """Run-level constants handed to every plug-in at construction time."""

    fs: float
    N: int
    M: int
    window_s: float
    step_s: float


@dataclass
class WindowContext:
    """Per-window state handed to every plug-in call (Lab Spec Section 2.1).

    ``gt_bpm`` is ``None`` unless ``run.allow_ground_truth_access`` is set; the
    leak guard (Lab Spec Section 2.4) depends on plug-ins never seeing it.
    """

    idx: int
    t_start_s: float
    acc: NDArray[np.float64] | None = None
    acc_bins_raw: set[int] = field(default_factory=set)
    acc_bins: set[int] = field(default_factory=set)
    prev_bin: int | None = None
    bpm_history: list[float] = field(default_factory=list)
    gt_bpm: float | None = None
    init_spectrum: NDArray[np.float64] | None = None
    """Spectrum the tracker should initialise from, set only for window 0.

    DEVIATION D6. The tracker otherwise sees the spectrum estimator's output,
    which sits after the second-order difference and is therefore biased towards
    high frequencies; see ASSUMPTION A20 in docs/assumption_log.md.
    """
