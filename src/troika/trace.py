"""Tracing of pipeline internals (Lab Spec Section 2.4).

Three levels:

``none``
    Metrics only. Nothing is stored.
``light``
    Per window: the estimate and its diagnostics, plus each stage's time-domain
    output and its spectrum truncated to 0-7 Hz, both as ``float32``. About
    20 kB per window, so roughly 3 MB for a 300 s recording, inside the 5 MB
    budget the spec sets.
``full``
    Everything in Lab Spec Section 2.3 for the chosen windows, plus the raw
    window signals. A full trace of every window would be about 1 GB per
    subject, because of the SSA matrices, so it is opt-in per window.

The leak guard lives here too: a trace that used ground-truth-derived state is
marked ``contaminated`` and the evaluation layer refuses to report it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = [
    "TRACE_LEVELS",
    "STAGE_ORDER",
    "StageTrace",
    "WindowTrace",
    "RunTrace",
    "light_spectrum",
]

#: Valid values of ``run.trace.level``, in increasing detail.
TRACE_LEVELS = ("none", "light", "full")

#: Stages whose output the light level stores, in pipeline order.
STAGE_ORDER = ("raw", "bandpass", "decomposition", "temporal_diff", "spectrum_estimator")

#: Upper edge of the stored light-level spectra, in Hz.
LIGHT_SPECTRUM_MAX_HZ = 7.0


def light_spectrum(
    signal: NDArray[np.float64], n_fft: int, fs: float
) -> NDArray[np.float32]:
    """Periodogram of ``signal`` truncated to 0-7 Hz, as ``float32``.

    Storing the whole one-sided spectrum would be 2049 bins per stage per
    window; everything the plots show sits below 7 Hz.
    """
    from troika.preprocessing.spectrum import periodogram

    spec = periodogram(signal, n_fft)
    hi = min(int(binmap.hz_to_bin(LIGHT_SPECTRUM_MAX_HZ, fs, n_fft)), spec.size - 1)
    return spec[: hi + 1].astype(np.float32)


@dataclass
class StageTrace:
    """One stage's output at the light level."""

    signal: NDArray[np.float32]
    spectrum: NDArray[np.float32]

    def nbytes(self) -> int:
        """Bytes held by this record."""
        return int(self.signal.nbytes + self.spectrum.nbytes)


@dataclass
class WindowTrace:
    """Everything recorded for one window.

    ``stages`` holds :class:`StageTrace` records at the light level. ``diag``
    holds each stage's full ``diag`` payload, and is populated only for windows
    traced at the full level.
    """

    idx: int
    t_start_s: float
    level: str = "light"
    stages: dict[str, StageTrace] = field(default_factory=dict)
    diag: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    ctx: WindowContext | None = None
    static: StaticContext | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    bpm_est: float = float("nan")
    bin_cur: int = -1
    case: int | None = None
    rule1_fired: bool = False
    rule2_fired: bool = False
    gt_bpm: float | None = None
    contaminated: bool = False

    @property
    def is_full(self) -> bool:
        """True when this window carries the Section 2.3 payload."""
        return self.level == "full"

    def require(self, stage: str, keys: Iterable[str], plot: str) -> dict[str, Any]:
        """Fetch a stage's ``diag``, or raise naming exactly what is missing.

        Lab Spec Section 2.3: a plot whose keys are absent must fail with a clear
        message rather than a stack trace.
        """
        keys = list(keys)
        if stage not in self.diag:
            raise KeyError(
                f"{plot} needs the {stage!r} diag of window {self.idx}, which this "
                f"trace does not hold. Re-run with a full trace, for example "
                f"lab.get_window_trace(rec, cfg, w={self.idx})."
            )
        payload = self.diag[stage]
        missing = [k for k in keys if k not in payload]
        if missing:
            method = payload.get("_method", stage)
            raise KeyError(
                f"{plot} needs diag keys {missing} from {stage} {method!r}, which "
                f"does not provide them; use the generic plot for this stage instead."
            )
        return payload

    def nbytes(self) -> int:
        """Approximate bytes held by this window's records."""
        total = sum(record.nbytes() for record in self.stages.values())
        total += sum(int(a.nbytes) for a in self.raw.values())
        return int(total)


@dataclass
class RunTrace:
    """Every traced window of one recording."""

    subject_id: int
    level: str = "light"
    windows: list[WindowTrace] = field(default_factory=list)
    static: StaticContext | None = None
    config_hash: str | None = None
    contaminated: bool = False
    protocol: str | None = None
    fs: float = 125.0
    n_samples: int = 0

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> WindowTrace:
        """Window by its index, not by position in the list."""
        for window in self.windows:
            if window.idx == idx:
                return window
        raise KeyError(
            f"window {idx} is not in this trace; it holds "
            f"{[w.idx for w in self.windows[:5]]}{' ...' if len(self.windows) > 5 else ''}"
        )

    @property
    def full_windows(self) -> list[int]:
        """Indices traced at the full level."""
        return [w.idx for w in self.windows if w.is_full]

    def stage_matrix(self, stage: str) -> NDArray[np.float32]:
        """Stack one stage's light spectra over windows, shape ``(W, n_bins)``.

        Used by the whole-recording spectrogram views (Lab Spec Section 5.2 G).
        """
        rows = [w.stages[stage].spectrum for w in self.windows if stage in w.stages]
        if not rows:
            raise KeyError(
                f"no window in this trace holds stage {stage!r}; "
                f"available: {sorted({k for w in self.windows for k in w.stages})}"
            )
        width = min(r.size for r in rows)
        return np.stack([r[:width] for r in rows])

    def nbytes(self) -> int:
        """Approximate size of the trace in bytes."""
        return int(sum(w.nbytes() for w in self.windows))
