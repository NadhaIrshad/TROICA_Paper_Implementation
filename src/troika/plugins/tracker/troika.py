"""TROIKA spectral peak tracker plug-in (paper Section III-D)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from troika.plugins.base import Tracker
from troika.registry import register
from troika.tracking.tracker import SpectralPeakTracker
from troika.types import StageOutput, StaticContext, TrackerResult, WindowContext

__all__ = ["TroikaTracker"]


def _default_verification() -> dict[str, Any]:
    return {
        "enabled": True,
        "theta_bins": 6,  # PAPER
        "tau_bins": 2,  # PAPER
        "stall_windows_h": 3,  # PAPER
        "trend_history_windows": 20,  # PAPER
        "trend_poly_order": 3,  # PAPER
        "trend_threshold_bpm": 3,  # PAPER
    }


@dataclass
class TroikaTrackerParams:
    """Options of the spectral peak tracker."""

    init_mode: str = "max_peak"  # max_peak | ground_truth (debug; marks CONTAMINATED)
    init_spectrum: str = "decomposition"  # ASSUMPTION A20: decomposition | bandpass | estimator
    init_band_bpm: list = field(default_factory=lambda: [40, 200])  # ASSUMPTION A10
    delta_s: int = 16  # PAPER
    delta_s_wide: int = 20  # PAPER
    max_peaks_per_range: int = 3  # PAPER
    eta_frac: float = 0.30  # PAPER
    harm_tol_bins: int = 2  # ASSUMPTION A7
    pair_tiebreak: str = "max_power"  # ASSUMPTION A8
    half_rounding: str = "round"  # ASSUMPTION A17
    verification: dict = field(default_factory=_default_verification)

    def __post_init__(self) -> None:
        if self.init_spectrum not in ("decomposition", "bandpass", "estimator"):
            raise ValueError(
                f"init_spectrum must be 'decomposition', 'bandpass' or 'estimator', "
                f"got {self.init_spectrum!r}"
            )
        if self.init_mode not in ("max_peak", "ground_truth"):
            raise ValueError(
                f"init_mode must be 'max_peak' or 'ground_truth', got {self.init_mode!r}"
            )
        if len(self.init_band_bpm) != 2 or self.init_band_bpm[0] >= self.init_band_bpm[1]:
            raise ValueError(f"init_band_bpm must be [low, high], got {self.init_band_bpm}")
        if self.delta_s < 1 or self.delta_s_wide < 1:
            raise ValueError("delta_s and delta_s_wide must be >= 1")
        if self.max_peaks_per_range < 1:
            raise ValueError(f"max_peaks_per_range must be >= 1, got {self.max_peaks_per_range}")
        if not 0.0 <= self.eta_frac <= 1.0:
            raise ValueError(f"eta_frac must be in [0, 1], got {self.eta_frac}")
        if self.pair_tiebreak not in ("max_power", "closest_to_prev"):
            raise ValueError(f"unknown pair_tiebreak {self.pair_tiebreak!r}")
        if self.half_rounding not in ("round", "floor"):
            raise ValueError(f"unknown half_rounding {self.half_rounding!r}")
        known = set(_default_verification())
        unknown = set(self.verification) - known
        if unknown:
            raise ValueError(
                f"unknown verification key(s) {sorted(unknown)}; known: {sorted(known)}"
            )
        self.verification = {**_default_verification(), **self.verification}

    @property
    def uses_ground_truth(self) -> bool:
        """True when this configuration would contaminate a run."""
        return self.init_mode == "ground_truth"


@register("tracker", "troika")
class TroikaTracker(Tracker):
    """Initialisation, Cases 1-3 and the two verification rules.

    Stateful: ``initialize`` handles the first window and ``step`` every later
    one, so windows must arrive in order.
    """

    Params = TroikaTrackerParams

    def __init__(self, params: TroikaTrackerParams, static: StaticContext) -> None:
        super().__init__(params, static)
        verification = params.verification
        self._tracker = SpectralPeakTracker(
            fs=static.fs,
            n_fft=static.N,
            init_mode=params.init_mode,
            init_band_bpm=tuple(params.init_band_bpm),
            delta_s=params.delta_s,
            delta_s_wide=params.delta_s_wide,
            max_peaks_per_range=params.max_peaks_per_range,
            eta_frac=params.eta_frac,
            harm_tol_bins=params.harm_tol_bins,
            pair_tiebreak=params.pair_tiebreak,
            half_rounding=params.half_rounding,
            verification_enabled=bool(verification["enabled"]),
            theta_bins=int(verification["theta_bins"]),
            tau_bins=int(verification["tau_bins"]),
            stall_windows_h=int(verification["stall_windows_h"]),
            trend_history_windows=int(verification["trend_history_windows"]),
            trend_poly_order=int(verification["trend_poly_order"]),
            trend_threshold_bpm=float(verification["trend_threshold_bpm"]),
        )

    @property
    def prev_bin(self) -> int | None:
        """Previously estimated 0-based HR bin."""
        return self._tracker.prev_bin

    @property
    def bpm_history(self) -> list[float]:
        """Every BPM committed so far, oldest first."""
        return list(self._tracker.state.bpm_history)

    def _output(self, k_cur: int, info: dict[str, Any]) -> StageOutput:
        result = TrackerResult(
            bin_cur=int(k_cur),
            bpm=self._tracker.bpm(k_cur),
            case=info["case"],
            rule1_fired=bool(info["rule1_fired"]),
            rule2_fired=bool(info["rule2_fired"]),
        )
        return StageOutput(data=result, diag=dict(info))

    def initialize(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Handle the first window (paper Section III-D.1).

        ASSUMPTION A20: unless ``init_spectrum`` is ``estimator``, the pipeline
        supplies a pre-difference PPG spectrum through ``ctx.init_spectrum`` and
        that is what the highest peak is taken from.
        """
        source = spectrum
        if self.params.init_spectrum != "estimator" and ctx.init_spectrum is not None:
            source = ctx.init_spectrum
        k_cur, info = self._tracker.initialize(source, gt_bpm=ctx.gt_bpm)
        return self._output(k_cur, info)

    def step(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Handle a subsequent window (paper Section III-D.2 and III-D.3)."""
        k_cur, info = self._tracker.step(spectrum)
        return self._output(k_cur, info)
