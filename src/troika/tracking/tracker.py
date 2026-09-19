"""Stateful spectral peak tracker (paper Section III-D, Blueprint Section 5.6).

This is the only object in the pipeline that carries state across windows, which
is why the windows of one recording must be processed in order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.tracking.selection import select_bin
from troika.tracking.verification import limit_jump, predict_trend

__all__ = ["TrackerState", "SpectralPeakTracker"]


@dataclass
class TrackerState:
    """Everything the tracker remembers between windows."""

    prev_bin: int | None = None
    bpm_history: list[float] = field(default_factory=list)
    same_count: int = 0
    wide_search: bool = False


class SpectralPeakTracker:
    """Initialisation, peak selection and verification over a recording.

    Parameters come from the ``tracker`` slot's ``params``; see
    ``configs/default.yaml`` for which are [PAPER] and which are assumptions.
    """

    def __init__(
        self,
        fs: float,
        n_fft: int,
        *,
        init_mode: str = "max_peak",
        init_band_bpm: tuple[float, float] = (40.0, 200.0),
        delta_s: int = 16,
        delta_s_wide: int = 20,
        max_peaks_per_range: int = 3,
        eta_frac: float = 0.30,
        harm_tol_bins: int = 2,
        pair_tiebreak: str = "max_power",
        half_rounding: str = "round",
        verification_enabled: bool = True,
        theta_bins: int = 6,
        tau_bins: int = 2,
        stall_windows_h: int = 3,
        trend_history_windows: int = 20,
        trend_poly_order: int = 3,
        trend_threshold_bpm: float = 3.0,
    ) -> None:
        self.fs = float(fs)
        self.n_fft = int(n_fft)
        self.init_mode = init_mode
        self.init_band_bpm = tuple(init_band_bpm)
        self.delta_s = int(delta_s)
        self.delta_s_wide = int(delta_s_wide)
        self.max_peaks_per_range = int(max_peaks_per_range)
        self.eta_frac = float(eta_frac)
        self.harm_tol_bins = int(harm_tol_bins)
        self.pair_tiebreak = pair_tiebreak
        self.half_rounding = half_rounding
        self.verification_enabled = bool(verification_enabled)
        self.theta_bins = int(theta_bins)
        self.tau_bins = int(tau_bins)
        self.stall_windows_h = int(stall_windows_h)
        self.trend_history_windows = int(trend_history_windows)
        self.trend_poly_order = int(trend_poly_order)
        self.trend_threshold_bpm = float(trend_threshold_bpm)
        self.state = TrackerState()

    # ---------------------------------------------------------------- helpers

    @property
    def prev_bin(self) -> int | None:
        """Previously estimated 0-based bin, or ``None`` before initialisation."""
        return self.state.prev_bin

    def bpm(self, k: int) -> float:
        """BPM of a bin on the shared grid."""
        return float(binmap.bin_to_bpm(int(k), self.fs, self.n_fft))

    def _commit(self, k_cur: int, n_bins: int) -> int:
        k_cur = int(np.clip(k_cur, 0, n_bins - 1))
        self.state.prev_bin = k_cur
        self.state.bpm_history.append(self.bpm(k_cur))
        return k_cur

    # --------------------------------------------------------- initialisation

    def initialize(
        self, spectrum: NDArray[np.float64], gt_bpm: float | None = None
    ) -> tuple[int, dict[str, Any]]:
        """First window: take the highest spectral peak (paper Section III-D.1).

        The wearer is asked to keep the hand still for the first few seconds, so
        the dominant peak is the heart rate. ASSUMPTION A10 restricts the search
        to a plausible band, 40-200 BPM by default.

        ``init_mode: ground_truth`` is a debugging aid only. It marks the run
        contaminated and must never be used for a reported result.
        """
        s = np.asarray(spectrum, dtype=np.float64)

        if self.init_mode == "ground_truth":
            if gt_bpm is None:
                raise ValueError(
                    "init_mode 'ground_truth' needs a ground-truth BPM; enable "
                    "run.allow_ground_truth_access"
                )
            k = int(binmap.bpm_to_bin(gt_bpm, self.fs, self.n_fft))
        elif self.init_mode == "max_peak":
            lo = int(binmap.bpm_to_bin(self.init_band_bpm[0], self.fs, self.n_fft))
            hi = int(binmap.bpm_to_bin(self.init_band_bpm[1], self.fs, self.n_fft))
            lo, hi = max(0, lo), min(s.size - 1, hi)
            k = int(lo + np.argmax(s[lo : hi + 1]))
        else:
            raise ValueError(
                f"init_mode must be 'max_peak' or 'ground_truth', got {self.init_mode!r}"
            )

        k_cur = self._commit(k, s.size)
        return k_cur, {
            "k_prev": None,
            "k_b": k_cur,
            "k_cur": k_cur,
            "case": None,
            "rule1_fired": False,
            "rule2_fired": False,
            "same_count": 0,
            "trend": 0,
            "delta_s_used": self.delta_s,
            "R0": (0, s.size - 1),
            "R1": (0, s.size - 1),
            "eta": 0.0,
            "P0": np.zeros(0, dtype=np.int64),
            "P1": np.zeros(0, dtype=np.int64),
            "pairs": [],
            "candidates": np.array([k_cur], dtype=np.int64),
        }

    # ------------------------------------------------------------------ step

    def step(self, spectrum: NDArray[np.float64]) -> tuple[int, dict[str, Any]]:
        """One subsequent window: select, then verify."""
        if self.state.prev_bin is None:
            raise RuntimeError("tracker.step called before initialize")

        s = np.asarray(spectrum, dtype=np.float64)
        k_prev = int(self.state.prev_bin)

        # ASSUMPTION A13: the widened range applies to the window after the
        # stall was detected, because this window's search has to run first.
        delta_s_used = self.delta_s_wide if self.state.wide_search else self.delta_s

        k_b, case, info = select_bin(
            s,
            k_prev,
            delta_s=delta_s_used,
            eta_frac=self.eta_frac,
            max_peaks=self.max_peaks_per_range,
            harm_tol=self.harm_tol_bins,
            pair_tiebreak=self.pair_tiebreak,
            half_rounding=self.half_rounding,
        )

        # Stall counter includes the current window (ASSUMPTION A13).
        same_count = self.state.same_count + 1 if k_b == k_prev else 0

        rule1_fired = False
        rule2_fired = False
        trend = 0
        k_cur = k_b

        if self.verification_enabled:
            k_cur, rule1_fired = limit_jump(k_b, k_prev, self.theta_bins, self.tau_bins)

            if same_count >= self.stall_windows_h:
                trend, _ = predict_trend(
                    self.state.bpm_history,
                    self.trend_poly_order,
                    self.trend_history_windows,
                    self.trend_threshold_bpm,
                )
                k_cur = k_prev + 2 * trend  # paper Eq. 17
                rule2_fired = True
                self.state.wide_search = True
                same_count = 0
            else:
                self.state.wide_search = False

        self.state.same_count = same_count
        k_cur = self._commit(k_cur, s.size)

        info.update(
            {
                "k_prev": k_prev,
                "k_b": int(k_b),
                "k_cur": int(k_cur),
                "case": int(case),
                "rule1_fired": bool(rule1_fired),
                "rule2_fired": bool(rule2_fired),
                "same_count": int(self.state.same_count),
                "trend": int(trend),
                "delta_s_used": int(delta_s_used),
            }
        )
        return k_cur, info
