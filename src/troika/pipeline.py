"""The TROIKA pipeline (Blueprint Section 4, Lab Spec Section 2).

Stages are built from the registry according to the config, then run window by
window. The order is fixed:

    raw -> bandpass -> acc_dominant (helper) -> decomposition -> temporal_diff
        -> spectrum_estimator -> tracker -> BPM

The pipeline is stateful across windows, because the previous estimate feeds
both the motion-artifact refinement and the tracker, so windows of one recording
must be processed in order. Parallelism belongs across subjects.
"""

from __future__ import annotations

import time
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

import troika.plugins  # noqa: F401  (registers the built-in slot plug-ins)
from troika import registry
from troika.config import Config
from troika.decomposition.motion_removal import acc_dominant_bins, refine_acc_bins
from troika.preprocessing.bandpass import apply_bandpass, make_bandpass
from troika.preprocessing.windowing import iter_windows, n_windows
from troika.trace import STAGE_ORDER, RunTrace, StageTrace, WindowTrace, light_spectrum
from troika.types import (
    Recording,
    RunResult,
    StageOutput,
    StaticContext,
    TrackerResult,
    WindowContext,
    WindowResult,
)

__all__ = ["TroikaEstimator"]


class TroikaEstimator:
    """Runs the pipeline over one recording.

    One estimator instance holds one tracker, so it processes one recording.
    :meth:`run` rebuilds the stages, which keeps repeated calls independent.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.fs = float(cfg.signal.fs)
        self.N = int(cfg.grid.n_fft)
        self.window_s = float(cfg.signal.window_s)
        self.step_s = float(cfg.signal.step_s)
        self.M = int(round(self.window_s * self.fs))
        self.static = StaticContext(
            fs=self.fs,
            N=self.N,
            M=self.M,
            window_s=self.window_s,
            step_s=self.step_s,
        )
        self._check_band_consistency()

    # ------------------------------------------------------------ construction

    def _check_band_consistency(self) -> None:
        """The Eq. 12 pruning band must match the band-pass corners.

        Paper Eq. 12 derives the kept columns from the band-pass corners, so a
        config where the two disagree would prune away signal the filter passed.
        """
        if self.cfg.slot_method("spectrum_estimator") != "focuss":
            return
        band = self.cfg.bandpass.params
        ssr = self.cfg.spectrum_estimator.params
        if (band.low_hz, band.high_hz) != (ssr.low_hz, ssr.high_hz):
            raise ValueError(
                "spectrum_estimator.params low_hz/high_hz must match "
                "bandpass.params, because paper Eq. 12 prunes the dictionary to "
                f"the band-pass band; got SSR {ssr.low_hz}-{ssr.high_hz} Hz "
                f"against band-pass {band.low_hz}-{band.high_hz} Hz"
            )

    def _build(self, slot: str):
        """Instantiate the plug-in configured for ``slot``."""
        cls = registry.get(slot, self.cfg.slot_method(slot))
        return cls(self.cfg.slot_params(slot), self.static)

    @property
    def contaminated(self) -> bool:
        """True when this configuration lets ground truth reach the pipeline."""
        params = self.cfg.slot_params("tracker")
        return bool(getattr(params, "uses_ground_truth", False))

    # -------------------------------------------------------------------- run

    def run(
        self,
        rec: Recording,
        *,
        trace: str | None = None,
        full_windows: Sequence[int] | None = None,
        prev_bin_override: int | None = None,
        stop_after: int | None = None,
    ) -> RunResult:
        """Estimate heart rate for every window of ``rec``.

        ``trace`` overrides ``run.trace.level``. ``full_windows`` lists window
        indices to record at the full level. ``prev_bin_override`` seeds the
        tracker's previous bin for isolated-window exploration, which marks the
        result contaminated. ``stop_after`` ends the run early, which the
        isolated-window helper uses.
        """
        level = str(trace if trace is not None else self.cfg.run.trace.level)
        wanted_full = set(int(w) for w in (full_windows or ()))
        allow_gt = bool(self.cfg.run.allow_ground_truth_access)

        bandpass = self._build("bandpass")
        decomposition = self._build("decomposition")
        temporal_diff = self._build("temporal_diff")
        estimator = self._build("spectrum_estimator")
        tracker = self._build("tracker")

        total = n_windows(rec.n_samples, self.fs, self.window_s, self.step_s)
        if stop_after is not None:
            total = min(total, int(stop_after))

        # ASSUMPTION A2: `global` filters the recording once instead of per window.
        prefiltered = self._prefilter(rec) if self._global_mode() else None

        contaminated = self.contaminated or prev_bin_override is not None
        run_trace = (
            RunTrace(
                subject_id=rec.subject_id,
                level=level,
                static=self.static,
                config_hash=self.cfg.hash(),
                contaminated=contaminated,
                protocol=rec.protocol,
                fs=self.fs,
                n_samples=rec.n_samples,
            )
            if level != "none"
            else None
        )

        bpm_est = np.full(total, np.nan)
        windows: list[WindowResult] = []

        for w, start, stop in iter_windows(rec.n_samples, self.fs, self.window_s, self.step_s):
            if w >= total:
                break
            t_start_s = start / self.fs
            # The trace and the plots may see the truth; plug-ins may not
            # (Lab Spec Section 2.4). `gt` goes into the trace, `ctx_gt` into
            # the WindowContext the plug-ins receive.
            gt = self._ground_truth_for(rec, w)
            ctx_gt = gt if allow_gt else None

            timings: dict[str, float] = {}
            stage_signals: dict[str, NDArray[np.float64]] = {}
            stage_diags: dict[str, dict[str, Any]] = {}
            is_full = level == "full" or (level != "none" and w in wanted_full)

            # --- band-pass -------------------------------------------------
            ctx = WindowContext(idx=w, t_start_s=t_start_s, gt_bpm=ctx_gt)
            raw_ppg = rec.ppg[start:stop]
            raw_acc = rec.acc[:, start:stop]
            stacked = np.vstack([raw_ppg[None, :], raw_acc])

            if prefiltered is not None:
                filtered = prefiltered[:, start:stop]
                bp_out = StageOutput(data=filtered, diag={})
            else:
                bp_out = self._timed(timings, "bandpass", bandpass, stacked, ctx)
            ppg_f = np.ascontiguousarray(bp_out.data[0])
            acc_f = np.ascontiguousarray(bp_out.data[1:])

            # --- accelerometer dominant frequencies ------------------------
            t0 = time.perf_counter()
            raw_bins = acc_dominant_bins(
                acc_f,
                self.N,
                self.fs,
                float(self.cfg.acc_dominant.rel_threshold),
                bool(self.cfg.acc_dominant.restrict_to_band),
                threshold_domain=str(self.cfg.acc_dominant.threshold_domain),
            )
            prev_bin = tracker.prev_bin if prev_bin_override is None or w > 0 else prev_bin_override
            refined = refine_acc_bins(
                raw_bins,
                prev_bin,
                int(self.cfg.acc_dominant.exclude_delta_bins),
                int(self.cfg.acc_dominant.n_harmonics),
                self.N,
            )
            timings["acc_dominant"] = (time.perf_counter() - t0) * 1e3

            ctx = WindowContext(
                idx=w,
                t_start_s=t_start_s,
                acc=acc_f,
                acc_bins_raw=raw_bins,
                acc_bins=refined,
                prev_bin=prev_bin,
                bpm_history=list(getattr(tracker, "bpm_history", [])),
                gt_bpm=ctx_gt,
            )

            # --- decomposition, difference, spectrum -----------------------
            dec_out = self._timed(timings, "decomposition", decomposition, ppg_f, ctx)
            diff_out = self._timed(timings, "temporal_diff", temporal_diff, dec_out.data, ctx)
            spec_out = self._timed(
                timings, "spectrum_estimator", estimator, diff_out.data, ctx
            )

            # ASSUMPTION A20: the first window's peak comes from a pre-difference
            # PPG spectrum, because the second-order difference weights power by
            # about f^4 and would otherwise hand a low heart rate's harmonic the
            # highest peak. Only the initialisation window needs it.
            if tracker.prev_bin is None:
                ctx.init_spectrum = self._init_spectrum(ppg_f, dec_out.data)

            # --- tracking ---------------------------------------------------
            t0 = time.perf_counter()
            if w == 0 and prev_bin_override is not None:
                tracker._tracker.state.prev_bin = int(prev_bin_override)  # seeded
                track_out = tracker.step(spec_out.data, ctx)
            elif tracker.prev_bin is None:
                track_out = tracker.initialize(spec_out.data, ctx)
            else:
                track_out = tracker.step(spec_out.data, ctx)
            timings["tracker"] = (time.perf_counter() - t0) * 1e3

            result: TrackerResult = track_out.data
            bpm_est[w] = result.bpm

            removed = dec_out.diag.get("removed_mask")
            n_removed = int(np.sum(removed)) if removed is not None else 0
            n_groups = int(len(removed)) if removed is not None else 0

            windows.append(
                WindowResult(
                    idx=w,
                    t_start_s=t_start_s,
                    bin_cur=result.bin_cur,
                    bpm_est=result.bpm,
                    case=result.case,
                    rule1_fired=result.rule1_fired,
                    rule2_fired=result.rule2_fired,
                    f_acc=sorted(refined),
                    n_groups=n_groups,
                    n_groups_removed=n_removed,
                    timings_ms=dict(timings),
                )
            )

            if run_trace is not None:
                stage_signals = {
                    "raw": raw_ppg,
                    "bandpass": ppg_f,
                    "decomposition": dec_out.data,
                    "temporal_diff": diff_out.data,
                }
                stage_diags = {
                    "bandpass": bp_out.diag,
                    "decomposition": dec_out.diag,
                    "temporal_diff": diff_out.diag,
                    "spectrum_estimator": spec_out.diag,
                    "tracker": track_out.diag,
                }
                run_trace.windows.append(
                    self._make_window_trace(
                        w,
                        t_start_s,
                        is_full,
                        stage_signals,
                        stage_diags,
                        spec_out.data,
                        ctx,
                        timings,
                        result,
                        gt,
                        contaminated,
                        raw_acc,
                    )
                )

        return RunResult(
            subject_id=rec.subject_id,
            bpm_est=bpm_est,
            bpm_gt=None if rec.bpm_gt is None else np.asarray(rec.bpm_gt[:total], dtype=float),
            windows=windows,
            config_hash=self.cfg.hash(),
            contaminated=contaminated,
            trace=run_trace,
        )

    # ---------------------------------------------------------------- helpers

    def _init_spectrum(
        self, ppg_f: NDArray[np.float64], ppg_cleansed: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Spectrum the tracker initialises from (ASSUMPTION A20).

        Paper Section III-D.1 says the first estimate is "the highest spectral
        peak in a PPG spectrum", which reads as a spectrum of the PPG rather
        than of its second-order difference. ``estimator`` keeps the blueprint's
        original reading and returns ``None``, so the tracker uses the spectrum
        it is handed.
        """
        from troika.preprocessing.spectrum import periodogram

        choice = str(getattr(self.cfg.tracker.params, "init_spectrum", "decomposition"))
        if choice == "estimator":
            return None
        source = ppg_cleansed if choice == "decomposition" else ppg_f
        return periodogram(source, self.N)

    def _global_mode(self) -> bool:
        """True when ASSUMPTION A2 selects whole-recording filtering."""
        return str(getattr(self.cfg.bandpass.params, "mode", "per_window")) == "global"

    def _prefilter(self, rec: Recording) -> NDArray[np.float64]:
        """Band-pass the whole recording once (ASSUMPTION A2, ``mode: global``)."""
        params = self.cfg.bandpass.params
        design = self.cfg.slot_method("bandpass")
        filt = make_bandpass(self.fs, params.low_hz, params.high_hz, design, params.order)
        stacked = np.vstack([rec.ppg[None, :], rec.acc])
        return apply_bandpass(stacked, filt)

    @staticmethod
    def _timed(timings: dict[str, float], name: str, stage, x, ctx) -> StageOutput:
        """Call a stage and record how long it took, in milliseconds."""
        start = time.perf_counter()
        out = stage(x, ctx)
        timings[name] = (time.perf_counter() - start) * 1e3
        if not isinstance(out, StageOutput):
            out = StageOutput(data=out)
        return out

    @staticmethod
    def _ground_truth_for(rec: Recording, w: int) -> float | None:
        if rec.bpm_gt is None or w >= len(rec.bpm_gt):
            return None
        return float(rec.bpm_gt[w])

    def _make_window_trace(
        self,
        w: int,
        t_start_s: float,
        is_full: bool,
        signals: dict[str, NDArray[np.float64]],
        diags: dict[str, dict[str, Any]],
        spectrum: NDArray[np.float64],
        ctx: WindowContext,
        timings: dict[str, float],
        result: TrackerResult,
        gt: float | None,
        contaminated: bool,
        raw_acc: NDArray[np.float64],
    ) -> WindowTrace:
        """Build one window's trace record at the requested level."""
        stages = {
            name: StageTrace(
                signal=np.asarray(sig, dtype=np.float32),
                spectrum=light_spectrum(sig, self.N, self.fs),
            )
            for name, sig in signals.items()
        }
        hi = min(int(spectrum.size - 1), int(round(7.0 * self.N / self.fs)))
        stages["spectrum_estimator"] = StageTrace(
            signal=np.asarray(signals["temporal_diff"], dtype=np.float32),
            spectrum=np.asarray(spectrum[: hi + 1], dtype=np.float32),
        )

        window = WindowTrace(
            idx=w,
            t_start_s=t_start_s,
            level="full" if is_full else "light",
            stages=stages,
            timings_ms=dict(timings),
            bpm_est=result.bpm,
            bin_cur=result.bin_cur,
            case=result.case,
            rule1_fired=result.rule1_fired,
            rule2_fired=result.rule2_fired,
            gt_bpm=gt,
            contaminated=contaminated,
            static=self.static,
            ctx=ctx,
        )
        if is_full:
            window.diag = {
                name: {**payload, "_method": self.cfg.slot_method(name)}
                for name, payload in diags.items()
            }
            window.raw = {
                "ppg": np.asarray(signals["raw"], dtype=np.float64),
                "acc": np.asarray(raw_acc, dtype=np.float64),
                "spectrum": np.asarray(spectrum, dtype=np.float64),
            }
        return window


_ = STAGE_ORDER  # re-exported for the visualisation layer
