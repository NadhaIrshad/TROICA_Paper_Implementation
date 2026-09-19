"""Band-pass filtering, 0.4-5 Hz (Blueprint Section 5.1).

Paper Section III: before TROIKA starts, the raw PPG signal and the acceleration
signal in a given time window are band-pass filtered from 0.4 Hz to 5 Hz. This
removes noise and motion artifact outside the band of interest and sparsifies
the spectra, which SSR depends on (Remark 1).

The design is ASSUMPTION A1 and the scope is ASSUMPTION A2; both are config
options. The IIR path keeps second-order sections, which stay well conditioned;
the FIR path keeps its taps in transfer-function form, because converting a
few-hundred-tap FIR to sections loses all precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, firwin, freqz, sosfiltfilt, sosfreqz

__all__ = ["BandpassFilter", "make_bandpass", "apply_bandpass", "frequency_response"]


@dataclass(frozen=True)
class BandpassFilter:
    """A designed band-pass filter.

    Exactly one representation is populated: ``sos`` for an IIR design, ``taps``
    for an FIR one. The remaining fields describe the design for the diagnostic
    plots and error messages.
    """

    fs: float
    low_hz: float
    high_hz: float
    design: str
    order: int
    sos: NDArray[np.float64] | None = None
    taps: NDArray[np.float64] | None = None

    @property
    def is_fir(self) -> bool:
        """True for the FIR design."""
        return self.taps is not None

    @property
    def description(self) -> str:
        """One-line summary for the ``diag`` payload and plot titles."""
        kind = f"FIR (Hamming, {self.order} taps)" if self.is_fir else f"Butterworth order {self.order}"
        return f"{kind}, {self.low_hz}-{self.high_hz} Hz, zero-phase at {self.fs:g} Hz"

    @property
    def min_length(self) -> int:
        """Shortest input the zero-phase pass accepts for this filter.

        ``filtfilt`` needs more than three times the filter length, because it
        extends the signal by that much at each end before the backward pass.
        """
        if self.is_fir:
            return 3 * int(self.taps.size) + 1
        n_sections = int(self.sos.shape[0])
        return 3 * (2 * n_sections + 1)


@lru_cache(maxsize=64)
def _design(
    fs: float, low_hz: float, high_hz: float, design: str, order: int
) -> tuple[NDArray[np.float64] | None, NDArray[np.float64] | None]:
    """Design the filter once per parameter set; cached because it runs per window."""
    nyq = 0.5 * float(fs)
    lo, hi = float(low_hz) / nyq, float(high_hz) / nyq
    if not 0 < lo < hi < 1:
        raise ValueError(
            f"band {low_hz}-{high_hz} Hz is not valid for fs={fs} Hz "
            f"(normalised {lo:.3f}-{hi:.3f})"
        )
    if design == "butter":
        return np.asarray(butter(int(order), [lo, hi], btype="band", output="sos")), None
    if design == "fir":
        # firwin needs an odd tap count for a symmetric band-pass.
        numtaps = int(order) if int(order) % 2 == 1 else int(order) + 1
        taps = np.asarray(firwin(numtaps, [lo, hi], pass_zero=False, window="hamming"))
        return None, taps
    raise ValueError(f"unknown bandpass design {design!r}; use 'butter' or 'fir'")


def make_bandpass(
    fs: float,
    low_hz: float = 0.4,
    high_hz: float = 5.0,
    design: str = "butter",
    order: int = 4,
) -> BandpassFilter:
    """Design the 0.4-5 Hz band-pass (ASSUMPTION A1).

    ``butter`` is the default: a 4th-order Butterworth applied zero-phase, so the
    effective response is squared and there is no phase distortion to move peak
    locations. ``fir`` gives a linear-phase Hamming-window FIR of ``order`` taps;
    see :func:`make_bandpass` notes in ``docs/assumption_log.md`` for why its
    lower edge is much softer inside an 8 s window.
    """
    sos, taps = _design(float(fs), float(low_hz), float(high_hz), str(design), int(order))
    return BandpassFilter(
        fs=float(fs),
        low_hz=float(low_hz),
        high_hz=float(high_hz),
        design=str(design),
        order=int(order),
        sos=sos,
        taps=taps,
    )


def apply_bandpass(x: NDArray[np.float64], filt: BandpassFilter) -> NDArray[np.float64]:
    """Apply the filter zero-phase along the last axis.

    The filter runs forwards and backwards, so the result has no phase
    distortion. Works on a 1-D signal or on a stacked ``(c, M)`` array.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] < filt.min_length:
        raise ValueError(
            f"signal of length {x.shape[-1]} is too short for this filter "
            f"({filt.description}); it needs at least {filt.min_length} samples"
        )
    if filt.is_fir:
        out = filtfilt(filt.taps, np.array([1.0]), x, axis=-1)
    else:
        out = sosfiltfilt(filt.sos, x, axis=-1)
    return np.ascontiguousarray(out)


def frequency_response(filt: BandpassFilter, n: int = 2048) -> dict[str, Any]:
    """Magnitude and phase of the filter, for ``diag['filter_response']``.

    The magnitude is squared before conversion to dB, because the zero-phase
    pass applies the filter twice; this is the response the signal actually
    sees. The phase is identically zero for the same reason.
    """
    if filt.is_fir:
        w, h = freqz(filt.taps, [1.0], worN=n, fs=filt.fs)
    else:
        w, h = sosfreqz(filt.sos, worN=n, fs=filt.fs)
    effective = np.abs(h) ** 2  # forward and backward
    with np.errstate(divide="ignore"):
        mag_db = 20.0 * np.log10(np.maximum(effective, 1e-20))
    freqs = np.asarray(w, dtype=np.float64)
    return {
        "f_hz": freqs,
        "mag_db": np.asarray(mag_db, dtype=np.float64),
        "phase_rad": np.zeros_like(freqs),  # zero-phase by construction
        "mag_linear": np.asarray(effective, dtype=np.float64),
    }


_ = field  # dataclasses re-export, used by plug-ins importing from here
