"""Band-pass plug-in that puts a fused two-channel PPG in place of the one channel (DEVIATIONS.md D16)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly

from troika.plugins.bandpass.butter import ButterBandpass, ButterParams
from troika.registry import register
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = ["FusedBandpass"]


@dataclass
class FusedParams(ButterParams):
    """The Butterworth options, plus the file of fused windows for the recording being run.

    ``windows_path`` is a ``.npy`` array ``(n_windows, m)`` with one fused PPG
    window per pipeline window, already band-passed, ``m`` dividing the window
    length. Left empty, the plug-in is the plain Butterworth band-pass.
    """

    windows_path: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.windows_path and self.mode != "per_window":
            raise ValueError("fused windows are per window; mode must be 'per_window'")


@register("bandpass", "fused")
class FusedBandpass(ButterBandpass):
    """Band-pass the accelerometer as usual and replace the PPG row by a fused window.

    The pipeline carries one PPG channel, so a fusion of the two cannot be
    computed inside it. It is computed beforehand for every window of a
    recording (``experiments/train_fused_ssa_ss_mlp_hmm.py``) and looked up here
    by window index. The fused window is resampled to the pipeline's sampling
    rate; it lies below 5 Hz, so nothing is lost. Every later stage then sees
    the fused signal where it would have seen PPG channel 1.
    """

    Params = FusedParams

    def __init__(self, params: FusedParams, static: StaticContext) -> None:
        super().__init__(params, static)
        self._windows: NDArray[np.float64] | None = None
        self._up = 1
        if params.windows_path:
            path = Path(params.windows_path)
            if not path.is_file():
                raise FileNotFoundError(f"fused PPG windows not found: {path.resolve()}")
            windows = np.asarray(np.load(path), dtype=np.float64)
            if windows.ndim != 2 or static.M % windows.shape[1] != 0:
                raise ValueError(
                    f"{path.name}: expected (n_windows, m) with m dividing {static.M}, got shape {windows.shape}"
                )
            self._windows = windows
            self._up = static.M // windows.shape[1]

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Filter the ``(4, M)`` stack; row 0 becomes the fused window ``ctx.idx``."""
        out = super().__call__(x, ctx)
        if self._windows is None:
            return out
        if not 0 <= ctx.idx < len(self._windows):
            raise IndexError(f"no fused window {ctx.idx}; the file holds {len(self._windows)}")
        fused = self._windows[ctx.idx]
        data = np.array(out.data, dtype=np.float64, copy=True)
        data[0] = resample_poly(fused, self._up, 1) if self._up > 1 else fused
        return StageOutput(
            data=data,
            diag={**out.diag, "description": f"{out.diag['description']}; PPG row replaced by a fused window"},
        )
