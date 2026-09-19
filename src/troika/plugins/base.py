"""Abstract base class per slot (Lab Spec Section 2.1).

Every plug-in is constructed once per run with its validated ``Params`` and the
run-level :class:`~troika.types.StaticContext`, then called once per window with
the stage input and the per-window :class:`~troika.types.WindowContext`.

Shapes, in pipeline order (M = window samples, M' = M - diff order, N = grid):

===================  ==========================  ==========================
Slot                 Input                       Output ``StageOutput.data``
===================  ==========================  ==========================
``bandpass``         ``(4, M)`` [PPG; ACC x,y,z]  ``(4, M)``
``decomposition``    PPG ``(M,)``                cleansed PPG ``(M,)``
``temporal_diff``    ``(M,)``                    ``(M',)``
``spectrum_estimator`` ``(M',)``                 power spectrum ``(N//2+1,)``
``tracker``          spectrum ``(N//2+1,)``      :class:`TrackerResult`
===================  ==========================  ==========================

No plug-in may read ``ctx.gt_bpm``; it is ``None`` unless the leak guard is
explicitly disabled (Lab Spec Sections 2.4 and 11).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from troika.types import StageOutput, StaticContext, TrackerResult, WindowContext

__all__ = [
    "Plugin",
    "Bandpass",
    "Decomposition",
    "TemporalDiff",
    "SpectrumEstimator",
    "Tracker",
]


@dataclass
class _NoParams:
    """Empty parameter set, for plug-ins that take no options."""


class Plugin(ABC):
    """Common construction protocol for every slot plug-in."""

    #: Dataclass of this plug-in's options; config ``params`` validate against it.
    Params: ClassVar[type] = _NoParams

    def __init__(self, params, static: StaticContext) -> None:
        self.params = params
        self.static = static

    @property
    def name(self) -> str:
        """Registered name of this plug-in."""
        return getattr(type(self), "_name", type(self).__name__)


class Bandpass(Plugin):
    """Band-pass the stacked ``(4, M)`` array of PPG and three ACC axes.

    Paper Section III: PPG and acceleration are band-pass filtered 0.4-5 Hz
    before TROIKA starts. Blueprint Section 5.1; A1 (design) and A2 (scope).

    ``diag`` keys: ``filter_response`` = {``f_hz``, ``mag_db``, ``phase_rad``},
    ``description``.
    """

    @abstractmethod
    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Filter ``x`` of shape ``(4, M)`` and return the same shape."""

    def prepare(self, recording_len: int) -> None:
        """Optional hook for ``mode: global`` filtering of a whole recording."""


class Decomposition(Plugin):
    """Decompose the band-passed PPG window and drop motion-artifact components.

    Paper Section III-A. Blueprint Sections 5.2-5.3.

    Generic ``diag`` keys (any method, including a user-written EMD):
    ``components (g, M)``, ``component_labels``, ``component_dominant_bins (g,)``,
    ``removed_mask (g,)``. SSA adds ``trajectory``, ``U``, ``s``, ``Vt``,
    ``groups``, ``group_dominant_bins``, ``group_singular_range``, ``L``, ``K``,
    ``grouping_strategy``.
    """

    @abstractmethod
    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return the cleansed PPG window, same length as ``x``."""


class TemporalDiff(Plugin):
    """Temporal difference between decomposition and spectrum estimation.

    Paper Section III-B: the second-order difference keeps the heartbeat
    fundamental and harmonics while suppressing aperiodic motion artifact.
    Blueprint Section 5.4; A3 (normalisation).

    ``diag`` keys: ``pre_normalization (M',)``, ``order``.
    """

    @abstractmethod
    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return a series of length ``len(x) - order``."""


class SpectrumEstimator(Plugin):
    """Estimate the power spectrum on the shared ``N``-point grid.

    Paper Section III-C. Blueprint Section 5.5.

    ``diag`` keys: ``iterates (n_iter+1, n_cols)`` complex, ``kept_bins``,
    ``sparsity_per_iter``.
    """

    @abstractmethod
    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return a one-sided power spectrum of length ``N // 2 + 1``."""


class Tracker(Plugin):
    """Stateful spectral peak tracking with verification.

    Paper Section III-D. Blueprint Section 5.6. The tracker owns the only state
    that persists across windows, so windows must be processed in order.

    ``diag`` keys: ``R0``, ``R1``, ``eta``, ``P0``, ``P1``, ``pairs``,
    ``candidates``, ``k_b``, ``case``, ``k_cur``, ``rule1_fired``,
    ``rule2_fired``, ``same_count``, ``trend``, ``delta_s_used``, ``k_prev``.
    """

    @abstractmethod
    def initialize(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Handle the first window; ``StageOutput.data`` is a :class:`TrackerResult`."""

    @abstractmethod
    def step(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Handle a subsequent window; ``StageOutput.data`` is a :class:`TrackerResult`."""

    @property
    @abstractmethod
    def prev_bin(self) -> int | None:
        """Previously estimated 0-based HR bin, or ``None`` before initialisation."""

    def __call__(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Dispatch to :meth:`initialize` on the first window, else :meth:`step`."""
        if self.prev_bin is None:
            return self.initialize(spectrum, ctx)
        return self.step(spectrum, ctx)


_ = TrackerResult  # re-exported through troika.types; referenced in docstrings
