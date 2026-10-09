"""Learned candidate-ranking tracker scored by a TinyEfficientNet1D (DEVIATIONS.md D10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.plugins.base import Tracker
from troika.registry import register
from troika.tracking.xgboost_features import (
    FEATURE_NAMES,
    candidate_bins,
    feature_matrix,
    guarded_choice,
)
from troika.types import StageOutput, StaticContext, TrackerResult, WindowContext


@dataclass
class EfficientNet1DTrackerParams:
    """Inference options. ``model_path`` is produced by the training script."""

    model_path: str = "models/efficientnet1d_tracker.pt"
    candidate_count: int = 8
    low_bpm: float = 40.0
    high_bpm: float = 200.0
    max_jump_bins: int = 6  # same size as the paper's theta (Section III-D.3)
    jump_patience: int = 3  # windows a far candidate must win; 1 disables the guard

    def __post_init__(self) -> None:
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.low_bpm >= self.high_bpm:
            raise ValueError("low_bpm must be less than high_bpm")
        if self.max_jump_bins < 0:
            raise ValueError("max_jump_bins must be >= 0")
        if self.jump_patience < 1:
            raise ValueError("jump_patience must be >= 1")


@register("tracker", "efficientnet1d")
class EfficientNet1DTracker(Tracker):
    """Rank SSR peak candidates using a pre-trained TinyEfficientNet1D.

    Identical to the XGBoost tracker except for the scorer: the network reads a
    spectrum crop around each candidate and takes the XGBoost feature row as
    ``aux``. Initialisation, candidates and the jump guard are shared.
    """

    Params = EfficientNet1DTrackerParams

    def __init__(self, params: EfficientNet1DTrackerParams, static: StaticContext) -> None:
        super().__init__(params, static)
        try:
            from troika.tracking.efficientnet1d import CandidateScorer
        except ImportError as exc:
            raise ImportError("Install the deep-learning extra: pip install -e '.[dl]'") from exc
        path = Path(params.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"EfficientNet1D model not found: {path.resolve()}")
        self.scorer = CandidateScorer.load(path)
        self._prev_bin: int | None = None
        self._history: list[float] = []
        self._pending: tuple[int, int] | None = None  # far candidate awaiting confirmation

    @property
    def prev_bin(self) -> int | None:
        return self._prev_bin

    @property
    def bpm_history(self) -> list[float]:
        return list(self._history)

    def _commit(self, k: int) -> int:
        self._prev_bin = int(np.clip(k, 0, self.static.N // 2))
        self._history.append(float(binmap.bin_to_bpm(self._prev_bin, self.static.fs, self.static.N)))
        return self._prev_bin

    def initialize(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        source = ctx.init_spectrum if ctx.init_spectrum is not None else spectrum
        lo = int(binmap.bpm_to_bin(self.params.low_bpm, self.static.fs, self.static.N))
        hi = int(binmap.bpm_to_bin(self.params.high_bpm, self.static.fs, self.static.N))
        k = self._commit(lo + int(np.argmax(source[lo : hi + 1])))
        return StageOutput(
            data=TrackerResult(bin_cur=k, bpm=self._history[-1]),
            diag={"method": "efficientnet1d", "candidates": np.array([k]), "scores": np.array([1.0]), "k_cur": k},
        )

    def step(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        candidates = candidate_bins(spectrum, self.static.fs, self.static.N, self.params.low_bpm, self.params.high_bpm, self.params.candidate_count)
        aux = feature_matrix(spectrum, ctx, candidates, self.static.fs, self.static.N)
        scores = self.scorer.score(spectrum, candidates, aux)
        choice, self._pending, held = guarded_choice(
            candidates, scores, self._prev_bin, self._pending,
            self.params.max_jump_bins, self.params.jump_patience,
        )
        k = self._commit(choice)
        return StageOutput(
            data=TrackerResult(bin_cur=k, bpm=self._history[-1]),
            diag={"method": "efficientnet1d", "feature_names": FEATURE_NAMES, "candidates": candidates, "scores": scores, "k_cur": k, "jump_held": held},
        )
