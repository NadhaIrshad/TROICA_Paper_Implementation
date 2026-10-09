"""Hidden-Markov path tracker over scored SSR peak candidates (DEVIATIONS.md D12)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from troika import binmap
from troika.plugins.base import Tracker
from troika.registry import register
from troika.tracking.hmm import emission_log, transition_log, viterbi_step
from troika.tracking.xgboost_features import FEATURE_NAMES, candidate_bins, feature_matrix
from troika.types import StageOutput, StaticContext, TrackerResult, WindowContext

SCORERS = ("spectrum", "xgboost", "efficientnet1d", "mlp")


@dataclass
class HMMTrackerParams:
    """Inference options.

    ``scorer`` says what scores the candidates: ``spectrum`` uses their relative
    SSR power and needs no model; the others load ``model_path`` as written by
    the matching training script.
    """

    scorer: str = "spectrum"
    model_path: str = ""
    candidate_count: int = 8
    low_bpm: float = 40.0
    high_bpm: float = 200.0
    transition_sigma_bins: float = 3.0  # typical move between windows 2 s apart
    jump_prob: float = 0.01  # chance per window of a move to anywhere in the band
    emission_sigma_bins: float = 1.0  # how far a candidate's support spreads
    emission_floor: float = 0.05  # likelihood of a bin with no candidate nearby

    def __post_init__(self) -> None:
        if self.scorer not in SCORERS:
            raise ValueError(f"scorer must be one of {SCORERS}, got {self.scorer!r}")
        if self.candidate_count < 1:
            raise ValueError("candidate_count must be >= 1")
        if self.low_bpm >= self.high_bpm:
            raise ValueError("low_bpm must be less than high_bpm")
        if self.transition_sigma_bins <= 0 or self.emission_sigma_bins <= 0:
            raise ValueError("transition_sigma_bins and emission_sigma_bins must be > 0")
        if not 0.0 < self.jump_prob < 1.0:
            raise ValueError("jump_prob must be in (0, 1)")
        if self.emission_floor <= 0:
            raise ValueError("emission_floor must be > 0")


@register("tracker", "hmm")
class HMMTracker(Tracker):
    """Follow the most probable heart-rate path through the scored candidates.

    Candidates and first-window initialisation are those of the XGBoost
    tracker. In place of its jump guard, a hidden Markov model over the bins of
    the heart-rate band weighs each window's scores against how far the rate
    can move in one step (``troika.tracking.hmm``). Each window's estimate is
    the end of the best path so far, so the tracker is causal.
    """

    Params = HMMTrackerParams

    def __init__(self, params: HMMTrackerParams, static: StaticContext) -> None:
        super().__init__(params, static)
        self._lo = int(binmap.bpm_to_bin(params.low_bpm, static.fs, static.N))
        self._hi = min(static.N // 2, int(binmap.bpm_to_bin(params.high_bpm, static.fs, static.N)))
        self._n_states = self._hi - self._lo + 1
        self._log_trans = transition_log(self._n_states, params.transition_sigma_bins, params.jump_prob)
        self._score = self._build_scorer(params)
        self._log_delta: NDArray[np.float64] | None = None
        self._prev_bin: int | None = None
        self._history: list[float] = []

    def _build_scorer(self, params: HMMTrackerParams) -> Callable[..., NDArray[np.float64]]:
        """``(spectrum, candidates, ctx) -> scores`` for the configured scorer."""
        if params.scorer == "spectrum":
            return lambda s, c, ctx: s[c] / max(float(np.max(s)), np.finfo(float).eps)

        path = Path(params.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"{params.scorer} model not found: {path.resolve()}")
        fs, n_fft = self.static.fs, self.static.N
        if params.scorer == "xgboost":
            try:
                from xgboost import XGBClassifier
            except ImportError as exc:
                raise ImportError("Install the ML extra: pip install -e '.[ml]'") from exc
            model = XGBClassifier()
            model.load_model(path)
            return lambda s, c, ctx: model.predict_proba(feature_matrix(s, ctx, c, fs, n_fft))[:, 1]

        try:
            from troika.tracking.efficientnet1d import CandidateScorer
            from troika.tracking.mlp import MLPScorer
        except ImportError as exc:
            raise ImportError("Install the deep-learning extra: pip install -e '.[dl]'") from exc
        if params.scorer == "mlp":
            mlp = MLPScorer.load(path)
            return lambda s, c, ctx: mlp.score(feature_matrix(s, ctx, c, fs, n_fft))
        network = CandidateScorer.load(path)
        return lambda s, c, ctx: network.score(s, c, feature_matrix(s, ctx, c, fs, n_fft))

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

    def _emission(self, candidates: NDArray[np.int64], scores: NDArray[np.float64]) -> NDArray[np.float64]:
        return emission_log(
            self._n_states, np.asarray(candidates) - self._lo, scores,
            self.params.emission_sigma_bins, self.params.emission_floor,
        )

    def initialize(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        source = ctx.init_spectrum if ctx.init_spectrum is not None else spectrum
        k = self._commit(self._lo + int(np.argmax(source[self._lo : self._hi + 1])))
        # The first window counts as one candidate of score 1 at the highest
        # peak; every other bin keeps the floor, so a wrong start can be left.
        log_emit = self._emission(np.array([k]), np.array([1.0]))
        self._log_delta = log_emit - log_emit.max()
        return StageOutput(
            data=TrackerResult(bin_cur=k, bpm=self._history[-1]),
            diag={"method": "hmm", "scorer": self.params.scorer, "candidates": np.array([k]), "scores": np.array([1.0]), "k_cur": k},
        )

    def step(self, spectrum: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        candidates = candidate_bins(spectrum, self.static.fs, self.static.N, self.params.low_bpm, self.params.high_bpm, self.params.candidate_count)
        scores = np.asarray(self._score(spectrum, candidates, ctx), dtype=np.float64)
        self._log_delta = viterbi_step(self._log_delta, self._log_trans, self._emission(candidates, scores))
        k = self._commit(self._lo + int(np.argmax(self._log_delta)))
        return StageOutput(
            data=TrackerResult(bin_cur=k, bpm=self._history[-1]),
            diag={"method": "hmm", "scorer": self.params.scorer, "feature_names": FEATURE_NAMES, "candidates": candidates, "scores": scores, "k_cur": k, "log_delta": self._log_delta},
        )
