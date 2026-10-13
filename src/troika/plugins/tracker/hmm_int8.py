"""The HMM tracker scored by the 8-bit MLP (DEVIATIONS.md D15)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from troika.plugins.tracker.hmm import HMMTracker, HMMTrackerParams
from troika.registry import register
from troika.tracking.mlp_int8 import QuantisedMLP
from troika.tracking.xgboost_features import feature_matrix


@dataclass
class HMMInt8TrackerParams(HMMTrackerParams):
    """The HMM tracker's options; ``model_path`` is a ``QuantisedMLP`` file."""

    scorer: str = "mlp_int8"

    def __post_init__(self) -> None:
        if self.scorer != "mlp_int8":
            raise ValueError(f"scorer must be 'mlp_int8', got {self.scorer!r}")
        # The remaining checks are the parent's, which knows only its own scorers.
        HMMTrackerParams(**{**asdict(self), "scorer": "spectrum"})


@register("tracker", "hmm_int8")
class HMMInt8Tracker(HMMTracker):
    """``HMMTracker`` with its candidates scored by ``troika.tracking.mlp_int8``.

    Everything but the scorer is inherited, so a run differs from the ``mlp``
    scorer's only by what quantising the MLP to 8 bits changes.
    """

    Params = HMMInt8TrackerParams

    def _build_scorer(self, params: HMMInt8TrackerParams) -> Callable[..., NDArray[np.float64]]:
        path = Path(params.model_path)
        if not path.is_file():
            raise FileNotFoundError(f"quantised MLP model not found: {path.resolve()}")
        model = QuantisedMLP.load(path)
        fs, n_fft = self.static.fs, self.static.N
        return lambda s, c, ctx: model.score(feature_matrix(s, ctx, c, fs, n_fft))
