"""MLP candidate scorer shared by the plug-in and its training script.

The ablation of DEVIATIONS.md D11: the head of the D10 network
(``troika.tracking.efficientnet1d.TinyEfficientNet1D``) on its own. Each SSR
peak candidate is scored from the truth-free feature row XGBoost uses
(``troika.tracking.xgboost_features.feature_matrix``), standardised; there are
no convolutions and no spectrum crop, so the gap to D10 is what the crop adds.

This module imports ``torch`` at the top, so import it lazily from anything that
must load without the optional dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

__all__ = ["CandidateMLP", "MLPScorer"]

AUX_CLIP = 5.0  # standardised features are clipped to +/- this many deviations
HIDDEN = 32  # same width as the D10 network's head


class CandidateMLP(nn.Module):
    """One hidden layer over the feature row, as in the D10 head.

    ``forward(aux)`` takes ``aux`` of shape ``(batch, aux_dim)`` and returns one
    logit per row.
    """

    def __init__(self, aux_dim, hidden=HIDDEN):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(aux_dim, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, aux):
        return self.head(aux).squeeze(-1)


class MLPScorer:
    """A ``CandidateMLP`` plus the feature scaling it was trained with."""

    def __init__(
        self, model: CandidateMLP, aux_mean: NDArray[np.float64], aux_std: NDArray[np.float64],
    ) -> None:
        self.model = model.double().eval()
        self.aux_mean = torch.as_tensor(aux_mean, dtype=torch.float64)
        self.aux_std = torch.as_tensor(aux_std, dtype=torch.float64)

    @classmethod
    def untrained(cls, aux: NDArray[np.float64]) -> "MLPScorer":
        """A freshly initialised scorer whose scaling is fitted to ``aux`` rows."""
        aux = np.asarray(aux, dtype=np.float64)
        std = aux.std(axis=0)
        return cls(CandidateMLP(aux.shape[1]), aux.mean(axis=0), np.where(std > 0, std, 1.0))

    def standardise(self, aux: NDArray[np.float64]) -> torch.Tensor:
        """Scale feature rows as in training, clipped to ``AUX_CLIP`` deviations."""
        z = (torch.as_tensor(np.asarray(aux), dtype=torch.float64) - self.aux_mean) / self.aux_std
        return z.clamp(-AUX_CLIP, AUX_CLIP)

    def score(self, aux: NDArray[np.float64]) -> NDArray[np.float64]:
        """Probability that each candidate is the heart-rate peak."""
        with torch.no_grad():
            logits = self.model(self.standardise(aux))
        return torch.sigmoid(logits).numpy()

    def save(self, path: str | Path) -> None:
        """Write the weights and scaling as one file ``load`` can read back."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "aux_mean": self.aux_mean,
                "aux_std": self.aux_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPScorer":
        """Read a file written by ``save``."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = CandidateMLP(blob["aux_mean"].numel()).double()
        model.load_state_dict(blob["state_dict"])
        return cls(model, blob["aux_mean"], blob["aux_std"])
