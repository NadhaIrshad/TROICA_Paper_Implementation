"""TinyEfficientNet1D candidate scorer shared by the plug-in and its training script.

The network replaces the XGBoost classifier of DEVIATIONS.md D9, not the stages
before it (D10). Each SSR peak candidate is scored from two inputs:

* a crop of the SSR spectrum centred on the candidate, which the 1-D
  convolutions read;
* the same truth-free feature row XGBoost uses
  (``troika.tracking.xgboost_features.feature_matrix``), standardised and joined
  to the pooled convolution output as ``aux``.

This module imports ``torch`` at the top, so import it lazily from anything that
must load without the optional dependency.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

__all__ = [
    "SqueezeExcite1D",
    "MBConv1D",
    "TinyEfficientNet1D",
    "candidate_crops",
    "CandidateScorer",
]

AUX_CLIP = 5.0  # standardised features are clipped to +/- this many deviations


class SqueezeExcite1D(nn.Module):
    """Channel re-weighting from the time-averaged activation."""

    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, 1)
        self.fc2 = nn.Conv1d(hidden, channels, 1)

    def forward(self, x):
        s = x.mean(dim=-1, keepdim=True)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class MBConv1D(nn.Module):
    """Inverted-residual block: expand, depthwise convolution, SE, project."""

    def __init__(self, in_ch, out_ch, kernel_size=5, stride=1, expand_ratio=3):
        super().__init__()
        self.use_residual = stride == 1 and in_ch == out_ch
        hidden = in_ch * expand_ratio if expand_ratio != 1 else in_ch

        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv1d(in_ch, hidden, 1, bias=False),
                       nn.BatchNorm1d(hidden), nn.SiLU()]
        layers += [nn.Conv1d(hidden, hidden, kernel_size, stride=stride,
                              padding=kernel_size // 2, groups=hidden, bias=False),
                   nn.BatchNorm1d(hidden), nn.SiLU()]
        self.block = nn.Sequential(*layers)
        self.se = SqueezeExcite1D(hidden)
        self.project = nn.Sequential(
            nn.Conv1d(hidden, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch)
        )

    def forward(self, x):
        out = self.project(self.se(self.block(x)))
        return out + x if self.use_residual else out


class TinyEfficientNet1D(nn.Module):
    """Three MBConv stages over a 1-channel series, plus ``aux`` at the head.

    ``forward(x, aux)`` takes ``x`` of shape ``(batch, in_len)`` and optional
    ``aux`` of shape ``(batch, aux_dim)``, and returns one logit per row.
    """

    def __init__(self, in_len, base_ch=12,
                 stage_cfg=((16, 1, 1), (24, 2, 3), (32, 2, 3)), aux_dim=0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(base_ch), nn.SiLU(),
        )
        blocks, in_ch = [], base_ch
        for out_ch, stride, expand in stage_cfg:
            blocks.append(MBConv1D(in_ch, out_ch, kernel_size=5, stride=stride, expand_ratio=expand))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(in_ch + aux_dim, 32), nn.SiLU(), nn.Linear(32, 1)
        )

    def forward(self, x, aux=None):
        x = x.unsqueeze(1)
        x = self.blocks(self.stem(x))
        x = self.pool(x).squeeze(-1)
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        return self.head(x).squeeze(-1)


def candidate_crops(
    spectrum: NDArray[np.float64], candidates: NDArray[np.int64], half_width: int,
) -> NDArray[np.float64]:
    """Spectrum crops of ``2 * half_width + 1`` bins, one row per candidate.

    Each row is centred on its candidate bin, scaled by the spectrum maximum
    (as the hand-made features are) and zero-padded past either end.
    """
    s = np.asarray(spectrum, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.int64)
    scale = max(float(np.max(s)), np.finfo(float).eps)
    padded = np.pad(s / scale, int(half_width))
    offsets = np.arange(2 * int(half_width) + 1)
    return padded[candidates[:, None] + offsets[None, :]]


class CandidateScorer:
    """A ``TinyEfficientNet1D`` plus the feature scaling it was trained with."""

    def __init__(
        self, model: TinyEfficientNet1D, in_len: int,
        aux_mean: NDArray[np.float64], aux_std: NDArray[np.float64],
    ) -> None:
        self.model = model.double().eval()
        self.in_len = int(in_len)
        self.aux_mean = torch.as_tensor(aux_mean, dtype=torch.float64)
        self.aux_std = torch.as_tensor(aux_std, dtype=torch.float64)

    @classmethod
    def untrained(cls, half_width: int, aux: NDArray[np.float64]) -> "CandidateScorer":
        """A freshly initialised scorer whose scaling is fitted to ``aux`` rows."""
        aux = np.asarray(aux, dtype=np.float64)
        std = aux.std(axis=0)
        in_len = 2 * int(half_width) + 1
        model = TinyEfficientNet1D(in_len=in_len, aux_dim=aux.shape[1])
        return cls(model, in_len, aux.mean(axis=0), np.where(std > 0, std, 1.0))

    @property
    def half_width(self) -> int:
        """Bins either side of the candidate that the network reads."""
        return (self.in_len - 1) // 2

    def standardise(self, aux: NDArray[np.float64]) -> torch.Tensor:
        """Scale feature rows as in training, clipped to ``AUX_CLIP`` deviations."""
        z = (torch.as_tensor(np.asarray(aux), dtype=torch.float64) - self.aux_mean) / self.aux_std
        return z.clamp(-AUX_CLIP, AUX_CLIP)

    def score(
        self, spectrum: NDArray[np.float64], candidates: NDArray[np.int64],
        aux: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Probability that each candidate is the heart-rate peak."""
        crops = torch.from_numpy(candidate_crops(spectrum, candidates, self.half_width))
        with torch.no_grad():
            logits = self.model(crops, self.standardise(aux))
        return torch.sigmoid(logits).numpy()

    def save(self, path: str | Path) -> None:
        """Write the weights and scaling as one file ``load`` can read back."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "in_len": self.in_len,
                "aux_mean": self.aux_mean,
                "aux_std": self.aux_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "CandidateScorer":
        """Read a file written by ``save``."""
        blob = torch.load(Path(path), map_location="cpu", weights_only=True)
        model = TinyEfficientNet1D(in_len=blob["in_len"], aux_dim=blob["aux_mean"].numel()).double()
        model.load_state_dict(blob["state_dict"])
        return cls(model, blob["in_len"], blob["aux_mean"], blob["aux_std"])
