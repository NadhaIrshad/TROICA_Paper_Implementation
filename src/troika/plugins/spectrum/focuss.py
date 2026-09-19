"""FOCUSS sparse spectrum estimation, the paper default (Section III-C)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.plugins.base import SpectrumEstimator
from troika.registry import register
from troika.ssr.basis import get_dictionary
from troika.ssr.focuss import focuss
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = ["FocussSpectrum"]


@dataclass
class FocussParams:
    """Options of the FOCUSS spectrum estimator."""

    p: float = 0.8  # PAPER
    lam: float = 0.1  # PAPER regularization parameter
    n_iter: int = 5  # PAPER
    x0: str = "ones"  # ASSUMPTION A11: ones | matched_filter
    prune_columns: bool = True  # PAPER Eq. 12
    low_hz: float = 0.4  # PAPER, band used for the Eq. 12 pruning
    high_hz: float = 5.0  # PAPER

    def __post_init__(self) -> None:
        if not 0.0 < self.p <= 2.0:
            raise ValueError(f"p must be in (0, 2], got {self.p}")
        if self.lam <= 0.0:
            raise ValueError(f"lam must be positive, got {self.lam}")
        if self.n_iter < 1:
            raise ValueError(f"n_iter must be >= 1, got {self.n_iter}")
        if self.x0 not in ("ones", "matched_filter"):
            raise ValueError(f"x0 must be 'ones' or 'matched_filter', got {self.x0!r}")


@register("spectrum_estimator", "focuss")
class FocussSpectrum(SpectrumEstimator):
    """Sparse spectrum on the shared N-point grid via Regularized M-FOCUSS.

    The estimated spectrum is ``s[k] = |x_k|^2`` (paper Eq. 10), mapped back onto
    the one-sided grid with zeros at the pruned bins. Peak tracking only ever
    looks at positive-frequency bins.
    """

    Params = FocussParams

    def __init__(self, params: FocussParams, static: StaticContext) -> None:
        super().__init__(params, static)
        # M' = M - difference order; resolved lazily on the first call, since the
        # temporal-difference stage decides it.
        self._cached_len: int | None = None

    def _dictionary(self, length: int):
        return get_dictionary(
            length,
            self.static.N,
            self.static.fs,
            self.params.low_hz,
            self.params.high_hz,
            self.params.prune_columns,
        )

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Return a one-sided power spectrum of length ``N // 2 + 1``."""
        z = np.asarray(x, dtype=np.float64)
        Phi, G, bins = self._dictionary(z.size)
        self._cached_len = z.size

        coeffs, iterates = focuss(
            z,
            Phi,
            G,
            p=self.params.p,
            lam=self.params.lam,
            n_iter=self.params.n_iter,
            x0=self.params.x0,
            return_iterates=True,
        )

        half = self.static.N // 2
        spectrum = np.zeros(half + 1, dtype=np.float64)
        positive = bins <= half
        np.add.at(spectrum, bins[positive], np.abs(coeffs[positive]) ** 2)

        power = np.abs(iterates) ** 2
        peak = power.max(axis=1, keepdims=True)
        sparsity = (power > 0.01 * np.where(peak > 0, peak, 1.0)).sum(axis=1)

        return StageOutput(
            data=spectrum,
            diag={
                "iterates": iterates,
                "kept_bins": bins,
                "sparsity_per_iter": np.asarray(sparsity, dtype=np.int64),
                "coefficients": coeffs,
            },
        )
