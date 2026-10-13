"""8-bit simulation of the MLP candidate scorer (DEVIATIONS.md D15).

Post-training quantisation of ``troika.tracking.mlp.MLPScorer`` in the scheme
embedded runtimes such as TensorFlow Lite Micro use: weights are symmetric int8
with one scale per layer, biases are int32, and every tensor between layers is
int8 with a scale and zero point taken from the range it covers on calibration
rows. The arithmetic of the two layers is done on the integers.

It is a simulation, written to measure what 8 bits cost in tracking accuracy
before anything is put on a device: a real runtime rounds its fixed-point
multipliers and its sigmoid table slightly differently, so scores can differ
from this by a least-significant bit.

Feature standardisation stays in floating point, outside the quantised model.
The 13 raw features differ in scale by orders of magnitude, and an int8 input
has a single scale, so quantising them raw would erase the small ones.

This module needs only NumPy and SciPy; the scorer handed to :func:`quantise`
is read through its attributes, so ``torch`` is not imported here.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.special import expit

__all__ = ["QuantisedMLP", "quantise"]

AUX_CLIP = 5.0  # as in troika.tracking.mlp: standardised features are clipped to +/- this
PROB_SCALE, PROB_ZERO = 1.0 / 256.0, -128  # the fixed int8 output of a sigmoid in TFLite


def _affine(lo: float, hi: float) -> tuple[float, int]:
    """Scale and zero point of an int8 tensor covering ``[lo, hi]``, zero included."""
    lo, hi = min(float(lo), 0.0), max(float(hi), 0.0)
    scale = max(hi - lo, 1e-12) / 255.0
    return scale, int(np.clip(round(-128.0 - lo / scale), -128, 127))


def _symmetric(w: NDArray[np.float64]) -> tuple[NDArray[np.int8], float]:
    """Weights as int8 in ``[-127, 127]`` with one scale and no zero point."""
    scale = max(float(np.max(np.abs(w))), 1e-12) / 127.0
    return np.clip(np.rint(w / scale), -127, 127).astype(np.int8), scale


def _q(x: NDArray[np.float64], scale: float, zero: int) -> NDArray[np.int64]:
    return np.clip(np.rint(x / scale) + zero, -128, 127).astype(np.int64)


def _dq(q: NDArray[np.int64], scale: float, zero: int) -> NDArray[np.float64]:
    return (q - zero) * scale


@dataclass
class QuantisedMLP:
    """The quantised weights, the tensor scales and the float feature scaling.

    ``scales`` and ``zeros`` are those of the input, the hidden layer before
    and after its activation, and the output logit, in that order.
    """

    w1: NDArray[np.int8]
    b1: NDArray[np.int32]
    w2: NDArray[np.int8]
    b2: NDArray[np.int32]
    weight_scales: NDArray[np.float64]
    scales: NDArray[np.float64]
    zeros: NDArray[np.int64]
    aux_mean: NDArray[np.float64]
    aux_std: NDArray[np.float64]

    @property
    def n_bytes(self) -> int:
        """Storage of the weights and biases, in bytes."""
        return int(self.w1.nbytes + self.b1.nbytes + self.w2.nbytes + self.b2.nbytes)

    def standardise(self, aux: NDArray[np.float64]) -> NDArray[np.float64]:
        """Scale feature rows as in training, clipped to ``AUX_CLIP`` deviations."""
        z = (np.asarray(aux, dtype=np.float64) - self.aux_mean) / self.aux_std
        return np.clip(z, -AUX_CLIP, AUX_CLIP)

    def score(self, aux: NDArray[np.float64]) -> NDArray[np.float64]:
        """Probability that each candidate is the heart-rate peak, in steps of 1/256."""
        (s_in, s_h, s_a, s_o), (z_in, z_h, z_a, z_o) = self.scales, (int(z) for z in self.zeros)
        s_w1, s_w2 = self.weight_scales

        x = _q(self.standardise(aux), s_in, z_in)
        h = _q(((x - z_in) @ self.w1.astype(np.int64).T + self.b1) * (s_in * s_w1), s_h, z_h)
        # SiLU as an embedded runtime computes it: an int8 sigmoid, then an int8 product.
        gate = _q(expit(_dq(h, s_h, z_h)), PROB_SCALE, PROB_ZERO)
        a = _q(_dq(h, s_h, z_h) * _dq(gate, PROB_SCALE, PROB_ZERO), s_a, z_a)
        logit = _q(((a - z_a) @ self.w2.astype(np.int64).T + self.b2) * (s_a * s_w2), s_o, z_o)
        p = _q(expit(_dq(logit, s_o, z_o)), PROB_SCALE, PROB_ZERO)
        return _dq(p, PROB_SCALE, PROB_ZERO).ravel()

    def save(self, path: str | Path) -> None:
        """Write every field to one ``.npz`` file ``load`` can read back."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{f.name: getattr(self, f.name) for f in fields(self)})

    @classmethod
    def load(cls, path: str | Path) -> "QuantisedMLP":
        """Read a file written by ``save``."""
        with np.load(Path(path)) as blob:
            return cls(**{f.name: blob[f.name] for f in fields(cls)})


def quantise(scorer, calibration_aux: NDArray[np.float64]) -> QuantisedMLP:
    """Quantise an ``MLPScorer``, taking the tensor ranges from ``calibration_aux``.

    ``calibration_aux`` holds raw feature rows ``(n, 13)``; they should be
    training rows, never those of the subject the model is then tested on.
    """
    calibration_aux = np.asarray(calibration_aux, dtype=np.float64)
    if calibration_aux.ndim != 2 or calibration_aux.shape[0] == 0:
        raise ValueError("calibration_aux must hold at least one feature row")

    first, second = scorer.model.head[0], scorer.model.head[2]
    w1 = first.weight.detach().cpu().numpy().astype(np.float64)
    b1 = first.bias.detach().cpu().numpy().astype(np.float64)
    w2 = second.weight.detach().cpu().numpy().astype(np.float64)
    b2 = second.bias.detach().cpu().numpy().astype(np.float64)
    mean = np.asarray(scorer.aux_mean, dtype=np.float64)
    std = np.asarray(scorer.aux_std, dtype=np.float64)

    # Float forward pass over the calibration rows, for the range of each tensor.
    z = np.clip((calibration_aux - mean) / std, -AUX_CLIP, AUX_CLIP)
    h = z @ w1.T + b1
    a = h * expit(h)
    logit = a @ w2.T + b2

    w1q, s_w1 = _symmetric(w1)
    w2q, s_w2 = _symmetric(w2)
    params = [_affine(-AUX_CLIP, AUX_CLIP), _affine(h.min(), h.max()), _affine(a.min(), a.max()), _affine(logit.min(), logit.max())]
    scales = np.array([s for s, _ in params], dtype=np.float64)
    zeros = np.array([k for _, k in params], dtype=np.int64)
    return QuantisedMLP(
        w1=w1q,
        b1=np.rint(b1 / (scales[0] * s_w1)).astype(np.int32),
        w2=w2q,
        b2=np.rint(b2 / (scales[2] * s_w2)).astype(np.int32),
        weight_scales=np.array([s_w1, s_w2], dtype=np.float64),
        scales=scales,
        zeros=zeros,
        aux_mean=mean,
        aux_std=std,
    )
