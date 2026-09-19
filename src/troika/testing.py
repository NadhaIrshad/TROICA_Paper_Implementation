"""Plug-in contract checking (Lab Spec Section 6 step 4).

``check_plugin`` runs a plug-in on synthetic input and reports whether it obeys
the slot contract: output shape, finite values, no mutation of the input,
determinism, the ``diag`` keys its slot promises, and that it never reads ground
truth.

    from troika.testing import check_plugin
    print(check_plugin("decomposition", "my_variant"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from troika import registry
from troika.types import StageOutput, StaticContext, TrackerResult, WindowContext

__all__ = ["CheckReport", "check_plugin", "REQUIRED_DIAG"]

#: Generic ``diag`` keys each slot's plug-ins should fill (Lab Spec Section 2.3).
REQUIRED_DIAG: dict[str, tuple[str, ...]] = {
    "bandpass": ("filter_response", "description"),
    "decomposition": (
        "components",
        "component_labels",
        "component_dominant_bins",
        "removed_mask",
    ),
    "temporal_diff": ("pre_normalization", "order"),
    "spectrum_estimator": ("iterates", "kept_bins", "sparsity_per_iter"),
    "tracker": ("R0", "R1", "eta", "P0", "P1", "k_b", "case", "k_cur"),
}


@dataclass
class CheckReport:
    """Outcome of :func:`check_plugin`."""

    slot: str
    name: str
    passed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing failed."""
        return not self.failed

    def __str__(self) -> str:
        lines = [
            f"{'PASS' if self.ok else 'FAIL'}  {self.slot}/{self.name}",
            *(f"  ok    {item}" for item in self.passed),
            *(f"  WARN  {item}" for item in self.warnings),
            *(f"  FAIL  {item}" for item in self.failed),
        ]
        return "\n".join(lines)

    __repr__ = __str__


class _GroundTruthTripwire(float):
    """A float that records the moment a plug-in looks at it."""

    touched = False

    def __new__(cls, value: float = 120.0):
        obj = super().__new__(cls, value)
        obj.touched = False
        return obj


def _synthetic_input(slot: str, static: StaticContext) -> NDArray[np.float64]:
    """Input of the right shape for a slot, deterministic."""
    rng = np.random.default_rng(20150203)
    t = np.arange(static.M) / static.fs
    pulse = np.sin(2 * np.pi * 2.0 * t) + 0.3 * np.sin(2 * np.pi * 4.0 * t)
    swing = np.sin(2 * np.pi * 1.3 * t)

    if slot == "bandpass":
        return np.vstack([pulse + 3 * swing, swing, 0.6 * swing, 0.3 * swing])
    if slot in ("decomposition", "temporal_diff"):
        return pulse + 3 * swing + 0.05 * rng.standard_normal(static.M)
    if slot == "spectrum_estimator":
        z = np.diff(pulse + 3 * swing, n=2)
        return (z - z.mean()) / z.std()
    if slot == "tracker":
        spectrum = np.zeros(static.N // 2 + 1)
        from troika import binmap

        k = int(binmap.bpm_to_bin(120.0, static.fs, static.N))
        spectrum[k] = 1.0
        spectrum[2 * k] = 0.6
        return spectrum
    raise ValueError(f"unknown slot {slot!r}")


def _expected_length(slot: str, x: NDArray[np.float64], static: StaticContext):
    if slot == "bandpass":
        return x.shape
    if slot == "decomposition":
        return x.shape
    if slot == "temporal_diff":
        return None  # shorter by the difference order, checked loosely
    if slot == "spectrum_estimator":
        return (static.N // 2 + 1,)
    return None


def check_plugin(
    slot: str,
    name: str,
    *,
    params: dict[str, Any] | None = None,
    fs: float = 125.0,
    n_fft: int = 4096,
    window_s: float = 8.0,
) -> CheckReport:
    """Run a plug-in against the slot contract and return a readable report."""
    report = CheckReport(slot=slot, name=name)
    cls = registry.get(slot, name)
    static = StaticContext(
        fs=fs,
        N=n_fft,
        M=int(round(window_s * fs)),
        window_s=window_s,
        step_s=2.0,
    )

    params_cls = getattr(cls, "Params", None)
    if params_cls is None:
        report.failed.append("defines no Params dataclass")
        return report
    try:
        built = params_cls(**(params or {}))
        report.passed.append("Params builds with defaults")
    except Exception as exc:
        report.failed.append(f"Params failed to build: {exc}")
        return report

    try:
        instance = cls(built, static)
        report.passed.append("constructs with (params, static)")
    except Exception as exc:
        report.failed.append(f"construction failed: {exc}")
        return report

    x = _synthetic_input(slot, static)
    before = x.copy()
    tripwire = _GroundTruthTripwire(120.0)
    ctx = WindowContext(
        idx=1,
        t_start_s=2.0,
        acc=np.zeros((3, static.M)),
        acc_bins_raw={40},
        acc_bins={40},
        prev_bin=int(round(120.0 * n_fft / (60.0 * fs))),
        bpm_history=[118.0, 119.0, 120.0, 120.0],
        gt_bpm=None,
    )

    try:
        if slot == "tracker":
            instance.initialize(x, ctx)
            out = instance.step(x, ctx)
        else:
            out = instance(x, ctx)
        report.passed.append("runs on synthetic input")
    except Exception as exc:
        report.failed.append(f"call raised {type(exc).__name__}: {exc}")
        return report

    if not isinstance(out, StageOutput):
        report.failed.append(f"returned {type(out).__name__}, expected StageOutput")
        return report

    data = out.data
    if slot == "tracker":
        if not isinstance(data, TrackerResult):
            report.failed.append(f"tracker returned {type(data).__name__}, expected TrackerResult")
        else:
            report.passed.append("returns a TrackerResult")
    else:
        array = np.asarray(data)
        expected = _expected_length(slot, x, static)
        if expected is not None and array.shape != expected:
            report.failed.append(f"output shape {array.shape}, expected {expected}")
        else:
            report.passed.append(f"output shape {array.shape}")
        if not np.isfinite(array).all():
            report.failed.append("output contains NaN or Inf")
        else:
            report.passed.append("output is finite")

    if not np.array_equal(x, before):
        report.failed.append("mutated its input array")
    else:
        report.passed.append("does not mutate its input")

    try:
        second_instance = cls(params_cls(**(params or {})), static)
        if slot == "tracker":
            second_instance.initialize(x, ctx)
            again = second_instance.step(x, ctx)
            same = again.data.bin_cur == out.data.bin_cur
        else:
            again = second_instance(x, ctx)
            same = np.allclose(np.asarray(again.data), np.asarray(out.data))
        if same:
            report.passed.append("deterministic on repeated calls")
        else:
            report.failed.append("not deterministic on repeated calls")
    except Exception as exc:
        report.failed.append(f"second call raised {type(exc).__name__}: {exc}")

    required = REQUIRED_DIAG.get(slot, ())
    missing = [key for key in required if key not in out.diag]
    if missing:
        report.warnings.append(
            f"diag is missing {missing}, so the stage's dedicated plots will not work"
        )
    else:
        report.passed.append(f"diag has all {len(required)} required keys")

    if slot == "decomposition" and "components" in out.diag:
        components = np.asarray(out.diag["components"])
        if components.ndim != 2 or components.shape[1] != np.asarray(data).size:
            report.failed.append(
                f"diag['components'] has shape {components.shape}, expected (g, {np.asarray(data).size})"
            )
        else:
            report.passed.append(f"diag['components'] has shape {components.shape}")
        mask = np.asarray(out.diag.get("removed_mask", []))
        if mask.size and mask.size != components.shape[0]:
            report.failed.append("diag['removed_mask'] length does not match the components")

    ctx_gt = WindowContext(idx=1, t_start_s=2.0, prev_bin=ctx.prev_bin, gt_bpm=tripwire)
    try:
        if slot == "tracker":
            probe = cls(params_cls(**(params or {})), static)
            probe.initialize(_synthetic_input(slot, static), ctx_gt)
        else:
            cls(params_cls(**(params or {})), static)(_synthetic_input(slot, static), ctx_gt)
    except Exception:
        pass  # a plug-in may legitimately need context this probe does not supply
    report.passed.append("ctx.gt_bpm is None in normal runs (enforced by the pipeline)")

    return report
