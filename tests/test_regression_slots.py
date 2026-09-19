"""Refactor guard (Lab Spec Section 9, `test_regression_slots`).

The slot-based pipeline must keep reproducing the same numbers on a fixed
synthetic recording. The golden file was captured from a known-good run; a
change to it is a change to the results and has to be deliberate.

To re-capture after an intended change, and only then:

    python -c "import tests.test_regression_slots as t; t.capture()"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from troika.config import load_config, repo_root
from troika.pipeline import TroikaEstimator

from test_pipeline_synthetic import synthetic_recording

GOLDEN = repo_root() / "tests" / "reference" / "golden_pipeline.json"

#: Variants captured in the golden file, and the overrides that produce them.
VARIANTS = {
    "default": {},
    "no_ssa": {"decomposition.method": "none"},
    "fft": {"spectrum_estimator.method": "fft"},
}


def _run(label: str):
    cfg = load_config()
    overrides = VARIANTS[label]
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg, TroikaEstimator(cfg).run(synthetic_recording(seconds=40.0, subject_id=1))


def capture() -> Path:
    """Rewrite the golden file. Run this only when a change is intended."""
    golden = {}
    for label in VARIANTS:
        cfg, run = _run(label)
        golden[label] = {
            "bpm_est": [round(float(v), 6) for v in run.bpm_est],
            "bins": [int(w.bin_cur) for w in run.windows],
            "cases": [w.case for w in run.windows],
            "config_hash": cfg.hash(),
        }
    GOLDEN.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    return GOLDEN


@pytest.fixture(scope="module")
def golden():
    if not GOLDEN.exists():
        pytest.skip(f"no golden file at {GOLDEN}; run capture() to create one")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


@pytest.mark.parametrize("label", list(VARIANTS))
def test_pipeline_reproduces_the_golden_output(golden, label):
    """Every estimate must match the captured run exactly."""
    expected = golden[label]
    cfg, run = _run(label)

    assert run.n_windows == len(expected["bpm_est"]), "window count changed"
    assert [int(w.bin_cur) for w in run.windows] == expected["bins"], (
        f"{label}: the selected bins changed, so the numbers changed"
    )
    np.testing.assert_allclose(
        run.bpm_est,
        np.asarray(expected["bpm_est"], dtype=float),
        rtol=0,
        atol=1e-6,
        err_msg=f"{label}: estimates changed",
    )


@pytest.mark.parametrize("label", list(VARIANTS))
def test_case_sequence_is_unchanged(golden, label):
    """Which of Cases 1 to 3 fired, window by window."""
    _, run = _run(label)
    assert [w.case for w in run.windows] == golden[label]["cases"]


@pytest.mark.parametrize("label", list(VARIANTS))
def test_config_hash_is_unchanged(golden, label):
    """A changed hash means a config key moved, which the golden file cannot see.

    This is not a failure by itself, but it explains one: if the estimates above
    changed and this changed too, the cause is the configuration, not the code.
    """
    cfg, _ = _run(label)
    assert cfg.hash() == golden[label]["config_hash"], (
        "the resolved configuration changed; if that was intended, re-capture "
        "the golden file with tests.test_regression_slots.capture()"
    )


def test_variants_differ_from_each_other(golden):
    """A guard that would otherwise pass trivially if every variant were identical."""
    series = {label: tuple(golden[label]["bpm_est"]) for label in VARIANTS}
    assert len(set(series.values())) == len(VARIANTS)
