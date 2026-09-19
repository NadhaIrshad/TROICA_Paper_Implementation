"""The plug-in contract checker (Lab Spec Section 9, `test_plugin_contract`)."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest

import troika.plugins  # noqa: F401
from troika import registry
from troika.plugins.base import Decomposition
from troika.plugins.new import scaffold
from troika.testing import REQUIRED_DIAG, check_plugin
from troika.types import StageOutput


@pytest.fixture
def clean_slot():
    saved = dict(registry._REGISTRY["decomposition"])
    yield "decomposition"
    registry._REGISTRY["decomposition"].clear()
    registry._REGISTRY["decomposition"].update(saved)


@pytest.mark.parametrize(
    "slot,name",
    [(slot, name) for slot in registry.SLOTS for name in registry.available(slot)],
)
def test_every_builtin_passes_the_contract(slot, name):
    report = check_plugin(slot, name)
    assert report.ok, str(report)


@pytest.mark.parametrize("slot", registry.SLOTS)
def test_paper_default_fills_its_diag_schema(slot):
    """Lab Spec Section 2.3: paper-default plug-ins fill every listed key."""
    from troika.config import load_config

    name = load_config().slot_method(slot)
    report = check_plugin(slot, name)
    assert report.ok, str(report)
    assert not report.warnings, f"{slot}/{name}: {report.warnings}"
    assert REQUIRED_DIAG[slot]


def test_wrong_output_length_fails(clean_slot):
    @registry.register(clean_slot, "too_short", override=True)
    class TooShort(Decomposition):
        @dataclass
        class Params:
            pass

        def __call__(self, x, ctx):
            return StageOutput(data=x[:-10])

    report = check_plugin(clean_slot, "too_short")
    assert not report.ok
    assert any("shape" in item for item in report.failed)


def test_nan_output_fails(clean_slot):
    @registry.register(clean_slot, "nan_maker", override=True)
    class NanMaker(Decomposition):
        @dataclass
        class Params:
            pass

        def __call__(self, x, ctx):
            out = x.copy()
            out[5] = np.nan
            return StageOutput(data=out)

    report = check_plugin(clean_slot, "nan_maker")
    assert not report.ok
    assert any("NaN" in item for item in report.failed)


def test_input_mutation_fails(clean_slot):
    @registry.register(clean_slot, "mutator", override=True)
    class Mutator(Decomposition):
        @dataclass
        class Params:
            pass

        def __call__(self, x, ctx):
            x *= 2.0  # in place, which is the bug
            return StageOutput(data=x)

    report = check_plugin(clean_slot, "mutator")
    assert not report.ok
    assert any("mutated" in item for item in report.failed)


def test_nondeterminism_fails(clean_slot):
    @registry.register(clean_slot, "random_maker", override=True)
    class RandomMaker(Decomposition):
        @dataclass
        class Params:
            pass

        def __call__(self, x, ctx):
            return StageOutput(data=x + np.random.default_rng().standard_normal(x.size))

    report = check_plugin(clean_slot, "random_maker")
    assert not report.ok
    assert any("deterministic" in item for item in report.failed)


def test_missing_diag_keys_warn(clean_slot):
    @registry.register(clean_slot, "no_diag", override=True)
    class NoDiag(Decomposition):
        @dataclass
        class Params:
            pass

        def __call__(self, x, ctx):
            return StageOutput(data=x.copy())

    report = check_plugin(clean_slot, "no_diag")
    assert report.ok  # the output is fine
    assert any("diag is missing" in item for item in report.warnings)


def test_raising_plugin_fails_cleanly(clean_slot):
    @registry.register(clean_slot, "exploder", override=True)
    class Exploder(Decomposition):
        @dataclass
        class Params:
            pass

        def __call__(self, x, ctx):
            raise RuntimeError("boom")

    report = check_plugin(clean_slot, "exploder")
    assert not report.ok
    assert any("RuntimeError" in item for item in report.failed)


def test_report_is_readable(clean_slot):
    text = str(check_plugin("decomposition", "ssa"))
    assert text.startswith("PASS")
    assert "decomposition/ssa" in text


# ------------------------------------------------------------------ scaffold


def test_scaffold_writes_both_files(tmp_path):
    written = scaffold("decomposition", "my_variant", root=tmp_path)
    assert len(written) == 2
    assert all(path.exists() for path in written)

    plugin_text = written[0].read_text(encoding="utf-8")
    assert "@register(\"decomposition\", \"my_variant\")" in plugin_text
    assert "class MyVariant(Decomposition)" in plugin_text
    assert "removed_mask" in plugin_text  # the diag keys are documented
    assert "test_my_variant" in written[1].read_text(encoding="utf-8")


def test_scaffold_output_is_valid_python(tmp_path):
    written = scaffold("tracker", "my_tracker", root=tmp_path)
    for path in written:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_scaffold_refuses_to_overwrite(tmp_path):
    scaffold("decomposition", "dup", root=tmp_path)
    with pytest.raises(FileExistsError):
        scaffold("decomposition", "dup", root=tmp_path)
    scaffold("decomposition", "dup", root=tmp_path, force=True)


def test_scaffold_rejects_a_bad_slot_or_name(tmp_path):
    with pytest.raises(ValueError, match="unknown slot"):
        scaffold("nope", "x", root=tmp_path)
    with pytest.raises(ValueError, match="identifier"):
        scaffold("decomposition", "not a name", root=tmp_path)


def test_scaffold_cli_runs(tmp_path):
    """The command line in Lab Spec Section 6 step 5."""
    result = subprocess.run(
        [sys.executable, "-m", "troika.plugins.new", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "slot" in result.stdout


# --------------------------------------------------------- notebook plug-ins


def test_builtin_detection(clean_slot):
    """A notebook-defined plug-in must be distinguishable from a packaged one."""
    assert registry.is_builtin("decomposition", "ssa")

    @registry.register_function(clean_slot, "from_a_cell", params={}, override=True)
    def from_a_cell(x, ctx):
        return x.copy()

    assert not registry.is_builtin(clean_slot, "from_a_cell")
