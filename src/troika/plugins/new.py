"""Scaffold a new plug-in file and its test (Lab Spec Section 6 step 5).

    python -m troika.plugins.new decomposition my_variant

Writes ``src/troika/plugins/<slot>/<name>.py`` and
``tests/plugins/test_<name>.py``, then prints what to do next. Paste the code
from the notebook cell into the generated ``__call__``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from troika import registry
from troika.config import repo_root
from troika.testing import REQUIRED_DIAG

__all__ = ["scaffold", "main"]

_BASE_CLASS = {
    "bandpass": "Bandpass",
    "decomposition": "Decomposition",
    "temporal_diff": "TemporalDiff",
    "spectrum_estimator": "SpectrumEstimator",
    "tracker": "Tracker",
}

_SLOT_DIR = {
    "bandpass": "bandpass",
    "decomposition": "decomposition",
    "temporal_diff": "temporal_diff",
    "spectrum_estimator": "spectrum",
    "tracker": "tracker",
}

_PLUGIN_TEMPLATE = '''"""{title} plug-in for the {slot} slot.

Fill in :meth:`{cls}.__call__`. The stage contract, including the shapes in and
out, is in ``troika/plugins/base.py``; the ``diag`` keys the plots rely on are
Lab Spec Section 2.3 and are listed below.

Required diag keys: {diag}
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from troika.plugins.base import {base}
from troika.registry import register
from troika.types import StageOutput, StaticContext, WindowContext

__all__ = ["{cls}"]


@dataclass
class {cls}Params:
    """Options of this plug-in. Every field becomes a config key."""

    # example: n_modes: int = 6
    pass

    def __post_init__(self) -> None:
        """Validate ranges here so a bad config fails at load time."""


@register("{slot}", "{name}")
class {cls}({base}):
    """One sentence saying what this does and why.

    Cite the paper section or the reference it implements.
    """

    Params = {cls}Params

    def __init__(self, params: {cls}Params, static: StaticContext) -> None:
        super().__init__(params, static)
        # Precompute anything that does not change between windows.

    def __call__(self, x: NDArray[np.float64], ctx: WindowContext) -> StageOutput:
        """Transform ``x`` for one window.

        ``ctx.acc_bins`` holds the refined accelerometer dominant bins, and
        ``ctx.prev_bin`` the previous estimate. Never read ``ctx.gt_bpm``.
        """
        raise NotImplementedError("paste the notebook implementation here")
'''

_TEST_TEMPLATE = '''"""Contract check for the {slot}/{name} plug-in."""

from __future__ import annotations

import troika.plugins  # noqa: F401  (registers the built-ins and this plug-in)
from troika.testing import check_plugin


def test_{name}_obeys_the_slot_contract():
    report = check_plugin("{slot}", "{name}")
    assert report.ok, str(report)
'''


def scaffold(slot: str, name: str, *, root: Path | None = None, force: bool = False) -> list[Path]:
    """Write the plug-in and test files, returning the paths written."""
    if slot not in registry.SLOTS:
        raise ValueError(f"unknown slot {slot!r}; slots are {list(registry.SLOTS)}")
    if not name.isidentifier():
        raise ValueError(f"{name!r} is not a valid Python identifier")

    root = root or repo_root()
    class_name = "".join(part.capitalize() for part in name.split("_"))
    plugin_path = root / "src" / "troika" / "plugins" / _SLOT_DIR[slot] / f"{name}.py"
    test_path = root / "tests" / "plugins" / f"test_{name}.py"

    for path in (plugin_path, test_path):
        if path.exists() and not force:
            raise FileExistsError(f"{path} already exists; pass --force to overwrite")

    plugin_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    plugin_path.write_text(
        _PLUGIN_TEMPLATE.format(
            title=name.replace("_", " ").capitalize(),
            slot=slot,
            name=name,
            cls=class_name,
            base=_BASE_CLASS[slot],
            diag=", ".join(REQUIRED_DIAG.get(slot, ())) or "none",
        ),
        encoding="utf-8",
    )
    test_path.write_text(
        _TEST_TEMPLATE.format(slot=slot, name=name), encoding="utf-8"
    )
    return [plugin_path, test_path]


def main(argv: list[str] | None = None) -> int:
    """Command line for the scaffold generator."""
    parser = argparse.ArgumentParser(
        prog="python -m troika.plugins.new",
        description="Create a new slot plug-in and its contract test.",
    )
    parser.add_argument("slot", choices=list(registry.SLOTS))
    parser.add_argument("name", help="plug-in name, a valid Python identifier")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args(argv)

    written = scaffold(args.slot, args.name, force=args.force)
    print("wrote:")
    for path in written:
        print(f"  {path}")
    print(
        f"\nnext: implement __call__, then try it with\n"
        f"  cfg2 = cfg.with_overrides({{'{args.slot}.method': '{args.name}'}})\n"
        f"  lab.run_all(cfg2, subjects)\n"
        f"and check the contract with\n"
        f"  python -m pytest tests/plugins/test_{args.name}.py"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
