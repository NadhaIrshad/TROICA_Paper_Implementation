"""Built-in slot plug-ins, auto-imported so the registry is populated on import.

Every module under ``plugins/<slot>/`` is imported; there is no manual list, so
dropping a new file into a slot folder is enough to register it
(Lab Spec Section 2.2).
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

from troika.plugins import base  # noqa: F401  (re-exported for plug-in authors)

__all__ = ["base", "load_builtin_plugins"]


def load_builtin_plugins() -> list[str]:
    """Import every built-in plug-in module and return the imported module names."""
    here = Path(__file__).parent
    imported: list[str] = []
    for slot_dir in sorted(p for p in here.iterdir() if p.is_dir() and not p.name.startswith("_")):
        pkg = f"{__name__}.{slot_dir.name}"
        try:
            slot_pkg = importlib.import_module(pkg)
        except ModuleNotFoundError:
            continue
        for mod in pkgutil.iter_modules(slot_pkg.__path__):
            if mod.name.startswith("_"):
                continue
            importlib.import_module(f"{pkg}.{mod.name}")
            imported.append(f"{pkg}.{mod.name}")
    return imported


load_builtin_plugins()
