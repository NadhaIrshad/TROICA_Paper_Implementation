"""Test helpers shared across modules (kept out of conftest so it can be imported)."""

from __future__ import annotations

import pytest

from troika.config import load_config
from troika.io import loader


def dataset_available() -> bool:
    """True when the real recordings are present under ``data.dir``."""
    try:
        loader.list_subjects(load_config())
    except Exception:
        return False
    return True


#: Skip marker for tests that need the real recordings (Blueprint Section 9).
needs_data = pytest.mark.skipif(
    not dataset_available(), reason="dataset not present under data.dir"
)
