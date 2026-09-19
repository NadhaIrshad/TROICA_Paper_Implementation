"""Shared fixtures. Real-data tests skip when the dataset is absent."""

from __future__ import annotations

import numpy as np
import pytest

from troika.config import load_config
from troika.io import loader


@pytest.fixture(scope="session")
def cfg():
    """The paper-default configuration."""
    return load_config()


@pytest.fixture(scope="session")
def subject_paths(cfg):
    """``(subject_id, path)`` pairs of the training split."""
    return loader.list_subjects(cfg)


@pytest.fixture(scope="session")
def subject5(cfg):
    """Subject 5, the paper's illustrative good case (Figs. 8 and 10)."""
    subjects = dict(loader.list_subjects(cfg))
    return loader.load_recording(subjects[5], 5, cfg)


@pytest.fixture
def rng():
    """Seeded generator; all synthetic data in the suite is reproducible."""
    return np.random.default_rng(20150203)
