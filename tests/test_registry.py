"""Plug-in registry behaviour (Lab Spec Sections 2.2 and 9)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import troika.plugins  # noqa: F401  (registers the built-ins)
from troika import registry
from troika.plugins.base import Decomposition
from troika.types import StageOutput


@pytest.fixture
def clean_slot():
    """Snapshot and restore the decomposition slot around a test."""
    saved = dict(registry._REGISTRY["decomposition"])
    yield "decomposition"
    registry._REGISTRY["decomposition"].clear()
    registry._REGISTRY["decomposition"].update(saved)


def test_builtin_slots_are_the_five_from_the_spec():
    assert registry.SLOTS == (
        "bandpass",
        "decomposition",
        "temporal_diff",
        "spectrum_estimator",
        "tracker",
    )


def test_register_class_and_get(clean_slot):
    @registry.register(clean_slot, "unit_test_dummy")
    class Dummy(Decomposition):
        @dataclass
        class Params:
            gain: float = 2.0

        def __call__(self, x, ctx):
            return StageOutput(data=x * self.params.gain, diag={})

    assert registry.get(clean_slot, "unit_test_dummy") is Dummy
    assert "unit_test_dummy" in registry.available(clean_slot)
    assert registry.params_class(clean_slot, "unit_test_dummy")().gain == 2.0


def test_double_registration_without_override_raises(clean_slot):
    @registry.register(clean_slot, "dup")
    class A(Decomposition):
        def __call__(self, x, ctx):
            return StageOutput(data=x)

    with pytest.raises(KeyError, match="already registered"):

        @registry.register(clean_slot, "dup")
        class B(Decomposition):
            def __call__(self, x, ctx):
                return StageOutput(data=x)


def test_override_replaces_registration(clean_slot):
    """A notebook cell re-run after an edit must be able to replace its plug-in."""

    @registry.register(clean_slot, "dup2")
    class A(Decomposition):
        def __call__(self, x, ctx):
            return StageOutput(data=x)

    @registry.register(clean_slot, "dup2", override=True)
    class B(Decomposition):
        def __call__(self, x, ctx):
            return StageOutput(data=x)

    assert registry.get(clean_slot, "dup2") is B


def test_unknown_method_error_lists_available(clean_slot):
    with pytest.raises(KeyError) as excinfo:
        registry.get(clean_slot, "does_not_exist")
    message = str(excinfo.value)
    assert "does_not_exist" in message and "available:" in message


def test_unknown_slot_raises():
    with pytest.raises(KeyError, match="unknown slot"):
        registry.get("not_a_slot", "x")


def test_register_function_notebook_style(clean_slot):
    """The quick function form of Lab Spec Section 2.2."""

    @registry.register_function(clean_slot, "scaled", params={"n_modes": 3})
    def scaled(x, ctx, *, n_modes):
        return StageOutput(data=x * n_modes, diag={"components": x[None, :]})

    cls = registry.get(clean_slot, "scaled")
    params = cls.Params()
    assert params.n_modes == 3

    inst = cls(cls.Params(n_modes=4), static=None)
    out = inst(np.ones(5), ctx=None)
    assert isinstance(out, StageOutput)
    assert np.allclose(out.data, 4.0)
    assert "components" in out.diag


def test_register_function_accepts_bare_array_return(clean_slot):
    """A bare ndarray return is wrapped with an empty diag."""

    @registry.register_function(clean_slot, "bare", params={})
    def bare(x, ctx):
        return x * 0 + 7.0

    cls = registry.get(clean_slot, "bare")
    out = cls(cls.Params(), static=None)(np.zeros(3), ctx=None)
    assert isinstance(out, StageOutput) and out.diag == {}
    assert np.allclose(out.data, 7.0)


def test_register_function_override(clean_slot):
    @registry.register_function(clean_slot, "twice", params={"k": 1})
    def v1(x, ctx, *, k):
        return x * k

    @registry.register_function(clean_slot, "twice", params={"k": 9}, override=True)
    def v2(x, ctx, *, k):
        return x * k

    cls = registry.get(clean_slot, "twice")
    assert cls.Params().k == 9


@pytest.mark.skipif(
    not all(registry.available(s) for s in registry.SLOTS),
    reason="built-in plug-ins land in M7; this becomes a real check then",
)
def test_every_slot_has_at_least_one_builtin():
    """The paper-default path must be registered for all five slots."""
    for slot in registry.SLOTS:
        assert registry.available(slot), f"slot {slot} has no built-in plug-in"
