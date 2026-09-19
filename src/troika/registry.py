"""Plug-in registry for the five pipeline slots (Lab Spec Section 2.2).

A slot is a pipeline stage whose implementation is chosen by name in the config.
Built-in plug-ins live in ``src/troika/plugins/<slot>/<name>.py`` and are
auto-imported; notebook-defined plug-ins register at run time with the same API.
"""

from __future__ import annotations

import dataclasses
from dataclasses import make_dataclass
from typing import Any, Callable

__all__ = [
    "SLOTS",
    "register",
    "register_function",
    "get",
    "available",
    "params_class",
    "clear",
]

#: The five swappable stages, in pipeline order (Lab Spec Section 2).
SLOTS: tuple[str, ...] = (
    "bandpass",
    "decomposition",
    "temporal_diff",
    "spectrum_estimator",
    "tracker",
)

_REGISTRY: dict[str, dict[str, type]] = {slot: {} for slot in SLOTS}


def _check_slot(slot: str) -> None:
    if slot not in _REGISTRY:
        raise KeyError(f"unknown slot {slot!r}; known slots are {list(SLOTS)}")


def register(slot: str, name: str, *, override: bool = False) -> Callable[[type], type]:
    """Class decorator registering a plug-in class under ``slot``/``name``.

    Pass ``override=True`` to replace an existing registration, which is what a
    notebook cell needs when it is re-run after an edit.
    """
    _check_slot(slot)

    def _decorator(cls: type) -> type:
        if name in _REGISTRY[slot] and not override:
            raise KeyError(
                f"{slot!r} plug-in {name!r} is already registered; "
                f"pass override=True to replace it"
            )
        cls._slot = slot  # type: ignore[attr-defined]
        cls._name = name  # type: ignore[attr-defined]
        _REGISTRY[slot][name] = cls
        return cls

    return _decorator


def register_function(
    slot: str,
    name: str,
    params: dict[str, Any] | None = None,
    *,
    override: bool = False,
) -> Callable[[Callable], Callable]:
    """Register a plain function as a plug-in (Lab Spec Section 2.2).

    The function is called as ``fn(x, ctx, **params)`` and may return either a
    :class:`~troika.types.StageOutput` or a bare array, which is wrapped with an
    empty ``diag``. ``params`` gives the parameter names and their defaults; they
    become the generated ``Params`` dataclass used for config validation.
    """
    _check_slot(slot)
    param_defaults = dict(params or {})

    def _decorator(fn: Callable) -> Callable:
        from troika.types import StageOutput

        fields = [
            (key, type(value) if value is not None else Any, dataclasses.field(default=value))
            for key, value in param_defaults.items()
        ]
        params_cls = make_dataclass(f"{name.title()}Params", fields)

        class _FunctionPlugin:
            """Adapter wrapping a notebook-defined function as a slot plug-in."""

            Params = params_cls
            _wrapped = staticmethod(fn)

            def __init__(self, params: Any, static: Any) -> None:
                self.params = params
                self.static = static

            def __call__(self, x: Any, ctx: Any) -> StageOutput:
                kwargs = dataclasses.asdict(self.params)
                out = fn(x, ctx, **kwargs)
                if isinstance(out, StageOutput):
                    return out
                return StageOutput(data=out)

        _FunctionPlugin.__name__ = f"{name}_plugin"
        _FunctionPlugin.__qualname__ = _FunctionPlugin.__name__
        _FunctionPlugin.__doc__ = fn.__doc__ or _FunctionPlugin.__doc__
        register(slot, name, override=override)(_FunctionPlugin)
        return fn

    return _decorator


def get(slot: str, name: str) -> type:
    """Look up a registered plug-in class.

    Raises ``KeyError`` listing the available names when ``name`` is unknown.
    """
    _check_slot(slot)
    try:
        return _REGISTRY[slot][name]
    except KeyError:
        raise KeyError(
            f"unknown {slot!r} method {name!r}; available: {available(slot)}"
        ) from None


def available(slot: str) -> list[str]:
    """Names registered for ``slot``, sorted."""
    _check_slot(slot)
    return sorted(_REGISTRY[slot])


def params_class(slot: str, name: str) -> type:
    """The ``Params`` dataclass of a plug-in, used to validate config ``params``."""
    cls = get(slot, name)
    par = getattr(cls, "Params", None)
    if par is None:
        raise AttributeError(f"{slot}/{name} defines no Params dataclass")
    return par


def clear(slot: str | None = None) -> None:
    """Remove registrations. Test helper; not used by the pipeline."""
    if slot is None:
        for key in _REGISTRY:
            _REGISTRY[key].clear()
    else:
        _check_slot(slot)
        _REGISTRY[slot].clear()
