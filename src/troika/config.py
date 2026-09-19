"""Configuration: slot layout, merging, overrides, validation, hashing.

Exploration Lab Spec Section 3 (which supersedes Blueprint Section 6). Every
swappable stage is a *slot* with ``method`` plus ``params``; ``params`` are
validated against the chosen plug-in's ``Params`` dataclass, so a typo or an
out-of-range value fails at load time rather than mid-run.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from troika import registry

__all__ = [
    "Config",
    "ConfigError",
    "load_config",
    "default_config_path",
    "repo_root",
]


class ConfigError(ValueError):
    """Raised for an unknown key, a wrong type, or an out-of-range value."""


def repo_root() -> Path:
    """Repository root, resolved from this file's location."""
    return Path(__file__).resolve().parents[2]


def default_config_path() -> Path:
    """Path of ``configs/default.yaml``."""
    return repo_root() / "configs" / "default.yaml"


def _read_yaml(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level of a config must be a mapping")
    return data


def _deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``over`` onto ``base``; ``over`` wins on conflict."""
    out = dict(copy.deepcopy(dict(base)))
    for key, value in over.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


class _Node(Mapping):
    """Read-only dotted-attribute view over a nested config mapping."""

    __slots__ = ("_d", "_path")

    def __init__(self, d: Mapping[str, Any], path: str = "") -> None:
        object.__setattr__(self, "_d", d)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, item: str) -> Any:
        try:
            value = self._d[item]
        except KeyError:
            where = self._path or "config"
            raise AttributeError(f"{where} has no key {item!r}") from None
        if isinstance(value, dict):
            return _Node(value, f"{self._path}.{item}" if self._path else item)
        return value

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __iter__(self):
        return iter(self._d)

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return f"_Node({self._path or 'config'}: {sorted(self._d)})"

    def to_dict(self) -> dict[str, Any]:
        """A deep copy of the underlying mapping."""
        return copy.deepcopy(dict(self._d))


def _flatten(d: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(_flatten(value, dotted))
        else:
            out[dotted] = value
    return out


def _set_dotted(d: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = d
    for part in parts[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    node[parts[-1]] = value


def _get_dotted(d: Mapping[str, Any], dotted: str) -> Any:
    node: Any = d
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(dotted)
        node = node[part]
    return node


def _has_dotted(d: Mapping[str, Any], dotted: str) -> bool:
    try:
        _get_dotted(d, dotted)
    except KeyError:
        return False
    return True


def _parse_scalar(text: str) -> Any:
    """Parse a ``--set key=value`` right-hand side with YAML scalar rules."""
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return text


def _type_ok(value: Any, expected: Any) -> bool:
    if expected is Any or expected is None:
        return True
    if expected is float:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected is bool:
        return isinstance(value, bool)
    if expected is str:
        return isinstance(value, str)
    if expected in (list, tuple):
        return isinstance(value, (list, tuple))
    return True


def _coerce(value: Any, expected: Any) -> Any:
    if expected is float and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if expected in (list, tuple) and isinstance(value, tuple):
        return list(value)
    return value


class Config:
    """A fully resolved configuration (``default.yaml`` merged with a user file).

    Access is by dotted attribute, e.g. ``cfg.signal.fs`` or
    ``cfg.decomposition.params.L``. The object is treated as immutable;
    :meth:`with_overrides` returns a new instance.
    """

    def __init__(
        self,
        data: Mapping[str, Any],
        *,
        source: str | None = None,
        validate: bool = True,
    ) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self.source = source
        if validate:
            self.validate()

    # ---------------------------------------------------------------- access

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(item)
        try:
            value = self._data[item]
        except KeyError:
            raise AttributeError(f"config has no section {item!r}") from None
        return _Node(value, item) if isinstance(value, dict) else value

    def __getitem__(self, dotted: str) -> Any:
        return _get_dotted(self._data, dotted)

    def __contains__(self, dotted: str) -> bool:
        return _has_dotted(self._data, dotted)

    def __repr__(self) -> str:
        return (
            f"Config(name={self.get('run.name')!r}, "
            f"hash={self.hash()}, source={self.source!r})"
        )

    def get(self, dotted: str, default: Any = None) -> Any:
        """Value at a dotted path, or ``default`` when absent."""
        try:
            return _get_dotted(self._data, dotted)
        except KeyError:
            return default

    def to_dict(self) -> dict[str, Any]:
        """Deep copy of the resolved configuration."""
        return copy.deepcopy(self._data)

    # ------------------------------------------------------------ slot access

    def slot_method(self, slot: str) -> str:
        """Configured plug-in name for ``slot``."""
        return str(self._data[slot]["method"])

    def slot_params(self, slot: str) -> Any:
        """Configured ``params`` of ``slot`` as the plug-in's ``Params`` instance."""
        method = self.slot_method(slot)
        params_cls = registry.params_class(slot, method)
        raw = dict(self._data[slot].get("params") or {})
        return _build_params(slot, method, params_cls, raw)

    # ------------------------------------------------------------- overrides

    def with_overrides(self, overrides: Mapping[str, Any] | Iterable[str]) -> "Config":
        """Return a new config with dotted-path overrides applied.

        Accepts a mapping (``{"bandpass.method": "fir"}``) or an iterable of
        ``"key=value"`` strings as produced by ``--set``.

        Two rules from Lab Spec Section 3.1:

        * **Shorthand.** ``decomposition.L`` resolves to
          ``decomposition.params.L`` when the short form is not itself a key.
        * **Method switch resets params.** Setting ``<slot>.method`` replaces
          that slot's ``params`` with the new plug-in's defaults, after which any
          explicitly given params of the same call are applied.
        """
        if not isinstance(overrides, Mapping):
            pairs: dict[str, Any] = {}
            for item in overrides:
                if "=" not in item:
                    raise ConfigError(f"override {item!r} is not of the form key=value")
                key, _, text = item.partition("=")
                pairs[key.strip()] = _parse_scalar(text.strip())
            overrides = pairs

        data = copy.deepcopy(self._data)

        # Apply method switches first so the params reset cannot clobber params
        # given in the same call.
        method_keys = {k: v for k, v in overrides.items() if k.split(".")[-1] == "method"}
        other_keys = {k: v for k, v in overrides.items() if k not in method_keys}

        for key, value in method_keys.items():
            slot = key.rsplit(".", 1)[0]
            if slot not in registry.SLOTS:
                raise ConfigError(
                    f"override {key!r}: {slot!r} is not a slot; "
                    f"slots are {list(registry.SLOTS)}"
                )
            params_cls = registry.params_class(slot, str(value))
            data[slot] = {"method": str(value), "params": _params_defaults(params_cls)}

        for key, value in other_keys.items():
            resolved = self._resolve_key(data, key)
            _set_dotted(data, resolved, value)

        return Config(data, source=self.source)

    @staticmethod
    def _resolve_key(data: Mapping[str, Any], key: str) -> str:
        """Expand slot shorthand, e.g. ``decomposition.L`` to ``decomposition.params.L``."""
        if _has_dotted(data, key):
            return key
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] in registry.SLOTS:
            # Allow setting a params key the plug-in defines but the YAML omits;
            # validation rejects it later if the plug-in does not define it.
            return ".".join([parts[0], "params", *parts[1:]])
        raise ConfigError(f"unknown config key {key!r}")

    # ------------------------------------------------------------ comparison

    def diff_from_default(self) -> dict[str, Any]:
        """Nested dict holding only what differs from ``configs/default.yaml``.

        A slot whose ``method`` differs is emitted whole (method plus its full
        resolved ``params``), so the result reloads without relying on the old
        method's defaults.
        """
        base = _read_yaml(default_config_path())
        return _nested_diff(base, self._data)

    def hash(self, length: int = 12) -> str:
        """Stable short hash of the resolved config, for cache keys and folders."""
        blob = json.dumps(self._data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:length]

    # --------------------------------------------------------------- writing

    def to_yaml(self, path: str | Path | None = None, *, diff_only: bool = False) -> str:
        """Serialise the config; writes to ``path`` when given and returns the text."""
        payload = self.diff_from_default() if diff_only else self._data
        text = yaml.safe_dump(
            payload, sort_keys=False, default_flow_style=False, allow_unicode=True
        )
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------ validation

    def validate(self) -> None:
        """Check section names, slot methods and every slot's ``params``."""
        base = _read_yaml(default_config_path())

        unknown = set(self._data) - set(base)
        if unknown:
            raise ConfigError(
                f"unknown config section(s) {sorted(unknown)}; "
                f"known sections are {sorted(base)}"
            )

        for section, value in self._data.items():
            if section in registry.SLOTS:
                continue
            self._validate_plain_section(section, base.get(section), value)

        for slot in registry.SLOTS:
            if slot not in self._data:
                raise ConfigError(f"config is missing the {slot!r} slot")
            node = self._data[slot]
            if not isinstance(node, dict) or "method" not in node:
                raise ConfigError(f"slot {slot!r} must be a mapping with a 'method' key")
            method = str(node["method"])
            if not registry.available(slot):
                # Plug-ins are not imported yet (bootstrapping); skip params check.
                continue
            params_cls = registry.params_class(slot, method)
            _build_params(slot, method, params_cls, dict(node.get("params") or {}))

    @staticmethod
    def _validate_plain_section(section: str, base: Any, value: Any) -> None:
        if not isinstance(base, dict) or not isinstance(value, dict):
            return
        unknown = set(value) - set(base)
        if unknown:
            raise ConfigError(
                f"unknown key(s) {sorted(unknown)} in section {section!r}; "
                f"known keys are {sorted(base)}"
            )
        for key, sub in value.items():
            if isinstance(sub, dict):
                Config._validate_plain_section(f"{section}.{key}", base.get(key), sub)


def _params_defaults(params_cls: type) -> dict[str, Any]:
    """Default ``params`` mapping of a plug-in's ``Params`` dataclass."""
    out = dataclasses.asdict(params_cls())
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in out.items()}


def _build_params(slot: str, method: str, params_cls: type, raw: Mapping[str, Any]) -> Any:
    """Validate a raw ``params`` mapping and build the plug-in's ``Params``."""
    fields = {f.name: f for f in dataclasses.fields(params_cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(
            f"{slot}.params: unknown key(s) {sorted(unknown)} for method {method!r}; "
            f"known keys are {sorted(fields)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        expected = fields[name].type
        if isinstance(value, Mapping):
            kwargs[name] = dict(value)
            continue
        if not _type_ok(value, expected):
            raise ConfigError(
                f"{slot}.params.{name}: expected "
                f"{getattr(expected, '__name__', expected)}, "
                f"got {type(value).__name__} ({value!r})"
            )
        kwargs[name] = _coerce(value, expected)
    try:
        return params_cls(**kwargs)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{slot}.params ({method}): {exc}") from None


def _nested_diff(base: Mapping[str, Any], cur: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in cur.items():
        if key not in base:
            out[key] = copy.deepcopy(value)
            continue
        ref = base[key]
        if isinstance(value, dict) and isinstance(ref, dict):
            # A slot whose method changed is emitted whole; stale params of the
            # old method would otherwise leak into the diff.
            if "method" in value and "method" in ref and value["method"] != ref["method"]:
                out[key] = copy.deepcopy(value)
                continue
            sub = _nested_diff(ref, value)
            if sub:
                out[key] = sub
        elif value != ref:
            out[key] = copy.deepcopy(value)
    return out


def load_config(
    path: str | Path | None = None,
    *,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
) -> Config:
    """Load ``configs/default.yaml`` and merge ``path`` on top of it.

    Partial files are fine; anything omitted falls back to the default. Passing
    ``None`` returns the paper defaults. ``overrides`` are applied last, with the
    dotted-path rules of :meth:`Config.with_overrides`.
    """
    import troika.plugins  # noqa: F401  (importing registers the built-in plug-ins)

    base = _read_yaml(default_config_path())
    source = None
    if path is not None:
        source = str(path)
        user = _read_yaml(path)
        base = _deep_merge(base, user)
        # A user file that switches a slot's method must not inherit the old
        # method's params (Lab Spec Section 3.1).
        defaults = _read_yaml(default_config_path())
        for slot in registry.SLOTS:
            if not registry.available(slot):
                continue  # plug-ins not imported yet (bootstrapping)
            if slot not in user or "method" not in user[slot]:
                continue
            method = str(user[slot]["method"])
            if method == str(defaults[slot]["method"]):
                # Same method as the default, so default.yaml's params are the
                # right base and the merge above already did the work. Resetting
                # here would inject plug-in defaults for keys the file omits and
                # break the to_yaml round trip.
                continue
            defaults_for_method = _params_defaults(registry.params_class(slot, method))
            given = dict(user[slot].get("params") or {})
            base[slot] = {
                "method": method,
                "params": _deep_merge(defaults_for_method, given),
            }

    cfg = Config(base, source=source)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return cfg
