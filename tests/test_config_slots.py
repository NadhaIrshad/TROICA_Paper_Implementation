"""Slot-based configuration (Lab Spec Sections 3, 3.1 and 9)."""

from __future__ import annotations

import textwrap

import pytest
import yaml

import troika.plugins  # noqa: F401  (registers the built-ins)
from troika import registry
from troika.config import Config, ConfigError, default_config_path, load_config


#: Config validation and method switching need the slot plug-ins, which land in
#: later milestones. The guard disappears by itself once all five are registered.
_PLUGINS_READY = all(registry.available(slot) for slot in registry.SLOTS)
needs_plugins = pytest.mark.skipif(
    not _PLUGINS_READY, reason="slot plug-ins land in M2-M7"
)


def _write(tmp_path, text: str, name: str = "cfg.yaml"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text), encoding="utf-8")
    return path


def test_default_config_loads_with_paper_values():
    """Every [PAPER] parameter must match the paper exactly."""
    cfg = load_config()
    assert cfg.signal.fs == 125
    assert cfg.signal.window_s == 8
    assert cfg.signal.step_s == 2
    assert cfg.grid.n_fft == 4096
    assert cfg.bandpass.params.low_hz == 0.4
    assert cfg.bandpass.params.high_hz == 5.0
    assert cfg.decomposition.params.L == 400
    assert cfg.acc_dominant.rel_threshold == 0.5
    assert cfg.acc_dominant.exclude_delta_bins == 10
    assert cfg.temporal_diff.params.order == 2
    assert cfg.spectrum_estimator.params.p == 0.8
    assert cfg.spectrum_estimator.params.lam == 0.1
    assert cfg.spectrum_estimator.params.n_iter == 5
    assert cfg.tracker.params.delta_s == 16
    assert cfg.tracker.params.delta_s_wide == 20
    assert cfg.tracker.params.max_peaks_per_range == 3
    assert cfg.tracker.params.eta_frac == 0.30
    verification = cfg.tracker.params.verification
    assert verification.theta_bins == 6
    assert verification.tau_bins == 2
    assert verification.stall_windows_h == 3
    assert verification.trend_history_windows == 20
    assert verification.trend_poly_order == 3
    assert verification.trend_threshold_bpm == 3
    assert cfg.evaluation.std_ddof == 1


def test_default_config_has_the_deviation_paths():
    """DEVIATION: the specs omit data and results paths; default.yaml adds them."""
    cfg = load_config()
    assert cfg.data.dir
    assert cfg.results.dir == "results"


def test_default_diff_is_empty():
    assert load_config().diff_from_default() == {}


def test_partial_file_merges_over_default(tmp_path):
    """Lab Spec Section 3: partial files are fine; omitted keys fall back."""
    path = _write(
        tmp_path,
        """
        run:
          name: partial
        decomposition:
          params:
            L: 300
        """,
    )
    cfg = load_config(path)
    assert cfg.run.name == "partial"
    assert cfg.decomposition.params.L == 300
    assert cfg.decomposition.params.grouping == "frequency_pairing"  # untouched
    assert cfg.signal.fs == 125
    assert cfg.diff_from_default() == {
        "run": {"name": "partial"},
        "decomposition": {"params": {"L": 300}},
    }


def test_shipped_my_experiment_config_loads():
    """configs/my_experiment.yaml must stay loadable as the editable template."""
    cfg = load_config(default_config_path().with_name("my_experiment.yaml"))
    assert cfg.run.name == "my_experiment"
    assert cfg.signal.fs == 125


@pytest.mark.parametrize(
    "name", ["no_ssa.yaml", "fft_instead_of_ssr.yaml", "no_verification.yaml"]
)
@needs_plugins
def test_ablation_configs_load_and_change_one_thing(name):
    """Blueprint Section 8.2 ablations, in the new slot keys."""
    path = default_config_path().parent / "ablations" / name
    cfg = load_config(path)
    diff = cfg.diff_from_default()
    diff.pop("run", None)
    assert len(diff) == 1, f"{name} should change exactly one stage, changed {sorted(diff)}"


def test_unknown_section_raises():
    with pytest.raises(ConfigError, match="unknown config section"):
        Config({**load_config().to_dict(), "nonsense": {}})


def test_unknown_key_in_plain_section_raises(tmp_path):
    path = _write(tmp_path, "signal:\n  smpling_rate: 125\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


@needs_plugins
def test_unknown_param_key_raises(tmp_path):
    path = _write(tmp_path, "decomposition:\n  params:\n    LL: 300\n")
    with pytest.raises(ConfigError) as excinfo:
        load_config(path)
    assert "LL" in str(excinfo.value)


@needs_plugins
def test_wrong_param_type_raises(tmp_path):
    path = _write(tmp_path, "decomposition:\n  params:\n    L: not-a-number\n")
    with pytest.raises(ConfigError, match="expected int"):
        load_config(path)


@needs_plugins
def test_unknown_method_raises_listing_options(tmp_path):
    path = _write(tmp_path, "spectrum_estimator:\n  method: wavelet\n")
    with pytest.raises(KeyError) as excinfo:
        load_config(path)
    assert "available:" in str(excinfo.value)


@needs_plugins
def test_out_of_range_param_raises(tmp_path):
    """Params dataclasses validate ranges in __post_init__."""
    path = _write(tmp_path, "spectrum_estimator:\n  params:\n    p: 5.0\n")
    with pytest.raises(ConfigError):
        load_config(path)


def test_override_with_mapping():
    cfg = load_config().with_overrides({"signal.ppg_channel": 1, "run.n_jobs": 1})
    assert cfg.signal.ppg_channel == 1
    assert cfg.run.n_jobs == 1


def test_override_shorthand_resolves_into_params():
    """Lab Spec Section 3.1: decomposition.L means decomposition.params.L."""
    cfg = load_config().with_overrides({"decomposition.L": 300})
    assert cfg.decomposition.params.L == 300


@needs_plugins
def test_override_cli_string_form():
    """The --set form used by experiments/run_all.py."""
    cfg = load_config().with_overrides(["bandpass.method=fir", "decomposition.L=300"])
    assert cfg.bandpass.method == "fir"
    assert cfg.decomposition.params.L == 300


@needs_plugins
def test_changing_method_resets_params():
    """Lab Spec Section 3.1: a method switch drops the old method's params."""
    cfg = load_config().with_overrides({"decomposition.L": 300})
    switched = cfg.with_overrides({"decomposition.method": "none"})
    assert switched.decomposition.method == "none"
    assert "L" not in dict(switched.decomposition.params)


@needs_plugins
def test_method_switch_and_params_in_one_call():
    """Params given alongside a method switch survive the reset."""
    cfg = load_config().with_overrides(
        {"bandpass.method": "fir", "bandpass.order": 101}
    )
    assert cfg.bandpass.method == "fir"
    assert cfg.bandpass.params.order == 101


@needs_plugins
def test_method_switch_in_a_file_resets_params(tmp_path):
    path = _write(tmp_path, "decomposition:\n  method: none\n")
    cfg = load_config(path)
    assert cfg.decomposition.method == "none"
    assert "L" not in dict(cfg.decomposition.params)


def test_unknown_override_key_raises():
    with pytest.raises(ConfigError, match="unknown config key"):
        load_config().with_overrides({"nonsense.key": 1})


@needs_plugins
def test_round_trip_diff_to_yaml_and_reload(tmp_path):
    """diff_from_default -> to_yaml -> reload reproduces the same config."""
    cfg = load_config().with_overrides(
        {"decomposition.L": 200, "bandpass.method": "fir", "run.name": "rt"}
    )
    path = tmp_path / "promoted.yaml"
    cfg.to_yaml(path, diff_only=True)
    reloaded = load_config(path)
    assert reloaded.to_dict() == cfg.to_dict()
    assert reloaded.hash() == cfg.hash()


def test_full_to_yaml_round_trip(tmp_path):
    cfg = load_config().with_overrides({"decomposition.L": 250})
    path = tmp_path / "full.yaml"
    cfg.to_yaml(path)
    assert yaml.safe_load(path.read_text(encoding="utf-8")) == cfg.to_dict()
    assert load_config(path).hash() == cfg.hash()


def test_hash_is_stable_and_sensitive():
    a = load_config()
    b = load_config()
    assert a.hash() == b.hash()
    assert a.with_overrides({"decomposition.L": 399}).hash() != a.hash()


@needs_plugins
def test_slot_params_builds_the_plugin_params_dataclass():
    cfg = load_config()
    for slot in registry.SLOTS:
        params = cfg.slot_params(slot)
        assert isinstance(params, registry.params_class(slot, cfg.slot_method(slot)))


def test_config_is_not_mutated_by_overrides():
    cfg = load_config()
    before = cfg.to_dict()
    cfg.with_overrides({"decomposition.L": 111})
    assert cfg.to_dict() == before
