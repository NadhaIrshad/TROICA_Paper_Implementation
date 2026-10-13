"""EEMD decomposition (DEVIATIONS.md D13)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.decomposition.eemd import default_n_imfs, eemd, emd, pulsatile_mask
from troika.plugins.decomposition.eemd import EemdDecomposition, EemdParams
from troika.preprocessing.spectrum import periodogram
from troika.testing import check_plugin
from troika.types import StaticContext, WindowContext

FS = 125.0
N = 4096
M = 1000
T = np.arange(M) / FS
TOL_BINS = 8  # about 0.25 Hz


def _tone(hz: float, amplitude: float = 1.0) -> np.ndarray:
    return amplitude * np.sin(2 * np.pi * hz * T)


def _peak_bin(x: np.ndarray) -> int:
    return int(np.argmax(periodogram(x, N)))


def _plugin(**params) -> EemdDecomposition:
    static = StaticContext(fs=FS, N=N, M=M, window_s=8.0, step_s=2.0)
    return EemdDecomposition(EemdParams(**params), static)


CTX = WindowContext(idx=1, t_start_s=2.0)


# ------------------------------------------------------------------------ emd


def test_default_mode_count_follows_wu_and_huang():
    assert default_n_imfs(1000) == 8
    assert default_n_imfs(1024) == 9


def test_emd_rows_sum_to_the_signal():
    x = _tone(8.0) + _tone(1.0) + 0.3 * T
    modes = emd(x, n_imfs=8)
    assert modes.shape == (9, M)
    assert modes.sum(axis=0) == pytest.approx(x, abs=1e-9)


def test_emd_separates_a_fast_tone_from_a_slow_one():
    modes = emd(_tone(8.0) + _tone(1.0), n_imfs=8)
    assert abs(_peak_bin(modes[0]) - int(binmap.hz_to_bin(8.0, FS, N))) <= TOL_BINS
    rest = modes[1:]
    strongest = rest[int(np.argmax((rest**2).sum(axis=1)))]
    assert abs(_peak_bin(strongest) - int(binmap.hz_to_bin(1.0, FS, N))) <= TOL_BINS


def test_emd_of_a_signal_without_extrema_is_all_residual():
    ramp = 0.01 * T
    modes = emd(ramp, n_imfs=4)
    assert not modes[:-1].any()
    assert modes[-1] == pytest.approx(ramp)


# ----------------------------------------------------------------------- eemd


def test_eemd_is_reproducible_for_a_seed_and_changes_with_it():
    x = _tone(2.0) + _tone(9.0, 0.5)
    first = eemd(x, ensemble_size=6, seed=3)
    assert np.array_equal(first, eemd(x, ensemble_size=6, seed=3))
    assert not np.allclose(first, eemd(x, ensemble_size=6, seed=4))


def test_eemd_rows_sum_to_the_signal_up_to_the_averaged_noise():
    x = _tone(2.0) + _tone(9.0, 0.5)
    modes = eemd(x, ensemble_size=8, noise_std=0.2)
    assert modes.shape == (default_n_imfs(M) + 1, M)
    # The leftover is the mean of 8 noise draws of 0.2 standard deviations.
    assert np.mean(np.abs(modes.sum(axis=0) - x)) < 0.15 * np.std(x)


def test_eemd_without_noise_is_plain_emd_and_a_flat_window_stays_finite():
    x = _tone(2.0) + _tone(9.0, 0.5)
    assert np.array_equal(eemd(x, noise_std=0.0, n_imfs=5), emd(x, 5))
    flat = eemd(np.zeros(M), ensemble_size=4)
    assert np.isfinite(flat).all() and not flat.any()


# ------------------------------------------------------------- mode selection


def test_only_modes_peaking_in_the_band_are_kept():
    modes = np.stack([_tone(2.0), _tone(6.0), _tone(0.2), np.zeros(M)])
    keep, dominant = pulsatile_mask(modes, FS, N, 0.5, 4.0)
    assert keep.tolist() == [True, False, False, False]
    assert abs(int(dominant[0]) - int(binmap.hz_to_bin(2.0, FS, N))) <= 2


# -------------------------------------------------------------------- plug-in


@pytest.mark.parametrize(
    "bad",
    [
        {"low_hz": 4.0, "high_hz": 0.5}, {"low_hz": -1.0}, {"ensemble_size": 0},
        {"noise_std": -0.1}, {"n_sifts": 0}, {"n_imfs": -1},
    ],
)
def test_plugin_params_validate(bad):
    with pytest.raises(ValueError):
        EemdParams(**bad)


def test_plugin_output_is_the_sum_of_the_kept_modes():
    x = _tone(1.5) + _tone(10.0, 2.0)
    out = _plugin(ensemble_size=8)(x, CTX)
    components, removed = out.diag["components"], out.diag["removed_mask"]
    assert out.data.shape == x.shape
    assert components.shape == (default_n_imfs(M) + 1, M)
    assert len(out.diag["component_labels"]) == removed.size == components.shape[0]
    assert out.data == pytest.approx(components[~removed].sum(axis=0))


def test_plugin_removes_a_tone_outside_the_band():
    x = _tone(1.5) + _tone(10.0, 2.0)
    assert abs(_peak_bin(x) - int(binmap.hz_to_bin(10.0, FS, N))) <= TOL_BINS
    out = _plugin(ensemble_size=8)(x, CTX)
    assert out.diag["removed_mask"].any()
    assert abs(_peak_bin(out.data) - int(binmap.hz_to_bin(1.5, FS, N))) <= TOL_BINS


def test_plugin_is_deterministic_and_leaves_its_input_alone():
    x = _tone(1.5) + _tone(10.0, 2.0)
    before = x.copy()
    first = _plugin(ensemble_size=4)(x, CTX).data
    assert np.array_equal(first, _plugin(ensemble_size=4)(x, CTX).data)
    assert np.array_equal(x, before)


def test_plugin_passes_the_slot_contract():
    report = check_plugin("decomposition", "eemd", params={"ensemble_size": 4})
    assert report.ok, str(report)
