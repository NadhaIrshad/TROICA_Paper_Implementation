"""Spectral-subtraction spectrum estimation (DEVIATIONS.md D14)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.plugins.spectrum.spectral_subtraction import (
    SpectralSubtractionParams,
    SpectralSubtractionSpectrum,
)
from troika.preprocessing.spectral_subtraction import (
    amplitude_spectrum,
    bpm_prior,
    harmonic_sum,
    spectral_subtraction,
)
from troika.testing import check_plugin
from troika.types import StaticContext, WindowContext

FS = 125.0
N = 4096
M = 1000
T = np.arange(M) / FS
PULSE_HZ, SWING_HZ = 2.0, 1.3
PULSE_BIN = int(binmap.hz_to_bin(PULSE_HZ, FS, N))
SWING_BIN = int(binmap.hz_to_bin(SWING_HZ, FS, N))
NO_HARMONICS = [1.0, 0.0, 0.0]

SWING = np.sin(2 * np.pi * SWING_HZ * T)
PPG = np.sin(2 * np.pi * PULSE_HZ * T) + 3.0 * SWING  # the artifact dominates
ACC = np.stack([SWING, 0.6 * SWING, 0.3 * SWING])


def _plugin(**params) -> SpectralSubtractionSpectrum:
    static = StaticContext(fs=FS, N=N, M=M, window_s=8.0, step_s=2.0)
    return SpectralSubtractionSpectrum(SpectralSubtractionParams(**params), static)


def _ctx(acc) -> WindowContext:
    return WindowContext(idx=1, t_start_s=2.0, acc=acc)


# ----------------------------------------------------------------------- core


def test_amplitude_spectrum_runs_along_the_last_axis():
    assert amplitude_spectrum(PPG, N).shape == (N // 2 + 1,)
    stacked = amplitude_spectrum(ACC, N)
    assert stacked.shape == (3, N // 2 + 1)
    assert stacked[0] == pytest.approx(amplitude_spectrum(SWING, N))


def test_subtraction_scales_each_spectrum_by_its_sum_and_clips():
    ppg = np.array([2.0, 2.0, 0.0, 0.0])
    acc = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    assert spectral_subtraction(ppg, acc).tolist() == [0.5, 0.0, 0.0, 0.0]
    assert spectral_subtraction(ppg, acc.sum(axis=0)).tolist() == [0.5, 0.0, 0.0, 0.0]


def test_subtraction_without_a_usable_accelerometer_only_normalises():
    ppg = np.array([1.0, 3.0, 0.0, 0.0])
    assert spectral_subtraction(ppg, None).tolist() == [0.25, 0.75, 0.0, 0.0]
    assert spectral_subtraction(ppg, np.zeros((3, 4))).tolist() == [0.25, 0.75, 0.0, 0.0]
    assert spectral_subtraction(np.zeros(4), np.ones((3, 4))).tolist() == [0.0] * 4


def test_harmonic_sum_adds_the_second_and_third_harmonics():
    y = np.zeros(10)
    y[[2, 4, 6]] = 1.0
    out = harmonic_sum(y, [1.0, 0.66, 0.33])
    assert out[2] == pytest.approx(1.99)  # itself, bin 4 and bin 6
    assert out[1] == pytest.approx(0.66)  # bin 2 as its second harmonic
    assert out[3] == pytest.approx(0.66)  # bin 6 as its second harmonic
    assert out[4] == pytest.approx(1.0)  # bins 8 and 12 hold nothing
    assert out[6] == pytest.approx(1.0)  # bins 12 and 18 are past the end
    assert harmonic_sum(y, NO_HARMONICS) == pytest.approx(y)


def test_bpm_prior_is_a_positive_bump_in_the_exercise_range():
    prior = bpm_prior(N // 2 + 1, FS, N, 130.0, 30.0, 0.6)
    assert np.isfinite(prior).all() and (prior >= 0).all()
    assert 120.0 < float(binmap.bin_to_bpm(int(np.argmax(prior)), FS, N)) < 160.0


# -------------------------------------------------------------------- plug-in


@pytest.mark.parametrize(
    "bad",
    [
        {"harmonic_weights": []}, {"harmonic_weights": [0.0, 1.0]},
        {"harmonic_weights": [1.0, -0.5]}, {"prior_scale_bpm": 0.0},
    ],
)
def test_plugin_params_validate(bad):
    with pytest.raises(ValueError):
        SpectralSubtractionParams(**bad)


def test_plugin_removes_the_artifact_the_accelerometer_carries():
    plugin = _plugin(harmonic_weights=NO_HARMONICS)
    without = plugin(PPG, _ctx(None)).data
    assert abs(int(np.argmax(without)) - SWING_BIN) <= 3
    cleaned = plugin(PPG, _ctx(ACC))
    assert cleaned.data.shape == (N // 2 + 1,)
    assert (cleaned.data >= 0).all() and np.isfinite(cleaned.data).all()
    assert abs(int(np.argmax(cleaned.data)) - PULSE_BIN) <= 3
    assert cleaned.diag["fallback"] is None


def test_plugin_falls_back_when_the_subtraction_leaves_nothing():
    acc = np.stack([PPG, np.zeros(M), np.zeros(M)])  # the accelerometer equals the PPG
    out = _plugin(harmonic_weights=NO_HARMONICS)(PPG, _ctx(acc))
    assert out.diag["fallback"] == "all_subtracted"
    assert out.data == pytest.approx(spectral_subtraction(amplitude_spectrum(PPG, N), None))


def test_plugin_options_change_the_spectrum():
    plain = _plugin(harmonic_weights=NO_HARMONICS)(PPG, _ctx(ACC)).data
    summed = _plugin()(PPG, _ctx(ACC)).data
    assert summed[PULSE_BIN // 2] > plain[PULSE_BIN // 2]  # the pulse shows up at half its rate
    assert _plugin(harmonic_weights=NO_HARMONICS, output_power=True)(PPG, _ctx(ACC)).data == pytest.approx(plain**2)
    weighted = _plugin(harmonic_weights=NO_HARMONICS, bpm_prior=True)(PPG, _ctx(ACC)).data
    assert np.isfinite(weighted).all() and not np.allclose(weighted, plain)


def test_plugin_passes_the_slot_contract():
    report = check_plugin("spectrum_estimator", "spectral_subtraction")
    assert report.ok, str(report)
