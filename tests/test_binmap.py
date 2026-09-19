"""Bin <-> Hz <-> BPM conversions (Blueprint Sections 3 and 9)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap

FS = 125.0
N = 4096


def test_bin_width_matches_paper_grid():
    """Blueprint Section 3: df = 0.0305 Hz = 1.831 BPM for fs=125, N=4096."""
    assert binmap.bin_width_hz(FS, N) == pytest.approx(0.030517578125, rel=1e-12)
    assert binmap.bin_width_bpm(FS, N) == pytest.approx(1.8310546875, rel=1e-12)


def test_remark3_grid_step_is_not_one_bpm():
    """Remark 3 of the paper claims ~1 BPM per grid step; it is really 1.83."""
    assert binmap.bin_width_bpm(FS, N) > 1.8


def test_theta_six_bins_is_about_eleven_bpm():
    """Paper Section III-D.3: a change of 6 locations means about 11 BPM."""
    assert binmap.bin_to_bpm(6, FS, N) == pytest.approx(10.986, abs=0.01)


def test_delta_ten_bins_is_about_point_three_hz():
    """Paper Section III-A: Delta = 10 corresponds to about 0.3 Hz."""
    assert binmap.bin_to_hz(10, FS, N) == pytest.approx(0.305, abs=0.01)


def test_delta_s_and_tau_in_bpm():
    """Sanity numbers from Blueprint Section 3."""
    assert binmap.bin_to_bpm(2, FS, N) == pytest.approx(3.66, abs=0.01)
    assert binmap.bin_to_bpm(16, FS, N) == pytest.approx(29.30, abs=0.01)


def test_bin_zero_is_dc():
    """Paper Eq. 11: the first frequency bin corresponds to 0 Hz."""
    assert binmap.bin_to_hz(0, FS, N) == 0.0
    assert binmap.bin_to_bpm(0, FS, N) == 0.0


@pytest.mark.parametrize("k", [1, 17, 100, 229, 2048])
def test_bin_bpm_round_trip(k):
    """bpm_to_bin inverts bin_to_bpm exactly on grid points."""
    assert binmap.bpm_to_bin(binmap.bin_to_bpm(k, FS, N), FS, N) == k


@pytest.mark.parametrize("k", [1, 17, 100, 229])
def test_bin_hz_round_trip(k):
    """hz_to_bin inverts bin_to_hz exactly on grid points."""
    assert binmap.hz_to_bin(binmap.bin_to_hz(k, FS, N), FS, N) == k


def test_vectorised_calls_return_arrays():
    """Array input gives array output; scalar input gives a Python float/int."""
    k = np.array([0, 10, 100])
    out = binmap.bin_to_bpm(k, FS, N)
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    assert isinstance(binmap.bin_to_bpm(10, FS, N), float)
    assert isinstance(binmap.bpm_to_bin(60.0, FS, N), int)


def test_band_to_bins_is_strictly_inside_the_band():
    """band_to_bins rounds the low edge up and the high edge down."""
    lo, hi = binmap.band_to_bins(0.4, 5.0, FS, N)
    assert (lo, hi) == (14, 163)
    assert binmap.bin_to_hz(lo, FS, N) >= 0.4
    assert binmap.bin_to_hz(hi, FS, N) <= 5.0
    assert binmap.bin_to_hz(lo - 1, FS, N) < 0.4
    assert binmap.bin_to_hz(hi + 1, FS, N) > 5.0


def test_band_to_bins_clips_to_one_sided_spectrum():
    """Edges beyond Nyquist clip to N // 2."""
    lo, hi = binmap.band_to_bins(0.0, 1000.0, FS, N)
    assert lo == 0 and hi == N // 2


def test_hr_band_bins():
    """The 40-200 BPM initialisation band (A10) maps to a sane bin range."""
    lo = binmap.bpm_to_bin(40, FS, N)
    hi = binmap.bpm_to_bin(200, FS, N)
    assert 20 < lo < 24 and 107 < hi < 111
    assert binmap.bin_to_bpm(lo, FS, N) == pytest.approx(40, abs=1.0)
    assert binmap.bin_to_bpm(hi, FS, N) == pytest.approx(200, abs=1.0)


def test_scaling_fs_keeps_physical_meaning():
    """Blueprint Section 6 and paper Section III-E.

    At fs = 25 Hz the paper takes N to about 1/5 of 4096 so that Delta_s, Delta,
    theta and tau keep the same physical meaning. 4096 / 5 is not an integer, so
    the match is close rather than exact.
    """
    assert binmap.bin_to_bpm(16, 25.0, N // 5) == pytest.approx(
        binmap.bin_to_bpm(16, FS, N), rel=1e-3
    )
