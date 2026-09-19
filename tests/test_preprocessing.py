"""Band-pass, periodogram, peaks and temporal difference (Blueprint Sections 5.1-5.4)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.preprocessing import bandpass, spectrum, temporal_diff

FS = 125.0
N = 4096
M = 1000


def _tone(f_hz, n=M, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f_hz * t + phase)


# ------------------------------------------------------------------- band-pass


@pytest.mark.parametrize("design,order", [("butter", 4), ("fir", 301)])
def test_passband_gain_is_about_one(design, order):
    """Blueprint Section 9: gain ~1 inside the passband."""
    filt = bandpass.make_bandpass(FS, 0.4, 5.0, design, order)
    for f in (1.5, 2.0, 3.0):
        x = _tone(f)
        y = bandpass.apply_bandpass(x, filt)
        gain = np.std(y[100:-100]) / np.std(x[100:-100])
        assert gain == pytest.approx(1.0, abs=0.1), f"{design} at {f} Hz: gain {gain:.3f}"


@pytest.mark.parametrize("design,order", [("butter", 4), ("fir", 301)])
def test_stopband_is_strongly_attenuated(design, order):
    """Blueprint Section 9: gain much less than 1 at 0.1 and 10 Hz."""
    filt = bandpass.make_bandpass(FS, 0.4, 5.0, design, order)
    for f in (0.1, 10.0):
        x = _tone(f)
        y = bandpass.apply_bandpass(x, filt)
        gain = np.std(y[200:-200]) / np.std(x[200:-200])
        assert gain < 0.1, f"{design} at {f} Hz: gain {gain:.3f}"


def test_fir_lower_edge_is_softer_than_the_butterworth():
    """A1 trade-off, measured rather than assumed.

    An 8 s window caps the FIR at about 333 taps, which is far too few to place
    a sharp corner at 0.4 Hz. The FIR therefore attenuates the bottom of the
    heart-rate band, where the Butterworth does not. Recorded so a later change
    of the A1 default is a deliberate choice.
    """
    x = _tone(0.5)
    fir = bandpass.apply_bandpass(x, bandpass.make_bandpass(FS, 0.4, 5.0, "fir", 301))
    iir = bandpass.apply_bandpass(x, bandpass.make_bandpass(FS, 0.4, 5.0, "butter", 4))
    interior = slice(250, -250)
    gain_fir = np.std(fir[interior]) / np.std(x[interior])
    gain_iir = np.std(iir[interior]) / np.std(x[interior])
    assert gain_fir < 0.5 < gain_iir


def test_fir_too_long_for_the_window_raises():
    """The tap count is bounded by the window length, and says so."""
    filt = bandpass.make_bandpass(FS, 0.4, 5.0, "fir", 401)
    with pytest.raises(ValueError, match="too short"):
        bandpass.apply_bandpass(np.zeros(M), filt)


def test_bandpass_removes_dc_offset():
    """A constant offset is out of band and must not survive."""
    filt = bandpass.make_bandpass(FS)
    x = _tone(1.5) + 500.0
    y = bandpass.apply_bandpass(x, filt)
    assert abs(y[200:-200].mean()) < 1.0


def test_bandpass_is_zero_phase():
    """sosfiltfilt must not shift the signal in time."""
    filt = bandpass.make_bandpass(FS)
    x = _tone(1.5)
    y = bandpass.apply_bandpass(x, filt)
    interior = slice(200, -200)
    lag = int(np.argmax(np.correlate(y[interior], x[interior], mode="same"))) - len(
        x[interior]
    ) // 2
    assert lag == 0


def test_bandpass_handles_stacked_channels():
    """The slot filters a (4, M) stack of PPG and three ACC axes at once."""
    filt = bandpass.make_bandpass(FS)
    stack = np.stack([_tone(1.5), _tone(2.0), _tone(0.1), _tone(10.0)])
    out = bandpass.apply_bandpass(stack, filt)
    assert out.shape == stack.shape
    for row in range(4):
        single = bandpass.apply_bandpass(stack[row], filt)
        assert np.allclose(out[row], single)


def test_bandpass_does_not_mutate_its_input():
    filt = bandpass.make_bandpass(FS)
    x = _tone(1.5)
    before = x.copy()
    bandpass.apply_bandpass(x, filt)
    assert np.array_equal(x, before)


def test_invalid_band_raises():
    with pytest.raises(ValueError, match="not valid"):
        bandpass.make_bandpass(FS, 5.0, 0.4)
    with pytest.raises(ValueError, match="not valid"):
        bandpass.make_bandpass(FS, 0.4, 70.0)


def test_unknown_design_raises():
    with pytest.raises(ValueError, match="unknown bandpass design"):
        bandpass.make_bandpass(FS, design="chebyshev")


def test_too_short_signal_raises():
    filt = bandpass.make_bandpass(FS)
    with pytest.raises(ValueError, match="too short"):
        bandpass.apply_bandpass(np.zeros(5), filt)


def test_frequency_response_shape_and_band_edges():
    """diag['filter_response'] must carry the keys the plots rely on."""
    filt = bandpass.make_bandpass(FS)
    resp = bandpass.frequency_response(filt, n=1024)
    assert set(resp) >= {"f_hz", "mag_db", "phase_rad"}
    assert resp["f_hz"].shape == resp["mag_db"].shape == (1024,)
    mid = np.argmin(np.abs(resp["f_hz"] - 2.0))
    lo = np.argmin(np.abs(resp["f_hz"] - 0.1))
    hi = np.argmin(np.abs(resp["f_hz"] - 15.0))
    assert resp["mag_db"][mid] > -3.0
    assert resp["mag_db"][lo] < -20.0
    assert resp["mag_db"][hi] < -20.0


# ------------------------------------------------------------------ periodogram


def test_periodogram_length_and_peak_location():
    """A 1.5 Hz tone peaks at the bin nearest 1.5 Hz."""
    s = spectrum.periodogram(_tone(1.5), N)
    assert s.shape == (N // 2 + 1,)
    assert int(np.argmax(s)) == binmap.hz_to_bin(1.5, FS, N)


def test_periodogram_is_nonnegative():
    rng = np.random.default_rng(0)
    assert (spectrum.periodogram(rng.standard_normal(M), N) >= 0).all()


def test_periodogram_of_empty_signal_raises():
    with pytest.raises(ValueError, match="empty"):
        spectrum.periodogram(np.zeros(0), N)


def test_local_maxima_finds_interior_peaks():
    s = np.array([0.0, 1.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    assert spectrum.local_maxima(s).tolist() == [1, 3, 5]


def test_local_maxima_includes_dominant_endpoints():
    """A peak on the edge of a narrow search range must still be visible."""
    s = np.array([5.0, 1.0, 0.0, 1.0, 4.0])
    assert spectrum.local_maxima(s).tolist() == [0, 4]


def test_local_maxima_respects_the_range():
    s = np.array([0.0, 9.0, 0.0, 2.0, 0.0, 3.0, 0.0])
    assert spectrum.local_maxima(s, 2, 6).tolist() == [3, 5]


def test_local_maxima_on_empty_or_single_bin_range():
    s = np.arange(10.0)
    assert spectrum.local_maxima(s, 5, 4).size == 0
    assert spectrum.local_maxima(s, 5, 5).tolist() == [5]


def test_dominant_bins_applies_the_fifty_percent_rule():
    """Paper Section III-A: keep peaks above 50 % of the maximum."""
    s = np.zeros(21)
    s[5], s[10], s[15] = 10.0, 4.0, 6.0
    assert spectrum.dominant_bins(s, rel_threshold=0.5).tolist() == [5, 15]


def test_dominant_bins_of_a_flat_spectrum_is_empty():
    assert spectrum.dominant_bins(np.zeros(50)).size == 0


def test_dominant_bins_on_two_acc_tones():
    """Two strong acceleration tones both survive the threshold."""
    s = spectrum.periodogram(_tone(1.2) + _tone(2.4, amp=0.9), N)
    lo, hi = binmap.band_to_bins(0.4, 5.0, FS, N)
    found = spectrum.dominant_bins(s, lo, hi, 0.5)
    hz = binmap.bin_to_hz(found, FS, N)
    assert any(abs(f - 1.2) < 0.05 for f in hz)
    assert any(abs(f - 2.4) < 0.05 for f in hz)


def test_dominant_bin_returns_the_argmax_in_range():
    s = np.array([0.0, 5.0, 0.0, 3.0, 0.0])
    assert spectrum.dominant_bin(s) == 1
    assert spectrum.dominant_bin(s, 2, 4) == 3


def test_dominant_bin_empty_range_raises():
    with pytest.raises(ValueError, match="empty search range"):
        spectrum.dominant_bin(np.zeros(5), 4, 2)


# ----------------------------------------------------------- temporal difference


def test_second_order_difference_length():
    """Paper: M = 1000 becomes M' = 998 (Blueprint Section 11)."""
    out, pre = temporal_diff.temporal_difference(np.arange(M, dtype=float), order=2)
    assert out.shape == (998,)
    assert pre.shape == (998,)


def test_difference_preserves_the_fundamental_and_harmonics():
    """Paper Section III-B: differencing keeps the periodic component's lines."""
    x = _tone(1.5) + 0.4 * _tone(3.0)
    out, _ = temporal_diff.temporal_difference(x, order=2, normalize=True)
    s = spectrum.periodogram(out, N)
    assert int(np.argmax(s)) in (
        binmap.hz_to_bin(1.5, FS, N),
        binmap.hz_to_bin(3.0, FS, N),
    )
    assert s[binmap.hz_to_bin(1.5, FS, N)] > 0
    assert s[binmap.hz_to_bin(3.0, FS, N)] > 0


def test_difference_suppresses_a_slow_aperiodic_trend():
    """A slow drift loses far more power than the heartbeat tone."""
    t = np.arange(M) / FS
    beat = _tone(2.0)
    drift = 3.0 * np.sin(2 * np.pi * 0.05 * t)

    def band_power(sig, f_hz, width=0.1):
        s = spectrum.periodogram(sig, N)
        lo, hi = binmap.band_to_bins(f_hz - width, f_hz + width, FS, N)
        return s[lo : hi + 1].sum()

    before = band_power(beat + drift, 0.05) / band_power(beat + drift, 2.0)
    after_sig, _ = temporal_diff.temporal_difference(beat + drift, order=2, normalize=False)
    after = band_power(after_sig, 0.05) / band_power(after_sig, 2.0)
    assert after < before / 100


def test_normalisation_gives_zero_mean_unit_variance():
    """ASSUMPTION A3, matching the normalised signals of paper Fig. 9."""
    rng = np.random.default_rng(1)
    out, _ = temporal_diff.temporal_difference(rng.standard_normal(M) * 37.0 + 5.0)
    assert out.mean() == pytest.approx(0.0, abs=1e-12)
    assert out.std() == pytest.approx(1.0, abs=1e-12)


def test_normalisation_can_be_switched_off():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(M) * 37.0
    out, pre = temporal_diff.temporal_difference(x, normalize=False)
    assert np.array_equal(out, pre)
    assert out.std() > 5.0


def test_pre_normalization_is_the_unscaled_difference():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(M)
    out, pre = temporal_diff.temporal_difference(x, order=2, normalize=True)
    assert np.allclose(pre, np.diff(x, n=2))
    assert not np.allclose(out, pre)


def test_constant_input_does_not_divide_by_zero():
    out, _ = temporal_diff.temporal_difference(np.full(M, 7.0), order=2, normalize=True)
    assert np.all(out == 0.0)


def test_order_zero_is_the_identity_up_to_normalisation():
    x = _tone(1.5)
    out, pre = temporal_diff.temporal_difference(x, order=0, normalize=False)
    assert np.array_equal(out, x)
    assert out is not x


def test_too_short_signal_for_order_raises():
    with pytest.raises(ValueError, match="too short"):
        temporal_diff.temporal_difference(np.zeros(2), order=2)


def test_negative_order_raises():
    with pytest.raises(ValueError, match=">= 0"):
        temporal_diff.temporal_difference(np.zeros(10), order=-1)
