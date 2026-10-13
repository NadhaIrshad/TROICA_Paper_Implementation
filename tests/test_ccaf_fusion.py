"""CCAF channel fusion and the band-pass plug-in that feeds it to the pipeline (DEVIATIONS.md D16)."""

from __future__ import annotations

import numpy as np
import pytest

from troika.plugins.bandpass.butter import ButterBandpass, ButterParams
from troika.plugins.bandpass.fused import FusedBandpass, FusedParams
from troika.testing import check_plugin
from troika.types import StaticContext, WindowContext

FS = 125.0
M = 1000
STATIC = StaticContext(fs=FS, N=4096, M=M, window_s=8.0, step_s=2.0)
T = np.arange(M) / FS


def _stack() -> np.ndarray:
    """A ``(4, M)`` window: PPG, then three accelerometer axes."""
    swing = np.sin(2 * np.pi * 1.3 * T)
    return np.vstack([np.sin(2 * np.pi * 2.0 * T) + 3 * swing, swing, 0.6 * swing, 0.3 * swing])


# ------------------------------------------------------------------- plug-in


def test_plugin_without_a_file_is_the_plain_band_pass():
    x = _stack()
    plain = ButterBandpass(ButterParams(), STATIC)(x, WindowContext(idx=0, t_start_s=0.0)).data
    fused = FusedBandpass(FusedParams(), STATIC)(x, WindowContext(idx=0, t_start_s=0.0)).data
    assert np.array_equal(fused, plain)
    assert check_plugin("bandpass", "fused").ok


def test_plugin_replaces_the_ppg_row_and_leaves_the_accelerometer(tmp_path):
    t25 = np.arange(200) / 25.0
    windows = np.stack([np.sin(2 * np.pi * 1.5 * t25), np.sin(2 * np.pi * 2.5 * t25)])
    np.save(tmp_path / "fused.npy", windows.astype(np.float32))
    plugin = FusedBandpass(FusedParams(windows_path=str(tmp_path / "fused.npy")), STATIC)
    x = _stack()
    plain = ButterBandpass(ButterParams(), STATIC)(x, WindowContext(idx=1, t_start_s=2.0)).data

    out = plugin(x, WindowContext(idx=1, t_start_s=2.0)).data
    assert out.shape == (4, M)
    assert np.array_equal(out[1:], plain[1:])
    # Window 1 of the file, at five times the rate: a 2.5 Hz sine. The resampling
    # filter rings at the two ends, so compare the middle.
    middle = slice(100, 900)
    assert out[0][middle] == pytest.approx(np.sin(2 * np.pi * 2.5 * T)[middle], abs=0.02)
    assert not np.allclose(out[0], plain[0])


def test_plugin_rejects_a_bad_file_or_window(tmp_path):
    with pytest.raises(FileNotFoundError, match="nowhere.npy"):
        FusedBandpass(FusedParams(windows_path=str(tmp_path / "nowhere.npy")), STATIC)
    np.save(tmp_path / "odd.npy", np.zeros((3, 333)))
    with pytest.raises(ValueError, match="dividing"):
        FusedBandpass(FusedParams(windows_path=str(tmp_path / "odd.npy")), STATIC)
    np.save(tmp_path / "two.npy", np.zeros((2, 200)))
    plugin = FusedBandpass(FusedParams(windows_path=str(tmp_path / "two.npy")), STATIC)
    with pytest.raises(IndexError, match="no fused window 2"):
        plugin(_stack(), WindowContext(idx=2, t_start_s=4.0))
    with pytest.raises(ValueError, match="per_window"):
        FusedParams(windows_path="x.npy", mode="global")


# --------------------------------------------------------------- the block

torch = pytest.importorskip("torch")

from troika.preprocessing.ccaf import (  # noqa: E402
    CCAF,
    ccaf_windows,
    fuse,
    snr_np,
    train_ccaf,
)


def _two_channels(seconds: float = 30.0) -> np.ndarray:
    """Two PPG channels: a 2 Hz pulse in both, a 1.3 Hz artifact mostly in the second."""
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * FS)) / FS
    pulse, swing = np.sin(2 * np.pi * 2.0 * t), np.sin(2 * np.pi * 1.3 * t)
    return np.stack([pulse + 0.3 * swing, pulse + 3.0 * swing]) + 0.05 * rng.standard_normal((2, t.size))


def test_windows_follow_the_pipeline_geometry_and_are_z_scored():
    w = ccaf_windows(_two_channels(30.0))
    assert w.shape == (12, 2, 200) and w.dtype == np.float32  # (30 - 8) / 2 + 1 windows at 25 Hz
    assert w.mean(-1) == pytest.approx(np.zeros((12, 2)), abs=1e-5)
    assert w.std(-1) == pytest.approx(np.ones((12, 2)), abs=1e-4)
    with pytest.raises(ValueError, match="two PPG channels"):
        ccaf_windows(np.zeros((1, 4000)))


def test_block_returns_one_fused_window_and_its_maps():
    torch.manual_seed(0)
    w = ccaf_windows(_two_channels())
    y, a1, g = fuse(CCAF(), w, return_maps=True)
    assert y.shape == a1.shape == g.shape == (len(w), 200)
    assert ((a1 > 0) & (a1 < 1)).all() and ((g > 0) & (g < 1)).all()
    assert fuse(CCAF(), w).shape == (len(w), 200)


def test_snr_is_higher_for_the_cleaner_channel():
    w = ccaf_windows(_two_channels())
    hr = np.full(len(w), 120.0)
    assert np.median(snr_np(w[:, 0], hr)) > np.median(snr_np(w[:, 1], hr)) + 3.0


def test_training_runs_and_is_reproducible():
    w = ccaf_windows(_two_channels())
    hr = np.full(len(w), 120.0)
    first = fuse(train_ccaf(w, hr, epochs=2, qat_epochs=1), w)
    again = fuse(train_ccaf(w, hr, epochs=2, qat_epochs=1), w)
    assert np.isfinite(first).all()
    assert first == pytest.approx(again, abs=1e-4)
