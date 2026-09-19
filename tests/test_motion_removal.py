"""Motion-artifact identification and removal (Blueprint Sections 5.2-5.3, 9)."""

from __future__ import annotations

import numpy as np
import pytest

from troika import binmap
from troika.config import load_config
from troika.decomposition.motion_removal import (
    acc_dominant_bins,
    refine_acc_bins,
    select_motion_components,
)
from troika.registry import get
from troika.types import StaticContext, WindowContext

FS = 125.0
N = 4096
M = 1000


def _tone(f_hz, n=M, fs=FS, amp=1.0, phase=0.0):
    t = np.arange(n) / fs
    return amp * np.sin(2 * np.pi * f_hz * t + phase)


def _bin(f_hz):
    return binmap.hz_to_bin(f_hz, FS, N)


def _static():
    return StaticContext(fs=FS, N=N, M=M, window_s=8.0, step_s=2.0)


def _ssa(**params):
    cls = get("decomposition", "ssa")
    merged = {"L": 400, "grouping": "frequency_pairing", "match_tol_bins": 2, **params}
    return cls(cls.Params(**merged), _static())


def _ctx(acc_bins=(), prev_bin=None):
    return WindowContext(idx=0, t_start_s=0.0, acc_bins=set(acc_bins), prev_bin=prev_bin)


# ------------------------------------------------------------------- F_acc


def test_acc_dominant_bins_finds_the_swing_frequency():
    """Paper Section III-A step 1: peaks above 50 % of each axis's maximum."""
    acc = np.stack([_tone(1.3), _tone(1.3, amp=0.5), _tone(2.6)])
    found = acc_dominant_bins(acc, N, FS, rel_threshold=0.5)
    hz = sorted(binmap.bin_to_hz(np.array(sorted(found)), FS, N))
    assert any(abs(f - 1.3) < 0.05 for f in hz)
    assert any(abs(f - 2.6) < 0.05 for f in hz)


def test_acc_dominant_bins_is_a_union_over_axes():
    """Each axis is thresholded against its own maximum, then unioned."""
    only_x = acc_dominant_bins(np.stack([_tone(1.3), np.zeros(M), np.zeros(M)]), N, FS)
    only_z = acc_dominant_bins(np.stack([np.zeros(M), np.zeros(M), _tone(2.6)]), N, FS)
    both = acc_dominant_bins(np.stack([_tone(1.3), np.zeros(M), _tone(2.6)]), N, FS)
    assert only_x <= both and only_z <= both


def test_weak_axis_peak_still_counts_because_the_rule_is_per_axis():
    """A quiet axis is thresholded against itself, not against the loud axis."""
    acc = np.stack([_tone(1.3, amp=100.0), _tone(3.1, amp=0.01), np.zeros(M)])
    found = acc_dominant_bins(acc, N, FS)
    assert _bin(3.1) in found or (_bin(3.1) - 1) in found or (_bin(3.1) + 1) in found


def test_acc_dominant_bins_of_silence_is_empty():
    assert acc_dominant_bins(np.zeros((3, M)), N, FS) == set()


def test_band_restriction_keeps_out_of_band_peaks_out():
    """ASSUMPTION A19: restrict the peak search to 0.4-5 Hz, or do not."""
    acc = np.stack([_tone(10.0), np.zeros(M), np.zeros(M)])
    restricted = acc_dominant_bins(acc, N, FS, restrict_to_band=True)
    unrestricted = acc_dominant_bins(acc, N, FS, restrict_to_band=False)
    lo, hi = binmap.band_to_bins(0.4, 5.0, FS, N)
    assert _bin(10.0) in unrestricted
    assert _bin(10.0) not in restricted
    assert all(lo <= b <= hi for b in restricted)


def test_the_fifty_percent_rule_is_relative_so_a_quiet_axis_still_yields_peaks():
    """A property of the paper's rule, recorded rather than worked around.

    The threshold is 50 % of the maximum *within the same spectrum*, so an axis
    with no real rhythmic content still produces dominant bins, drawn from its
    noise and leakage floor. On real running data every axis carries a stride
    peak, and the refinement plus the all-removed fallback contain the damage,
    so the paper's rule is kept as written.
    """
    quiet = np.stack([_tone(10.0), np.zeros(M), np.zeros(M)])
    assert len(acc_dominant_bins(quiet, N, FS, restrict_to_band=True)) > 1

    loud = np.stack([_tone(1.3, amp=50.0), np.zeros(M), np.zeros(M)])
    assert len(acc_dominant_bins(loud, N, FS, restrict_to_band=True)) <= 3


# --------------------------------------------------------------- refinement


def test_refine_excludes_the_previous_hr_neighbourhood():
    """Paper Section III-A: exclude {N_p - Delta, ..., N_p + Delta} from F_acc."""
    prev = 60
    f_acc = {40, 55, 60, 65, 80, 120}
    refined = refine_acc_bins(f_acc, prev, delta=10, n_harmonics=1, n_fft=N)
    assert refined == {40, 80, 120}


def test_refine_also_protects_the_first_harmonic():
    """ASSUMPTION A6: N_p holds the fundamental and the first harmonic."""
    prev = 60
    f_acc = {40, 60, 118, 122, 200}
    refined = refine_acc_bins(f_acc, prev, delta=10, n_harmonics=2, n_fft=N)
    assert refined == {40, 200}


def test_refine_with_one_harmonic_leaves_the_harmonic_alone():
    refined = refine_acc_bins({120}, 60, delta=10, n_harmonics=1, n_fft=N)
    assert refined == {120}


def test_refine_is_identity_on_the_first_window():
    """No previous estimate exists, so F_acc passes through unchanged."""
    f_acc = {10, 20, 30}
    assert refine_acc_bins(f_acc, None, delta=10) == f_acc


def test_refine_does_not_mutate_its_input():
    f_acc = {10, 20, 30}
    refine_acc_bins(f_acc, 20, delta=5)
    assert f_acc == {10, 20, 30}


def test_refine_stops_at_nyquist():
    """A harmonic beyond N/2 must not be constructed."""
    refined = refine_acc_bins({5}, prev_bin=1500, delta=2, n_harmonics=3, n_fft=N)
    assert refined == {5}


# ------------------------------------------------------- component selection


def test_component_matching_uses_a_tolerance():
    """ASSUMPTION A5: membership is |d_p - a| <= match_tol_bins."""
    dominant = np.array([50, 52, 55, 90])
    mask = select_motion_components(dominant, {50}, match_tol_bins=2)
    assert mask.tolist() == [True, True, False, False]


def test_exact_matching_when_tolerance_is_zero():
    dominant = np.array([50, 51])
    assert select_motion_components(dominant, {50}, 0).tolist() == [True, False]


def test_nothing_is_removed_when_f_acc_is_empty():
    mask = select_motion_components(np.array([10, 20, 30]), set(), 2)
    assert not mask.any()


# ---------------------------------------------------- end-to-end on synthetic


def test_ma_component_is_removed_and_the_heartbeat_survives():
    """Blueprint Section 9: HR sinusoid plus MA sinusoid, acc carries the MA.

    The motion component is five times the heartbeat, which is the regime the
    paper describes; after removal the heartbeat must dominate the spectrum.
    """
    hr_hz, ma_hz = 2.0, 1.25
    ppg = _tone(hr_hz, amp=1.0) + _tone(ma_hz, amp=5.0)
    acc_bins = acc_dominant_bins(np.stack([_tone(ma_hz)] * 3), N, FS)

    out = _ssa()(ppg, _ctx(acc_bins=acc_bins))
    cleansed = out.data

    from troika.preprocessing.spectrum import periodogram

    before = periodogram(ppg, N)
    after = periodogram(cleansed, N)
    assert int(np.argmax(before)) == pytest.approx(_bin(ma_hz), abs=2)
    assert int(np.argmax(after)) == pytest.approx(_bin(hr_hz), abs=2)
    assert out.diag["removed_mask"].any()


def test_excluding_the_previous_hr_protects_a_cadence_that_matches_it():
    """Paper Section III-A: the refinement exists for exactly this case.

    Cadence and heart rate coincide at 2 Hz. Without the refinement the
    accelerometer peak would delete the heartbeat; with it, the component
    survives.
    """
    shared_hz = 2.0
    ppg = _tone(shared_hz, amp=1.0) + _tone(3.3, amp=2.0)
    raw = acc_dominant_bins(np.stack([_tone(shared_hz)] * 3), N, FS)
    refined = refine_acc_bins(raw, prev_bin=_bin(shared_hz), delta=10, n_harmonics=2, n_fft=N)

    assert any(abs(b - _bin(shared_hz)) <= 2 for b in raw)
    assert not any(abs(b - _bin(shared_hz)) <= 2 for b in refined)

    from troika.preprocessing.spectrum import periodogram

    unprotected = _ssa()(ppg, _ctx(acc_bins=raw)).data
    protected = _ssa()(ppg, _ctx(acc_bins=refined)).data
    hr_bin = _bin(shared_hz)
    assert periodogram(protected, N)[hr_bin] > 10 * periodogram(unprotected, N)[hr_bin]


def test_all_removed_falls_back_to_the_bandpassed_window():
    """Blueprint Section 5.3 fallback: never hand the estimator an empty signal."""
    ppg = _tone(2.0)
    every_bin = set(range(0, N // 2))
    out = _ssa()(ppg, _ctx(acc_bins=every_bin))
    assert np.allclose(out.data, ppg)
    assert out.diag["fallback"] == "all_removed"
    assert not out.diag["removed_mask"].any()


def test_no_match_is_the_identity():
    """With nothing to remove, the sum of all groups is the input."""
    ppg = _tone(2.0) + 0.5 * _tone(3.0)
    out = _ssa()(ppg, _ctx(acc_bins=set()))
    assert np.allclose(out.data, ppg, atol=1e-8)
    assert not out.diag["removed_mask"].any()


def test_none_decomposition_is_the_ablation_identity():
    """Table I row 2 removes the decomposition entirely."""
    cls = get("decomposition", "none")
    ppg = _tone(2.0)
    out = cls(cls.Params(), _static())(ppg, _ctx(acc_bins={_bin(2.0)}))
    assert np.array_equal(out.data, ppg)
    assert not out.diag["removed_mask"].any()


# --------------------------------------------------------------- diag schema


def test_ssa_fills_every_diag_key_the_plots_need():
    """Lab Spec Section 2.3: paper-default plug-ins fill every listed key."""
    out = _ssa()(_tone(2.0) + _tone(1.3, amp=3.0), _ctx(acc_bins={_bin(1.3)}))
    generic = {"components", "component_labels", "component_dominant_bins", "removed_mask"}
    specific = {
        "trajectory",
        "U",
        "s",
        "Vt",
        "groups",
        "group_dominant_bins",
        "group_singular_range",
        "L",
        "K",
        "grouping_strategy",
    }
    assert generic <= set(out.diag)
    assert specific <= set(out.diag)

    g = len(out.diag["groups"])
    assert out.diag["components"].shape == (g, M)
    assert out.diag["component_dominant_bins"].shape == (g,)
    assert out.diag["removed_mask"].shape == (g,)
    assert len(out.diag["component_labels"]) == g
    assert out.diag["trajectory"].shape == (400, 601)
    assert out.diag["L"] == 400 and out.diag["K"] == 601


def test_components_sum_to_the_input_when_nothing_is_removed():
    ppg = _tone(2.0) + _tone(1.3, amp=3.0)
    out = _ssa()(ppg, _ctx())
    assert np.allclose(out.diag["components"].sum(axis=0), ppg, atol=1e-8)


def test_l_above_half_the_window_is_rejected():
    """Paper footnote 2: L must be below M/2 for a stable decomposition."""
    cls = get("decomposition", "ssa")
    with pytest.raises(ValueError, match="exceeds M/2"):
        cls(cls.Params(L=600), _static())


def test_plugin_does_not_mutate_its_input():
    ppg = _tone(2.0) + _tone(1.3, amp=3.0)
    before = ppg.copy()
    _ssa()(ppg, _ctx(acc_bins={_bin(1.3)}))
    assert np.array_equal(ppg, before)


def test_plugin_is_deterministic():
    ppg = _tone(2.0) + _tone(1.3, amp=3.0)
    a = _ssa()(ppg, _ctx(acc_bins={_bin(1.3)})).data
    b = _ssa()(ppg, _ctx(acc_bins={_bin(1.3)})).data
    assert np.array_equal(a, b)


def test_config_default_params_build_the_plugin():
    """The shipped default.yaml must construct the paper-default decomposition."""
    cfg = load_config()
    params = cfg.slot_params("decomposition")
    assert params.L == 400 and params.grouping == "frequency_pairing"
    get("decomposition", cfg.slot_method("decomposition"))(params, _static())
