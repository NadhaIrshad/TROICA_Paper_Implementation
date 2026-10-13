"""CCAF attention fusion of the two PPG channels (DEVIATIONS.md D16).

The fusion block of ``Channel Fusion/main.py``, taken on its own: the network,
its loss, its training loop with the int8 fine-tuning stage, and the
preprocessing it expects. The code is copied from that file with only its
dependence on the script's global configuration turned into arguments; nothing
else of that pipeline (EEMD-lite, its spectral subtraction, EMA, TinyTracker,
its HMM) is used here.

The block works at 25 Hz on 8-second windows of 200 samples, both channels
band-passed and z-scored per window, and returns one fused 200-sample window.
It is trained to maximise the spectral signal-to-noise ratio at the ECG heart
rate, so it must be fitted on training subjects only.

This module imports ``torch`` at the top, so import it lazily from anything that
must load without the optional dependency. As in the source file it works in
float32, unlike the float64 of the rest of the package.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi

from troika.preprocessing.windowing import n_windows

__all__ = ["CCAF", "ccaf_windows", "zscore", "spectral_snr_db", "snr_np", "train_ccaf", "fuse"]

QMAX = 127


# ----------------------------------------------------------- preprocessing


def zscore(x, axis=-1):
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-8)


def ccaf_windows(
    ppg_all: NDArray[np.float64],
    fs_raw: float = 125.0,
    window_s: float = 8.0,
    step_s: float = 2.0,
    low_hz: float = 0.4,
    high_hz: float = 5.0,
    order: int = 4,
    decim: int = 5,
) -> NDArray[np.float32]:
    """The block's input for a whole recording, ``(n_windows, 2, window_s * fs_raw / decim)``.

    As ``StreamingPreprocessor`` and ``make_windows`` of the source file: a
    causal Butterworth band-pass over the recording, started from the first
    sample's steady state, every ``decim``-th sample kept, then the windows cut
    and each channel of each window z-scored. Filtering the recording in one
    pass gives the same samples as that file's chunk-by-chunk filter.
    """
    raw = np.asarray(ppg_all, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] != 2:
        raise ValueError(f"ppg_all must hold the two PPG channels as (2, n), got shape {raw.shape}")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs_raw, output="sos")
    zi = sosfilt_zi(sos)[:, None, :] * raw[None, :, :1]
    filtered, _ = sosfilt(sos, raw, axis=-1, zi=zi)
    x = filtered[:, ::decim]

    fs = fs_raw / decim
    win, step = int(round(window_s * fs)), int(round(step_s * fs))
    count = n_windows(raw.shape[1], fs_raw, window_s, step_s)
    idx = np.arange(count)[:, None] * step + np.arange(win)[None, :]
    return zscore(x[:, idx].transpose(1, 0, 2)).astype(np.float32)


# ------------------------------------------- int8 quantisation-aware layers


def fake_quant(x, scale):
    q = torch.clamp(torch.round(x / scale), -QMAX, QMAX) * scale
    return x + (q - x).detach()                                 # straight-through estimator


class QConv1d(nn.Conv1d):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.qat = False
        self.register_buffer("act_max", torch.tensor(0.0))
        self.momentum = 0.05

    def w_scale(self):
        return (self.weight.detach().abs().amax(dim=(1, 2), keepdim=True) / QMAX).clamp_min(1e-8)

    def a_scale(self):
        return (self.act_max / QMAX).clamp_min(1e-8)

    def forward(self, x):
        if not self.qat:
            return super().forward(x)
        if self.training:
            m = x.detach().abs().max()
            self.act_max.copy_(m if self.act_max == 0 else (1 - self.momentum) * self.act_max + self.momentum * m)
        return F.conv1d(fake_quant(x, self.a_scale()), fake_quant(self.weight, self.w_scale()), self.bias,
                        self.stride, self.padding, self.dilation, self.groups)


def set_qat(model, on=True):
    for m in model.modules():
        if isinstance(m, QConv1d):
            m.qat = on
    return model


@torch.no_grad()
def calibrate(model, batches):
    was = model.training
    model.train()
    for xb in batches:
        model(xb)
    model.train(was)


# ------------------------------------------------------------------ the block


class CCAF(nn.Module):
    def __init__(self, C=8, k=9, smooth=13):
        super().__init__()
        self.enc_dw = QConv1d(1, C, k, padding=k // 2)
        self.enc_pw = QConv1d(C, C, 1)
        self.score = QConv1d(3 * C, 1, 1)
        self.gate = QConv1d(2 * C, 1, 1)
        self.smooth = smooth

    def _ma(self, z):
        return F.avg_pool1d(z, self.smooth, 1, self.smooth // 2, count_include_pad=False)

    def forward(self, x, return_maps=False):
        B, _, T = x.shape
        h = F.relu(self.enc_pw(F.relu(self.enc_dw(x.reshape(B * 2, 1, T))))).reshape(B, 2, -1, T)
        h1, h2 = h[:, 0], h[:, 1]
        agree, diff = h1 * h2, (h1 - h2).abs()
        s1 = self.score(torch.cat([h1, agree, diff], 1))
        s2 = self.score(torch.cat([h2, agree, diff], 1))
        a1 = torch.sigmoid(self._ma(s1 - s2))
        g = torch.sigmoid(self._ma(self.gate(torch.cat([agree, diff], 1))))
        y = (g * (a1 * x[:, :1] + (1 - a1) * x[:, 1:2])).squeeze(1)
        return (y, a1.squeeze(1), g.squeeze(1)) if return_maps else y


def spectral_snr_db(y, hr_bpm, fs=25, nfft=512, delta_hz=0.2):
    y = y - y.mean(-1, keepdim=True)
    P = torch.fft.rfft(y * torch.hann_window(y.shape[-1], periodic=False, dtype=y.dtype), n=nfft).abs() ** 2
    f = torch.fft.rfftfreq(nfft, 1 / fs).to(y.dtype)
    f0 = (hr_bpm.to(y.dtype) / 60.0)[:, None]
    sig = ((f[None] - f0).abs() <= delta_hz) | ((f[None] - 2 * f0).abs() <= delta_hz)
    inb = ((f >= 0.5) & (f <= 8.0))[None]
    return 10 * torch.log10(((P * (sig & inb)).sum(-1) + 1e-10) / ((P * (~sig & inb)).sum(-1) + 1e-10))


def snr_np(y, hr):
    return spectral_snr_db(torch.from_numpy(np.ascontiguousarray(np.atleast_2d(y))).double(),
                           torch.as_tensor(np.atleast_1d(hr)).double()).numpy()


def train_ccaf(ppg, hr, *, fs=25, channels=8, kernel=9, smooth=13, epochs=40, qat_epochs=10, lr=3e-3, seed=0):
    """Fit the block on windows ``ppg`` ``(n, 2, T)`` and their heart rates ``hr`` ``(n,)``.

    ``train_ccaf`` of the source file with its configuration as keyword
    arguments, defaults as there: ``epochs`` in float32, then ``qat_epochs``
    with int8 fake quantisation at a tenth of the learning rate.
    """
    torch.manual_seed(seed)
    model = CCAF(channels, kernel, smooth)
    X, Y = torch.from_numpy(ppg), torch.from_numpy(hr).float()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def run(n_epochs, rate):
        for gr in opt.param_groups:
            gr["lr"] = rate
        for _ in range(n_epochs):
            model.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 64):
                b = perm[i:i + 64]
                xb = X[b]
                if torch.rand(1) < 0.5:
                    xb = xb.flip(1)
                y, a, g = model(xb, return_maps=True)
                loss = -spectral_snr_db(y, Y[b], fs).mean() + 2.0 * F.relu(0.5 - g.mean())
                opt.zero_grad()
                loss.backward()
                opt.step()

    run(epochs, lr)
    set_qat(model, True)
    calibrate(model, [X[i:i + 256] for i in range(0, len(X), 256)])
    run(qat_epochs, lr / 10)
    return model.eval()


@torch.no_grad()
def fuse(model, ppg, return_maps=False):
    out = model.eval()(torch.from_numpy(ppg), return_maps=True)
    return tuple(o.numpy() for o in out) if return_maps else out[0].numpy()
