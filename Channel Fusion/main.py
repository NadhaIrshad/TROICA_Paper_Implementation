# %%
import copy
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from scipy.signal import butter, sosfilt, sosfilt_zi, stft
from scipy.interpolate import CubicSpline
from joblib import Parallel, delayed


T_START = time.time()


def log(msg):
    """Print a progress line with the elapsed time since the start of the script."""
    print(f"[{time.time() - T_START:8.1f} s] {msg}", flush=True)


# %% Configuration 
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
NPY_DIR = ROOT / "competition_data" / "npy_data"
OUT_DIR = ROOT / "results_all"
OUT_DIR.mkdir(exist_ok=True)

USE_EEMD = False      # True - EEMD-Lite used  | False - skipped (lite pipeline)
USE_EMA = True      # True - EMA input channel is used  | False - not used (lite pipeline)
QUICK = False         # True - very short training, only to check that everything runs
LOG_EVERY = 10        # print training progress every N epochs

# block switches (reference pipeline = the values below; the ablation study changes one at a time)
FUSION = "ccaf"       # [1] "ccaf" = attention | "ch1" / "ch2" = one PPG channel only | "mean" = plain average
SS_SUBTRACT = True    # [3] True = subtract ACC spectrum   | False = PPG spectrum only
SS_HARMONICS = True   # [3] True = 2x/3x harmonic boost   | False = no boost
USE_HINT = True       # [4] True = previous-HR hint row   | False = hint always flat
USE_DAGGER = True     # [4] True = DAgger rounds          | False = no DAgger
USE_KD = True         # [4] True = student learns from teacher (KD) | False = student trained alone
USE_HMM = True        # [5] True = HMM decoding           | False = CNN best guess each window
HMM_START_PRIOR = True  # [5] True = HMM starts at ~95 +/- 20 BPM | False = flat start
RUN_ABLATION = False  # True = after the main run, run the ablation study (section at the end of the file)
RUN_PARAMS_OPS_ONLY = False  # True = ONLY print/save the parameter + operation table per block (no training), then stop
if QUICK:             # quick test results go to a separate folder so they are never mistaken for real results
    OUT_DIR = ROOT / "results_all_in_one_QUICK_TEST"
    OUT_DIR.mkdir(exist_ok=True)


@dataclass
class Config:
    use_eemd: bool = USE_EEMD
    use_ema: bool = USE_EMA
    fusion: str = FUSION
    ss_subtract: bool = SS_SUBTRACT
    ss_harmonics: bool = SS_HARMONICS
    use_hint: bool = USE_HINT
    use_dagger: bool = USE_DAGGER
    use_kd: bool = USE_KD
    use_hmm: bool = USE_HMM
    hmm_start_prior: bool = HMM_START_PRIOR
    # raw data
    fs_raw: int = 125
    win_sec: int = 8
    step_sec: int = 2
    # preprocessing
    bp_low: float = 0.4
    bp_high: float = 5.0
    bp_order: int = 4
    decim: int = 5                        # 125 Hz - 25 Hz
    # attention
    attn_ch: int = 8
    attn_kernel: int = 9
    attn_smooth: int = 13
    attn_epochs: int = 40
    attn_qat_epochs: int = 10
    attn_lr: float = 3e-3
    # EEMD-lite
    eemd_trials: int = 8
    eemd_noise: float = 0.2
    eemd_max_imfs: int = 5
    eemd_sifts: int = 6
    imf_band: tuple = (0.5, 4.0)
    # spectral subtraction
    nfft: int = 1024
    ss_band: tuple = (0.5, 4.0)
    harm_w: tuple = (1.0, 0.66, 0.33)
    # EMA
    ema_beta: float = 0.7
    # tracker
    bpm_min: float = 40.0
    bpm_max: float = 220.0
    n_bins: int = 128
    prior_sigma: float = 8.0
    trans_sigma: float = 5.0
    hmm_eps: float = 1e-2
    init_sd: float = 20.0
    teacher_width: int = 48
    student_width: int = 16
    tracker_epochs: int = 60
    kd_epochs: int = 60
    finetune_epochs: int = 15
    prune_amount: float = 0.5
    tracker_lr: float = 3e-3
    seed: int = 0
    folds: list = field(default_factory=lambda: [["01", "02", "03"], ["04", "05", "06"],
                                                 ["07", "08", "09"], ["10", "11", "12"]])

    @property
    def fs(self):   return self.fs_raw // self.decim          # 25 Hz
    @property
    def win(self):  return self.win_sec * self.fs              # 200 samples
    @property
    def step(self): return self.step_sec * self.fs             # 50 samples
    @property
    def n_in(self): return 3 + (1 if self.use_ema else 0) + 1  


cfg = Config()
if QUICK:
    cfg = Config(attn_epochs=3, attn_qat_epochs=1, tracker_epochs=4, kd_epochs=4, finetune_epochs=2)
log(f"config: FUSION={cfg.fusion} EEMD={cfg.use_eemd} SS_SUBTRACT={cfg.ss_subtract} HARMONICS={cfg.ss_harmonics} "
    f"EMA={cfg.use_ema} HINT={cfg.use_hint} DAGGER={cfg.use_dagger} KD={cfg.use_kd} HMM={cfg.use_hmm} "
    f"START_PRIOR={cfg.hmm_start_prior}  CNN input rows={cfg.n_in}  QUICK={QUICK}")


# %% data loading + preprocessing
CHANNELS = ["ppg1", "ppg2", "acc1", "acc2", "acc3"]


def load_subject(pid, cfg):
    d = np.load(NPY_DIR / f"participant_{pid}.npy", allow_pickle=True).item()
    hop = cfg.step_sec * cfg.fs_raw
    sig = np.stack([np.concatenate([d[k][0], d[k][1:, -hop:].ravel()]) for k in CHANNELS])
    return {"pid": pid, "raw": sig.astype(np.float64), "hr": d["heart_rate"].astype(np.float64),
            "n_win": d["ppg1"].shape[0]}


class StreamingPreprocessor:
    def __init__(self, cfg, n_ch=5):
        self.cfg = cfg
        self.sos = butter(cfg.bp_order, [cfg.bp_low, cfg.bp_high], btype="bandpass", fs=cfg.fs_raw, output="sos")
        self.zi, self.offset = None, 0

    def process(self, chunk):
        if self.zi is None:
            self.zi = sosfilt_zi(self.sos)[:, None, :] * chunk[None, :, :1]
        y, self.zi = sosfilt(self.sos, chunk, axis=-1, zi=self.zi)
        j0 = (-self.offset) % self.cfg.decim
        self.offset += chunk.shape[-1]
        return y[:, j0::self.cfg.decim]


def preprocess_stream(raw, cfg):
    pre = StreamingPreprocessor(cfg, raw.shape[0])
    hop, first = cfg.step_sec * cfg.fs_raw, cfg.win_sec * cfg.fs_raw
    out = [pre.process(raw[:, :first])]
    for s in range(first, raw.shape[1], hop):
        out.append(pre.process(raw[:, s:s + hop]))
    return np.concatenate(out, axis=1)


def zscore(x, axis=-1):
    return (x - x.mean(axis, keepdims=True)) / (x.std(axis, keepdims=True) + 1e-8)


def make_windows(subj, cfg):
    x = preprocess_stream(subj["raw"], cfg)
    idx = np.arange(subj["n_win"])[:, None] * cfg.step + np.arange(cfg.win)[None, :]
    w = x[:, idx].transpose(1, 0, 2)
    return zscore(w[:, :2]).astype(np.float32), zscore(w[:, 2:]).astype(np.float32), subj["hr"]


def build_dataset(cfg):
    ds = {}
    for i in range(1, 13):
        pid = f"{i:02d}"
        ppg, acc, hr = make_windows(load_subject(pid, cfg), cfg)
        ds[pid] = {"ppg": ppg, "acc": acc, "hr": hr}
    return ds


# %% Edge toolkit — int8 QAT, pruning, profiling, C export
QMAX = 127


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


def prunable(model):
    return [(m, "weight") for m in model.modules() if isinstance(m, nn.Conv1d) and m.weight.numel() > 16]


def apply_pruning(model, amount):
    prune.global_unstructured(prunable(model), pruning_method=prune.L1Unstructured, amount=amount)
    return model


def remove_pruning(model):
    for m, n in prunable(model):
        if prune.is_pruned(m):
            prune.remove(m, n)
    return model


def profile(model, example, int8=False):
    macs = []

    def hook(m, i, o):
        nz = (m.weight != 0).sum().item()
        n_run = o.shape[0]      # a layer can run on several items per input (CCAF encoder: once per PPG channel)
        macs.append((n_run * o.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0],
                     n_run * o.shape[-1] * nz))

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        model.eval()
        model(example)
    for h in hs:
        h.remove()
    n_param = sum(p.numel() for p in model.parameters())
    n_w = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv1d))
    size = (n_w + 4 * (n_param - n_w)) if int8 else 4 * n_param
    return {"params": n_param, "nonzero": sum((p != 0).sum().item() for p in model.parameters()),
            "MACs": sum(a for a, _ in macs), "sparse_MACs": sum(b for _, b in macs), "size_kB": size / 1024}


def quantized_layers(model):
    out = []
    for name, m in model.named_modules():
        if isinstance(m, QConv1d):
            sw = m.w_scale().reshape(-1).numpy()
            sa = float(m.a_scale())
            wq = torch.clamp(torch.round(m.weight.detach() / m.w_scale()), -QMAX, QMAX).to(torch.int8).numpy()
            bq = None if m.bias is None else np.round(m.bias.detach().numpy() / (sa * sw)).astype(np.int32)
            out.append({"name": name, "w": wq, "b": bq, "s_w": sw, "s_in": sa, "groups": m.groups,
                        "dilation": m.dilation[0], "padding": m.padding[0]})
    return out


def export_c_header(model, path, prefix):
    lines, total = [f"// int8 model {prefix}", "#pragma once", "#include <stdint.h>", ""], 0
    for L in quantized_layers(model):
        n, w = L["name"].replace(".", "_"), L["w"]
        total += w.size + (0 if L["b"] is None else 4 * L["b"].size)
        lines.append(f"static const int8_t {prefix}_{n}_w[{w.size}] = {{{','.join(map(str, w.ravel()))}}};")
        if L["b"] is not None:
            lines.append(f"static const int32_t {prefix}_{n}_b[{L['b'].size}] = {{{','.join(map(str, L['b']))}}};")
        lines.append(f"static const float {prefix}_{n}_s_w[{L['s_w'].size}] = {{{','.join(f'{v:.8e}f' for v in L['s_w'])}}};")
        lines.append(f"static const float {prefix}_{n}_s_in = {L['s_in']:.8e}f;")
    Path(path).write_text("\n".join(lines))
    return total


def int8_conv1d(x_float, L):
    xq = np.clip(np.round(x_float / L["s_in"]), -QMAX, QMAX).astype(np.int32)
    w = L["w"].astype(np.int32)
    cout, cin_g, k = w.shape
    g, d, p = L["groups"], L["dilation"], L["padding"]
    xp = np.pad(xq, ((0, 0), (0, 0), (p, p)))
    T = xq.shape[-1] + 2 * p - d * (k - 1)
    acc = np.zeros((xq.shape[0], cout, T), dtype=np.int32)
    for o in range(cout):
        for c in range(cin_g):
            ch = (o // (cout // g)) * cin_g + c
            for j in range(k):
                acc[:, o] += w[o, c, j] * xp[:, ch, j * d:j * d + T]
    if L["b"] is not None:
        acc += L["b"][None, :, None]
    return acc * (L["s_in"] * L["s_w"])[None, :, None]


# %% CCAF attention fusion
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


def train_ccaf(ppg, hr, cfg):
    torch.manual_seed(cfg.seed)
    model = CCAF(cfg.attn_ch, cfg.attn_kernel, cfg.attn_smooth)
    X, Y = torch.from_numpy(ppg), torch.from_numpy(hr).float()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.attn_lr)

    def run(epochs, lr, name):
        for gr in opt.param_groups:
            gr["lr"] = lr
        t0 = time.time()
        for ep in range(epochs):
            model.train()
            perm = torch.randperm(len(X))
            tot = 0.0
            for i in range(0, len(X), 64):
                b = perm[i:i + 64]
                xb = X[b]
                if torch.rand(1) < 0.5:
                    xb = xb.flip(1)
                y, a, g = model(xb, return_maps=True)
                loss = -spectral_snr_db(y, Y[b], cfg.fs).mean() + 2.0 * F.relu(0.5 - g.mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if (ep + 1) % LOG_EVERY == 0 or ep + 1 == epochs:
                log(f"    {name:12s} epoch {ep + 1:3d}/{epochs}  loss {tot / len(X):8.3f}  ({time.time() - t0:5.1f} s)")

    run(cfg.attn_epochs, cfg.attn_lr, "CCAF fp32")
    set_qat(model, True)
    calibrate(model, [X[i:i + 256] for i in range(0, len(X), 256)])
    run(cfg.attn_qat_epochs, cfg.attn_lr / 10, "CCAF int8")
    return model.eval()


@torch.no_grad()
def fuse(model, ppg, return_maps=False):
    out = model.eval()(torch.from_numpy(ppg), return_maps=True)
    return tuple(o.numpy() for o in out) if return_maps else out[0].numpy()


# %% EEMD-lite (used only if USE_EEMD = True)
def _extrema(h):
    d = np.diff(h)
    mx = np.where((d[:-1] > 0) & (d[1:] <= 0))[0] + 1
    mn = np.where((d[:-1] < 0) & (d[1:] >= 0))[0] + 1
    return mx, mn


def _envelope(idx, h, n):
    t = np.concatenate([[-idx[0]], idx, [2 * (n - 1) - idx[-1]]])
    v = np.concatenate([[h[idx[0]]], h[idx], [h[idx[-1]]]])
    return CubicSpline(t, v, bc_type="natural")(np.arange(n))


def emd(x, max_imfs=5, n_sift=6):
    n = len(x)
    imfs = np.zeros((max_imfs, n))
    r = x.copy()
    for i in range(max_imfs):
        h, ok = r.copy(), True
        for _ in range(n_sift):
            mx, mn = _extrema(h)
            if len(mx) < 2 or len(mn) < 2:
                ok = False
                break
            h = h - 0.5 * (_envelope(mx, h, n) + _envelope(mn, h, n))
        if not ok and i > 0:
            break
        imfs[i] = h
        r = r - h
    return imfs


class EEMDLite:
    def __init__(self, cfg):
        self.cfg = cfg
        half = np.random.default_rng(cfg.seed).standard_normal((cfg.eemd_trials // 2, cfg.win))
        self.noise = np.concatenate([half, -half])
        self.win = np.hamming(cfg.win)
        self.freqs = np.fft.rfftfreq(cfg.nfft, 1 / cfg.fs)

    def __call__(self, x):
        c = self.cfg
        s = c.eemd_noise * x.std()
        imfs = np.mean([emd(x + s * nz, c.eemd_max_imfs, c.eemd_sifts) for nz in self.noise], axis=0)
        P = np.abs(np.fft.rfft(imfs * self.win, n=c.nfft)) ** 2
        fpk, ppk = self.freqs[P.argmax(-1)], P.max(-1)
        keep = (fpk >= c.imf_band[0]) & (fpk <= c.imf_band[1]) & (ppk > 1e-6 * ppk.max())
        if not keep.any():
            keep[np.argmax(ppk)] = True
        return imfs[keep].sum(0)


# %% spectral subtraction
class SpectralSubtraction:
    def __init__(self, cfg):
        self.cfg = cfg
        self.w = np.hanning(cfg.win)
        self.f = np.fft.rfftfreq(cfg.nfft, 1 / cfg.fs)
        self.band = (self.f >= cfg.ss_band[0]) & (self.f <= cfg.ss_band[1])
        self.grid = np.linspace(cfg.bpm_min, cfg.bpm_max, cfg.n_bins)
        k = np.arange(len(self.f))
        self.h2, self.h3 = np.minimum(2 * k, len(k) - 1), np.minimum(3 * k, len(k) - 1)

    def spectra(self, pulse, acc):
        Yp = np.abs(np.fft.rfft(pulse * self.w, n=self.cfg.nfft))
        Ya = np.abs(np.fft.rfft(acc * self.w, n=self.cfg.nfft)).sum(0)
        Yp_n = Yp / (Yp[self.band].sum() + 1e-12)
        Ya_n = Ya / (Ya[self.band].sum() + 1e-12)
        ss = np.maximum(Yp_n - Ya_n, 0.0) if self.cfg.ss_subtract else Yp_n      # switch: ACC subtraction
        if self.cfg.ss_harmonics:                                                 # switch: harmonic boost
            a1, a2, a3 = self.cfg.harm_w
            ss = a1 * ss + a2 * ss[self.h2] + a3 * ss[self.h3]
        return ss, Yp_n, Ya_n

    def to_grid(self, s):
        g = np.interp(self.grid / 60.0, self.f, s)
        return g / (g.max() + 1e-12)

    def __call__(self, pulse, acc):
        return np.stack([self.to_grid(s) for s in self.spectra(pulse, acc)]).astype(np.float32)


def short_time_psd(x, fs=25, sub_sec=3.0, hop_sec=0.25):
    n = int(sub_sec * fs)
    f, t, Z = stft(x, fs=fs, window="hann", nperseg=n, noverlap=n - int(hop_sec * fs), nfft=512,
                   boundary=None, padded=False)
    return f, t, np.abs(Z) ** 2


def short_time_snr(f, P, hr_bpm, delta_hz):
    f0 = hr_bpm / 60.0
    sig = (np.abs(f - f0) <= delta_hz) | (np.abs(f - 2 * f0) <= delta_hz)
    inb = (f >= 0.5) & (f <= 8.0)
    return 10 * np.log10((P[sig & inb].sum(0) + 1e-12) / (P[~sig & inb].sum(0) + 1e-12))


# %% EMA channel (used only if USE_EMA = True) + feature extraction
def add_ema(feats, beta):
    ema = np.empty_like(feats[:, :1])
    acc = feats[0, 0]
    for i in range(len(feats)):
        acc = beta * acc + (1 - beta) * feats[i, 0] if i else feats[0, 0]
        ema[i, 0] = acc / (acc.max() + 1e-12)
    return np.concatenate([feats, ema], 1)


def extract_features(sig1, acc, cfg):
    ss = SpectralSubtraction(cfg)
    if cfg.use_eemd:                                          # EEMD-lite
        eemd = EEMDLite(cfg)
        pulses = np.stack([eemd(x) for x in sig1])
    else:
        pulses = sig1.copy()
    feats = np.stack([ss(p, a) for p, a in zip(pulses, acc)])  # (n, 3, 128) = [SS, PPG, ACC]
    if cfg.use_ema:                                           # EMA - (n, 4, 128)
        feats = add_ema(feats, cfg.ema_beta)
    return pulses, feats


# %% TinyTracker CNN + HMM
class DSBlock(nn.Module):
    def __init__(self, cin, cout, k, dil):
        super().__init__()
        self.dw = QConv1d(cin, cin, k, padding=dil * (k // 2), dilation=dil, groups=cin)
        self.pw = QConv1d(cin, cout, 1)

    def forward(self, x):
        return F.relu(self.pw(self.dw(x)))


class TinyTracker(nn.Module):
    def __init__(self, in_ch, width, k, dilations):
        super().__init__()
        chans = [in_ch] + [width] * len(dilations)
        self.blocks = nn.Sequential(*[DSBlock(chans[i], chans[i + 1], k, d) for i, d in enumerate(dilations)])
        self.head = QConv1d(width, 1, 1)

    def forward(self, x):
        return self.head(self.blocks(x)).squeeze(1)


def make_teacher(cfg): return TinyTracker(cfg.n_in, cfg.teacher_width, 9, (1, 2, 4, 8))
def make_student(cfg): return TinyTracker(cfg.n_in, cfg.student_width, 7, (1, 2, 4, 8))


def prior_bump(grid, hr_prev, sigma):
    hr_prev = np.asarray(hr_prev, dtype=np.float64)
    p = np.exp(-0.5 * ((grid[None, :] - np.nan_to_num(hr_prev)[:, None]) / sigma) ** 2)
    p[np.isnan(hr_prev)] = 1.0
    return p.astype(np.float32)


def decode(logits, grid, half=3):
    logits = np.atleast_2d(logits)
    out = np.empty(len(logits))
    for i, ki in enumerate(logits.argmax(-1)):
        lo, hi = max(ki - half, 0), min(ki + half + 1, logits.shape[-1])
        z = logits[i, lo:hi]
        p = np.exp(z - z.max())
        out[i] = (p * grid[lo:hi]).sum() / p.sum()
    return out


def soft_target(hr, grid, sigma_bins=1.5):
    t = np.exp(-0.5 * ((grid[None, :] - hr[:, None]) / (sigma_bins * (grid[1] - grid[0]))) ** 2)
    return (t / t.sum(-1, keepdims=True)).astype(np.float32)


class HMMFilter:
    def __init__(self, grid, trans_sigma=5.0, eps=1e-2, init=None):
        d = grid[:, None] - grid[None, :]
        K = np.exp(-0.5 * (d / trans_sigma) ** 2)
        K[np.abs(d) > 4 * trans_sigma] = 0
        self.K, self.eps, self.grid, self.belief = K / K.sum(0, keepdims=True), eps, grid, None
        if init is not None:
            b = np.exp(-0.5 * ((grid - init[0]) / init[1]) ** 2)
            self.belief = b / b.sum()

    def update(self, logits):
        p = np.exp(logits - logits.max())
        p = p / p.sum()
        p = (1 - self.eps) * p + self.eps / len(p)
        b = p if self.belief is None else p * (self.K @ self.belief)
        self.belief = b / b.sum()
        return decode(np.log(self.belief + 1e-30)[None], self.grid)[0]


@torch.no_grad()
def track(model, feats, cfg, grid, hmm=None, start_prior=None):
    """Run the tracker over ONE recording in time order. hmm / start_prior default to the cfg switches."""
    hmm = cfg.use_hmm if hmm is None else hmm
    start_prior = cfg.hmm_start_prior if start_prior is None else start_prior
    model.eval()
    est = np.empty(len(feats))
    prev = np.nan
    init = getattr(model, "hmm_init", None) if start_prior else None
    filt = HMMFilter(grid, cfg.trans_sigma, cfg.hmm_eps, init=init) if hmm else None
    for i in range(len(feats)):
        hint_hr = prev if cfg.use_hint else np.nan                     # switch: hint
        x = np.concatenate([feats[i], prior_bump(grid, [hint_hr], cfg.prior_sigma)], 0)[None]
        lg = model(torch.from_numpy(x)).numpy()
        est[i] = filt.update(lg[0]) if hmm else decode(lg, grid)[0]
        prev = est[i]
    return est


def track_classical(feats, grid, search=12.0):
    """TROIKA-like rule tracker (no CNN, no HMM): largest SS peak within +/- 12 BPM of the last answer."""
    est = np.empty(len(feats))
    prev = np.nan
    for i, Fi in enumerate(feats):
        s = Fi[0]
        if np.isnan(prev):
            k = s.argmax()
        else:
            m = np.abs(grid - prev) <= search
            k = np.where(m)[0][s[m].argmax()]
        est[i] = prev = grid[k]
    return est


# %% Tracker training — teacher -> DAgger -> KD student -> pruning -> int8 QAT
def _noisy_prev(hr_prev, rng):
    n, u = len(hr_prev), rng.random(len(hr_prev))
    noise = np.where(u < 0.7, rng.normal(0, 2, n), np.where(u < 0.95, rng.normal(0, 6, n), rng.uniform(-40, 40, n)))
    p = hr_prev + noise
    p[rng.random(n) < 0.03] = np.nan
    return p


class TrackerData:
    def __init__(self, feats_list, hr_list):
        self.feats = feats_list
        self.F = torch.from_numpy(np.concatenate(feats_list))
        self.HR = np.concatenate(hr_list)
        self.prev_gt = np.concatenate([np.r_[np.nan, h[:-1]] for h in hr_list])
        self.self_prev = None

    def priors(self, rng, grid, cfg):
        if not cfg.use_hint:                                            # switch: hint always flat
            return torch.ones(len(self.HR), len(grid))
        p = _noisy_prev(np.where(np.isnan(self.prev_gt), self.HR, self.prev_gt), rng)
        p[np.isnan(self.prev_gt)] = np.nan
        if self.self_prev is not None:
            use = rng.random(len(p)) < 0.5
            p[use] = self.self_prev[use]
        return torch.from_numpy(prior_bump(grid, p, cfg.prior_sigma))


def _fit(model, data, cfg, grid, epochs, lr, name, teacher=None, T=2.0, alpha=0.7, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    tgt = torch.from_numpy(soft_target(data.HR, grid))
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        X = torch.cat([data.F, data.priors(rng, grid, cfg)[:, None]], 1)
        if teacher is not None:
            with torch.no_grad():
                t_logits = teacher(X)
        perm = torch.randperm(len(X))
        tot = 0.0
        for i in range(0, len(X), 128):
            b = perm[i:i + 128]
            lg = model(X[b])
            loss = -(tgt[b] * F.log_softmax(lg, -1)).sum(-1).mean()
            if teacher is not None:
                kd = F.kl_div(F.log_softmax(lg / T, -1), F.softmax(t_logits[b] / T, -1), reduction="batchmean")
                loss = (1 - alpha) * loss + alpha * T * T * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        sched.step()
        if (ep + 1) % LOG_EVERY == 0 or ep + 1 == epochs:
            log(f"    {name:12s} epoch {ep + 1:3d}/{epochs}  loss {tot / len(X):8.4f}  ({time.time() - t0:5.1f} s)")
    return model


def _dagger(model, data, cfg, grid, name):
    if not cfg.use_dagger:                                              # switch: no DAgger
        log(f"    {name:12s} DAgger skipped (USE_DAGGER = False)")
        return
    t0 = time.time()
    data.self_prev = np.concatenate([np.r_[np.nan, track(model, f, cfg, grid)[:-1]] for f in data.feats])
    log(f"    {name:12s} DAgger: recorded own predictions ({time.time() - t0:.1f} s)")


def train_tracker_suite(feats_list, hr_list, cfg):
    grid = np.linspace(cfg.bpm_min, cfg.bpm_max, cfg.n_bins)
    data = TrackerData(feats_list, hr_list)
    init = (float(np.mean([h[0] for h in hr_list])), cfg.init_sd)

    teacher = make_teacher(cfg)
    teacher.hmm_init = init
    _fit(teacher, data, cfg, grid, cfg.tracker_epochs, cfg.tracker_lr, "teacher", seed=cfg.seed)
    _dagger(teacher, data, cfg, grid, "teacher")
    _fit(teacher, data, cfg, grid, cfg.tracker_epochs // 2, cfg.tracker_lr / 3, "teacher", seed=cfg.seed + 1)

    kd_t = teacher if cfg.use_kd else None                             # switch: knowledge distillation
    student = make_student(cfg)
    student.hmm_init = init
    _fit(student, data, cfg, grid, cfg.kd_epochs, cfg.tracker_lr, "student", teacher=kd_t, seed=cfg.seed + 2)
    _dagger(student, data, cfg, grid, "student")
    _fit(student, data, cfg, grid, cfg.kd_epochs // 2, cfg.tracker_lr / 3, "student", teacher=kd_t, seed=cfg.seed + 3)

    pruned = copy.deepcopy(student)
    apply_pruning(pruned, cfg.prune_amount)
    _fit(pruned, data, cfg, grid, cfg.finetune_epochs, cfg.tracker_lr / 3, "pruned", teacher=kd_t, seed=cfg.seed + 4)
    remove_pruning(pruned)

    q = copy.deepcopy(pruned)
    mask = {n: (p != 0).float() for n, p in q.named_parameters() if p.dim() == 3}
    set_qat(q, True)
    Xc = torch.cat([data.F, torch.from_numpy(prior_bump(grid, data.HR, cfg.prior_sigma))[:, None]], 1)
    calibrate(q, [Xc[i:i + 512] for i in range(0, len(Xc), 512)])
    hooks = [p.register_hook(lambda g, m=mask[n]: g * m) for n, p in q.named_parameters() if n in mask]
    _fit(q, data, cfg, grid, cfg.finetune_epochs, cfg.tracker_lr / 10, "int8 QAT", teacher=kd_t, seed=cfg.seed + 5)
    for h in hooks:
        h.remove()
    return {"teacher_fp32": teacher, "student_fp32": student, "student_pruned": pruned, "student_int8": q}, grid


def int8_reference_forward(model, x):
    L = quantized_layers(model)
    h = x.astype(np.float64)
    for i in range(0, len(L) - 1, 2):
        h = np.maximum(int8_conv1d(int8_conv1d(h, L[i]), L[i + 1]), 0)
    return int8_conv1d(h, L[-1])[:, 0]


# %% Cross-validation (4 folds, leave 3 subjects out)
def metrics(est, hr):
    e = est - hr
    return {"MAE": np.mean(np.abs(e)), "AAE%": 100 * np.mean(np.abs(e) / hr),
            "RMSE": np.sqrt(np.mean(e ** 2)), "r": np.corrcoef(est, hr)[0, 1]}


def run_cv(ds, cfg):
    res = {"est": {}, "fused": {}, "pulse": {}, "feats": {}, "models": {}}
    for k, test in enumerate(cfg.folds):
        tf = time.time()
        train = [p for p in ds if p not in test]
        log(f"===== FOLD {k + 1}/4   train {train}   test {test} =====")

        if cfg.fusion == "ccaf":                                        # switch: attention / one channel / mean
            log("  [1] training CCAF attention")
            attn = train_ccaf(np.concatenate([ds[p]["ppg"] for p in train]),
                              np.concatenate([ds[p]["hr"] for p in train]), cfg)
            fused = {p: fuse(attn, ds[p]["ppg"]) for p in ds}
        else:
            log(f"  [1] attention OFF -> using '{cfg.fusion}'")
            attn = None
            pick = {"ch1": lambda x: x[:, 0].copy(), "ch2": lambda x: x[:, 1].copy(),
                    "mean": lambda x: zscore(x.mean(1))}[cfg.fusion]
            fused = {p: pick(ds[p]["ppg"]) for p in ds}

        t0 = time.time()
        out = Parallel(n_jobs=-1 if cfg.use_eemd else 1)(
            delayed(extract_features)(fused[p], ds[p]["acc"], cfg) for p in ds)
        pulses = {p: o[0] for p, o in zip(ds, out)}
        feats = {p: o[1] for p, o in zip(ds, out)}
        log(f"  [2]/[3] features (EEMD={cfg.use_eemd}, EMA={cfg.use_ema}) shape {feats[test[0]].shape[1:]} "
            f"({time.time() - t0:.1f} s)")

        log("  [4] training tracker: teacher -> student (KD) -> pruned -> int8")
        models, grid = train_tracker_suite([feats[p] for p in train], [ds[p]["hr"] for p in train], cfg)
        res["models"][k] = {"attn": attn, **models}

        log("  testing on unseen subjects")
        for p in test:
            t0 = time.time()
            res["fused"][p], res["pulse"][p], res["feats"][p] = fused[p], pulses[p], feats[p]
            res["est"][p] = {n: track(m, feats[p], cfg, grid) for n, m in models.items()}
            # decoding variants of the SAME int8 model (no retraining needed) -> used by the ablation study
            res["est"][p]["int8 no HMM"] = track(models["student_int8"], feats[p], cfg, grid, hmm=False)
            res["est"][p]["int8 HMM flat start"] = track(models["student_int8"], feats[p], cfg, grid, start_prior=False)
            res["est"][p]["rule tracker"] = track_classical(feats[p], grid)
            mae = {n: round(float(metrics(res["est"][p][n], ds[p]["hr"])["MAE"]), 2) for n in models}
            log(f"    subject {p}: MAE {mae}  ({time.time() - t0:.1f} s)")
        log(f"  fold {k + 1} done in {time.time() - tf:.0f} s")
    res["grid"] = grid
    return res


# %% Parameter and operation count per block
def _conv_counts(model, example):
    macs = [0, 0]                                            # [dense MACs, MACs with non-zero weights]

    def hook(m, i, o):
        n_run = o.shape[0]
        macs[0] += n_run * o.shape[-1] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0]
        macs[1] += n_run * o.shape[-1] * int((m.weight != 0).sum())

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv1d)]
    with torch.no_grad():
        model.eval()
        model(example)
    for h in hs:
        h.remove()
    params = sum(p.numel() for p in model.parameters())
    conv_w = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Conv1d))
    return params, conv_w, macs[0], macs[1]


def params_ops_table(cfg, models=None, save=True):
    T, nb, hop = cfg.win, cfg.n_bins, cfg.step_sec * cfg.fs_raw
    fft = 2.5 * cfg.nfft * np.log2(cfg.nfft)                     # one real FFT
    n_sos = butter(cfg.bp_order, [cfg.bp_low, cfg.bp_high], btype="bandpass", fs=cfg.fs_raw, output="sos").shape[0]
    rows = []

    def add(block, params, nonzero, mem_bytes, ops, note):
        rows.append({"block": block, "parameters": int(params), "non-zero parameters": int(nonzero),
                     "memory (bytes, int8)": int(mem_bytes), "operations per 2-s update": int(round(ops)),
                     "est. Cortex-M4 80 MHz (ms)": ops / 80e3, "note": note})

    # [0] preprocessing: 5 channels x 250 new samples x biquads x 5 MAC
    add("[0] Pre-processing", 0, 0, 0, 5 * hop * n_sos * 5, f"{n_sos} biquads/channel, fixed coefficients")

    # [1] fusion
    if cfg.fusion == "ccaf":
        attn = models["attn"] if models else CCAF(cfg.attn_ch, cfg.attn_kernel, cfg.attn_smooth)
        p, conv_w, macs, _ = _conv_counts(attn, torch.zeros(1, 2, T))
        C, S = cfg.attn_ch, cfg.attn_smooth
        elementwise = C * T + 2 * C * T + 2 * S * T + 2 * T + 4 * T   # agree, |diff|, 2 moving averages, 2 sigmoids, mix
        add("[1] CCAF attention", p, p, conv_w + 4 * (p - conv_w), macs + elementwise,
            f"{macs:,} conv MACs + {elementwise:,} element-wise")
    else:
        add(f"[1] fusion = {cfg.fusion}", 0, 0, 0, 2 * T if cfg.fusion == "mean" else 0, "attention switched off")

    # [2] EEMD-lite
    if cfg.use_eemd:
        knots = 2 * 8 * cfg.win_sec // 2
        spline = 8 * knots + 4 * T
        sift = 2 * spline + 3 * T
        ops = cfg.eemd_trials * cfg.eemd_max_imfs * cfg.eemd_sifts * sift + cfg.eemd_max_imfs * fft
        add("[2] EEMD-lite", 0, 0, 0, ops,
            f"{cfg.eemd_trials} trials x {cfg.eemd_max_imfs} IMFs x {cfg.eemd_sifts} sifts")
    else:
        add("[2] EEMD-lite", 0, 0, 0, 0, "switched off")

    # [3] spectral subtraction: 4 FFTs (1 PPG + 3 ACC) + vector ops + resampling to the 128-bin grid
    add("[3] Spectral subtraction", 0, 0, 0, 4 * fft + 10 * cfg.nfft + 3 * 4 * nb, f"4 FFTs of {cfg.nfft}")

    # EMA channel
    add("EMA channel", 0, 0, 0, 3 * nb if cfg.use_ema else 0, "on" if cfg.use_ema else "switched off")

    # [4] hint + CNN (deployed: pruned int8 student)
    add("[4] Hint (prior bump)", 0, 0, 0, 4 * nb if cfg.use_hint else 0, "Gaussian at previous HR")
    x = torch.zeros(1, cfg.n_in, nb)
    if models:
        q = models["student_int8"]
        p, conv_w, dense, sparse = _conv_counts(q, x)
        nz = sum(int((t != 0).sum()) for t in q.parameters())
    else:
        st = make_student(cfg)
        p, conv_w, dense, _ = _conv_counts(st, x)
        prunable_w = sum(m.weight.numel() for m, _ in prunable(st))
        removed = int(round(cfg.prune_amount * prunable_w))
        nz = p - removed
        sparse = nb * (conv_w - removed)                       # every tracker conv runs once over the 128 bins
    add("[4] TinyTracker student (int8, deployed)", p, nz, conv_w + 4 * (p - conv_w), sparse,
        f"{dense:,} MACs dense, {sparse:,} with non-zero weights")

    # [5] HMM forward filter: banded transition mat-vec + softmax/floor/normalise
    if cfg.use_hmm:
        grid = np.linspace(cfg.bpm_min, cfg.bpm_max, nb)
        k_nz = int((HMMFilter(grid, cfg.trans_sigma, cfg.hmm_eps).K != 0).sum())
        add("[5] HMM filter", 0, 0, 0, k_nz + 6 * nb, f"{k_nz:,} non-zero transitions (banded)")
    else:
        add("[5] HMM filter", 0, 0, 0, 0, "switched off")

    tab = pd.DataFrame(rows)
    total = {c: tab[c].sum() for c in ["parameters", "non-zero parameters", "memory (bytes, int8)",
                                       "operations per 2-s update", "est. Cortex-M4 80 MHz (ms)"]}
    tab.loc[len(tab)] = {"block": "TOTAL (deployed pipeline)", **total, "note": ""}

    # reference only (not in the total): teacher model and the original TROIKA blocks
    tp, _, tmacs, _ = _conv_counts(models["teacher_fp32"] if models else make_teacher(cfg), x)
    L, K = 100, T - 100 + 1
    ref = [("ref: TinyTracker teacher (fp32)", tp, tp, 4 * tp, tmacs, "not deployed"),
           ("ref: TROIKA SSA (one channel)", 0, 0, 0, 4 * L * L * K + 22 * L ** 3 + L * L * K, "SVD + reconstruction"),
           ("ref: TROIKA FOCUSS", 0, 0, 0, 5 * (T * T * 2 * 4096 + T ** 3 / 3 + 2 * T * 2 * 4096), "5 iterations, N=4096")]
    for b, p_, nz_, mem, ops, note in ref:
        tab.loc[len(tab)] = {"block": b, "parameters": p_, "non-zero parameters": nz_, "memory (bytes, int8)": mem,
                             "operations per 2-s update": int(ops), "est. Cortex-M4 80 MHz (ms)": ops / 80e3, "note": note}

    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 60)
    print("\nPARAMETERS AND OPERATIONS PER BLOCK  (switches: "
          f"FUSION={cfg.fusion} EEMD={cfg.use_eemd} EMA={cfg.use_ema} HINT={cfg.use_hint} HMM={cfg.use_hmm})\n")
    print(tab.round(3).to_string(index=False))
    if save:
        path = OUT_DIR / "parameters_and_operations.csv"
        tab.round(4).to_csv(path, index=False)
        log(f"saved {path}")
    return tab


if RUN_PARAMS_OPS_ONLY:                                    # test run: table only, no data loading, no training
    params_ops_table(cfg)
    raise SystemExit("RUN_PARAMS_OPS_ONLY = True -> stopped after the parameter/operation table")


# %% RUN
log("loading and preprocessing data ...")
ds = build_dataset(cfg)
log(f"loaded {len(ds)} subjects, {sum(len(d['hr']) for d in ds.values())} windows")
res = run_cv(ds, cfg)
log("cross-validation finished")

# %% Results — accuracy
pids = sorted(ds)
stages = ["teacher_fp32", "student_fp32", "student_pruned", "student_int8"]
mae = pd.DataFrame({s: [metrics(res["est"][p][s], ds[p]["hr"])["MAE"] for p in pids] for s in stages}, index=pids)
mae["TROIKA (paper)"] = [2.29, 2.19, 2.00, 2.15, 2.01, 2.76, 1.67, 1.93, 1.86, 4.70, 1.72, 2.84]
mae.loc["mean"], mae.loc["std"] = mae.iloc[:12].mean(), mae.iloc[:12].std()
pd.set_option("display.width", 200)
print("\nMAE (BPM) per subject\n", mae.round(2).to_string())
mae.round(3).to_csv(OUT_DIR / "mae_per_subject.csv")
allest = np.concatenate([res["est"][p]["student_int8"] for p in pids])
allhr = np.concatenate([ds[p]["hr"] for p in pids])
ov = metrics(allest, allhr)
print("int8 all windows:", {k: round(float(v), 3) for k, v in ov.items()},
      f"| within 5 %: {100 * np.mean(np.abs(allest - allhr) / allhr <= 0.05):.1f} %")

# %% Attention SNR (before: ch-1, ch-2 / after: single output)
snr = {"ch1": [], "ch2": [], "mean": [], "attention": []}
for p in pids:
    snr["ch1"].append(snr_np(ds[p]["ppg"][:, 0], ds[p]["hr"]))
    snr["ch2"].append(snr_np(ds[p]["ppg"][:, 1], ds[p]["hr"]))
    snr["mean"].append(snr_np(zscore(ds[p]["ppg"].mean(1)), ds[p]["hr"]))
    snr["attention"].append(snr_np(res["fused"][p], ds[p]["hr"]))
snr_tab = pd.DataFrame({n: [np.median(v) for v in vals] for n, vals in snr.items()}, index=pids)
snr_tab.loc["median (all windows)"] = {n: np.median(np.concatenate(v)) for n, v in snr.items()}
print("\nmedian SNR (dB)\n", snr_tab.round(2).to_string())
snr_tab.round(3).to_csv(OUT_DIR / "snr_per_subject.csv")

# %% Figures (saved to results_all_in_one/)
fig, axs = plt.subplots(3, 4, figsize=(15, 8), sharey=True)
for ax, p in zip(axs.ravel(), pids):
    t = np.arange(len(ds[p]["hr"])) * 2
    ax.plot(t, ds[p]["hr"], "k", lw=1.8, label="ECG ground truth")
    ax.plot(t, res["est"][p]["student_int8"], color="#1baf7a", lw=1.2, label="int8 pipeline")
    ax.set_title(f"subject {p}  MAE {mae.loc[p, 'student_int8']:.2f} BPM")
axs[0, 0].legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "hr_tracks.png", dpi=130)

subj, seg = "01", 60
sigs = {"PPG ch-1": ds[subj]["ppg"][seg, 0], "PPG ch-2": ds[subj]["ppg"][seg, 1],
        f"fused output ({cfg.fusion})": res["fused"][subj][seg]}
hr = ds[subj]["hr"][seg]
fig, axs = plt.subplots(2, 3, figsize=(14, 6))
for j, (n, s) in enumerate(sigs.items()):
    axs[0, j].plot(np.arange(cfg.win) / cfg.fs, s)
    axs[0, j].set_title(n)
    f, tt, P = short_time_psd(s, cfg.fs)
    m = f <= 6
    axs[1, j].pcolormesh(tt, f[m] * 60, 10 * np.log10(P[m] / P.max() + 1e-6), shading="auto", cmap="Greys",
                         vmin=-30, vmax=0)
    axs[1, j].axhline(hr, color="r")
    axs[1, j].set_title(f"3 s PSD, mean SNR {short_time_snr(f, P, hr, 1 / 3).mean():.1f} dB")
fig.suptitle(f"subject {subj}, segment {seg}, true HR {hr:.1f} BPM")
fig.tight_layout()
fig.savefig(OUT_DIR / "attention_segment.png", dpi=130)
plt.close("all")

# %% Edge profile + C export + integer check (fold-1 models)
m0 = res["models"][0]
prof = {"teacher fp32": profile(m0["teacher_fp32"], torch.zeros(1, cfg.n_in, cfg.n_bins)),
        "student fp32": profile(m0["student_fp32"], torch.zeros(1, cfg.n_in, cfg.n_bins)),
        "student pruned int8": profile(m0["student_int8"], torch.zeros(1, cfg.n_in, cfg.n_bins), int8=True)}
if m0["attn"] is not None:
    prof = {"CCAF attention (int8)": profile(m0["attn"], torch.zeros(1, 2, cfg.win), int8=True), **prof}
    print("C header bytes: attention", export_c_header(m0["attn"], OUT_DIR / "ccaf_int8.h", "ccaf"))
print("\nmodel sizes\n", pd.DataFrame(prof).T.round(3).to_string())
print("C header bytes: tracker", export_c_header(m0["student_int8"], OUT_DIR / "tracker_int8.h", "trk"))
p0 = cfg.folds[0][0]
xin = np.concatenate([res["feats"][p0][:32], prior_bump(res["grid"], ds[p0]["hr"][:32], cfg.prior_sigma)[:, None]], 1)
with torch.no_grad():
    ref = m0["student_int8"](torch.from_numpy(xin)).numpy()
print(f"integer-only vs QAT: argmax agreement "
      f"{100 * np.mean(ref.argmax(-1) == int8_reference_forward(m0['student_int8'], xin).argmax(-1)):.0f} %")
params_ops_table(cfg, m0)      

# %% Real-time streaming monitor + latency
class StreamingHRMonitor:
    def __init__(self, attn, tracker, cfg):
        self.cfg, self.tracker = cfg, tracker.eval()
        self.attn = attn.eval() if attn is not None else None
        self.pre = StreamingPreprocessor(cfg, 5)
        self.ss = SpectralSubtraction(cfg)
        self.eemd = EEMDLite(cfg) if cfg.use_eemd else None
        self.ema = None
        self.buf, self.filled, self.prev = np.zeros((5, cfg.win)), 0, np.nan
        init = getattr(tracker, "hmm_init", None) if cfg.hmm_start_prior else None
        self.hmm = HMMFilter(self.ss.grid, cfg.trans_sigma, cfg.hmm_eps, init=init) if cfg.use_hmm else None
        self.timing = {k: [] for k in ["preproc", "attention", "eemd", "spectral_sub", "tracker"]}

    @torch.no_grad()
    def push(self, chunk):
        t = time.perf_counter()
        x = self.pre.process(chunk)
        self.buf = np.concatenate([self.buf[:, x.shape[1]:], x], 1)
        self.filled += x.shape[1]
        t1 = time.perf_counter(); self.timing["preproc"].append(t1 - t)
        if self.filled < self.cfg.win:
            return None
        w = zscore(self.buf).astype(np.float32)
        if self.cfg.fusion == "ccaf":
            y = self.attn(torch.from_numpy(w[None, :2])).numpy()[0]
        elif self.cfg.fusion in ("ch1", "ch2"):
            y = w[0 if self.cfg.fusion == "ch1" else 1].copy()
        else:
            y = zscore(w[:2].mean(0)).astype(np.float32)
        t2 = time.perf_counter(); self.timing["attention"].append(t2 - t1)
        if self.cfg.use_eemd:
            y = self.eemd(y)
        t3 = time.perf_counter(); self.timing["eemd"].append(t3 - t2)
        f = self.ss(y, w[2:])
        if self.cfg.use_ema:
            b = self.cfg.ema_beta
            self.ema = f[0] if self.ema is None else b * self.ema + (1 - b) * f[0]
            f = np.concatenate([f, (self.ema / (self.ema.max() + 1e-12))[None]], 0).astype(np.float32)
        t4 = time.perf_counter(); self.timing["spectral_sub"].append(t4 - t3)
        hint_hr = self.prev if self.cfg.use_hint else np.nan
        xin = np.concatenate([f, prior_bump(self.ss.grid, [hint_hr], self.cfg.prior_sigma)], 0)[None]
        lg = self.tracker(torch.from_numpy(xin)).numpy()
        hr = self.hmm.update(lg[0]) if self.hmm is not None else decode(lg, self.ss.grid)[0]
        self.timing["tracker"].append(time.perf_counter() - t4)
        self.prev = hr
        return hr

    def run(self, raw):
        hop, first = self.cfg.step_sec * self.cfg.fs_raw, self.cfg.win_sec * self.cfg.fs_raw
        out = [self.push(raw[:, :first])] + [self.push(raw[:, s:s + hop]) for s in range(first, raw.shape[1], hop)]
        return np.array([o for o in out if o is not None])

torch.set_num_threads(1)
mon = StreamingHRMonitor(m0["attn"], m0["student_int8"], cfg)
est_stream = mon.run(load_subject(p0, cfg)["raw"])
print(f"\nstreaming vs offline (subject {p0}): max difference "
      f"{np.abs(est_stream - res['est'][p0]['student_int8']).max():.2e} BPM")
lat = pd.Series({k: 1e3 * np.median(v) for k, v in mon.timing.items()}, name="ms per 2 s update")
lat["total"] = lat.sum()
print(lat.round(3).to_string())
log(f"MAIN RUN DONE — results in {OUT_DIR}")


# %% ABLATION STUDY (runs only if RUN_ABLATION = True)
ABLATIONS = [
    # name                         block                     switches changed
    ("A1 ch-1 only",               "[1] attention",          {"fusion": "ch1"}),
    ("A2 ch-2 only",               "[1] attention",          {"fusion": "ch2"}),
    ("A3 mean(ch-1,ch-2)",         "[1] attention",          {"fusion": "mean"}),
    ("B no EEMD",                  "[2] EEMD",               {"use_eemd": False}),
    ("C1 no ACC subtraction",      "[3] spectral sub.",      {"ss_subtract": False}),
    ("C2 no harmonic boost",       "[3] spectral sub.",      {"ss_harmonics": False}),
    ("D no EMA",                   "[4] EMA input",          {"use_ema": False}),
    ("E no hint",                  "[5] tracker input",      {"use_hint": False}),
    ("G no DAgger",                "training",               {"use_dagger": False}),
    ("H no KD",                    "training",               {"use_kd": False}),
]
# decoding ablations come from the SAME trained reference models (no retraining):
DECODING = [("F1 no HMM", "[5] decoding", "int8 no HMM"),
            ("F2 HMM flat start", "[5] decoding", "int8 HMM flat start"),
            ("F3 rule tracker (TROIKA-like)", "[5] decoding", "rule tracker")]

if RUN_ABLATION:
    from scipy.stats import wilcoxon

    def subject_maes(r, key):
        return np.array([metrics(r["est"][p][key], ds[p]["hr"])["MAE"] for p in pids])

    def build_table():
        ref_m = per_subj["reference"]
        rows = []
        for name, (block, stage_means, m) in results.items():
            row = {"variant": name, "block": block, **stage_means,
                   "int8 MAE": m.mean(), "int8 std": m.std(ddof=1),
                   "Δ vs reference": m.mean() - ref_m.mean(),
                   "worse in (subjects)": int((m - ref_m > 0.05).sum()),
                   "Wilcoxon p": np.nan if name == "reference" or np.allclose(m, ref_m) else wilcoxon(m, ref_m).pvalue}
            rows.append(row)
        return pd.DataFrame(rows).set_index("variant")

    results, per_subj = {}, {}
    m_ref = subject_maes(res, "student_int8")
    results["reference"] = ("reference", {f"MAE {s}": subject_maes(res, s).mean() for s in stages}, m_ref)
    per_subj["reference"] = m_ref
    for name, block, key in DECODING:
        m = subject_maes(res, key)
        results[name], per_subj[name] = (block, {}, m), m

    log(f"ABLATION: {len(ABLATIONS)} variants, each a full cross-validation run")
    for name, block, over in ABLATIONS:
        if all(getattr(cfg, k) == v for k, v in over.items()):
            log(f"  skip '{name}': already like this in the reference configuration")
            continue
        log(f"########## ABLATION {name}  ({over}) ##########")
        r = run_cv(ds, replace(cfg, **over))
        m = subject_maes(r, "student_int8")
        results[name] = (block, {f"MAE {s}": subject_maes(r, s).mean() for s in stages}, m)
        per_subj[name] = m
        tab = build_table()                                            # save after every variant
        tab.round(3).to_csv(OUT_DIR / "ablation_mae.csv")
        pd.DataFrame(per_subj, index=pids).T.round(2).to_csv(OUT_DIR / "ablation_per_subject.csv")
        log(f"  {name}: int8 MAE {m.mean():.2f}  (reference {m_ref.mean():.2f})  -> saved ablation_mae.csv")

    tab = build_table()
    tab.round(3).to_csv(OUT_DIR / "ablation_mae.csv")
    print("\nABLATION RESULTS (int8 deployed model)\n", tab.round(2).to_string())

    fig, ax = plt.subplots(figsize=(10, 0.5 * len(tab) + 1.5))
    y = np.arange(len(tab))[::-1]
    for yi, (n, row) in zip(y, tab.iterrows()):
        d, p = row["Δ vs reference"], row["Wilcoxon p"]
        col = "#1baf7a" if n == "reference" else ("#eb6834" if d > 0.1 else "#b4b2a9")
        ax.barh(yi, row["int8 MAE"], color=col, height=0.6)
        star = "" if n == "reference" else (" *" if p < 0.05 else " (*)" if p < 0.1 else "")
        ax.text(row["int8 MAE"] * 1.05, yi, f"{row['int8 MAE']:.2f}  Δ {d:+.2f}{star}", va="center", fontsize=8)
    ax.set_yticks(y, [f"{n}  {row['block']}" for n, row in tab.iterrows()])
    ax.set_xscale("log")
    ax.axvline(tab.loc["reference", "int8 MAE"], color="#1baf7a", ls="--", lw=1)
    ax.set_xlabel("MAE of the int8 model (BPM, log scale)   orange = removing the block makes it worse by > 0.1 BPM")
    ax.set_title("Ablation study (4-fold leave-subjects-out CV)  * p<0.05, (*) p<0.1 paired Wilcoxon", loc="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation.png", dpi=150)
    plt.close("all")
    log(f"ABLATION DONE — ablation_mae.csv, ablation_per_subject.csv, ablation.png in {OUT_DIR}")
