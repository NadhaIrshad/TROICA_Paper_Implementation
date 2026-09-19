# %% [markdown]
# # 03. Temporal difference and sparse reconstruction
#
# What the second-order difference does to the spectrum, and what FOCUSS
# resolves that the periodogram cannot.

# %% setup
# %load_ext autoreload
# %autoreload 2
from troika import lab, viz

viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg, [5])
rec = subjects[5]
trace = lab.get_window_trace(rec, cfg, w=40)

# %% [markdown]
# ## The temporal difference
#
# Differencing keeps the heartbeat fundamental and its harmonics and suppresses
# aperiodic motion. It also weights power by roughly f to the fourth, so the
# harmonics come out far stronger than the fundamental. That is deliberate for
# tracking, and it is why the first window is initialised from a spectrum taken
# before the difference (ASSUMPTION A20).

# %%
viz.plot_temporal_diff(trace)

# %% [markdown]
# ## The pruned dictionary
#
# Equation 12 keeps only the columns inside the band-pass region plus a
# transition margin: 458 of 4096, symmetric under k to N-k, with DC excluded.

# %%
viz.plot_ssr_dictionary(trace)

# %% [markdown]
# ## Five iterations
#
# The paper stops at five because the large coefficients settle quickly. The
# lower panel counts coefficients above 1 % of the peak, so you can see whether
# that holds here.

# %%
viz.plot_ssr_iterations(trace)

# %% [markdown]
# ## Sparse spectrum against the periodogram
#
# The Figure 1 comparison on real data.

# %%
viz.plot_ssr_spectrum(trace)

# %% [markdown]
# ## Variant cell: FFT instead of sparse reconstruction
#
# This is the paper's second ablation. On an easy window the two agree; on a
# window with a strong nearby motion peak the periodogram smears them together.

# %%
variants = {
    "focuss (default)": cfg,
    "fft": cfg.with_overrides({"spectrum_estimator.method": "fft"}),
}
lab.compare_variants(rec, variants, window=40, stage="spectrum_estimator")

# %% [markdown]
# ## Sweeping lambda and the iteration count
#
# Both are paper values. Larger lambda means more regularisation and a less
# sparse answer; more iterations sharpen the peaks but cost time linearly.

# %%
sweep = {
    f"lam={lam}": cfg.with_overrides({"spectrum_estimator.lam": lam})
    for lam in (0.01, 0.1, 1.0)
}
lab.compare_variants(rec, sweep, window=40, stage="spectrum_estimator")

# %%
sweep = {
    f"n_iter={n}": cfg.with_overrides({"spectrum_estimator.n_iter": n})
    for n in (1, 5, 15)
}
lab.compare_variants(rec, sweep, window=40, stage="spectrum_estimator")
