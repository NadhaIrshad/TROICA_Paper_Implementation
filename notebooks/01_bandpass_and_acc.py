# %% [markdown]
# # 01. Band-pass filtering and the accelerometer
#
# The first two things TROIKA does to a window: band-pass both the PPG and the
# accelerometer to 0.4-5 Hz, then find which frequencies the accelerometer says
# are motion.

# %% setup
# %load_ext autoreload
# %autoreload 2
from troika import lab, viz

viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg, [5])
rec = subjects[5]

# %% [markdown]
# ## Trace one window
#
# `prev_bin="tracked"` runs windows 0 to w-1 at the light level first, so the
# state feeding window w is the state a real run would have. That is the honest
# default; seeding it directly marks the trace contaminated.

# %%
trace = lab.get_window_trace(rec, cfg, w=40, prev_bin="tracked")
print(f"window {trace.idx} at {trace.t_start_s:.0f} s")
print(f"estimate {trace.bpm_est:.1f} BPM, truth {trace.gt_bpm:.1f} BPM")
print(f"stage timings (ms): { {k: round(v, 1) for k, v in trace.timings_ms.items()} }")

# %% [markdown]
# ## The band-pass stage
#
# Input light, output bold, time on the left and frequency on the right.

# %%
viz.plot_stage(trace, "bandpass")

# %% [markdown]
# ## The filter itself
#
# The magnitude is squared because the filter runs forwards and backwards, so
# this is what the signal actually sees. The phase is identically zero, which is
# why peak locations do not move.

# %%
viz.plot_filter_response(trace)

# %% [markdown]
# ## Accelerometer dominant frequencies
#
# Per axis, the peaks above 50 % of that axis's maximum. The union is F_acc.
# The red band is the plus or minus Delta zone around the previous estimate,
# which is excluded so that a cadence coinciding with the heart rate does not
# take the heartbeat with it.

# %%
viz.plot_acc_dominant(trace)

# %% [markdown]
# ## Variant cell: FIR against Butterworth
#
# The FIR alternative cannot place a sharp corner at 0.4 Hz inside an 8 s
# window, so it attenuates the bottom of the heart-rate band. The response plot
# above, drawn for each variant, shows why.

# %%
variants = {
    "butter (default)": cfg,
    "fir": cfg.with_overrides({"bandpass.method": "fir"}),
}
lab.compare_variants(rec, variants, window=40, stage="bandpass")

# %% [markdown]
# Change `stage` to see the same variants further down the pipeline; a filter
# difference that looks small here can matter by the time it reaches tracking.

# %%
lab.compare_variants(rec, variants, window=40, stage="spectrum_estimator")
