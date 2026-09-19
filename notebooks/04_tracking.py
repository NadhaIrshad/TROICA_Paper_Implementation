# %% [markdown]
# # 04. Spectral peak tracking
#
# How one peak is chosen, and what happens over a whole recording. This is where
# failures become visible.

# %% setup
# %load_ext autoreload
# %autoreload 2
import numpy as np

from troika import lab, viz

viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg, [5, 6])

# %% [markdown]
# ## One window's decision
#
# The shaded ranges are R0 around the previous estimate and R1 around its first
# harmonic. The dotted line is eta, 30 % of the highest peak in R0. Triangles are
# the kept peaks, dotted links are harmonic pairs, and the three vertical lines
# are the previous bin, the selected bin and the bin after verification.

# %%
rec = subjects[5]
trace = lab.get_window_trace(rec, cfg, w=40)
viz.plot_tracking_window(trace)

# %% [markdown]
# ## A whole recording
#
# The estimate over the spectrum it was picked from, with the error below.
# Markers show where Case 2 or 3 was taken and where a verification rule fired.

# %%
run = lab.run_subject(cfg, rec, trace="light")
viz.plot_tracking_run(run)

# %%
viz.plot_verification_timeline(run)

# %% [markdown]
# ## A subject that goes wrong
#
# Subject 6 is the paper's hard case. Find its worst window and look at the
# decision that was made there.

# %%
rec6 = subjects[6]
run6 = lab.run_subject(cfg, rec6, trace="light")
error = np.abs(run6.bpm_est - run6.bpm_gt)
worst = int(np.nanargmax(error))
print(f"worst window {worst}: estimate {run6.bpm_est[worst]:.1f}, truth {run6.bpm_gt[worst]:.1f}")
viz.plot_tracking_run(run6)

# %%
viz.plot_tracking_window(lab.get_window_trace(rec6, cfg, w=worst))

# %% [markdown]
# The accelerometer view of the same window usually explains it: if the refined
# F_acc is empty, every motion frequency fell inside the excluded zone around
# the previous estimate, so nothing was removed.

# %%
viz.plot_acc_dominant(lab.get_window_trace(rec6, cfg, w=worst))

# %% [markdown]
# ## Isolating initialisation from tracking
#
# `init_mode: ground_truth` seeds the first window from the ECG. It marks the run
# contaminated and must never be used for a reported number, but it answers one
# question cleanly: did this subject fail because it started wrong, or because it
# lost the trail later?

# %%
debug = cfg.with_overrides(
    {"tracker.init_mode": "ground_truth", "run.allow_ground_truth_access": True}
)
run6_seeded = lab.run_subject(debug, rec6)
lab.evaluate({6: run6_seeded}, cfg, allow_contaminated=True)
