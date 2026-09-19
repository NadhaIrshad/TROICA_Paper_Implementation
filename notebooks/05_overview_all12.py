# %% [markdown]
# # 05. The whole pipeline, and all twelve subjects
#
# One window end to end, then the full run and the paper's summary figures.

# %% setup
# %load_ext autoreload
# %autoreload 2
from troika import lab, viz

viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg)

# %% [markdown]
# ## One window, every stage
#
# The true heart rate is marked at the same place in every frequency panel. The
# row where it stops being the obvious peak is the row that explains a failure.

# %%
trace = lab.get_window_trace(subjects[5], cfg, w=40)
viz.plot_pipeline_overview(trace)

# %% [markdown]
# ## All twelve subjects
#
# This takes a couple of minutes: about 200 ms per window, roughly 150 windows
# per subject, spread across `run.n_jobs` workers.

# %%
runs = lab.run_all(cfg, subjects, trace="light")
stats = lab.evaluate(runs, cfg)
stats.round(2)

# %% [markdown]
# ## Estimate against truth, every subject
#
# The paper's Figure 8, twelve times over.

# %%
viz.plot_all_subjects_tracking(runs)

# %% [markdown]
# ## Error per subject, against the paper

# %%
viz.plot_error_bars(stats)

# %% [markdown]
# ## The paper's Figures 5 and 6

# %%
viz.plot_bland_altman(runs)

# %%
viz.plot_scatter(runs)

# %% [markdown]
# ## Each subject's worst window, side by side
#
# `collect_traces` runs each subject, finds the window with the largest error and
# traces that one window fully.

# %%
traces = lab.collect_traces(cfg, subjects, selector="max_error", runs=runs)
viz.plot_all_subjects_stage(traces, stage="spectrum_estimator")

# %% [markdown]
# The same grid at an earlier stage usually shows whether the heart-rate peak was
# ever there to be found.

# %%
viz.plot_all_subjects_stage(traces, stage="decomposition")
