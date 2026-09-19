# %% [markdown]
# # 07. The explorer
#
# One panel for everything: pick a subject, a plot and a window.
#
# Each change re-traces the chosen window at the full level, which takes a second
# or two because the pipeline has to run up to that window honestly before the
# window itself can be traced.

# %% setup
# %load_ext autoreload
# %autoreload 2
from troika import lab, viz

viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg)

# %%
viz.explore(subjects, cfg)

# %% [markdown]
# Untick "honest previous bin" to seed the window from the ECG instead of
# tracking up to it. That is faster and useful for looking around, but it marks
# the trace contaminated and must never be used for a reported number.
