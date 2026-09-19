# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: troika
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 00. The data
#
# What the twelve recordings look like before any processing: the two PPG
# channels, the three accelerometer axes and the ECG, in time and in frequency.
#
# Run this top to bottom. The first cell is the same in every notebook.

# %% setup
# %load_ext autoreload
# %autoreload 2
from troika import lab, viz

viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg)
print(f"{len(subjects)} subjects: {sorted(subjects)}")

# %% [markdown]
# ## One subject in detail
#
# Time on the left, spectrum on the right, one row per channel. The shaded bands
# are the nominal speed segments; subject 1 ran the TYPE01 schedule and everyone
# else TYPE02, which is why the labels differ.

# %%
rec = subjects[5]
print(
    f"subject {rec.subject_id}: {rec.duration_s:.1f} s, {rec.n_samples} samples, "
    f"protocol {rec.protocol}, ground truth {rec.bpm_gt.min():.0f}-{rec.bpm_gt.max():.0f} BPM"
)
viz.plot_recording(rec)

# %% [markdown]
# ## Where the difficulty lives
#
# The spectrogram shows two ridges: the heart rate climbing with the treadmill
# speed, and the motion artifact from the stride. Where they cross is where the
# whole framework earns its keep.

# %%
viz.plot_spectrogram(rec, channel="ppg")

# %% [markdown]
# The accelerometer sees the stride and nothing else, which is what lets the
# decomposition stage tell the two ridges apart.

# %%
viz.plot_spectrogram(rec, channel="acc")

# %% [markdown]
# ## All twelve at once

# %%
viz.plot_all_subjects(subjects, channel="ppg", domain="time")

# %%
viz.plot_all_subjects(subjects, channel="ppg", domain="freq")

# %%
viz.plot_all_subjects(subjects, channel="ppg", domain="spectrogram")

# %% [markdown]
# ## The two PPG channels
#
# The paper uses a single channel; this dataset ships two, 2 cm apart. Which one
# is used is ASSUMPTION A15, and the channels are not interchangeable: their
# correlation ranges from -0.21 on subject 1 to 0.62 on subject 5.

# %%
import numpy as np
import pandas as pd

pd.DataFrame(
    [
        {
            "subject": s,
            "protocol": r.protocol,
            "duration_s": round(r.duration_s, 1),
            "windows": len(r.bpm_gt),
            "gt_min": round(float(r.bpm_gt.min()), 1),
            "gt_max": round(float(r.bpm_gt.max()), 1),
            "corr(ppg1, ppg2)": round(float(np.corrcoef(r.ppg_all[0], r.ppg_all[1])[0, 1]), 3),
        }
        for s, r in sorted(subjects.items())
    ]
).set_index("subject")

# %% [markdown]
# ## Ground truth
#
# The dataset ships a per-window BPM trace computed from the ECG. Recomputing it
# from R-peak intervals agrees to well under 1 BPM on eleven of twelve subjects;
# subject 11 disagrees only because its ECG clips.

# %%
from troika.io.ground_truth import compare_ground_truth, gt_from_ecg

pd.DataFrame(
    [
        {
            "subject": s,
            **compare_ground_truth(r.bpm_gt, gt_from_ecg(r.ecg, r.fs, 8, 2)),
        }
        for s, r in sorted(subjects.items())
    ]
).set_index("subject").round(3)
