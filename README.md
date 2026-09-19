# TROIKA: heart-rate estimation from wrist PPG during intensive exercise

A from-scratch Python implementation of

> Z. Zhang, Z. Pi and B. Liu, "TROIKA: A General Framework for Heart Rate
> Monitoring Using Wrist-Type Photoplethysmographic (PPG) Signals During
> Intensive Physical Exercise", *IEEE Transactions on Biomedical Engineering*
> 62(2):522-531, 2015. DOI 10.1109/TBME.2014.2359372

The pipeline estimates heart rate from a single-channel wrist PPG signal while
the wearer runs at up to 15 km/h, using three stages: signal decomposiTion with
singular spectrum analysis, sparse signal RecOnstructIon with FOCUSS, and
spectral peaK trAcking with verification.

Around that core sits an exploration lab: every stage is a swappable plug-in,
every internal signal can be traced and plotted interactively, and a variant
found in a notebook can be promoted into a config and evaluated on all 12
subjects.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[lab,dev]"
pytest
```

Python 3.10 or newer. The `lab` extra pulls in Plotly, Jupyter and joblib; the
core pipeline needs only NumPy, SciPy, pandas and PyYAML.

## Data

The 12-subject training set of the IEEE Signal Processing Cup 2015, which is the
dataset of the paper. Place it at `datasets/IEEE_SPC_2015/` with
`Training_data/DATA_01_TYPE01.mat` and friends. Data is never committed.

Each recording is a `6 x n` array `sig`: row 0 ECG, rows 1-2 the two PPG
channels, rows 3-5 accelerometer x, y and z, all at 125 Hz. Per-window ground
truth ships alongside as `DATA_xx_TYPExx_BPMtrace.mat`.

## Run

```bash
# paper defaults, all 12 subjects
python experiments/run_all.py --config configs/default.yaml

# one subject, with a parameter changed
python experiments/run_all.py --config configs/default.yaml --subjects 5 \
    --set decomposition.L=300

# your own parameters
python experiments/run_all.py --config configs/my_experiment.yaml
```

Results land in `results/<run name>_<timestamp>/`.

## Explore

Notebooks live in `notebooks/` as percent-format `.py` files, which VS Code runs
cell by cell. Install the Python and Jupyter extensions, select `.venv` as the
kernel, and click "Run Cell" above any `# %%` marker.

```python
from troika import lab, viz
viz.setup_notebook()
cfg      = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg)
trace    = lab.get_window_trace(subjects[5], cfg, w=40)
viz.plot_pipeline_overview(trace)
```

## Layout

| Path | What |
|---|---|
| `src/troika/` | the package: core algorithms, plug-ins, lab API, visualisation |
| `configs/` | `default.yaml` (paper values, never edited), `my_experiment.yaml`, ablations |
| `experiments/` | command-line entry points that reproduce the paper's tables and figures |
| `notebooks/` | step-by-step exploration, one per pipeline stage |
| `tests/` | unit tests per stage, plus synthetic end-to-end runs |
| `docs/` | the two specifications, and the assumption log |

`DEVIATIONS.md` lists everything this implementation does that the two
specifications do not state. `CLAUDE.md` holds the coding conventions.

## Milestones

- [x] **M0** scaffold: config with the slot layout, types, bin mapping, plug-in registry
