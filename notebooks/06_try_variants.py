# %% [markdown]
# # 06. Trying your own variant
#
# Write a plug-in in a cell, compare it against the paper default, and promote
# the winner into a config file that the command line can reproduce.

# %% setup
# %load_ext autoreload
# %autoreload 2
import numpy as np

from troika import lab, registry, viz
from troika.types import StageOutput

viz.setup_notebook()
cfg = lab.load_config("configs/default.yaml")
subjects = lab.load_subjects(cfg, [5, 6])
rec = subjects[5]

# %% [markdown]
# ## What is available
#
# Every slot, and the plug-ins registered for it.

# %%
{slot: registry.available(slot) for slot in registry.SLOTS}

# %% [markdown]
# ## A notebook plug-in
#
# The quick form: a function plus its parameters. Re-run this cell after editing
# it; `override=True` is what makes that work.
#
# This example is a placeholder that keeps only the strongest few SSA
# components. Replace the body with whatever you want to try, for instance
# empirical mode decomposition instead of singular spectrum analysis.

# %%
from troika.decomposition.grouping import component_dominant_bins, group_eigentriples
from troika.decomposition.motion_removal import select_motion_components
from troika.decomposition.ssa import embed, svd


@registry.register_function(
    "decomposition", "top_k_ssa", params={"L": 400, "keep": 40}, override=True
)
def top_k_ssa(x, ctx, *, L, keep):
    """Keep the strongest `keep` SSA components, then drop the motion ones."""
    M = x.size
    U, s, Vt = svd(embed(x, L))
    groups, elementary = group_eigentriples(U, s, Vt, M, "frequency_pairing")
    components = np.stack([elementary[np.asarray(g)].sum(axis=0) for g in groups])

    energy = (components**2).sum(axis=1)
    strongest = np.argsort(-energy)[:keep]
    components = components[strongest]

    dominant = component_dominant_bins(components, 4096, 125.0)
    removed = select_motion_components(dominant, ctx.acc_bins, 0)
    cleansed = components[~removed].sum(axis=0) if (~removed).any() else x.copy()

    return StageOutput(
        data=cleansed,
        diag={
            "components": components,
            "component_labels": [f"component {i}" for i in range(components.shape[0])],
            "component_dominant_bins": dominant,
            "removed_mask": removed,
        },
    )


# %% [markdown]
# ## Check the contract
#
# `check_plugin` runs it on synthetic input and reports shape, finiteness, input
# mutation, determinism and whether the diag keys the plots need are there.

# %%
from troika.testing import check_plugin

print(check_plugin("decomposition", "top_k_ssa"))

# %% [markdown]
# ## Compare it on one window
#
# Because it fills the generic diag keys, the generic decomposition plot works
# on it with no extra code.

# %%
variant = cfg.with_overrides({"decomposition.method": "top_k_ssa"})
lab.compare_variants(rec, {"ssa (default)": cfg, "top_k_ssa": variant}, window=40,
                     stage="decomposition")

# %%
viz.plot_decomposition_components(lab.get_window_trace(rec, variant, w=40), top=8)

# %% [markdown]
# ## Compare whole runs

# %%
runs_default = lab.run_all(cfg, subjects)
runs_variant = lab.run_all(variant, subjects)
lab.compare_runs({"default": runs_default, "top_k_ssa": runs_variant})

# %%
lab.evaluate(runs_variant, cfg).round(2)

# %% [markdown]
# ## Promote the winner
#
# `promote` writes only what differs from the default, so the file stays short
# and readable. Then the command line reproduces it:
#
#     python experiments/run_all.py --config configs/from_notebook.yaml
#
# This cell writes to `configs/from_notebook.yaml` rather than
# `configs/my_experiment.yaml`, so that running this notebook does not overwrite
# your own experiment file. Change the path when you mean to keep the variant.
#
# To move the plug-in out of the notebook and into the package, which is also
# what lets it run in parallel across subjects:
#
#     python -m troika.plugins.new decomposition top_k_ssa

# %%
lab.promote(variant, "configs/from_notebook.yaml")
