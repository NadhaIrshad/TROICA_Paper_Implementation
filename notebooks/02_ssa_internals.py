# %% [markdown]
# # 02. Inside the decomposition
#
# Singular spectrum analysis, step by step: embedding, the singular value
# decomposition, grouping, and which groups get thrown away as motion.

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
# ## Embedding
#
# The window becomes an L by K matrix whose anti-diagonals are constant. The
# numbered corner makes that structure visible.

# %%
viz.plot_ssa_embedding(trace, zoom=8)

# %% [markdown]
# ## Singular values
#
# Circles mark adjacent pairs whose singular values are nearly equal. A sinusoid
# produces exactly such a pair, which is what the default grouping looks for.

# %%
viz.plot_ssa_svd(trace)

# %% [markdown]
# ## Eigenvectors
#
# A sinusoid makes u_i against u_{i+1} trace a circle. That is the clearest
# visual test that two eigentriples belong together.

# %%
viz.plot_ssa_eigenvectors(trace, idx=range(8))

# %% [markdown]
# ## Grouping
#
# The table says, per group, which eigentriples it holds, its dominant frequency
# and whether it matched the accelerometer and was removed.

# %%
viz.plot_ssa_grouping(trace)

# %% [markdown]
# ## Weighted correlation
#
# The classical separability view. Blocks along the diagonal are components that
# cannot be pulled apart and therefore belong in one group.

# %%
viz.plot_ssa_wcorr(trace, top=40)

# %% [markdown]
# ## Components, and what survives

# %%
viz.plot_ssa_components(trace, top=10)

# %%
viz.plot_ssa_reconstruction(trace)

# %% [markdown]
# ## Change L and the grouping strategy in place
#
# L is a paper value (400); the grouping strategy is ASSUMPTION A4. Both change
# what the stage removes, so compare them on the same window.

# %%
variants = {
    "L=400, frequency_pairing (default)": cfg,
    "L=200": cfg.with_overrides({"decomposition.L": 200}),
    "wcorr_hclust": cfg.with_overrides({"decomposition.grouping": "wcorr_hclust"}),
    "singleton": cfg.with_overrides({"decomposition.grouping": "singleton"}),
}
lab.compare_variants(rec, variants, window=40, stage="decomposition")
