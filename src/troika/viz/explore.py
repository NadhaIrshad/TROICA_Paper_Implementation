"""Interactive explorer widget (Lab Spec Section 5.2 group J, milestone L7)."""

from __future__ import annotations

from typing import Mapping

from troika.config import Config
from troika.types import Recording

__all__ = ["explore"]

#: Plots offered by the explorer, mapped to how they are drawn.
_PLOTS = {
    "pipeline overview": ("window", "plot_pipeline_overview"),
    "stage: bandpass": ("stage", "bandpass"),
    "stage: decomposition": ("stage", "decomposition"),
    "stage: temporal difference": ("stage", "temporal_diff"),
    "stage: spectrum": ("stage", "spectrum_estimator"),
    "accelerometer dominant": ("window", "plot_acc_dominant"),
    "temporal difference detail": ("window", "plot_temporal_diff"),
    "SSA singular values": ("window", "plot_ssa_svd"),
    "SSA grouping": ("window", "plot_ssa_grouping"),
    "SSA components": ("window", "plot_ssa_components"),
    "SSA reconstruction": ("window", "plot_ssa_reconstruction"),
    "SSR dictionary": ("window", "plot_ssr_dictionary"),
    "SSR iterations": ("window", "plot_ssr_iterations"),
    "SSR spectrum": ("window", "plot_ssr_spectrum"),
    "tracking window": ("window", "plot_tracking_window"),
}


def explore(subjects: Mapping[int, Recording], cfg: Config):
    """An ipywidgets panel: pick a subject, a plot and a window, and see it.

    Works in a VS Code notebook. Each change re-traces the chosen window at the
    full level, which takes a second or two, because the pipeline must run up to
    that window honestly before the window itself can be traced.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:  # pragma: no cover - needs the lab extra
        raise ImportError(
            "viz.explore needs ipywidgets; install the lab extra with "
            "pip install -e .[lab]"
        ) from exc

    from troika import lab, viz

    ids = sorted(subjects)
    subject_dd = widgets.Dropdown(options=ids, value=ids[0], description="subject")
    plot_dd = widgets.Dropdown(
        options=list(_PLOTS), value="pipeline overview", description="plot"
    )
    window_slider = widgets.IntSlider(
        value=40, min=0, max=140, step=1, description="window", continuous_update=False
    )
    honest = widgets.Checkbox(value=True, description="honest previous bin")
    out = widgets.Output()

    def _max_window(subject_id: int) -> int:
        from troika.preprocessing.windowing import n_windows

        rec = subjects[subject_id]
        return max(
            0,
            n_windows(rec.n_samples, rec.fs, cfg.signal.window_s, cfg.signal.step_s) - 1,
        )

    def _render(*_args):
        with out:
            clear_output(wait=True)
            rec = subjects[subject_dd.value]
            window_slider.max = _max_window(subject_dd.value)
            trace = lab.get_window_trace(
                rec,
                cfg,
                window_slider.value,
                prev_bin="tracked" if honest.value else "ground_truth",
            )
            kind, target = _PLOTS[plot_dd.value]
            figure = (
                viz.plot_stage(trace, target)
                if kind == "stage"
                else getattr(viz, target)(trace)
            )
            display(figure)

    for control in (subject_dd, plot_dd, window_slider, honest):
        control.observe(_render, names="value")

    panel = widgets.VBox(
        [
            widgets.HBox([subject_dd, plot_dd]),
            widgets.HBox([window_slider, honest]),
            out,
        ]
    )
    _render()
    return panel
