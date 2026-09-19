"""Interactive Plotly figures for every pipeline stage (Lab Spec Section 5).

Every function returns a ``plotly.graph_objects.Figure`` and never calls
``.show()``; a notebook displays the last expression of a cell. Nothing here
recomputes pipeline results, beyond cheap derived views such as the periodogram
of a stored signal or a w-correlation matrix.
"""

from __future__ import annotations

from pathlib import Path

from troika.viz.common import add_bin_markers, add_gt_lines, add_protocol_shading, decimate
from troika.viz.compare import compare_runs, compare_stage
from troika.viz.data import plot_all_subjects, plot_recording, plot_spectrogram
from troika.viz.explore import explore
from troika.viz.overview import (
    plot_all_subjects_stage,
    plot_all_subjects_tracking,
    plot_bland_altman,
    plot_error_bars,
    plot_pipeline_overview,
    plot_scatter,
)
from troika.viz.ssa import (
    plot_ssa_components,
    plot_ssa_eigenvectors,
    plot_ssa_embedding,
    plot_ssa_grouping,
    plot_ssa_reconstruction,
    plot_ssa_svd,
    plot_ssa_wcorr,
)
from troika.viz.ssr import plot_ssr_dictionary, plot_ssr_iterations, plot_ssr_spectrum
from troika.viz.stages import (
    plot_acc_dominant,
    plot_decomposition_components,
    plot_filter_response,
    plot_stage,
    plot_stage_spectrogram,
    plot_stage_stitched,
    plot_temporal_diff,
)
from troika.viz.theme import COLORS, TEMPLATE
from troika.viz.tracking import (
    plot_tracking_run,
    plot_tracking_window,
    plot_verification_timeline,
)

__all__ = [
    "setup_notebook",
    "save",
    "COLORS",
    "TEMPLATE",
    # A. data
    "plot_recording",
    "plot_spectrogram",
    "plot_all_subjects",
    # B. per-window stages
    "plot_stage",
    "plot_filter_response",
    "plot_acc_dominant",
    "plot_temporal_diff",
    "plot_pipeline_overview",
    # C. SSA internals
    "plot_ssa_embedding",
    "plot_ssa_svd",
    "plot_ssa_eigenvectors",
    "plot_ssa_grouping",
    "plot_ssa_wcorr",
    "plot_ssa_components",
    "plot_ssa_reconstruction",
    # D. generic decomposition
    "plot_decomposition_components",
    # E. SSR internals
    "plot_ssr_dictionary",
    "plot_ssr_iterations",
    "plot_ssr_spectrum",
    # F. tracking
    "plot_tracking_window",
    "plot_tracking_run",
    "plot_verification_timeline",
    # G. whole-recording stage views
    "plot_stage_spectrogram",
    "plot_stage_stitched",
    # H. all subjects
    "plot_all_subjects_stage",
    "plot_all_subjects_tracking",
    "plot_error_bars",
    "plot_bland_altman",
    "plot_scatter",
    # I. comparison
    "compare_stage",
    "compare_runs",
    # J. explorer
    "explore",
    # helpers
    "add_gt_lines",
    "add_bin_markers",
    "add_protocol_shading",
    "decimate",
]


def setup_notebook(renderer: str | None = None) -> None:
    """Configure Plotly for a VS Code notebook.

    Call this once at the top of a notebook. Without a renderer argument it
    leaves Plotly's own detection alone, which already picks the right one in
    the VS Code interactive window; pass ``"notebook"`` or ``"vscode"`` to force
    one.
    """
    import plotly.io as pio

    pio.templates.default = TEMPLATE
    if renderer:
        pio.renderers.default = renderer


def save(fig, path: str | Path, *, scale: float = 2.0) -> Path:
    """Write a figure to disk.

    ``.html`` gives an interactive file that needs no Python to open. Other
    extensions go through ``kaleido``, which is the optional ``png`` extra; the
    error says so if it is missing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in ("", ".html", ".htm"):
        path = path.with_suffix(".html")
        fig.write_html(str(path), include_plotlyjs="cdn")
        return path
    try:
        fig.write_image(str(path), scale=scale)
    except Exception as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError(
            f"writing {path.suffix} needs kaleido; install it with "
            "pip install -e .[png], or save as .html instead"
        ) from exc
    return path
