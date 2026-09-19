"""One place to change how every figure looks (Lab Spec Section 5.1).

Colour roles are fixed so a reader learns them once: grey is the raw signal,
blue the band-passed one, green the decomposition, orange the temporal
difference, purple the spectrum estimator, red the tracker, black dashed the
ground truth, teal the accelerometer's dominant bins, light red the excluded
zone around the previous estimate.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "COLORS",
    "TEMPLATE",
    "STAGE_LABELS",
    "SPEED_SEGMENTS",
    "apply_layout",
    "stage_color",
]

TEMPLATE = "plotly_white"

COLORS: dict[str, str] = {
    "raw": "#8c8c8c",
    "bandpass": "#1f6fb4",
    "decomposition": "#2e9e5b",
    "temporal_diff": "#e8871a",
    "spectrum_estimator": "#7b52ab",
    "tracker": "#d62728",
    "ground_truth": "#000000",
    "acc": "#1aa3a3",
    "excluded": "rgba(214, 39, 40, 0.12)",
    "removed": "#d62728",
    "kept": "#2e9e5b",
    "grid": "rgba(0,0,0,0.08)",
    "annotation": "#555555",
}

STAGE_LABELS: dict[str, str] = {
    "raw": "raw PPG",
    "bandpass": "band-passed",
    "decomposition": "after decomposition",
    "temporal_diff": "after temporal difference",
    "spectrum_estimator": "spectrum estimate",
    "tracker": "tracking",
}

#: Nominal speed segments per recording type (DEVIATION D2). Boundaries are the
#: 30/90/150/210/270 s the Lab Spec gives; only the labels differ by type.
SPEED_SEGMENTS: dict[str, list[tuple[float, float, str]]] = {
    "TYPE01": [
        (0, 30, "rest"),
        (30, 90, "8 km/h"),
        (90, 150, "15 km/h"),
        (150, 210, "8 km/h"),
        (210, 270, "15 km/h"),
        (270, 300, "rest"),
    ],
    "TYPE02": [
        (0, 30, "rest"),
        (30, 90, "6 km/h"),
        (90, 150, "12 km/h"),
        (150, 210, "6 km/h"),
        (210, 270, "12 km/h"),
        (270, 300, "rest"),
    ],
}


def stage_color(stage: str) -> str:
    """Colour assigned to a pipeline stage, falling back to the raw grey."""
    return COLORS.get(stage, COLORS["raw"])


def apply_layout(fig, title: str | None = None, height: int | None = None, **kwargs: Any):
    """Apply the shared template, title and sizing to a figure."""
    fig.update_layout(
        template=TEMPLATE,
        title=title,
        height=height,
        margin=dict(l=60, r=30, t=60 if title else 30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="closest",
        **kwargs,
    )
    fig.update_xaxes(gridcolor=COLORS["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=COLORS["grid"], zeroline=False)
    return fig
