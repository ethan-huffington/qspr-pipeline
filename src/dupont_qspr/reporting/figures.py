"""Figures for the point-accuracy report.

Three charts, each answering one question, following the project's data-viz rules:
categorical hues assigned in fixed order and never cycled, one axis per chart, a
legend whenever two series share a panel and none when there is only one, thin
marks, recessive grid, and text in ink tokens rather than series colours.

Two consequences of those rules are worth stating because they shaped the designs.

**Five outer folds are not five series.** They are five repeated measurements of
one procedure, so they share a single hue. Colouring them separately would imply
an identity they do not have, and would also breach the palette's all-pairs cap -
only the first three categorical slots validate for scatter-type forms.

**Every figure renders in both light and dark.** The dark variants are stepped for
the dark surface from the same ramps, not an automatic inversion of the light ones.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from dupont_qspr.contracts import PROPERTIES, property_caption

__all__ = ["DARK", "LIGHT", "Theme", "render_all"]


@dataclass(frozen=True, slots=True)
class Theme:
    """Surfaces, ink and the two categorical slots this report uses.

    Slots 1 and 2 are a subset of the first three categorical hues, which are the
    ones documented to clear the all-pairs colour-vision floors in both modes.
    """

    name: str
    surface: str
    ink: str
    ink_soft: str
    grid: str
    series_1: str
    series_2: str


LIGHT = Theme(
    name="light",
    surface="#fcfcfb",
    ink="#0b0b0b",
    ink_soft="#52514e",
    grid="#dcdcd8",
    series_1="#2a78d6",
    series_2="#eb6834",
)

DARK = Theme(
    name="dark",
    surface="#1a1a19",
    ink="#ffffff",
    ink_soft="#c3c2b7",
    grid="#3a3a37",
    series_1="#3987e5",
    series_2="#d95926",
)


def _style(theme: Theme) -> dict[str, Any]:
    return {
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "savefig.facecolor": theme.surface,
        "text.color": theme.ink,
        "axes.labelcolor": theme.ink_soft,
        "axes.edgecolor": theme.grid,
        "xtick.color": theme.ink_soft,
        "ytick.color": theme.ink_soft,
        "grid.color": theme.grid,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }


def _finish(ax, theme: Theme) -> None:
    ax.grid(True, alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #


def figure_skill(estimate: dict[str, Any], theme: Theme, path: Path) -> Path:
    """RMSE ÷ SD per property, against the predict-the-mean line at 1.0.

    A horizontal bar because the job is comparing magnitudes across a handful of
    named categories with long labels. One series, so no legend - the title says
    what is plotted. The per-fold dots are the same measurement repeated, so they
    share the hue rather than becoming five series.

    Value labels sit in a reserved right-hand column at a fixed x rather than
    floating at each bar's end, because bar ends and fold dots occupy the same
    space and the two collide.
    """
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(7.6, 3.1), dpi=200)
        rows = [
            p for p in PROPERTIES if p in estimate and "rmse_over_sd" in estimate[p]
        ]
        y = np.arange(len(rows))

        means = [estimate[p]["rmse_over_sd"]["mean"] for p in rows]
        folds = [estimate[p]["rmse_over_sd"]["per_fold"] for p in rows]

        widest = max([*means, 1.0, *(v for f in folds for v in f)])
        label_x = widest * 1.16
        ax.set_xlim(0, widest * 1.30)
        ax.set_ylim(len(rows) - 0.45, -0.95)

        ax.barh(y, means, height=0.34, color=theme.series_1, zorder=3)
        for i, values in enumerate(folds):
            ax.scatter(
                values,
                np.full(len(values), i),
                s=38,
                color=theme.surface,
                edgecolor=theme.series_1,
                linewidth=1.6,
                zorder=5,
            )
        for i, value in enumerate(means):
            ax.text(
                label_x,
                i,
                f"{value:.3f}",
                va="center",
                ha="right",
                fontsize=10,
                color=theme.ink,
                zorder=6,
            )

        ax.axvline(1.0, color=theme.series_2, linewidth=1.8, linestyle="--", zorder=4)
        ax.text(
            1.0,
            -0.88,
            "predict-the-mean",
            color=theme.series_2,
            fontsize=9,
            va="bottom",
            ha="center",
        )

        ax.set_yticks(y, [property_caption(p) for p in rows])  # type: ignore[arg-type]
        ax.set_xlabel("RMSE ÷ standard deviation   (lower is better; 1.0 = no skill)")
        ax.set_title(
            "Nested estimate: how much better than guessing the mean?",
            loc="left",
            pad=16,
        )
        _finish(ax, theme)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def figure_parity(predictions: pl.DataFrame, theme: Theme, path: Path) -> Path:
    """Predicted against observed, one panel per property.

    Small multiples rather than one panel with three colours: the axes are in
    different units, and putting unrelated scales on shared axes is the mistake
    this rule exists to prevent. One hue per panel, so no legend is needed.
    """
    present = [
        p for p in PROPERTIES if predictions.filter(pl.col("property") == p).height
    ]
    with plt.rc_context(_style(theme)):
        fig, axes = plt.subplots(
            1, max(1, len(present)), figsize=(3.4 * len(present), 3.3), dpi=200
        )
        axes = np.atleast_1d(axes)

        for ax, prop in zip(axes, present, strict=True):
            sub = predictions.filter(pl.col("property") == prop)
            truth = sub.get_column("y_true").to_numpy()
            predicted = sub.get_column("y_pred").to_numpy()

            lo = float(min(truth.min(), predicted.min()))
            hi = float(max(truth.max(), predicted.max()))
            pad = (hi - lo) * 0.05
            ax.plot(
                [lo - pad, hi + pad],
                [lo - pad, hi + pad],
                color=theme.ink_soft,
                linewidth=1.2,
                linestyle="--",
                zorder=2,
            )
            ax.scatter(
                truth,
                predicted,
                s=7,
                alpha=0.28,
                linewidths=0,
                color=theme.series_1,
                zorder=3,
            )

            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(property_caption(prop), loc="left", fontsize=10)  # type: ignore[arg-type]
            ax.set_xlabel("observed")
            ax.set_ylabel("predicted" if prop == present[0] else "")
            ax.text(
                0.04,
                0.94,
                f"n = {sub.height:,}",
                transform=ax.transAxes,
                fontsize=9,
                color=theme.ink_soft,
                va="top",
            )
            _finish(ax, theme)

        fig.suptitle(
            "Out-of-fold predictions — every molecule scored by a model that never saw it",
            x=0.02,
            ha="left",
            fontsize=11,
            color=theme.ink,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def figure_ranking(
    ranking: dict[str, dict[str, float]], theme: Theme, path: Path
) -> Path:
    """Overall rank correlation against rank correlation in the top decile.

    Two series, so a legend is mandatory. The axis spans the full Spearman range
    including negatives with a zero reference, because a negative tail correlation
    is a real and important reading - it means the model orders the best candidates
    backwards - and an axis starting at zero would hide it entirely.
    """
    rows = [p for p in PROPERTIES if p in ranking]
    with plt.rc_context(_style(theme)):
        fig, ax = plt.subplots(figsize=(7.6, 3.3), dpi=200)
        y = np.arange(len(rows))
        height = 0.30
        gap = 0.02  # surface gap between adjacent fills

        overall = [ranking[p]["overall"] for p in rows]
        tail = [ranking[p]["top_decile"] for p in rows]
        finite = [v for v in (*overall, *tail) if np.isfinite(v)]
        low = min(0.0, min(finite)) if finite else 0.0

        ax.set_xlim(low - 0.30, 1.16)
        ax.set_ylim(len(rows) - 0.45, -0.75)

        ax.barh(
            y - (height / 2 + gap),
            overall,
            height=height,
            color=theme.series_1,
            label="all molecules",
            zorder=3,
        )
        ax.barh(
            y + (height / 2 + gap),
            tail,
            height=height,
            color=theme.series_2,
            label="top decile only",
            zorder=3,
        )
        ax.axvline(0.0, color=theme.ink_soft, linewidth=1.1, zorder=4)

        def _label(value: float, row: float) -> None:
            if not np.isfinite(value):
                return
            offset = 0.02 if value >= 0 else -0.02
            ax.text(
                value + offset,
                row,
                f"{value:.2f}",
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9,
                color=theme.ink,
                zorder=6,
            )

        for i, (a, b) in enumerate(zip(overall, tail, strict=True)):
            _label(a, i - (height / 2 + gap))
            _label(b, i + (height / 2 + gap))

        ax.set_yticks(y, [property_caption(p) for p in rows])  # type: ignore[arg-type]
        ax.set_xlabel(
            "Spearman rank correlation   (higher is better; 0 = no ordering signal)"
        )
        ax.set_title(
            "Ranking quality overall, and where it actually matters", loc="left", pad=16
        )
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            fontsize=9,
            labelcolor=theme.ink_soft,
        )
        _finish(ax, theme)
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def render_all(
    estimate: dict[str, Any],
    predictions: pl.DataFrame,
    ranking: dict[str, dict[str, float]],
    directory: Path,
) -> list[Path]:
    """Every figure, in both themes. Returns the paths written."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for theme in (LIGHT, DARK):
        written.append(
            figure_skill(estimate, theme, directory / f"skill-{theme.name}.png")
        )
        written.append(
            figure_parity(predictions, theme, directory / f"parity-{theme.name}.png")
        )
        written.append(
            figure_ranking(ranking, theme, directory / f"ranking-{theme.name}.png")
        )
    return written
