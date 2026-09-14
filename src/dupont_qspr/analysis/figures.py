"""Figures for steps 9 and 10, in the same visual system as the step-5 report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from dupont_qspr.contracts import PROPERTIES, property_caption
from dupont_qspr.reporting.figures import DARK, LIGHT, Theme, _finish, _style

__all__ = ["render_ablation", "render_conditional_coverage"]


def _panels(table: pl.DataFrame) -> list[str]:
    return [p for p in PROPERTIES if table.filter(pl.col("property") == p).height]


def figure_conditional_coverage(
    table: pl.DataFrame, nominal: float, title: str, theme: Theme, path: Path
) -> Path:
    """Coverage by distance band, one panel per property, one line per calibrator.

    Two series share each panel, so a legend is present. The dashed line is the
    nominal level: a calibrator that holds up stays on it as distance grows; one
    that only achieves marginal coverage sags below it on the far right.
    """
    props = _panels(table)
    with plt.rc_context(_style(theme)):
        fig, axes = plt.subplots(
            1, max(1, len(props)), figsize=(3.6 * len(props), 3.4), dpi=200
        )
        axes = np.atleast_1d(axes)
        for ax, prop in zip(axes, props, strict=True):
            for method, colour, label in (
                ("cqr", theme.series_1, "CQR (adaptive width)"),
                ("split", theme.series_2, "split conformal (constant width)"),
            ):
                sub = table.filter(
                    (pl.col("property") == prop) & (pl.col("method") == method)
                ).sort("band")
                if not sub.height:
                    continue
                ax.plot(
                    sub.get_column("mean_distance"),
                    sub.get_column("picp"),
                    color=colour,
                    linewidth=2,
                    marker="o",
                    markersize=6,
                    label=label,
                    zorder=3,
                )
            ax.axhline(
                nominal, color=theme.ink_soft, linewidth=1.2, linestyle="--", zorder=2
            )
            ax.set_ylim(0.0, 1.04)
            ax.set_title(property_caption(prop), loc="left", fontsize=10)  # type: ignore[arg-type]
            ax.set_xlabel("nearest-neighbour Tanimoto distance")
            ax.set_ylabel("coverage (PICP)" if prop == props[0] else "")
            _finish(ax, theme)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=9,
            labelcolor=theme.ink_soft,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(title, x=0.02, ha="left", fontsize=11, color=theme.ink)
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def figure_ablation(summary: pl.DataFrame, theme: Theme, path: Path) -> Path:
    """Normalised error against labelled training rows, one line per track."""
    props = _panels(summary)
    tracks = sorted(summary.get_column("track").unique().to_list())
    colours = [theme.series_1, theme.series_2]
    with plt.rc_context(_style(theme)):
        fig, axes = plt.subplots(
            1, max(1, len(props)), figsize=(3.6 * len(props), 3.4), dpi=200
        )
        axes = np.atleast_1d(axes)
        for ax, prop in zip(axes, props, strict=True):
            for track, colour in zip(tracks, colours, strict=False):
                sub = summary.filter(
                    (pl.col("property") == prop) & (pl.col("track") == track)
                ).sort("size")
                if not sub.height:
                    continue
                x = sub.get_column("size").to_numpy()
                y = sub.get_column("mean_rmse_over_sd").to_numpy()
                spread = sub.get_column("sd_rmse_over_sd").fill_null(0.0).to_numpy()
                ax.fill_between(
                    x,
                    y - spread,
                    y + spread,
                    color=colour,
                    alpha=0.14,
                    linewidth=0,
                    zorder=2,
                )
                ax.plot(
                    x,
                    y,
                    color=colour,
                    linewidth=2,
                    marker="o",
                    markersize=6,
                    label=track,
                    zorder=3,
                )
            ax.axhline(
                1.0, color=theme.ink_soft, linewidth=1.1, linestyle="--", zorder=1
            )
            ax.set_xscale("log")
            sizes = sorted(summary.get_column("size").unique().to_list())
            ax.set_xticks(sizes, [str(s) for s in sizes])
            ax.minorticks_off()
            ax.set_title(property_caption(prop), loc="left", fontsize=10)  # type: ignore[arg-type]
            ax.set_xlabel("labelled training molecules")
            ax.set_ylabel("RMSE ÷ SD" if prop == props[0] else "")
            _finish(ax, theme)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=2,
            frameon=False,
            fontsize=9,
            labelcolor=theme.ink_soft,
            bbox_to_anchor=(0.5, -0.02),
        )
        fig.suptitle(
            "Low-data ablation — does multi-task help most when labels are scarce?",
            x=0.02,
            ha="left",
            fontsize=11,
            color=theme.ink,
        )
        fig.tight_layout(rect=(0, 0.07, 1, 0.93))
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
    return path


def render_conditional_coverage(
    table: pl.DataFrame, nominal: float, label: str, directory: Path
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    title = f"Coverage as molecules move away from the training set — {label}"
    return [
        figure_conditional_coverage(
            table,
            nominal,
            title,
            theme,
            directory / f"coverage-{label}-{theme.name}.png",
        )
        for theme in (LIGHT, DARK)
    ]


def render_ablation(summary: pl.DataFrame, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    return [
        figure_ablation(summary, theme, directory / f"ablation-{theme.name}.png")
        for theme in (LIGHT, DARK)
    ]
