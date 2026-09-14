"""Step 9: do the intervals still hold far from the training set?

Conformal prediction delivers *marginal* coverage almost by construction, so a
pooled PICP near 0.90 is a weak diagnostic. The question that matters is
*conditional* coverage: whether molecules unlike anything in training are still
covered, or whether a good average hides intervals that quietly fail on exactly
the novel chemistry the system exists to evaluate. So coverage is broken out by
nearest-neighbour Tanimoto distance band.

The same distances also set the applicability-domain threshold. Error is plotted
against distance, and the threshold is the last distance before normalised error
climbs past a tolerance multiple of the error among the nearer half of molecules.
The threshold is chosen per property and the most conservative one is kept,
because a scored record carries a single in-domain flag for the whole molecule.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES

__all__ = [
    "band_labels",
    "choose_ad_threshold",
    "conditional_coverage",
    "error_vs_distance",
    "load_interval_frame",
]

_JOIN = ["property", "fold", "row_index"]


def load_interval_frame(cfg: Config, label: str) -> pl.DataFrame:
    """Calibrated intervals joined to each molecule's applicability-domain distance."""
    intervals = cfg.processed_dir / f"intervals_{label}.parquet"
    distance = cfg.processed_dir / f"ad_distance_{label}.parquet"
    for path in (intervals, distance):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run experiments/08_uncertainty.py for {label} first."
            )
    return pl.read_parquet(intervals).join(
        pl.read_parquet(distance), on=_JOIN, how="inner"
    )


def band_labels(edges: Sequence[float]) -> list[str]:
    bounds = [0.0, *edges]
    return [f"{lo:.2f}–{hi:.2f}" for lo, hi in itertools.pairwise(bounds)]


def conditional_coverage(frame: pl.DataFrame, edges: Sequence[float]) -> pl.DataFrame:
    """PICP and width per property, method and distance band."""
    edges = sorted(edges)
    breaks = np.asarray(edges[:-1], dtype=np.float64)
    band = np.searchsorted(
        breaks, frame.get_column("nn_distance").to_numpy(), side="left"
    )
    labels = band_labels(edges)
    return (
        frame.with_columns(pl.Series("band", band.astype(np.int64)))
        .group_by(["property", "method", "band"])
        .agg(
            pl.col("covered").mean().alias("picp"),
            pl.col("width").mean().alias("mean_width"),
            pl.col("nn_distance").mean().alias("mean_distance"),
            pl.len().alias("n"),
        )
        .with_columns(
            pl.col("band")
            .map_elements(lambda b: labels[int(b)], return_dtype=pl.String)
            .alias("band_label")
        )
        .sort(["property", "method", "band"])
    )


def error_vs_distance(
    predictions: pl.DataFrame, distance: pl.DataFrame, *, n_bins: int = 10
) -> pl.DataFrame:
    """Normalised absolute error in equal-count distance bins, per property.

    Error is divided by the property's standard deviation so that one tolerance
    means the same thing for log units and Kelvin.
    """
    joined = predictions.join(distance, on=_JOIN, how="inner")
    rows: list[dict[str, Any]] = []
    for prop in PROPERTIES:
        sub = joined.filter(pl.col("property") == prop)
        if sub.height < n_bins * 5:
            continue
        y = sub.get_column("y_true").to_numpy()
        error = np.abs(y - sub.get_column("y_pred").to_numpy()) / (
            float(np.std(y)) or 1.0
        )
        d = sub.get_column("nn_distance").to_numpy()
        for index, chunk in enumerate(
            np.array_split(np.argsort(d, kind="stable"), n_bins)
        ):
            rows.append(
                {
                    "property": prop,
                    "bin": index,
                    "d_low": float(d[chunk].min()),
                    "d_high": float(d[chunk].max()),
                    "mean_distance": float(d[chunk].mean()),
                    "norm_abs_error": float(error[chunk].mean()),
                    "n": int(chunk.size),
                }
            )
    return pl.DataFrame(rows)


def choose_ad_threshold(
    curve: pl.DataFrame, *, tolerance: float = 1.5
) -> tuple[float | None, dict[str, float]]:
    """The distance past which error materially climbs, most conservative across properties.

    Only the farther half of the bins is searched, because the reference is the
    nearer half; a noisy near bin should not be able to trigger the threshold. If
    error never climbs past the tolerance, every observed distance is in domain.
    """
    per_property: dict[str, float] = {}
    if curve.is_empty():
        return None, per_property
    for prop in curve.get_column("property").unique().to_list():
        c = curve.filter(pl.col("property") == prop).sort("bin")
        errors = c.get_column("norm_abs_error").to_numpy()
        counts = c.get_column("n").to_numpy()
        highs = c.get_column("d_high").to_numpy()
        half = max(1, errors.size // 2)
        reference = float(np.average(errors[:half], weights=counts[:half]))
        threshold = float(highs[-1])
        for index in range(half, errors.size):
            if errors[index] > tolerance * reference:
                threshold = float(highs[index - 1])
                break
        per_property[prop] = threshold
    return min(per_property.values()), per_property
