"""Result tables, reported per property and never pooled.

Pooling would average a 0.98 log-unit error against a 41 Kelvin one, which is
arithmetic on incompatible units and hides a weak property behind a strong one.
Every table here is keyed by property.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from dupont_qspr.contracts import PROPERTIES, PROPERTY_LABELS, PROPERTY_UNITS
from dupont_qspr.metrics.baselines import mean_baseline
from dupont_qspr.metrics.point import fold_interval, spearman, tail_spearman

__all__ = ["baseline_table", "ranking_summary", "results_table"]


def results_table(estimate: dict[str, Any]) -> pl.DataFrame:
    """Headline numbers with t-intervals across the outer folds."""
    rows = []
    for prop in PROPERTIES:
        entry = estimate.get(prop)
        if not entry:
            continue
        rmse = fold_interval(entry["rmse"]["per_fold"])
        ratio = fold_interval(entry["rmse_over_sd"]["per_fold"])
        rho = fold_interval(entry["spearman"]["per_fold"])
        rows.append(
            {
                "property": prop,
                "name": PROPERTY_LABELS[prop],  # type: ignore[index]
                "units": PROPERTY_UNITS[prop],  # type: ignore[index]
                "n": entry["n_test_total"],
                # Shown because the t-interval width is dominated by it: the
                # multiplier is 12.7 at 2 folds and 2.78 at 5.
                "folds": rmse.get("n", 0),
                "RMSE": round(rmse["mean"], 3),
                "RMSE 95% CI": f"[{rmse['lower']:.3f}, {rmse['upper']:.3f}]",
                "RMSE/SD": round(ratio["mean"], 3),
                "RMSE/SD 95% CI": f"[{ratio['lower']:.3f}, {ratio['upper']:.3f}]",
                "MAE": round(entry["mae"]["mean"], 3),
                "Spearman": round(rho["mean"], 3),
            }
        )
    return pl.DataFrame(rows)


def baseline_table(predictions: pl.DataFrame) -> pl.DataFrame:
    """Model against predict-the-mean, computed on identical held-out rows.

    The baseline is fitted per outer fold on that fold's training rows, exactly
    as the model was, so neither sees anything the other did not.
    """
    rows = []
    for prop in PROPERTIES:
        sub = predictions.filter(pl.col("property") == prop)
        if not sub.height:
            continue
        truth = sub.get_column("y_true").to_numpy()
        predicted = sub.get_column("y_pred").to_numpy()

        model_rmse = float(np.sqrt(np.mean((truth - predicted) ** 2)))
        # Leave-fold-out mean: for each fold, the mean of the other folds' rows.
        baseline_errors = []
        for fold in sub.get_column("fold").unique().to_list():
            held = sub.filter(pl.col("fold") == fold).get_column("y_true").to_numpy()
            rest = sub.filter(pl.col("fold") != fold).get_column("y_true").to_numpy()
            if rest.size and held.size:
                baseline_errors.append(
                    mean_baseline(rest, held)["rmse"] ** 2 * held.size
                )
        base_rmse = (
            float(np.sqrt(sum(baseline_errors) / truth.size))
            if baseline_errors
            else float("nan")
        )

        rows.append(
            {
                "property": prop,
                "name": PROPERTY_LABELS[prop],  # type: ignore[index]
                "units": PROPERTY_UNITS[prop],  # type: ignore[index]
                "model RMSE": round(model_rmse, 3),
                "predict-the-mean RMSE": round(base_rmse, 3),
                "improvement": f"{100 * (1 - model_rmse / base_rmse):.0f}%",
            }
        )
    return pl.DataFrame(rows)


def ranking_summary(predictions: pl.DataFrame) -> dict[str, dict[str, float]]:
    """Overall and top-decile rank correlation, pooled over out-of-fold predictions."""
    out: dict[str, dict[str, float]] = {}
    for prop in PROPERTIES:
        sub = predictions.filter(pl.col("property") == prop)
        if not sub.height:
            continue
        truth = sub.get_column("y_true").to_numpy()
        predicted = sub.get_column("y_pred").to_numpy()
        out[prop] = {
            "overall": spearman(truth, predicted),
            "top_decile": tail_spearman(truth, predicted),
            "n": float(sub.height),
        }
    return out
