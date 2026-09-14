"""The nested cross-validation driver, walked end to end with skeleton parts.

This is the control flow the real system keeps. Only the components it calls get
swapped out: synthetic data becomes curated data, hash features become RDKit and
ChemBERTa features, the ridge grid becomes Optuna over XGBoost and multi-task
heads, split conformal becomes CQR. The loop structure below - an independent
search inside every outer fold, a single refit, one scoring pass on data the
search never saw - is the part that must not drift, because it is what makes the
headline number honest.

Two properties of that structure are worth stating plainly, since both are easy to
break silently later:

* The hyperparameter search runs *inside* each outer fold, producing one study per
  fold. Different folds may pick different winners. That is correct: selecting a
  configuration on the same data used to report its score is how nested CV gets
  quietly turned back into a flattering single split.
* The nested run evaluates the *procedure*. The shipped model comes from one final
  search over all the data, refit once - not from the five fold models.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import (
    PROPERTIES,
    PROPERTY_UNITS,
    SMILES_COLUMN,
    FoldSpec,
    PropertyName,
    Split,
    targets_and_mask,
)
from dupont_qspr.skeleton import (
    HashFeatureCache,
    NearestNeighbourDomain,
    RidgeQSPRModel,
    SplitConformalCalibrator,
    grouped_nested_folds,
    rank_correlation,
    rmse,
    scaffold_ids_from_smiles,
    score_molecules,
    synthetic_union_table,
)
from dupont_qspr.tracking import RunLogger, start_run

__all__ = ["label_availability", "run_spine"]


def label_availability(
    M: npt.NDArray[np.bool_],
) -> dict[str, Any]:
    """Counts per property and per label-combination.

    Brief §4 asks for this as a first-class artifact rather than a log line,
    because it is the evidence for the multi-task design: if almost every row
    carried all three labels there would be nothing for a masked loss to buy.
    """
    combinations: dict[str, int] = {}
    for row in M:
        key = "+".join(p for p, present in zip(PROPERTIES, row, strict=True) if present)
        combinations[key] = combinations.get(key, 0) + 1
    return {
        "n_molecules": int(M.shape[0]),
        "per_property": {p: int(M[:, j].sum()) for j, p in enumerate(PROPERTIES)},
        "per_combination": dict(sorted(combinations.items(), key=lambda kv: -kv[1])),
        "n_complete_rows": int(M.all(axis=1).sum()),
    }


def _normalised_inner_score(
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float64],
    M: npt.NDArray[np.bool_],
    splits: Sequence[Split],
    alpha: float,
) -> float:
    """Mean over inner folds of RMSE divided by the training standard deviation.

    Normalising per property is what makes a single scalar legitimate here: melting
    point in Kelvin would otherwise dominate a raw average of three RMSEs and the
    search would tune for it alone. A value near 1 means no better than predicting
    the mean.
    """
    fold_scores: list[float] = []
    for split in splits:
        model = RidgeQSPRModel(alpha=alpha).fit(
            X[split.train_idx], Y[split.train_idx], M[split.train_idx]
        )
        predicted = model.predict(X[split.test_idx])
        per_property: list[float] = []
        for j in range(len(PROPERTIES)):
            rows = np.flatnonzero(M[split.test_idx, j])
            if rows.size < 2:
                continue
            truth = Y[split.test_idx][rows, j]
            spread = float(Y[split.train_idx][M[split.train_idx, j], j].std()) or 1.0
            per_property.append(rmse(truth, predicted[rows, j]) / spread)
        if per_property:
            fold_scores.append(float(np.mean(per_property)))
    return float(np.mean(fold_scores)) if fold_scores else float("inf")


def _out_of_fold_predictions(
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float64],
    M: npt.NDArray[np.bool_],
    splits: Sequence[Split],
    alpha: float,
    quantiles: Sequence[float],
) -> dict[PropertyName, tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]]:
    """Collect held-out quantile predictions from the inner folds, per property.

    Reusing the inner folds for calibration is the data-efficient path the brief
    calls for. Carving off a dedicated calibration split would take rows straight
    out of training at a scale where every row counts.
    """
    collected: dict[PropertyName, list[tuple[npt.NDArray, npt.NDArray]]] = {
        p: [] for p in PROPERTIES
    }
    for split in splits:
        model = RidgeQSPRModel(alpha=alpha).fit(
            X[split.train_idx], Y[split.train_idx], M[split.train_idx]
        )
        raw = model.predict_quantiles(X[split.test_idx], quantiles)
        for j, prop in enumerate(PROPERTIES):
            rows = np.flatnonzero(M[split.test_idx, j])
            if rows.size:
                collected[prop].append((raw[rows, j, :], Y[split.test_idx][rows, j]))

    out: dict[PropertyName, tuple[npt.NDArray, npt.NDArray]] = {}
    for prop, chunks in collected.items():
        if chunks:
            out[prop] = (
                np.concatenate([c[0] for c in chunks]),
                np.concatenate([c[1] for c in chunks]),
            )
    return out


def _evaluate_outer_fold(
    X: npt.NDArray[np.float32],
    Y: npt.NDArray[np.float64],
    M: npt.NDArray[np.bool_],
    split: Split,
    model: RidgeQSPRModel,
    calibrators: dict[PropertyName, SplitConformalCalibrator],
    quantiles: Sequence[float],
) -> dict[str, dict[str, float]]:
    """Per-property metrics on the held-out outer fold. Never pooled across properties."""
    predicted = model.predict(X[split.test_idx])
    raw = model.predict_quantiles(X[split.test_idx], quantiles)

    results: dict[str, dict[str, float]] = {}
    for j, prop in enumerate(PROPERTIES):
        rows = np.flatnonzero(M[split.test_idx, j])
        if rows.size < 2:
            results[prop] = {"n": float(rows.size)}
            continue
        truth = Y[split.test_idx][rows, j]
        point = predicted[rows, j]
        train_spread = float(Y[split.train_idx][M[split.train_idx, j], j].std()) or 1.0

        metrics = {
            "n": float(rows.size),
            "rmse": rmse(truth, point),
            "mae": float(np.mean(np.abs(truth - point))),
            "rmse_over_sd": rmse(truth, point) / train_spread,
            "spearman": rank_correlation(truth, point),
        }
        if prop in calibrators:
            bounds = calibrators[prop].transform(raw[rows, j, :])
            inside = (truth >= bounds[:, 0]) & (truth <= bounds[:, 1])
            metrics["picp"] = float(inside.mean())
            metrics["mean_interval_width"] = float(np.mean(bounds[:, 1] - bounds[:, 0]))
        results[prop] = metrics
    return results


def _fold_label_counts(
    M: npt.NDArray[np.bool_], spec: FoldSpec
) -> list[dict[str, int]]:
    return [
        {p: int(M[split.test_idx, j].sum()) for j, p in enumerate(PROPERTIES)}
        for split in spec.outer
    ]


def run_spine(cfg: Config, logger: RunLogger | None = None) -> dict[str, Any]:
    """Walk every stage once and return a summary.

    Returns the same dictionary it logs, so tests can assert on the structure
    without going through the tracking backend.
    """
    started = time.perf_counter()
    cfg.ensure_dirs()

    if logger is None:
        with start_run(cfg, "smoke-spine") as opened:
            return run_spine(cfg, opened)

    # ---- stages 1-3: data, features, folds ---------------------------------- #
    table = synthetic_union_table(cfg)
    smiles = table.get_column(SMILES_COLUMN).to_list()
    Y, M = targets_and_mask(table)

    availability = label_availability(M)
    logger.log_dict(availability, "label_availability")

    cache = HashFeatureCache()
    X = cache.transform(smiles)
    cache_misses_first_pass = cache.misses
    cache.transform(smiles)  # second pass must be served entirely from cache
    logger.log_metrics(
        {
            "cache_misses_first_pass": cache_misses_first_pass,
            "cache_misses_second_pass": cache.misses - cache_misses_first_pass,
            "n_features": cache.n_features,
        }
    )

    spec = grouped_nested_folds(scaffold_ids_from_smiles(smiles), cfg)
    spec.validate_disjoint()
    fold_counts = _fold_label_counts(M, spec)
    logger.log_dict(
        {
            "n_outer": spec.n_outer,
            "n_inner": spec.n_inner,
            "n_scaffolds": int(np.unique(spec.scaffold_id).size),
            "test_labels_per_fold": fold_counts,
            "min_required": cfg.splits.min_test_labels_per_property,
            "thin_folds": [
                {"fold": i, "property": p}
                for i, counts in enumerate(fold_counts)
                for p in PROPERTIES
                if counts[p] < cfg.splits.min_test_labels_per_property
            ],
        },
        "fold_report",
    )

    # ---- stages 4-9: the nested loop ---------------------------------------- #
    # A small grid of ridge penalties stands in for an Optuna study. One candidate
    # here is one trial there: scored by inner-fold mean, never by outer-test.
    candidates = np.geomspace(1e-2, 1e3, num=max(2, cfg.tuning.n_trials)).tolist()
    quantiles = list(cfg.uncertainty.quantiles)

    outer_results: list[dict[str, Any]] = []
    for fold_index, (split, inner_splits) in enumerate(
        zip(spec.outer, spec.inner, strict=True)
    ):
        # Inner splits index the union table, not the outer-train slice, so the
        # full arrays are passed through and the indexing stays global everywhere.
        scored = [
            (alpha, _normalised_inner_score(X, Y, M, inner_splits, alpha))
            for alpha in candidates
        ]
        best_alpha, best_inner = min(scored, key=lambda pair: pair[1])

        # Calibrate on inner out-of-fold predictions, then refit on all outer-train.
        oof = _out_of_fold_predictions(X, Y, M, inner_splits, best_alpha, quantiles)
        calibrators = {
            prop: SplitConformalCalibrator(
                nominal_coverage=cfg.uncertainty.nominal_coverage
            ).fit(raw, truth)
            for prop, (raw, truth) in oof.items()
        }
        model = RidgeQSPRModel(alpha=best_alpha).fit(
            X[split.train_idx], Y[split.train_idx], M[split.train_idx]
        )

        metrics = _evaluate_outer_fold(X, Y, M, split, model, calibrators, quantiles)
        outer_results.append(
            {
                "fold": fold_index,
                "best_params": {"alpha": best_alpha},
                "inner_score": best_inner,
                "n_train": int(split.train_idx.size),
                "n_test": int(split.test_idx.size),
                "metrics": metrics,
            }
        )
        for prop, values in metrics.items():
            logger.log_metrics(
                {f"outer/{prop}/{k}": v for k, v in values.items()}, step=fold_index
            )

    nested = _aggregate(outer_results)
    logger.log_dict(
        {"outer_folds": outer_results, "nested_estimate": nested}, "nested_results"
    )
    logger.log_metrics(
        {
            f"nested/{prop}/{k}": v["mean"]
            for prop, m in nested.items()
            for k, v in m.items()
        }
    )

    # ---- stage 11: one final fit over everything, then score a batch --------- #
    final_scored = [
        (alpha, _normalised_inner_score(X, Y, M, spec.outer, alpha))
        for alpha in candidates
    ]
    final_alpha = min(final_scored, key=lambda pair: pair[1])[0]
    final_oof = _out_of_fold_predictions(X, Y, M, spec.outer, final_alpha, quantiles)
    final_calibrators = {
        prop: SplitConformalCalibrator(
            nominal_coverage=cfg.uncertainty.nominal_coverage
        ).fit(raw, truth)
        for prop, (raw, truth) in final_oof.items()
    }
    final_model = RidgeQSPRModel(alpha=final_alpha).fit(X, Y, M)
    domain = NearestNeighbourDomain().fit(X)

    demo = [smiles[0], smiles[len(smiles) // 2], "S9999_M999", "not a molecule", ""]
    records = score_molecules(
        demo,
        model=final_model,
        cache=cache,
        calibrators=final_calibrators,
        domain=domain,
        cfg=cfg,
        model_version=f"skeleton-ridge-alpha{final_alpha:.4g}",
    )
    logger.log_dict({"inputs": demo, "records": records}, "scored_records")

    elapsed = time.perf_counter() - started
    summary = {
        "profile": cfg.profile,
        "elapsed_s": elapsed,
        "label_availability": availability,
        "fold_label_counts": fold_counts,
        "outer_folds": outer_results,
        "nested_estimate": nested,
        "final_params": {"alpha": final_alpha},
        "scored_records": records,
    }
    logger.log_metrics({"elapsed_s": elapsed})
    logger.log_dict(summary, "summary")
    return summary


def _aggregate(
    outer_results: Sequence[dict[str, Any]],
) -> dict[str, dict[str, dict[str, float]]]:
    """Mean and spread across outer folds, per property, per metric.

    The spread is reported rather than smoothed away: with five folds and thin
    per-property coverage, a wide interval is information, not an embarrassment.
    """
    out: dict[str, dict[str, dict[str, float]]] = {}
    for prop in PROPERTIES:
        collected: dict[str, list[float]] = {}
        for fold in outer_results:
            for key, value in fold["metrics"].get(prop, {}).items():
                if key != "n" and np.isfinite(value):
                    collected.setdefault(key, []).append(value)
        out[prop] = {
            key: {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n_folds": len(values),
                "units": PROPERTY_UNITS[prop] if key in {"rmse", "mae"} else "",
            }
            for key, values in collected.items()
        }
    return out


def summary_table(summary: dict[str, Any]) -> pl.DataFrame:
    """Nested estimate as a scannable table."""
    rows = [
        {
            "property": prop,
            "units": PROPERTY_UNITS[prop],
            **{
                key: round(stats["mean"], 4)
                for key, stats in metrics.items()
                if key
                in {
                    "rmse",
                    "mae",
                    "rmse_over_sd",
                    "spearman",
                    "picp",
                    "mean_interval_width",
                }
            },
        }
        for prop, metrics in summary["nested_estimate"].items()
    ]
    return pl.DataFrame(rows)
