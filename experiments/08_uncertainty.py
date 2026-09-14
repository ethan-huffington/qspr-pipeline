"""Calibrated prediction intervals and the applicability-domain distance.

    uv run python experiments/08_uncertainty.py --track xgb --profile dev
    uv run python experiments/08_uncertainty.py --track mtl --profile dev

Runs *after* the nested search and reuses the hyperparameters it selected, so this
is a cheap extra pass rather than a second search. For each outer fold:

  1. fit the raw-uncertainty model on the inner folds and predict their held-out
     rows  ->  calibration data, out-of-fold by construction
  2. calibrate a conformal interval on those scores
  3. fit on the whole outer-training set, predict the outer test fold, and apply
     the calibrated interval
  4. measure coverage and width on the outer test fold, which calibration never
     touched

The two tracks run as separate invocations because xgboost and torch cannot share
a process; ``--track`` decides which one is imported, and neither branch imports
the other.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import polars as pl

from dupont_qspr.config import load_config
from dupont_qspr.contracts import PROPERTIES, PROPERTY_LABELS
from dupont_qspr.dataset import load_prepared
from dupont_qspr.reporting.load import latest_nested_result
from dupont_qspr.tracking import start_run
from dupont_qspr.uncertainty import (
    ApplicabilityDomainIndex,
    ConformalizedQR,
    SplitConformal,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev", choices=("smoke", "dev", "full"))
    parser.add_argument("--track", default="xgb", choices=("xgb", "mtl"))
    parser.add_argument("--encoder", default=None, help="mtl only")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    started = time.perf_counter()

    if args.track == "xgb":
        rows, domain_rows = _run_xgb(cfg)
        label = "xgb"
    else:
        encoder = args.encoder or cfg.features.primary_encoder
        rows, domain_rows = _run_mtl(cfg, encoder)
        label = f"mtl_{encoder}"

    frame = pl.DataFrame(rows)
    with start_run(cfg, f"uncertainty-{label}") as logger:
        path = cfg.processed_dir / f"intervals_{label}.parquet"
        frame.write_parquet(path)
        pl.DataFrame(domain_rows).write_parquet(
            cfg.processed_dir / f"ad_distance_{label}.parquet"
        )
        logger.log_dict(_summary(frame, cfg), "coverage")

    _report(frame, cfg, time.perf_counter() - started, path)


def _calibrated_rows(
    data: Any,
    prop: str,
    j: int,
    fold: int,
    raw_cal,
    y_cal,
    raw_test,
    y_test,
    rows,
    coverage: float,
) -> list[dict[str, Any]]:
    """Calibrate both methods on the same scores, apply both to the same test rows."""
    out: list[dict[str, Any]] = []
    for name, calibrator in (
        ("split", SplitConformal(nominal_coverage=coverage)),
        ("cqr", ConformalizedQR(nominal_coverage=coverage)),
    ):
        bounds = calibrator.fit(raw_cal, y_cal).transform(raw_test)
        out += [
            {
                "property": prop,
                "fold": fold,
                "method": name,
                "row_index": int(row),
                "y_true": float(truth),
                "lower": float(lo),
                "upper": float(hi),
                "covered": bool(lo <= truth <= hi),
                "width": float(hi - lo),
            }
            for row, truth, (lo, hi) in zip(rows, y_test, bounds, strict=True)
        ]
    return out


def _run_xgb(cfg) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from dupont_qspr.models.xgb import XGBQuantileModel

    result, _ = latest_nested_result(cfg)
    best = {(c["property"], c["fold"]): c for c in result["cells"]}
    data = load_prepared(cfg, path="a")
    quantiles = tuple(cfg.uncertainty.quantiles)
    coverage = cfg.uncertainty.nominal_coverage

    rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for j, prop in enumerate(PROPERTIES):
        for fold in range(data.folds.n_outer):
            cell = best.get((prop, fold))
            if cell is None:
                continue
            params, trees = cell["best_params"], cell["n_estimators_refit"]
            outer = data.folds.outer[fold]

            # Calibration data: inner folds, out-of-fold by construction.
            cal_raw, cal_y = [], []
            for split in data.folds.inner[fold]:
                tr = data.labelled(split.train_idx, j)
                te = data.labelled(split.test_idx, j)
                if tr.size < 10 or te.size < 3:
                    continue
                model = XGBQuantileModel(
                    params=params, quantiles=quantiles, n_jobs=cfg.models.xgb_n_jobs
                ).fit(data.X[tr], data.Y[tr, j], n_estimators=trees)
                cal_raw.append(model.predict_quantiles(data.X[te]))
                cal_y.append(data.Y[te, j])
            if not cal_raw:
                continue

            train = data.labelled(outer.train_idx, j)
            test = data.labelled(outer.test_idx, j)
            final = XGBQuantileModel(
                params=params, quantiles=quantiles, n_jobs=cfg.models.xgb_n_jobs
            ).fit(data.X[train], data.Y[train, j], n_estimators=trees)

            rows += _calibrated_rows(
                data,
                prop,
                j,
                fold,
                np.vstack(cal_raw),
                np.concatenate(cal_y),
                final.predict_quantiles(data.X[test]),
                data.Y[test, j],
                test,
                coverage,
            )
            domain_rows += _domain_rows(data, prop, fold, train, test)
            print(
                f"  {prop:<6} fold {fold}  calibrated on {sum(len(c) for c in cal_y):,}",
                flush=True,
            )

    return rows, domain_rows


def _run_mtl(cfg, encoder: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import optuna

    from dupont_qspr.models.ensemble import DeepEnsemble
    from dupont_qspr.tuning.spaces import suggest_mtl

    result, _ = latest_nested_result(cfg, f"nested_mtl_{encoder}.json")
    data = load_prepared(cfg, path="b", encoder=encoder)
    coverage = cfg.uncertainty.nominal_coverage
    descriptors = load_prepared(cfg, path="a")  # ECFP bits for the domain index

    rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []

    for entry in result["folds"]:
        fold = entry["fold"]
        params = suggest_mtl(optuna.trial.FixedTrial(entry["best_params"]))
        outer = data.folds.outer[fold]

        # params bound as a default argument: capturing the loop variable by
        # reference would silently use the last fold's hyperparameters if anything
        # ever deferred this call.
        def build(train_idx, eval_idx, params=params):
            return DeepEnsemble(
                params=params,
                n_members=cfg.models.ensemble_seeds,
                max_epochs=cfg.models.mtl_max_epochs,
                patience=cfg.models.mtl_patience,
            ).fit(
                data.X[train_idx],
                data.Y[train_idx],
                data.M[train_idx],
                eval_set=(data.X[eval_idx], data.Y[eval_idx], data.M[eval_idx]),
            )

        cal = [
            (build(s.train_idx, s.test_idx), s.test_idx) for s in data.folds.inner[fold]
        ]
        holdout = data.folds.inner[fold][0]
        final = build(np.setdiff1d(outer.train_idx, holdout.test_idx), holdout.test_idx)

        for j, prop in enumerate(PROPERTIES):
            cal_raw, cal_y = [], []
            for ensemble, idx in cal:
                keep = data.M[idx, j]
                if keep.sum() < 3:
                    continue
                cal_raw.append(ensemble.predict_quantiles(data.X[idx[keep]])[:, j, :])
                cal_y.append(data.Y[idx[keep], j])
            if not cal_raw:
                continue
            test = data.labelled(outer.test_idx, j)
            rows += _calibrated_rows(
                data,
                prop,
                j,
                fold,
                np.vstack(cal_raw),
                np.concatenate(cal_y),
                final.predict_quantiles(data.X[test])[:, j, :],
                data.Y[test, j],
                test,
                coverage,
            )
            domain_rows += _domain_rows(
                descriptors, prop, fold, data.labelled(outer.train_idx, j), test
            )
        print(f"  fold {fold}  ensemble of {cfg.models.ensemble_seeds}", flush=True)

    return rows, domain_rows


def _domain_rows(data, prop: str, fold: int, train, test) -> list[dict[str, Any]]:
    """Distance from each test molecule to the closest training molecule."""
    index = ApplicabilityDomainIndex().fit(data.X[train])
    distances = index.distance(data.X[test])
    return [
        {"property": prop, "fold": fold, "row_index": int(r), "nn_distance": float(d)}
        for r, d in zip(test, distances, strict=True)
    ]


def _summary(frame: pl.DataFrame, cfg) -> dict[str, Any]:
    if frame.is_empty():
        return {}
    return {
        "nominal_coverage": cfg.uncertainty.nominal_coverage,
        "by_property_method": frame.group_by(["property", "method"])
        .agg(
            pl.col("covered").mean().alias("picp"),
            pl.col("width").mean().alias("mean_width"),
            pl.len().alias("n"),
        )
        .sort(["property", "method"])
        .to_dicts(),
    }


def _report(frame: pl.DataFrame, cfg, seconds: float, path) -> None:
    print()
    print(
        f"COVERAGE  (nominal {cfg.uncertainty.nominal_coverage:.0%}, {seconds / 60:.1f} min)"
    )
    header = (
        f"  {'property':<10}{'name':<22}{'method':<8}{'PICP':>8}{'width':>12}{'n':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    summary = _summary(frame, cfg).get("by_property_method", [])
    for row in summary:
        print(
            f"  {row['property']:<10}{PROPERTY_LABELS[row['property']]:<22}"
            f"{row['method']:<8}{row['picp']:>8.3f}{row['mean_width']:>12.3f}{row['n']:>8,}"
        )
    print()
    print("  coverage alone is gameable by widening intervals — read it with the width")
    print(f"  written: {path}")


if __name__ == "__main__":
    main()
