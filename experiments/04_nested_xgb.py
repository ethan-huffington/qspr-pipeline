"""Track 1: nested cross-validation of the XGBoost baseline.

    uv run python experiments/04_nested_xgb.py --profile dev

Reads the union table, folds and descriptor cache built by steps 1-3 and does no
featurization or splitting of its own. Runs one independent Optuna study per
(outer fold, property) - fifteen studies at the full geometry - refits each
winner on its outer-training set, and scores it once on the held-out fold.

The printed estimate is the mean of the outer scores. Numbers from the `dev`
profile are directional only; `full` is the profile whose numbers get reported.
"""

from __future__ import annotations

import argparse

from dupont_qspr.config import load_config
from dupont_qspr.contracts import PROPERTIES, PROPERTY_LABELS
from dupont_qspr.dataset import load_prepared
from dupont_qspr.tracking import start_run
from dupont_qspr.tuning.nested_run import run_nested_xgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev", choices=("smoke", "dev", "full"))
    args = parser.parse_args()

    cfg = load_config(args.profile)
    data = load_prepared(cfg, path="a")

    total_fits = (
        data.folds.n_outer
        * len(PROPERTIES)
        * (cfg.tuning.n_trials * data.folds.n_inner + 1)
    )
    print(f"profile    : {cfg.profile}")
    print(
        f"features   : {data.feature_name} {data.featurizer_version}  "
        f"({data.n_molecules:,} x {data.n_features:,})"
    )
    print(
        f"geometry   : {data.folds.n_outer} outer x {data.folds.n_inner} inner  "
        f"x {len(PROPERTIES)} properties"
    )
    print(
        f"budget     : {cfg.tuning.n_trials} trials/study, "
        f"{data.folds.n_outer * len(PROPERTIES)} studies, "
        f"up to {total_fits:,} fits before pruning"
    )
    print()

    with start_run(cfg, "nested-xgb") as logger:
        nested = run_nested_xgb(data, cfg, logger)

        # Out-of-fold predictions are the input to step 5's figures and step 8's
        # conformal calibration. Writing them here means neither has to re-run
        # three hours of search to get them back.
        predictions = nested.predictions_frame(data.smiles)
        path = cfg.processed_dir / "oof_predictions_xgb.parquet"
        predictions.write_parquet(path)
        logger.log_metrics({"oof_predictions": predictions.height})

    print(f"\n  out-of-fold predictions: {predictions.height:,} rows -> {path}")
    _report(nested)


def _report(nested) -> None:
    estimate = nested.estimate()
    print()
    print(f"NESTED ESTIMATE  ({nested.seconds / 60:.1f} min)")
    header = (
        f"  {'property':<8}{'name':<20}{'RMSE':>16}{'RMSE/SD':>14}"
        f"{'MAE':>10}{'Spearman':>16}{'n':>8}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prop in PROPERTIES:
        s = estimate.get(prop)
        if not s:
            continue
        r, n, m, rho = s["rmse"], s["rmse_over_sd"], s["mae"], s["spearman"]
        print(
            f"  {prop:<8}{PROPERTY_LABELS[prop]:<20}"
            f"{r['mean']:>9.3f} ±{r['std']:<5.3f}"
            f"{n['mean']:>8.3f} ±{n['std']:<5.3f}"
            f"{m['mean']:>10.3f}"
            f"{rho['mean']:>9.3f} ±{rho['std']:<5.3f}"
            f"{s['n_test_total']:>8,}"
        )
    print()
    print("  per-fold RMSE/SD (spread across folds is information, not noise):")
    for prop in PROPERTIES:
        s = estimate.get(prop)
        if s:
            print(f"    {prop:<8}{s['rmse_over_sd']['per_fold']}")
    print()
    print("  selected hyperparameters differ by fold, as nested CV requires:")
    for prop, entries in nested.by_property().items():
        depths = [e.best_params.get("max_depth") for e in entries]
        rates = [round(e.best_params.get("learning_rate", 0), 3) for e in entries]
        trees = [e.n_estimators_refit for e in entries]
        print(f"    {prop:<8} max_depth={depths}  lr={rates}  trees={trees}")


if __name__ == "__main__":
    main()
