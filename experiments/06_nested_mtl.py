"""Track 2: nested cross-validation of the multi-task neural heads.

    uv run python experiments/06_nested_mtl.py --profile dev

Runs once per configured encoder, so step 7 can separate two questions that would
otherwise be confounded: does multi-task learning help, and what did ChemBERTa-2's
broken tokenizer cost?

This process must never import xgboost — the two link separate OpenMP runtimes and
cannot share a process. Nothing imported below reaches it.
"""

from __future__ import annotations

import argparse

from dupont_qspr.config import load_config
from dupont_qspr.contracts import PROPERTIES, PROPERTY_LABELS
from dupont_qspr.dataset import load_prepared
from dupont_qspr.tracking import start_run
from dupont_qspr.tuning.nested_mtl import run_nested_mtl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev", choices=("smoke", "dev", "full"))
    parser.add_argument(
        "--encoder", default=None, help="default: every configured encoder"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    args = parser.parse_args()

    cfg = load_config(args.profile)
    encoders = [args.encoder] if args.encoder else list(cfg.features.encoders)

    for encoder in encoders:
        data = load_prepared(cfg, path="b", encoder=encoder)
        print(f"profile   : {cfg.profile}")
        print(f"encoder   : {encoder}  ({cfg.features.encoders[encoder]})")
        print(
            f"features  : {data.n_molecules:,} x {data.n_features:,}   device={args.device}"
        )
        print(
            f"geometry  : {data.folds.n_outer} outer x {data.folds.n_inner} inner, "
            f"{cfg.tuning.n_trials} trials  ->  {data.folds.n_outer} studies"
        )
        print("            one network per fold, all three properties at once")
        print()

        with start_run(cfg, f"nested-mtl-{encoder}") as logger:
            result = run_nested_mtl(
                data, cfg, logger, encoder=encoder, device=args.device
            )
            frame = result.predictions_frame(data.smiles, data.M)
            path = cfg.processed_dir / f"oof_predictions_mtl_{encoder}.parquet"
            frame.write_parquet(path)

        _report(result)
        print(f"\n  out-of-fold predictions: {frame.height:,} rows -> {path}\n")


def _report(result) -> None:
    estimate = result.estimate()
    print()
    print(f"NESTED ESTIMATE — {result.encoder}  ({result.seconds / 60:.1f} min)")
    header = f"  {'property':<8}{'name':<20}{'RMSE':>14}{'RMSE/SD':>14}{'Spearman':>14}{'n':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for prop in PROPERTIES:
        s = estimate.get(prop)
        if not s:
            continue
        print(
            f"  {prop:<8}{PROPERTY_LABELS[prop]:<20}"
            f"{s['rmse']['mean']:>8.3f} ±{s['rmse']['std']:<5.3f}"
            f"{s['rmse_over_sd']['mean']:>8.3f} ±{s['rmse_over_sd']['std']:<5.3f}"
            f"{s['spearman']['mean']:>8.3f} ±{s['spearman']['std']:<5.3f}"
            f"{s['n_test_total']:>8,}"
        )


if __name__ == "__main__":
    main()
