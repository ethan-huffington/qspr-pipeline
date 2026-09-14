"""Steps 7 and 9: compare the tracks, check conditional coverage, set the AD threshold.

    uv run python experiments/07_analyze.py --profile full

Run after 04, 06 and 08. Reads only persisted artifacts and trains nothing, so it
imports neither XGBoost nor PyTorch. Writes artifacts/<profile>/decision.json, which
experiments/11_final_fit.py reads to know which family to ship and which threshold
to embed.
"""

from __future__ import annotations

import argparse
import json

import polars as pl

from dupont_qspr.analysis import (
    choose_ad_threshold,
    comparison_table,
    conditional_coverage,
    decide_family,
    error_vs_distance,
    load_interval_frame,
    load_track_predictions,
)
from dupont_qspr.analysis.figures import render_conditional_coverage
from dupont_qspr.config import load_config
from dupont_qspr.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full", choices=("smoke", "dev", "full"))
    args = parser.parse_args()
    cfg = load_config(args.profile)

    tracks = load_track_predictions(cfg)
    table = comparison_table(tracks)
    decision = decide_family(tracks)

    with pl.Config(
        tbl_rows=30, tbl_cols=12, float_precision=3, tbl_hide_dataframe_shape=True
    ):
        print("STEP 7 — TRACKS ON IDENTICAL HELD-OUT MOLECULES")
        print(
            table.select(
                "property",
                "track",
                "n",
                "rmse",
                "rmse_ci_low",
                "rmse_ci_high",
                "rmse_over_sd",
                "spearman",
                "ef10_upper",
                "ef10_lower",
            )
        )
    print()
    for prop, interval in decision.differences.items():
        print(
            f"  {prop:<6} RMSE(xgb) − RMSE({decision.mtl_candidate}) = {interval['mean']:+.3f} "
            f"[{interval['lower']:+.3f}, {interval['upper']:+.3f}]  ->  {decision.winners[prop]}"
        )
    print(
        f"\n  SHIP: {decision.family}"
        + (f" ({decision.encoder})" if decision.encoder else "")
    )
    print(f"  {decision.rationale}\n")

    ship_label = "xgb" if decision.family == "xgb" else f"mtl_{decision.encoder}"
    coverage_tables: dict[str, pl.DataFrame] = {}
    figures = []
    for label in ["xgb", *(f"mtl_{e}" for e in cfg.features.encoders)]:
        try:
            frame = load_interval_frame(cfg, label)
        except FileNotFoundError:
            continue
        coverage = conditional_coverage(frame, cfg.uncertainty.ad_bands)
        coverage_tables[label] = coverage
        figures += render_conditional_coverage(
            coverage,
            cfg.uncertainty.nominal_coverage,
            label,
            cfg.artifacts_dir / "figures",
        )
        with pl.Config(tbl_rows=40, float_precision=3, tbl_hide_dataframe_shape=True):
            print(f"STEP 9 — COVERAGE BY DISTANCE BAND — {label}")
            print(
                coverage.select(
                    "property", "method", "band_label", "n", "picp", "mean_width"
                )
            )
        print()

    distance_path = next(
        (
            cfg.processed_dir / f"ad_distance_{lab}.parquet"
            for lab in (ship_label, "xgb")
            if (cfg.processed_dir / f"ad_distance_{lab}.parquet").exists()
        ),
        None,
    )
    threshold, per_property = None, {}
    if distance_path is not None:
        source = (
            tracks[ship_label] if ship_label in tracks else next(iter(tracks.values()))
        )
        curve = error_vs_distance(source, pl.read_parquet(distance_path))
        threshold, per_property = choose_ad_threshold(
            curve, tolerance=cfg.uncertainty.ad_error_tolerance
        )
        print(f"AD THRESHOLD  {threshold}   per property: {per_property}")
    else:
        print("AD THRESHOLD  not set — run experiments/08_uncertainty.py first")

    result = {
        **decision.to_dict(),
        "ad_threshold": threshold,
        "ad_threshold_per_property": per_property,
        "ad_error_tolerance": cfg.uncertainty.ad_error_tolerance,
        "comparison": table.to_dicts(),
    }
    path = cfg.artifacts_dir / "decision.json"
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str))

    with start_run(cfg, "analyze") as logger:
        logger.log_dict(result, "decision")
        for label, coverage in coverage_tables.items():
            logger.log_dict(coverage.to_dicts(), f"conditional_coverage_{label}")
        for figure in figures:
            logger.log_artifact(figure)
    print(f"\n  written: {path}")


if __name__ == "__main__":
    main()
