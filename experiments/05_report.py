"""Point-accuracy report: tables and figures from a completed nested run.

    uv run python experiments/05_report.py --profile dev

Reads what step 4 persisted and refits nothing. Writes figures to
artifacts/<profile>/figures/ in both light and dark variants.
"""

from __future__ import annotations

import argparse

import polars as pl

from dupont_qspr.config import load_config
from dupont_qspr.reporting import (
    baseline_table,
    latest_nested_result,
    load_oof_predictions,
    ranking_summary,
    render_all,
    results_table,
)
from dupont_qspr.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev", choices=("smoke", "dev", "full"))
    args = parser.parse_args()

    cfg = load_config(args.profile)
    result, source = latest_nested_result(cfg)
    predictions = load_oof_predictions(cfg)
    estimate = result["estimate"]

    print(f"profile   : {cfg.profile}")
    print(f"source    : {source.parent.name}")
    print(f"oof rows  : {predictions.height:,}")
    print()

    with pl.Config(tbl_rows=20, fmt_str_lengths=30, tbl_hide_dataframe_shape=True):
        print("HEADLINE  (mean across outer folds, 95% t-interval)")
        print(results_table(estimate))
        print()
        print("AGAINST THE REQUIRED BASELINE")
        print(baseline_table(predictions))
        print()

    ranking = ranking_summary(predictions)
    print("RANKING QUALITY  (the decision-relevant metric)")
    for prop, entry in ranking.items():
        print(
            f"  {prop:<6} overall rho={entry['overall']:.3f}   "
            f"top-decile rho={entry['top_decile']:.3f}   n={int(entry['n']):,}"
        )

    with start_run(cfg, "report") as logger:
        figures = render_all(
            estimate, predictions, ranking, cfg.artifacts_dir / "figures"
        )
        logger.log_dict(
            {
                "source_run": source.parent.name,
                "results": results_table(estimate).to_dicts(),
                "baselines": baseline_table(predictions).to_dicts(),
                "ranking": ranking,
            },
            "point_accuracy_report",
        )
        for path in figures:
            logger.log_artifact(path)

    print()
    print(f"FIGURES  ({len(figures)} written)")
    for path in figures:
        print(f"  {path}")


if __name__ == "__main__":
    main()
