"""Acquire, curate, and join the three source datasets.

    uv run python experiments/01_build_dataset.py --profile dev

Writes the union table and its audit trail to data/processed/, and logs the
label-availability matrix, the curation ledger, and the measurement-noise estimate
as run artifacts. A source that cannot be obtained is reported rather than fatal;
pass --require-all to make it fatal instead.
"""

from __future__ import annotations

import argparse

import polars as pl

from dupont_qspr.config import load_config
from dupont_qspr.data.build import build_dataset, write_dataset
from dupont_qspr.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev", choices=("smoke", "dev", "full"))
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="fail if any source is unavailable, rather than continuing without it",
    )
    args = parser.parse_args()

    cfg = load_config(args.profile)

    with start_run(cfg, "build-dataset") as logger:
        build = build_dataset(cfg, require_all=args.require_all)
        paths = write_dataset(build, cfg)

        logger.log_dict(build.availability, "label_availability")
        logger.log_dict(build.curation_summary(), "curation_summary")
        logger.log_dict(build.measurement_noise(), "measurement_noise")
        logger.log_dict(
            {
                "acquired": [a.describe() for a in build.acquisitions],
                "unavailable": build.unavailable,
                "written": paths,
            },
            "provenance",
        )
        logger.log_metrics(
            {
                "n_molecules": build.table.height,
                "n_rejected": build.ledger.height,
                "n_mp_conflicts": build.conflicts.height,
                **{
                    f"labels/{p}": v
                    for p, v in build.availability["per_property"].items()
                },
            }
        )

    _report(build, paths)


def _report(build, paths: dict[str, str]) -> None:
    print("SOURCES")
    for acquisition in build.acquisitions:
        print(f"  {acquisition.describe()}")
    for key, message in build.unavailable.items():
        first_line = message.strip().splitlines()[0]
        print(f"  {key:<14} UNAVAILABLE     {first_line}")

    availability = build.availability
    print()
    print(f"UNION TABLE  {availability['n_molecules']:,} molecules")
    print(f"  per property     {availability['per_property']}")
    print(f"  pairwise overlap {availability['pairwise_overlap']}")
    print(f"  all three        {availability['n_complete_rows']:,}")
    print()
    print("  label combinations")
    for combination, count in availability["per_combination"].items():
        share = 100 * count / availability["n_molecules"]
        print(f"    {combination:<22} {count:>7,}  {share:5.1f}%")

    print()
    print(f"CURATION LEDGER  {build.ledger.height:,} rows rejected or collapsed")
    if not build.ledger.is_empty():
        summary = (
            build.ledger.group_by(["source", "stage", "reason"])
            .len()
            .sort("len", descending=True)
            .rename({"len": "count"})
        )
        with pl.Config(tbl_rows=25, fmt_str_lengths=40):
            print(summary)

    noise = build.measurement_noise()
    if noise:
        print()
        print("MEASUREMENT NOISE  (floor on achievable RMSE)")
        for prop, stats in noise.items():
            bits = ", ".join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in stats.items()
                if k != "units"
            )
            print(f"  {prop:<8} {bits}  [{stats['units']}]")

    print()
    print("WRITTEN")
    for name, path in paths.items():
        print(f"  {name:<20} {path}")


if __name__ == "__main__":
    main()
