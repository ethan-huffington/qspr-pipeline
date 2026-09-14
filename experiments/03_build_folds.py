"""Build and audit the nested scaffold-disjoint cross-validation folds.

    uv run python experiments/03_build_folds.py --profile dev

Writes the fold assignment to data/processed/<profile>/ and prints the audit that
matters: per-property label counts in every outer test fold, and whether any fold
is too thin to score. Disjointness is validated inside the builder and raises on
failure, so reaching the report at all means no scaffold spans a boundary.
"""

from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from dupont_qspr.config import load_config
from dupont_qspr.contracts import PROPERTIES
from dupont_qspr.splits import build_nested_folds
from dupont_qspr.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev", choices=("smoke", "dev", "full"))
    args = parser.parse_args()

    cfg = load_config(args.profile)
    cfg.ensure_dirs()

    table_path = cfg.processed_dir / "union_table.parquet"
    if not table_path.exists():
        raise SystemExit(
            f"{table_path} not found. Run experiments/01_build_dataset.py "
            f"--profile {args.profile} first."
        )
    table = pl.read_parquet(table_path)

    with start_run(cfg, "build-folds") as logger:
        build = build_nested_folds(table, cfg)
        report = build.report(cfg.splits.min_test_labels_per_property)
        logger.log_dict(report, "fold_report")

        path = cfg.processed_dir / "folds.parquet"
        _write_folds(build, table, path)
        logger.log_metrics(
            {
                "n_groups": report["scaffolds"]["n_groups"],
                "largest_group": report["scaffolds"]["largest_group"],
                "n_thin_folds": len(build.thin_folds),
            }
        )

    _report(build, cfg, path)


def _write_folds(build, table: pl.DataFrame, path) -> None:
    """Persist fold membership per molecule, so later steps never re-split."""
    n = table.height
    outer_of = np.full(n, -1, dtype=np.int64)
    for f, split in enumerate(build.spec.outer):
        outer_of[split.test_idx] = f

    columns: dict[str, object] = {
        "smiles": table.get_column("smiles"),
        "scaffold_id": build.assignment.scaffold_id,
        "scaffold": [
            build.assignment.scaffold_smiles[i] for i in build.assignment.scaffold_id
        ],
        "outer_fold": outer_of,
    }
    for f, group in enumerate(build.spec.inner):
        inner_of = np.full(n, -1, dtype=np.int64)
        for j, split in enumerate(group):
            inner_of[split.test_idx] = j
        columns[f"inner_fold_{f}"] = inner_of
    pl.DataFrame(columns).write_parquet(path)


def _report(build, cfg, path) -> None:
    summary = build.assignment.summary()
    print(f"profile   : {cfg.profile}")
    print(f"molecules : {summary['n_molecules']:,}")
    print()
    print("SCAFFOLDS")
    print(f"  distinct groups   {summary['n_groups']:,}")
    print(f"  singletons        {summary['n_singletons']:,}")
    print(
        f"  acyclic           {summary['n_acyclic']:,}  (policy: {summary['acyclic_policy']})"
    )
    print(f"  largest group     {summary['largest_group']:,}")
    print("  biggest cores:")
    for entry in summary["largest_groups"][:5]:
        print(f"    {entry['n']:>6,}  {entry['scaffold'][:56]}")

    print()
    print(f"OUTER FOLDS  ({build.spec.n_outer} × {build.spec.n_inner} nested)")
    header = f"  {'fold':<6}{'molecules':>11}" + "".join(f"{p:>10}" for p in PROPERTIES)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for f, split in enumerate(build.spec.outer):
        counts = "".join(
            f"{build.label_counts[f, j]:>10,}" for j in range(len(PROPERTIES))
        )
        print(f"  {f:<6}{split.test_idx.size:>11,}{counts}")
    totals = build.label_counts.sum(axis=0)
    print("  " + "-" * (len(header) - 2))
    print(
        f"  {'total':<6}{sum(s.test_idx.size for s in build.spec.outer):>11,}"
        + "".join(f"{t:>10,}" for t in totals)
    )

    print()
    if build.thin_folds:
        print(f"THIN FOLDS  (below {cfg.splits.min_test_labels_per_property} labels)")
        for entry in build.thin_folds:
            print(
                f"  fold {entry['fold']} / {entry['property']}: {entry['n_labels']} labels"
            )
    else:
        print(
            f"THIN FOLDS  none — every fold clears "
            f"{cfg.splits.min_test_labels_per_property} labels per property"
        )

    print()
    print("  scaffold-disjoint at both levels: validated")
    print(f"  written: {path}")


if __name__ == "__main__":
    main()
