"""Build both feature caches for every molecule in the union table.

    uv run python experiments/02_featurize.py --profile dev

Idempotent: molecules already cached are skipped, so re-running costs almost
nothing. The second pass reported at the end is the proof of that - it re-requests
every molecule and should compute none of them.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import polars as pl

from dupont_qspr.config import load_config
from dupont_qspr.contracts import SMILES_COLUMN
from dupont_qspr.features import build_descriptor_cache, build_encoder_caches
from dupont_qspr.features.encoders import resolve_device
from dupont_qspr.tracking import start_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dev", choices=("smoke", "dev", "full"))
    parser.add_argument("--path", default="both", choices=("a", "b", "both"))
    args = parser.parse_args()

    cfg = load_config(args.profile)
    cfg.ensure_dirs()

    table_path = cfg.processed_dir / "union_table.parquet"
    if not table_path.exists():
        raise SystemExit(
            f"{table_path} not found. Run experiments/01_build_dataset.py "
            f"--profile {args.profile} first."
        )
    smiles = pl.read_parquet(table_path).get_column(SMILES_COLUMN).to_list()

    print(f"profile   : {cfg.profile}")
    print(f"molecules : {len(smiles):,}")
    print(f"cache dir : {cfg.features_dir}  (shared across profiles)")
    print(f"device    : {resolve_device(cfg.features.device)}")
    print()

    with start_run(cfg, "featurize") as logger:
        summary = {}
        if args.path in ("a", "both"):
            summary["path_a"] = _run_cache(
                build_descriptor_cache(cfg), smiles, "PATH A"
            )
        if args.path in ("b", "both"):
            for key, cache in build_encoder_caches(cfg).items():
                primary = " (primary)" if key == cfg.features.primary_encoder else ""
                summary[f"path_b_{key}"] = _run_cache(
                    cache, smiles, f"PATH B / {key}{primary}"
                )
        logger.log_dict(summary, "featurization")
        logger.log_metrics(
            {
                f"{path}/{key}": value
                for path, stats in summary.items()
                for key, value in stats.items()
                if isinstance(value, (int, float))
            }
        )


def _run_cache(cache, smiles: list[str], label: str) -> dict:
    already = cache.contains(smiles)
    print(f"{label}  {cache.name} {cache.version}")
    print(f"  width           {cache.n_features:,}")
    print(f"  already cached  {already:,} / {len(smiles):,}")

    started = time.perf_counter()
    features = cache.transform(smiles)
    first_pass = time.perf_counter() - started

    computed = cache.stats.misses
    started = time.perf_counter()
    again = cache.transform(smiles)
    second_pass = time.perf_counter() - started

    identical = np.array_equal(features, again, equal_nan=True)
    non_finite = int((~np.isfinite(features)).sum())
    rows_affected = int((~np.isfinite(features)).any(axis=1).sum())

    print(f"  shape           {features.shape}")
    print(f"  computed        {computed:,} molecules in {first_pass:.1f}s", end="")
    print(f"  ({computed / first_pass:,.0f}/s)" if first_pass > 0 and computed else "")
    print(
        f"  second pass     {second_pass:.2f}s, {cache.stats.misses - computed} recomputed"
    )
    print(f"  reproducible    {identical}")
    if non_finite:
        print(
            f"  non-finite      {non_finite:,} cells across {rows_affected:,} molecules"
        )
    print(f"  on disk         {cache.values_path.stat().st_size / 1e6:.1f} MB")
    print()

    return {
        "name": cache.name,
        "version": cache.version,
        "n_molecules": features.shape[0],
        "n_features": features.shape[1],
        "computed_this_run": computed,
        "first_pass_seconds": round(first_pass, 2),
        "second_pass_seconds": round(second_pass, 3),
        "recomputed_on_second_pass": cache.stats.misses - computed,
        "identical_between_passes": identical,
        "non_finite_cells": non_finite,
        "molecules_with_non_finite": rows_affected,
        "megabytes_on_disk": round(cache.values_path.stat().st_size / 1e6, 1),
    }


if __name__ == "__main__":
    main()
