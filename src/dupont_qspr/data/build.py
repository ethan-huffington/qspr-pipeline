"""Orchestrate acquisition, curation, and the join into one reproducible artifact.

Deliberately tolerant of a missing source. A dataset waiting on a manual download
should not stop the other two from being curated and joined - the union table's
schema is the same either way, and the label-availability matrix already says
which properties are represented. What must never happen is a source going missing
*quietly*, so anything unavailable is reported loudly and recorded in the build
summary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES, SMILES_COLUMN, UNION_TABLE_SCHEMA
from dupont_qspr.data.curate_mp import curate_bradley, load_bradley_raw
from dupont_qspr.data.download import Acquisition, SourceUnavailableError, ensure_source
from dupont_qspr.data.sources import AQSOLDB, BRADLEY, LIPOPHILICITY
from dupont_qspr.data.standardize import LEDGER_SCHEMA
from dupont_qspr.data.union import (
    build_union_table,
    label_availability_matrix,
    load_aqsoldb,
    load_lipophilicity,
)

__all__ = ["DatasetBuild", "build_dataset"]


@dataclass(slots=True)
class DatasetBuild:
    """Everything step 1 produces."""

    table: pl.DataFrame
    ledger: pl.DataFrame
    conflicts: pl.DataFrame
    availability: dict[str, Any]
    acquisitions: list[Acquisition] = field(default_factory=list)
    unavailable: dict[str, str] = field(default_factory=dict)

    def curation_summary(self) -> dict[str, Any]:
        """Rejection counts by source, stage, and reason - the audit trail."""
        if self.ledger.is_empty():
            by_reason: list[dict[str, Any]] = []
        else:
            by_reason = (
                self.ledger.group_by(["source", "stage", "reason"])
                .len()
                .sort("len", descending=True)
                .rename({"len": "count"})
                .to_dicts()
            )
        return {
            "n_rejected_total": self.ledger.height,
            "by_source_stage_reason": by_reason,
            "n_conflicting_compounds": self.conflicts.height,
        }

    def measurement_noise(self) -> dict[str, Any]:
        """Inter-source disagreement, which bounds the achievable RMSE.

        A reported error below this floor would not mean the model is better than
        the measurement; it would mean it had learned one source's idiosyncrasies.
        """
        out: dict[str, Any] = {}
        if "mp_spread_k" in self.table.columns:
            spread = self.table.get_column("mp_spread_k").drop_nulls()
            replicated = spread.filter(spread > 0)
            if replicated.len():
                out["mp_K"] = {
                    "n_with_replicates": replicated.len(),
                    "median_spread": float(replicated.median()),
                    "mean_spread": float(replicated.mean()),
                    "p90_spread": float(replicated.quantile(0.9)),
                    "units": "K",
                }
        if "logS_sd" in self.table.columns:
            sd = self.table.get_column("logS_sd").drop_nulls()
            replicated = sd.filter(sd > 0)
            if replicated.len():
                out["logS"] = {
                    "n_with_replicates": replicated.len(),
                    "median_sd": float(replicated.median()),
                    "mean_sd": float(replicated.mean()),
                    "p90_sd": float(replicated.quantile(0.9)),
                    "units": "log mol/L",
                }
        return out


def _subsample(table: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Cap the table for the smoke profile, keeping every property represented.

    A naive head() would order by SMILES and could return a slice covering only
    one source. Sampling is seeded so the smoke profile is still reproducible.
    """
    limit = cfg.data.max_molecules
    if limit is None or table.height <= limit:
        return table

    # Take a proportional slice of each label-combination stratum so the sparsity
    # structure of the full table survives into the smoke profile.
    stratum = pl.concat_str(
        [pl.col(p).is_not_null().cast(pl.Int8).cast(pl.String) for p in PROPERTIES]
    ).alias("_stratum")
    fraction = limit / table.height
    sampled = (
        table.with_columns(stratum)
        .group_by("_stratum")
        .map_groups(
            lambda group: group.sample(
                n=max(1, round(group.height * fraction)),
                seed=cfg.seed,
                shuffle=True,
            )
        )
        .drop("_stratum")
    )
    return sampled.sort(SMILES_COLUMN).head(limit)


def build_dataset(cfg: Config, *, require_all: bool = False) -> DatasetBuild:
    """Acquire, curate, and join. Returns everything needed to audit the result."""
    cfg.ensure_dirs()
    ledgers: list[pl.DataFrame] = []
    unavailable: dict[str, str] = {}
    acquisitions: list[Acquisition] = []

    solubility: pl.DataFrame | None = None
    lipophilicity: pl.DataFrame | None = None
    melting_point: pl.DataFrame | None = None
    conflicts = pl.DataFrame(
        schema={
            "smiles": pl.String,
            "n_measurements": pl.Int32,
            "min_k": pl.Float64,
            "max_k": pl.Float64,
            "median_k": pl.Float64,
            "spread_k": pl.Float64,
            "sources": pl.String,
        }
    )

    shared = {
        "mixture_ratio": cfg.data.mixture_fragment_ratio,
        "keep_stereochemistry": cfg.data.keep_stereochemistry,
    }

    # --- solubility --------------------------------------------------------- #
    try:
        acquired = ensure_source(AQSOLDB, cfg.raw_dir)
        acquisitions.append(acquired)
        solubility, ledger = load_aqsoldb(
            acquired.path, bounds=cfg.data.ranges["logS"], **shared
        )
        ledgers.append(ledger)
    except SourceUnavailableError as error:
        if require_all:
            raise
        unavailable[AQSOLDB.key] = str(error)

    # --- lipophilicity ------------------------------------------------------ #
    try:
        acquired = ensure_source(LIPOPHILICITY, cfg.raw_dir)
        acquisitions.append(acquired)
        lipophilicity, ledger = load_lipophilicity(
            acquired.path, bounds=cfg.data.ranges["logP"], **shared
        )
        ledgers.append(ledger)
    except SourceUnavailableError as error:
        if require_all:
            raise
        unavailable[LIPOPHILICITY.key] = str(error)

    # --- melting point ------------------------------------------------------ #
    try:
        acquired = ensure_source(BRADLEY, cfg.raw_dir)
        acquisitions.append(acquired)
        melting_point, ledger, conflicts = curate_bradley(
            load_bradley_raw(acquired.path),
            subset=cfg.data.bradley_subset,
            tolerance_k=cfg.data.mp_conflict_tolerance_k,
            **shared,
        )
        ledgers.append(ledger)
        melting_point, range_ledger = _screen_mp(melting_point, cfg)
        ledgers.append(range_ledger)
    except (SourceUnavailableError, FileNotFoundError) as error:
        if require_all:
            raise
        unavailable[BRADLEY.key] = str(error)

    if solubility is None and lipophilicity is None and melting_point is None:
        raise SourceUnavailableError(
            "No data source could be obtained:\n\n" + "\n\n".join(unavailable.values())
        )

    table = build_union_table(
        solubility=solubility, lipophilicity=lipophilicity, melting_point=melting_point
    )
    table = _subsample(table, cfg)

    ledger = (
        pl.concat(ledgers, how="vertical")
        if ledgers
        else pl.DataFrame(schema=LEDGER_SCHEMA)
    )
    return DatasetBuild(
        table=table,
        ledger=ledger,
        conflicts=conflicts,
        availability=label_availability_matrix(table),
        acquisitions=acquisitions,
        unavailable=unavailable,
    )


def _screen_mp(frame: pl.DataFrame, cfg: Config) -> tuple[pl.DataFrame, pl.DataFrame]:
    from dupont_qspr.data.standardize import screen_range

    return screen_range(
        frame,
        column="mp_K",
        prop="mp_K",
        bounds=cfg.data.ranges["mp_K"],
        source="bradley",
    )


def write_dataset(build: DatasetBuild, cfg: Config) -> dict[str, str]:
    """Persist the union table and its audit trail. Returns the paths written."""
    written: dict[str, str] = {}

    table_path = cfg.processed_dir / "union_table.parquet"
    build.table.write_parquet(table_path)
    written["union_table"] = str(table_path)

    ledger_path = cfg.processed_dir / "curation_ledger.parquet"
    build.ledger.write_parquet(ledger_path)
    written["curation_ledger"] = str(ledger_path)

    if build.conflicts.height:
        conflicts_path = cfg.processed_dir / "mp_conflicts.parquet"
        build.conflicts.write_parquet(conflicts_path)
        written["mp_conflicts"] = str(conflicts_path)

    # A stable, schema-only view for anything that just wants the target matrix.
    core_path = cfg.processed_dir / "union_table_core.parquet"
    build.table.select(list(UNION_TABLE_SCHEMA)).write_parquet(core_path)
    written["union_table_core"] = str(core_path)

    return written
