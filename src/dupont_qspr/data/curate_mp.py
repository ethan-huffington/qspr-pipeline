"""Curate the Bradley melting-point set, which is the messiest of the three.

Two things make this dataset different from the other two, and both are handled
here rather than swept aside.

**It carries explicit "do not use" flags.** Bradley annotated entries that are
salts, mixtures, decomposition points rather than melting points, or plainly
out-of-range transcriptions. Ignoring those flags is the most common way this
dataset gets misused.

**It carries multiple measurements per compound.** The same molecule appears with
values from several literature sources, and those sources disagree. The brief is
emphatic that the disagreement must be *recorded rather than averaged away*,
because it sets a floor on achievable accuracy: no model can be more accurate than
the data is self-consistent, and reporting an RMSE below that spread would mean
the model had learned the idiosyncrasies of one source rather than the physics.

So replicates are collapsed to a median - robust to a single bad transcription in
a way the mean is not - while the spread, the count, and the contributing sources
travel alongside as columns, and every disagreement wider than the tolerance is
written to a conflict report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import polars as pl

from dupont_qspr.data.standardize import LEDGER_SCHEMA, RejectReason, standardize_frame

__all__ = [
    "CONFLICT_SCHEMA",
    "KELVIN_OFFSET",
    "curate_bradley",
    "load_bradley_raw",
    "resolve_replicates",
]

KELVIN_OFFSET = 273.15

#: Reported alongside the curated table. Each row is one compound whose replicate
#: measurements disagree by more than the configured tolerance.
CONFLICT_SCHEMA: dict[str, pl.DataType] = {
    "smiles": pl.String(),
    "n_measurements": pl.Int32(),
    "min_k": pl.Float64(),
    "max_k": pl.Float64(),
    "median_k": pl.Float64(),
    "spread_k": pl.Float64(),
    "sources": pl.String(),
}

# The published workbook's column names, with the variants seen across releases.
_SMILES_COLUMNS = ("smiles", "smile", "canonical_smiles")
_VALUE_COLUMNS = ("mpc", "mp_c", "mp", "meltingpoint", "melting_point")
_FLAG_COLUMNS = ("donotuse", "do_not_use", "do not use")
_FLAG_REASON_COLUMNS = ("donotusebecause", "do_not_use_because", "donotusereason")
_SOURCE_COLUMNS = ("source", "reference")


def _resolve_column(
    frame: pl.DataFrame,
    candidates: tuple[str, ...],
    what: str,
    *,
    required: bool = True,
) -> str | None:
    """Find a column by normalized name, or explain what was actually there.

    The workbook is fetched from an external publisher whose column naming has
    drifted between releases. Guessing wrong should produce a readable error, not
    a KeyError three functions deeper.
    """
    lookup = {c.strip().lower().replace(" ", "_"): c for c in frame.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace(" ", "_")
        if key in lookup:
            return lookup[key]
    if not required:
        return None
    raise ValueError(
        f"Could not find the {what} column. Looked for {list(candidates)}; "
        f"the file actually has {frame.columns}."
    )


def load_bradley_raw(path: Path) -> pl.DataFrame:
    """Read the published workbook as-is, with no interpretation."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is not present. Run the downloader, or place the file by hand "
            "as described in dupont_qspr.data.sources.BRADLEY.manual_instructions."
        )
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pl.read_excel(path)
    return pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)


def resolve_replicates(
    frame: pl.DataFrame,
    *,
    value_column: str,
    tolerance: float,
    source_column: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collapse repeated measurements per molecule, keeping the disagreement visible.

    Returns ``(resolved, conflicts)``. ``resolved`` has one row per canonical
    SMILES with the median value, the number of measurements, and their spread.
    ``conflicts`` lists only those compounds whose spread exceeds ``tolerance``.

    The spread column is the useful output here, not just a diagnostic: averaged
    over the dataset it estimates the irreducible measurement noise, which is the
    number a reported RMSE has to be read against.
    """
    source_expr = (
        pl.col(source_column)
        .cast(pl.String)
        .unique()
        .sort()
        .str.join("; ")
        .alias("sources")
        if source_column is not None
        else pl.lit(None, dtype=pl.String).alias("sources")
    )

    grouped = (
        frame.group_by("smiles")
        .agg(
            pl.col(value_column).median().alias("median_k"),
            pl.col(value_column).min().alias("min_k"),
            pl.col(value_column).max().alias("max_k"),
            pl.len().cast(pl.Int32).alias("n_measurements"),
            source_expr,
        )
        .with_columns((pl.col("max_k") - pl.col("min_k")).alias("spread_k"))
        .sort("smiles")
    )

    conflicts = (
        grouped.filter(pl.col("spread_k") > tolerance)
        .select(
            "smiles",
            "n_measurements",
            "min_k",
            "max_k",
            "median_k",
            "spread_k",
            "sources",
        )
        .sort("spread_k", descending=True)
        .cast(CONFLICT_SCHEMA)  # type: ignore[arg-type]
    )
    return grouped, conflicts


def curate_bradley(
    raw: pl.DataFrame,
    *,
    subset: Literal["full", "double_plus_good"] = "full",
    tolerance_k: float = 5.0,
    mixture_ratio: float = 0.5,
    keep_stereochemistry: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Run the whole melting-point curation. Returns ``(curated, ledger, conflicts)``.

    ``subset="double_plus_good"`` keeps only compounds whose replicate measurements
    agree within ``tolerance_k``, approximating the published high-purity subset of
    roughly 3,000 entries. It is an approximation built from our own agreement
    criterion rather than a download of that exact file, so it is comparable in
    spirit but not row-identical - worth stating plainly in the final report.
    """
    smiles_column = _resolve_column(raw, _SMILES_COLUMNS, "SMILES")
    value_column = _resolve_column(raw, _VALUE_COLUMNS, "melting point")
    flag_column = _resolve_column(raw, _FLAG_COLUMNS, "do-not-use flag", required=False)
    reason_column = _resolve_column(
        raw, _FLAG_REASON_COLUMNS, "do-not-use reason", required=False
    )
    source_column = _resolve_column(raw, _SOURCE_COLUMNS, "source", required=False)

    ledgers: list[pl.DataFrame] = []
    frame = raw

    # 1. Honour the publisher's own exclusions before doing anything else.
    if flag_column is not None:
        flagged = frame.filter(
            pl.col(flag_column)
            .cast(pl.String)
            .str.strip_chars()
            .str.to_lowercase()
            .is_in(["x", "1", "true", "yes"])
        )
        if flagged.height:
            ledgers.append(
                pl.DataFrame(
                    {
                        "source": ["bradley"] * flagged.height,
                        "stage": ["donotuse_flag"] * flagged.height,
                        "reason": ["flagged_do_not_use"] * flagged.height,
                        "detail": (
                            flagged.get_column(reason_column).cast(pl.String).to_list()
                            if reason_column is not None
                            else [""] * flagged.height
                        ),
                        "smiles_in": flagged.get_column(smiles_column)
                        .cast(pl.String)
                        .to_list(),
                        "value": flagged.get_column(value_column)
                        .cast(pl.Float64, strict=False)
                        .to_list(),
                    },
                    schema=LEDGER_SCHEMA,
                )
            )
            frame = frame.join(flagged.select(pl.all()), on=frame.columns, how="anti")

    # 2. Standardize structures.
    frame, ledger = standardize_frame(
        frame,
        smiles_column=smiles_column,
        source="bradley",
        value_column=value_column,
        mixture_ratio=mixture_ratio,
        keep_stereochemistry=keep_stereochemistry,
    )
    ledgers.append(ledger)

    # 3. Harmonize units: the workbook records degrees Celsius.
    frame = frame.with_columns(
        (pl.col(value_column).cast(pl.Float64, strict=False) + KELVIN_OFFSET).alias(
            "mp_K"
        )
    )
    missing = frame.filter(pl.col("mp_K").is_null())
    if missing.height:
        ledgers.append(
            pl.DataFrame(
                {
                    "source": ["bradley"] * missing.height,
                    "stage": ["unit_harmonization"] * missing.height,
                    "reason": [str(RejectReason.MISSING_VALUE)] * missing.height,
                    "detail": ["no melting point recorded"] * missing.height,
                    "smiles_in": missing.get_column("smiles").to_list(),
                    "value": [None] * missing.height,
                },
                schema=LEDGER_SCHEMA,
            )
        )
        frame = frame.filter(pl.col("mp_K").is_not_null())

    # 4. Collapse replicates, keeping the disagreement.
    resolved, conflicts = resolve_replicates(
        frame,
        value_column="mp_K",
        tolerance=tolerance_k,
        source_column=source_column,
    )

    # 5. Optionally keep only the compounds whose replicates agree.
    if subset == "double_plus_good":
        rejected = resolved.filter(pl.col("spread_k") > tolerance_k)
        if rejected.height:
            ledgers.append(
                pl.DataFrame(
                    {
                        "source": ["bradley"] * rejected.height,
                        "stage": ["subset:double_plus_good"] * rejected.height,
                        "reason": ["replicates_disagree"] * rejected.height,
                        "detail": [
                            f"spread {s:.1f} K > {tolerance_k} K"
                            for s in rejected.get_column("spread_k").to_list()
                        ],
                        "smiles_in": rejected.get_column("smiles").to_list(),
                        "value": rejected.get_column("median_k").to_list(),
                    },
                    schema=LEDGER_SCHEMA,
                )
            )
        resolved = resolved.filter(pl.col("spread_k") <= tolerance_k)

    curated = resolved.select(
        pl.col("smiles"),
        pl.col("median_k").alias("mp_K"),
        pl.col("n_measurements").alias("mp_n_measurements"),
        pl.col("spread_k").alias("mp_spread_k"),
        pl.col("sources").alias("mp_sources"),
    )

    ledger = (
        pl.concat(ledgers, how="vertical")
        if ledgers
        else pl.DataFrame(schema=LEDGER_SCHEMA)
    )
    return curated, ledger, conflicts
