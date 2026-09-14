"""Join three partially overlapping datasets into one sparse target matrix.

The three sources cover different molecule sets with only partial overlap, so the
result is deliberately sparse: most rows carry one or two of the three labels and
very few carry all three. That is the design premise of the whole project, not a
defect. Missing targets are never imputed and rows with missing targets are never
dropped - the masked loss exists precisely to consume this structure, and the
label-availability matrix reported here is the evidence justifying it.

The join key is the RDKit canonical SMILES produced by
:mod:`dupont_qspr.data.standardize`, and every source passes through the same
standardizer before joining. Sources are canonicalized *before* the join, never
after: canonicalizing afterwards would join on raw strings and silently miss every
pair that merely wrote the same molecule two different ways.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from dupont_qspr.contracts import (
    PROPERTIES,
    SMILES_COLUMN,
    UNION_TABLE_SCHEMA,
    source_column,
    validate_union_table,
)
from dupont_qspr.data.standardize import (
    LEDGER_SCHEMA,
    resolve_duplicate_values,
    screen_range,
    standardize_frame,
)

__all__ = [
    "build_union_table",
    "label_availability_matrix",
    "load_aqsoldb",
    "load_lipophilicity",
]


def load_aqsoldb(
    path: Path,
    *,
    bounds: tuple[float, float],
    mixture_ratio: float = 0.5,
    keep_stereochemistry: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """AqSolDB: aqueous solubility as log mol/L, already one row per compound.

    The authors' curation carries two columns worth keeping. ``SD`` is the spread
    across the replicate measurements they merged, and ``Occurrences`` is how many
    there were - together the same measurement-noise signal the melting-point
    curation reconstructs by hand, but supplied ready-made. ``Group`` names which
    of the nine underlying source datasets a value came from.
    """
    raw = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)

    frame, ledger = standardize_frame(
        raw,
        smiles_column="SMILES",
        source="aqsoldb",
        value_column="Solubility",
        mixture_ratio=mixture_ratio,
        keep_stereochemistry=keep_stereochemistry,
    )
    frame = frame.with_columns(
        pl.col("Solubility").cast(pl.Float64, strict=False).alias("logS"),
        pl.col("SD").cast(pl.Float64, strict=False).alias("logS_sd"),
        pl.col("Occurrences").cast(pl.Int32, strict=False).alias("logS_n_measurements"),
        pl.col("Group").cast(pl.String).alias("logS_group"),
    )
    frame, range_ledger = screen_range(
        frame, column="logS", prop="logS", bounds=bounds, source="aqsoldb"
    )

    # Standardization can map two distinct raw entries onto one canonical key -
    # a free acid and its sodium salt, say - so duplicates are possible here even
    # though the published file has one row per compound.
    frame, dup_ledger = resolve_duplicate_values(
        frame,
        value_column="logS",
        source="aqsoldb",
        keep=["logS_sd", "logS_n_measurements", "logS_group"],
    )

    curated = frame.select(
        SMILES_COLUMN,
        "logS",
        "logS_sd",
        "logS_n_measurements",
        "logS_group",
    )
    return curated, pl.concat([ledger, range_ledger, dup_ledger], how="vertical")


def load_lipophilicity(
    path: Path,
    *,
    bounds: tuple[float, float],
    mixture_ratio: float = 0.5,
    keep_stereochemistry: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """AstraZeneca lipophilicity: experimental logD at pH 7.4.

    Strictly this is logD7.4 rather than logP - the distribution coefficient of a
    partly ionized species rather than the partition coefficient of the neutral
    one. They coincide for compounds that are neutral at pH 7.4 and diverge for
    ionizable ones. The brief treats this column as the lipophilicity target, and
    the distinction is carried into the report rather than into the modelling.
    """
    raw = pl.read_csv(path, infer_schema_length=10_000, ignore_errors=True)

    frame, ledger = standardize_frame(
        raw,
        smiles_column="smiles",
        source="lipophilicity",
        value_column="exp",
        mixture_ratio=mixture_ratio,
        keep_stereochemistry=keep_stereochemistry,
    )
    frame = frame.with_columns(
        pl.col("exp").cast(pl.Float64, strict=False).alias("logP")
    )
    frame, range_ledger = screen_range(
        frame, column="logP", prop="logP", bounds=bounds, source="lipophilicity"
    )
    frame, dup_ledger = resolve_duplicate_values(
        frame, value_column="logP", source="lipophilicity", keep=[]
    )

    curated = frame.select(SMILES_COLUMN, "logP")
    return curated, pl.concat([ledger, range_ledger, dup_ledger], how="vertical")


def build_union_table(
    *,
    solubility: pl.DataFrame | None,
    lipophilicity: pl.DataFrame | None,
    melting_point: pl.DataFrame | None,
) -> pl.DataFrame:
    """Outer-join the curated per-source tables on canonical SMILES.

    A source passed as ``None`` yields an all-null column rather than a missing
    one, so the union table's schema is identical whether or not every dataset was
    available. Downstream code then never branches on which sources happened to be
    present - the mask already encodes it.
    """
    contributions: list[tuple[str, pl.DataFrame | None, list[str]]] = [
        ("logS", solubility, ["logS", "logS_sd", "logS_n_measurements", "logS_group"]),
        ("logP", lipophilicity, ["logP"]),
        (
            "mp_K",
            melting_point,
            ["mp_K", "mp_n_measurements", "mp_spread_k", "mp_sources"],
        ),
    ]

    table: pl.DataFrame | None = None
    for prop, frame, columns in contributions:
        if frame is None or frame.is_empty():
            continue
        available = [c for c in columns if c in frame.columns]
        piece = frame.select(SMILES_COLUMN, *available).with_columns(
            pl.lit(_provenance_for(prop)).alias(source_column(prop))
        )
        table = (
            piece
            if table is None
            else table.join(piece, on=SMILES_COLUMN, how="full", coalesce=True)
        )

    if table is None:
        raise ValueError("no sources were available; cannot build a union table")

    # Fill in the schema for anything that did not contribute, then order columns.
    for prop in PROPERTIES:
        if prop not in table.columns:
            table = table.with_columns(pl.lit(None, dtype=pl.Float64).alias(prop))
        if source_column(prop) not in table.columns:
            table = table.with_columns(
                pl.lit(None, dtype=pl.String).alias(source_column(prop))
            )

    extras = [
        c for c in table.columns if c not in UNION_TABLE_SCHEMA and c != SMILES_COLUMN
    ]
    table = table.select(
        pl.col(SMILES_COLUMN).cast(pl.String),
        *[pl.col(p).cast(pl.Float64) for p in PROPERTIES],
        *[pl.col(source_column(p)).cast(pl.String) for p in PROPERTIES],
        *extras,
    ).sort(SMILES_COLUMN)

    validate_union_table(table.select(list(UNION_TABLE_SCHEMA)))
    return table


def _provenance_for(prop: str) -> str:
    return {
        "logS": "AqSolDB",
        "logP": "Lipophilicity_AstraZeneca",
        "mp_K": "Bradley_OpenMeltingPoint",
    }[prop]


def label_availability_matrix(table: pl.DataFrame) -> dict[str, Any]:
    """Counts per property and per label-combination.

    Reported as a first-class artifact because it is the evidence for the
    multi-task design. If nearly every row carried all three labels there would be
    nothing for a masked loss to buy, and three independent models would be the
    honest choice.
    """
    present = {p: table.get_column(p).is_not_null() for p in PROPERTIES}

    combination_counts: dict[str, int] = {}
    for row in zip(*(present[p].to_list() for p in PROPERTIES), strict=True):
        key = (
            "+".join(p for p, flag in zip(PROPERTIES, row, strict=True) if flag)
            or "none"
        )
        combination_counts[key] = combination_counts.get(key, 0) + 1

    pairwise = {}
    for i, a in enumerate(PROPERTIES):
        for b in PROPERTIES[i + 1 :]:
            pairwise[f"{a}&{b}"] = int((present[a] & present[b]).sum())

    return {
        "n_molecules": table.height,
        "per_property": {p: int(present[p].sum()) for p in PROPERTIES},
        "pairwise_overlap": pairwise,
        "per_combination": dict(
            sorted(combination_counts.items(), key=lambda kv: -kv[1])
        ),
        "n_complete_rows": int(
            (present["logS"] & present["logP"] & present["mp_K"]).sum()
        ),
    }


def empty_ledger() -> pl.DataFrame:
    return pl.DataFrame(schema=LEDGER_SCHEMA)
