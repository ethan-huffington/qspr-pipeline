"""The frozen seams every later build step targets.

Nothing here does real work. These are the interfaces that let the data pipeline,
the two model tracks, the uncertainty layer, and the serving wrapper be built
independently and still meet in the middle. Changing a signature here is a
deliberate act with downstream consequences; changing an implementation behind one
is routine.

Five contracts, in the order the pipeline uses them:

1. ``UNION_TABLE_SCHEMA``  - the sparse target matrix produced by curation
2. ``FeatureCache``        - content-addressed featurization, shared across folds
3. ``FoldSpec``            - nested scaffold-disjoint cross-validation folds
4. ``QSPRModel``           - what both model tracks implement
5. ``ScoredRecord``        - what the deployed pyfunc emits, per molecule
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = [
    "PROPERTIES",
    "PROPERTY_LABELS",
    "PROPERTY_UNITS",
    "SMILES_COLUMN",
    "UNION_TABLE_SCHEMA",
    "ApplicabilityDomain",
    "ErrorRecord",
    "FeatureCache",
    "FoldSpec",
    "IntervalCalibrator",
    "PropertyModel",
    "PropertyName",
    "PropertyPrediction",
    "QSPRModel",
    "ScoredRecord",
    "Split",
    "property_caption",
    "source_column",
    "targets_and_mask",
    "validate_union_table",
]

# --------------------------------------------------------------------------- #
# Properties
# --------------------------------------------------------------------------- #

PropertyName = Literal["logS", "logP", "mp_K"]

#: Canonical property order. Every (n, 3) target/prediction array in this project
#: uses these columns in this order. Do not reorder.
PROPERTIES: tuple[PropertyName, ...] = ("logS", "logP", "mp_K")

PROPERTY_UNITS: dict[PropertyName, str] = {
    "logS": "log mol/L",
    "logP": "log-ratio",
    "mp_K": "K",
}

#: What each symbol actually means, for anything a human reads. Symbols alone are
#: fine in code and unhelpful on a chart axis six months later.
PROPERTY_LABELS: dict[PropertyName, str] = {
    "logS": "aqueous solubility",
    "logP": "lipophilicity",
    "mp_K": "melting point",
}


def property_caption(prop: PropertyName) -> str:
    """e.g. ``"logS — aqueous solubility (log mol/L)"``."""
    return f"{prop} — {PROPERTY_LABELS[prop]} ({PROPERTY_UNITS[prop]})"


# --------------------------------------------------------------------------- #
# 1. Union table
# --------------------------------------------------------------------------- #

SMILES_COLUMN = "smiles"


def source_column(prop: PropertyName) -> str:
    """Name of the provenance column recording where ``prop``'s value came from."""
    return f"{prop}_source"


#: The sparse target matrix. One row per unique RDKit-canonical SMILES; each target
#: column is nullable because the three source datasets cover different, only
#: partially overlapping molecule sets. That sparsity is the design premise of the
#: multi-task track, not a defect to be imputed away.
UNION_TABLE_SCHEMA: dict[str, pl.DataType] = {
    SMILES_COLUMN: pl.String(),
    **{p: pl.Float64() for p in PROPERTIES},
    **{source_column(p): pl.String() for p in PROPERTIES},
}


def validate_union_table(table: pl.DataFrame) -> None:
    """Raise ``ValueError`` unless ``table`` satisfies the union-table contract.

    Checks the schema, that the SMILES key is unique and non-null, and that a
    target value is present exactly when its provenance is. The uniqueness check
    is the one that catches the failure this project is most exposed to: joining
    on SMILES that were never canonicalized, which silently produces duplicate
    keys and molecules that appear in two cross-validation folds at once.
    """
    missing = [c for c in UNION_TABLE_SCHEMA if c not in table.columns]
    if missing:
        raise ValueError(f"union table is missing columns: {missing}")

    for column, expected in UNION_TABLE_SCHEMA.items():
        actual = table.schema[column]
        if actual != expected:
            raise ValueError(
                f"union table column {column!r} has dtype {actual}, expected {expected}"
            )

    smiles = table.get_column(SMILES_COLUMN)
    if smiles.null_count():
        raise ValueError(f"union table has {smiles.null_count()} null SMILES")
    n_unique = smiles.n_unique()
    if n_unique != table.height:
        raise ValueError(
            f"union table SMILES are not unique: {table.height} rows, {n_unique} "
            "distinct values. The usual cause is joining on non-canonicalized SMILES."
        )

    for prop in PROPERTIES:
        disagree = (
            table.get_column(prop).is_null()
            != table.get_column(source_column(prop)).is_null()
        ).sum()
        if disagree:
            raise ValueError(
                f"{disagree} rows have a value for {prop!r} without provenance, or "
                "the reverse"
            )

    labelled = table.select(
        pl.any_horizontal(pl.col(PROPERTIES).is_not_null())
    ).to_series()
    if not labelled.all():
        raise ValueError(
            f"{(~labelled).sum()} union table rows have no labelled property at all"
        )


def targets_and_mask(
    table: pl.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """Split a union table into a dense target array and its label mask.

    Returns ``(Y, M)`` both shaped ``(n_rows, len(PROPERTIES))``. ``M[i, j]`` is
    ``True`` where row ``i`` has a measured value for ``PROPERTIES[j]``. Positions
    where ``M`` is ``False`` hold ``nan`` in ``Y`` and must never reach a loss
    term, a metric, or a fit call.
    """
    values = table.select(PROPERTIES).to_numpy().astype(np.float64)
    return values, ~np.isnan(values)


# --------------------------------------------------------------------------- #
# 2. Feature cache
# --------------------------------------------------------------------------- #


@runtime_checkable
class FeatureCache(Protocol):
    """Featurization that is computed once per molecule and reused everywhere.

    Implementations are content-addressed by ``(canonical SMILES, version)``, so a
    cache survives re-splitting and is shared across every outer fold, every inner
    fold, and every Optuna trial. That reuse is what makes nested cross-validation
    affordable here, and it is the reason the ChemBERTa trunk is frozen.
    """

    @property
    def version(self) -> str:
        """Identifier pinning the exact featurization recipe.

        Travels with the model into the registry. A model registered without this
        is not reproducible.
        """
        ...

    @property
    def n_features(self) -> int:
        """Width of the returned feature matrix."""
        ...

    def transform(self, smiles: Sequence[str]) -> npt.NDArray[np.float32]:
        """Featurize canonical SMILES, computing and storing anything not cached.

        Returns an ``(len(smiles), n_features)`` array in input order. Callers pass
        already-canonicalized SMILES; implementations do not canonicalize.
        """
        ...


# --------------------------------------------------------------------------- #
# 3. Folds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Split:
    """One train/test pair of positional indices into the union table."""

    train_idx: npt.NDArray[np.int64]
    test_idx: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        overlap = np.intersect1d(self.train_idx, self.test_idx)
        if overlap.size:
            raise ValueError(f"{overlap.size} indices appear in both train and test")


@dataclass(frozen=True, slots=True)
class FoldSpec:
    """Nested, scaffold-disjoint cross-validation folds over the union table.

    ``outer`` holds the folds producing the honest generalization estimate.
    ``inner[i]`` holds folds *within* ``outer[i].train_idx`` used to score
    hyperparameter configurations; they are built by re-splitting the outer-train
    scaffold groups, never by re-splitting individual molecules.
    """

    outer: tuple[Split, ...]
    inner: tuple[tuple[Split, ...], ...]
    scaffold_id: npt.NDArray[np.int64]
    seed: int

    def __post_init__(self) -> None:
        if len(self.outer) != len(self.inner):
            raise ValueError(
                f"{len(self.outer)} outer folds but {len(self.inner)} inner fold groups"
            )

    @property
    def n_outer(self) -> int:
        return len(self.outer)

    @property
    def n_inner(self) -> int:
        return len(self.inner[0]) if self.inner else 0

    def validate_disjoint(self) -> None:
        """Raise ``ValueError`` if any scaffold spans a train/test boundary.

        Scaffold leakage is silent and inflates every downstream number, so this is
        asserted rather than assumed. Checked at both nesting levels.
        """
        for level, split in self._iter_splits():
            train_scaffolds = set(self.scaffold_id[split.train_idx].tolist())
            test_scaffolds = set(self.scaffold_id[split.test_idx].tolist())
            shared = train_scaffolds & test_scaffolds
            if shared:
                raise ValueError(
                    f"{level}: {len(shared)} scaffolds appear in both train and test "
                    f"(e.g. {sorted(shared)[:5]})"
                )

    def _iter_splits(self) -> list[tuple[str, Split]]:
        out = [(f"outer fold {i}", s) for i, s in enumerate(self.outer)]
        for i, group in enumerate(self.inner):
            out += [
                (f"outer fold {i} / inner fold {j}", s) for j, s in enumerate(group)
            ]
        return out


# --------------------------------------------------------------------------- #
# 4. Models
# --------------------------------------------------------------------------- #


@runtime_checkable
class PropertyModel(Protocol):
    """A regressor for a single property, fitted on rows where it is labelled.

    This is what the XGBoost track is natively, and what the final hybrid artifact
    composes from: the brief expects a mixed outcome where different families win
    different properties.
    """

    def fit(
        self,
        X: npt.NDArray[np.float32],
        y: npt.NDArray[np.float64],
        *,
        eval_set: tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]] | None = None,
    ) -> Self:
        """Fit on labelled rows only. ``eval_set`` drives early stopping."""
        ...

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Point predictions, shape ``(n_samples,)``, in the property's native units."""
        ...

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Raw (uncalibrated) quantile predictions, shape ``(n_samples, n_quantiles)``.

        This is the signal the conformal layer calibrates, not a coverage guarantee
        in its own right.
        """
        ...


@runtime_checkable
class QSPRModel(Protocol):
    """All three properties at once - the interface evaluation code is written against.

    Both tracks implement this. The XGBoost track does so by holding three
    independent ``PropertyModel`` instances and honouring the mask per column; the
    multi-task track does so natively with a shared frozen trunk and three masked
    heads. Because the seam sits here, the nested CV harness, the metrics layer,
    and the conformal wrapper are each written exactly once.
    """

    @property
    def properties(self) -> tuple[PropertyName, ...]:
        """Which properties this model actually predicts, in ``PROPERTIES`` order."""
        ...

    def fit(
        self,
        X: npt.NDArray[np.float32],
        Y: npt.NDArray[np.float64],
        M: npt.NDArray[np.bool_],
        *,
        eval_set: tuple[
            npt.NDArray[np.float32], npt.NDArray[np.float64], npt.NDArray[np.bool_]
        ]
        | None = None,
    ) -> Self:
        """Fit on a sparse target matrix.

        ``Y`` is ``(n, 3)`` with ``nan`` wherever ``M`` is ``False``. Implementations
        must consume the mask: an unlabelled cell contributes nothing to any loss
        term and produces no gradient.
        """
        ...

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Point predictions, shape ``(n_samples, len(properties))``, native units."""
        ...

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Raw quantile predictions, shape ``(n_samples, n_properties, n_quantiles)``."""
        ...


# --------------------------------------------------------------------------- #
# 5. Uncertainty and the scored record
# --------------------------------------------------------------------------- #


@runtime_checkable
class IntervalCalibrator(Protocol):
    """Turns a raw uncertainty signal into intervals with a coverage guarantee.

    Kept separate from the models on purpose: the brief's §9 insists the three
    uncertainty mechanisms stay distinct. Nested CV estimates model performance,
    quantiles and ensembles produce the raw per-molecule signal, and this layer -
    and only this layer - makes "90%" mean the same thing across both families.
    """

    @property
    def nominal_coverage(self) -> float:
        """Target coverage, e.g. ``0.9``."""
        ...

    def fit(
        self,
        raw: npt.NDArray[np.float64],
        y_true: npt.NDArray[np.float64],
    ) -> Self:
        """Calibrate against held-out truth. ``raw`` is ``(n_samples, n_quantiles)``."""
        ...

    def transform(self, raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Return calibrated ``(n_samples, 2)`` lower/upper bounds."""
        ...


@dataclass(frozen=True, slots=True)
class PropertyPrediction:
    value: float
    lower: float
    upper: float
    nominal_coverage: float

    def to_dict(self) -> dict[str, float]:
        return {
            "value": self.value,
            "lower": self.lower,
            "upper": self.upper,
            "nominal_coverage": self.nominal_coverage,
        }


@dataclass(frozen=True, slots=True)
class ApplicabilityDomain:
    """Distance to the training set, and whether that distance is acceptable.

    The flag is load-bearing rather than decorative: the training data is
    drug-molecule-flavoured, so anything from electronics-materials chemistry is
    extrapolation and the consumer needs to be told so.
    """

    nn_tanimoto_distance: float
    in_domain: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "nn_tanimoto_distance": self.nn_tanimoto_distance,
            "in_domain": self.in_domain,
        }


@dataclass(frozen=True, slots=True)
class ScoredRecord:
    """One molecule's full scoring result - the pyfunc's unit of output."""

    smiles_canonical: str
    predictions: dict[PropertyName, PropertyPrediction]
    applicability_domain: ApplicabilityDomain
    model_version: str
    featurizer_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles_canonical": self.smiles_canonical,
            "predictions": {k: v.to_dict() for k, v in self.predictions.items()},
            "applicability_domain": self.applicability_domain.to_dict(),
            "model_version": self.model_version,
            "featurizer_version": self.featurizer_version,
        }

    def gse_residual(self) -> float | None:
        """Deviation from the General Solubility Equation, or ``None`` if incomplete.

        ``log S ~ 0.5 - 0.01*(MP_C - 25) - logP``. A free physical sanity check: a
        prediction set that grossly violates this relationship is suspect even when
        each property looks individually plausible.
        """
        if not all(p in self.predictions for p in PROPERTIES):
            return None
        mp_c = self.predictions["mp_K"].value - 273.15
        expected = 0.5 - 0.01 * (mp_c - 25.0) - self.predictions["logP"].value
        return self.predictions["logS"].value - expected


@dataclass(frozen=True, slots=True)
class ErrorRecord:
    """Structured failure for one input, so a bad SMILES cannot kill a batch."""

    smiles_input: str
    error: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles_input": self.smiles_input,
            "error": self.error,
            "predictions": None,
        }
