"""The contracts have to reject the failures they exist to catch.

These tests are about the guard rails, not the happy path. Each one corresponds to
a mistake that would otherwise be silent and would inflate every number downstream
of it.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dupont_qspr.contracts import (
    PROPERTIES,
    SMILES_COLUMN,
    ApplicabilityDomain,
    ErrorRecord,
    FoldSpec,
    PropertyPrediction,
    ScoredRecord,
    Split,
    source_column,
    targets_and_mask,
    validate_union_table,
)


def _table(rows: list[dict]) -> pl.DataFrame:
    frame = pl.DataFrame(
        rows,
        schema={
            SMILES_COLUMN: pl.String,
            **{p: pl.Float64 for p in PROPERTIES},
            **{source_column(p): pl.String for p in PROPERTIES},
        },
    )
    return frame


def _row(smiles: str, **values: float) -> dict:
    row: dict = {SMILES_COLUMN: smiles}
    for prop in PROPERTIES:
        row[prop] = values.get(prop)
        row[source_column(prop)] = "test" if prop in values else None
    return row


class TestUnionTable:
    def test_accepts_a_sparse_table(self) -> None:
        validate_union_table(
            _table([_row("CCO", logS=-0.3), _row("c1ccccc1", logP=2.1, mp_K=278.7)])
        )

    def test_rejects_duplicate_smiles(self) -> None:
        """The failure mode of joining on SMILES that were never canonicalized.

        Duplicates put the same molecule in two folds at once, which leaks and is
        invisible in every metric.
        """
        with pytest.raises(ValueError, match="not unique"):
            validate_union_table(
                _table([_row("CCO", logS=-0.3), _row("CCO", logP=1.0)])
            )

    def test_rejects_value_without_provenance(self) -> None:
        rows = [_row("CCO", logS=-0.3)]
        rows[0][source_column("logS")] = None
        with pytest.raises(ValueError, match="provenance"):
            validate_union_table(_table(rows))

    def test_rejects_row_with_no_labels(self) -> None:
        with pytest.raises(ValueError, match="no labelled property"):
            validate_union_table(_table([_row("CCO")]))

    def test_rejects_missing_column(self) -> None:
        table = _table([_row("CCO", logS=-0.3)]).drop("mp_K")
        with pytest.raises(ValueError, match="missing columns"):
            validate_union_table(table)

    def test_mask_marks_exactly_the_measured_cells(self) -> None:
        table = _table([_row("CCO", logS=-0.3), _row("c1ccccc1", logP=2.1, mp_K=278.7)])
        values, mask = targets_and_mask(table)

        assert mask.tolist() == [[True, False, False], [False, True, True]]
        assert values[0, 0] == pytest.approx(-0.3)
        # Unlabelled cells must be nan, never a silently imputed zero.
        assert np.isnan(values[~mask]).all()


class TestSplits:
    def test_split_rejects_overlapping_indices(self) -> None:
        with pytest.raises(ValueError, match="both train and test"):
            Split(train_idx=np.array([0, 1, 2]), test_idx=np.array([2, 3]))

    def test_validate_disjoint_catches_a_scaffold_spanning_the_boundary(self) -> None:
        """Scaffold leakage is the failure the whole split protocol exists to prevent."""
        scaffold_id = np.array([0, 0, 1, 1])
        leaky = Split(train_idx=np.array([0, 2]), test_idx=np.array([1, 3]))
        spec = FoldSpec(outer=(leaky,), inner=((),), scaffold_id=scaffold_id, seed=1)

        with pytest.raises(ValueError, match="both train and test"):
            spec.validate_disjoint()

    def test_validate_disjoint_accepts_grouped_folds(self) -> None:
        scaffold_id = np.array([0, 0, 1, 1])
        clean = Split(train_idx=np.array([0, 1]), test_idx=np.array([2, 3]))
        spec = FoldSpec(outer=(clean,), inner=((),), scaffold_id=scaffold_id, seed=1)
        spec.validate_disjoint()

    def test_validate_disjoint_checks_the_inner_level_too(self) -> None:
        scaffold_id = np.array([0, 0, 1, 1, 2, 2])
        outer = Split(train_idx=np.array([0, 1, 2, 3]), test_idx=np.array([4, 5]))
        leaky_inner = Split(train_idx=np.array([0, 2]), test_idx=np.array([1, 3]))
        spec = FoldSpec(
            outer=(outer,), inner=((leaky_inner,),), scaffold_id=scaffold_id, seed=1
        )

        with pytest.raises(ValueError, match="inner fold 0"):
            spec.validate_disjoint()

    def test_fold_count_mismatch_is_rejected(self) -> None:
        split = Split(train_idx=np.array([0]), test_idx=np.array([1]))
        with pytest.raises(ValueError, match="outer folds but"):
            FoldSpec(
                outer=(split, split), inner=((),), scaffold_id=np.array([0, 1]), seed=1
            )


class TestScoredRecord:
    def _record(
        self, logS: float = -3.60, logP: float = 3.30, mp_K: float = 353.4
    ) -> ScoredRecord:
        """Naphthalene by default: a real solid with all three properties measured."""
        return ScoredRecord(
            smiles_canonical="c1ccc2ccccc2c1",
            predictions={
                "logS": PropertyPrediction(logS, logS - 0.8, logS + 0.8, 0.9),
                "logP": PropertyPrediction(logP, logP - 0.7, logP + 0.7, 0.9),
                "mp_K": PropertyPrediction(mp_K, mp_K - 33.0, mp_K + 33.0, 0.9),
            },
            applicability_domain=ApplicabilityDomain(0.34, True),
            model_version="m1",
            featurizer_version="f1",
        )

    def test_serialises_to_the_brief_schema(self) -> None:
        payload = self._record().to_dict()

        assert set(payload) == {
            "smiles_canonical",
            "predictions",
            "applicability_domain",
            "model_version",
            "featurizer_version",
        }
        assert set(payload["predictions"]) == set(PROPERTIES)
        assert set(payload["predictions"]["logS"]) == {
            "value",
            "lower",
            "upper",
            "nominal_coverage",
        }
        assert set(payload["applicability_domain"]) == {
            "nn_tanimoto_distance",
            "in_domain",
        }

    def test_gse_residual_is_zero_for_a_perfectly_consistent_trio(self) -> None:
        """logS ~ 0.5 - 0.01*(MP_C - 25) - logP, the free sanity check on the trio."""
        mp_K, logP = 400.0, 4.0
        consistent = 0.5 - 0.01 * ((mp_K - 273.15) - 25.0) - logP

        residual = self._record(logS=consistent, logP=logP, mp_K=mp_K).gse_residual()
        assert residual == pytest.approx(0.0)

    def test_gse_residual_is_small_for_real_measured_values(self) -> None:
        """Naphthalene's measured trio sits within the GSE's usual accuracy."""
        residual = self._record().gse_residual()
        assert residual is not None
        assert abs(residual) < 0.5

    def test_gse_residual_flags_a_grossly_inconsistent_prediction(self) -> None:
        """The point of the check: catch prediction sets that cannot all be true."""
        residual = self._record(logS=2.0).gse_residual()
        assert residual is not None
        assert abs(residual) > 3.0

    def test_gse_residual_is_none_when_a_property_is_absent(self) -> None:
        record = self._record()
        partial = ScoredRecord(
            smiles_canonical=record.smiles_canonical,
            predictions={"logS": record.predictions["logS"]},
            applicability_domain=record.applicability_domain,
            model_version="m1",
            featurizer_version="f1",
        )
        assert partial.gse_residual() is None

    def test_error_record_is_distinguishable_from_a_scored_one(self) -> None:
        payload = ErrorRecord(
            smiles_input="not a molecule", error="unparseable"
        ).to_dict()
        assert payload["predictions"] is None
        assert payload["smiles_input"] == "not a molecule"
