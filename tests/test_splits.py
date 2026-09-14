"""Fold construction has exactly one job it cannot get wrong.

If a scaffold spans a train/test boundary, analogues of a test compound sit in
training, every metric downstream is inflated, and nothing anywhere raises. The
whole protocol rests on that not happening, so it is asserted at both nesting
levels rather than trusted.

The second concern is subtler: folds must also be *scoreable*. Balancing molecule
counts is not enough when label coverage is uneven across scaffolds, which it
badly is here.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES, SMILES_COLUMN, source_column
from dupont_qspr.splits.nested import allocate_groups, build_nested_folds
from dupont_qspr.splits.scaffold import assign_scaffolds, scaffold_for_smiles


def _table(
    rows: list[tuple[str, float | None, float | None, float | None]],
) -> pl.DataFrame:
    """Build a minimal union table from (smiles, logS, logP, mp_K) tuples."""
    data: dict[str, list] = {SMILES_COLUMN: [r[0] for r in rows]}
    for j, prop in enumerate(PROPERTIES):
        data[prop] = [r[j + 1] for r in rows]
        data[source_column(prop)] = [
            "test" if r[j + 1] is not None else None for r in rows
        ]
    return pl.DataFrame(data)


class TestScaffoldAssignment:
    def test_analogues_share_a_scaffold(self) -> None:
        """The property the whole split protocol depends on.

        Toluene, phenol and aniline are all a benzene ring with one substituent.
        Scaffold splitting must treat them as one family, or a model tested on
        aniline has already seen its analogues in training.
        """
        cores = [
            scaffold_for_smiles(s) for s in ["Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1"]
        ]
        assert len(set(cores)) == 1
        assert cores[0] == "c1ccccc1"

    def test_different_ring_systems_do_not(self) -> None:
        assert scaffold_for_smiles("c1ccccc1") != scaffold_for_smiles("c1ccncc1")

    def test_acyclic_molecules_have_an_empty_scaffold(self) -> None:
        assert scaffold_for_smiles("CCO") == ""
        assert scaffold_for_smiles("CCCCCC") == ""

    def test_generic_frameworks_merge_heteroatom_variants(self) -> None:
        """Off by default, but the harsher split should be available."""
        plain = {scaffold_for_smiles(s) for s in ["c1ccccc1", "c1ccncc1"]}
        generic = {
            scaffold_for_smiles(s, generic=True) for s in ["c1ccccc1", "c1ccncc1"]
        }
        assert len(plain) == 2
        assert len(generic) == 1

    def test_singleton_policy_isolates_acyclic_molecules(self) -> None:
        smiles = ["CCO", "CCCC", "CCCCCC", "c1ccccc1", "Cc1ccccc1"]
        assignment = assign_scaffolds(smiles, acyclic_policy="singleton")

        # Three acyclics get three groups; the two aromatics share one.
        assert assignment.n_groups == 4
        assert assignment.n_acyclic == 3
        assert assignment.scaffold_id[3] == assignment.scaffold_id[4]

    def test_shared_policy_pools_them(self) -> None:
        """The DeepChem convention, retained as an option.

        In this dataset it would produce one 4,471-member group carrying 27% of
        the solubility labels and 0.1% of the lipophilicity labels, which has to
        move between folds as a single indivisible piece.
        """
        smiles = ["CCO", "CCCC", "CCCCCC", "c1ccccc1", "Cc1ccccc1"]
        assignment = assign_scaffolds(smiles, acyclic_policy="shared")

        assert assignment.n_groups == 2
        assert len(set(assignment.scaffold_id[:3].tolist())) == 1

    def test_assignment_is_deterministic(self) -> None:
        smiles = ["Cc1ccccc1", "CCO", "c1ccncc1", "Oc1ccccc1"]
        first = assign_scaffolds(smiles)
        second = assign_scaffolds(smiles)
        assert np.array_equal(first.scaffold_id, second.scaffold_id)


class TestAllocator:
    def test_groups_are_never_split(self) -> None:
        """Atomicity is the non-negotiable part."""
        groups = [np.arange(0, 10), np.arange(10, 14), np.arange(14, 15)]
        labels = np.ones((15, 3), dtype=bool)

        bins = allocate_groups(groups, labels, 2)
        for group in groups:
            holders = {i for i, b in enumerate(bins) if np.intersect1d(b, group).size}
            assert len(holders) == 1, f"group {group[:3]}... was split across folds"

    def test_every_molecule_lands_exactly_once(self) -> None:
        groups = [np.arange(i * 5, i * 5 + 5) for i in range(8)]
        labels = np.ones((40, 3), dtype=bool)

        bins = allocate_groups(groups, labels, 3)
        gathered = np.sort(np.concatenate(bins))
        assert np.array_equal(gathered, np.arange(40))

    def test_balances_labels_rather_than_molecule_counts(self) -> None:
        """The reason the allocator is label-aware at all.

        Two large groups carry no logP labels and two small ones carry all of
        them. An allocator balancing molecule counts would happily put both
        logP-bearing groups in the same fold and leave another with none, which
        is unscoreable. Balancing labels must not do that.
        """
        groups = [
            np.arange(0, 40),
            np.arange(40, 80),
            np.arange(80, 84),
            np.arange(84, 88),
        ]
        labels = np.zeros((88, 3), dtype=bool)
        labels[:80, 0] = True  # logS on the two big groups
        labels[80:, 1] = True  # logP only on the two small ones

        bins = allocate_groups(groups, labels, 2)
        per_fold = [labels[b].sum(axis=0) for b in bins]
        for counts in per_fold:
            assert counts[1] > 0, "a fold received no logP labels at all"

    def test_refuses_to_leave_a_fold_empty(self) -> None:
        with pytest.raises(ValueError, match="received no molecules"):
            allocate_groups([np.arange(0, 10)], np.ones((10, 3), dtype=bool), 3)

    def test_requires_at_least_two_folds(self) -> None:
        with pytest.raises(ValueError, match="at least 2 folds"):
            allocate_groups([np.arange(4)], np.ones((4, 3), dtype=bool), 1)


class TestNestedFolds:
    @pytest.fixture
    def built(self, smoke_cfg: Config):
        rows: list[tuple[str, float | None, float | None, float | None]] = []
        # A few scaffold families, each with several substituted members, plus
        # acyclics — enough structure for a 2x2 nested split to be meaningful.
        families = {
            "c1ccccc1": ["Cc1ccccc1", "Oc1ccccc1", "Nc1ccccc1", "Clc1ccccc1"],
            "c1ccncc1": ["Cc1ccncc1", "Oc1ccncc1", "Nc1ccncc1"],
            "c1ccc2ccccc2c1": ["Cc1ccc2ccccc2c1", "Oc1ccc2ccccc2c1", "Nc1ccc2ccccc2c1"],
            "C1CCCCC1": ["CC1CCCCC1", "OC1CCCCC1", "NC1CCCCC1"],
        }
        i = 0
        for members in families.values():
            for smiles in members:
                rows.append((smiles, float(i), float(i) * 0.5, 300.0 + i))
                i += 1
        for chain in ["CCO", "CCCO", "CCCCO", "CCCCCO", "CCCCCCO", "CCCCCCCO"]:
            rows.append((chain, float(i), float(i) * 0.5, 300.0 + i))
            i += 1
        return build_nested_folds(_table(rows), smoke_cfg)

    def test_no_scaffold_spans_a_boundary_at_either_level(self, built) -> None:
        built.spec.validate_disjoint()  # raises on leakage

    def test_every_molecule_is_tested_exactly_once(self, built) -> None:
        gathered = np.concatenate([s.test_idx for s in built.spec.outer])
        assert gathered.size == np.unique(gathered).size

    def test_train_and_test_partition_the_dataset(self, built) -> None:
        total = sum(s.test_idx.size for s in built.spec.outer)
        for split in built.spec.outer:
            assert split.train_idx.size + split.test_idx.size == total

    def test_inner_folds_stay_inside_their_outer_training_set(self, built) -> None:
        """An inner fold reaching into outer-test would leak the held-out data."""
        for outer, group in zip(built.spec.outer, built.spec.inner, strict=True):
            allowed = set(outer.train_idx.tolist())
            for inner in group:
                assert set(inner.train_idx.tolist()) <= allowed
                assert set(inner.test_idx.tolist()) <= allowed

    def test_inner_folds_also_partition_the_outer_training_set(self, built) -> None:
        for outer, group in zip(built.spec.outer, built.spec.inner, strict=True):
            gathered = np.sort(np.concatenate([s.test_idx for s in group]))
            assert np.array_equal(gathered, outer.train_idx)

    def test_geometry_matches_the_configuration(self, built, smoke_cfg: Config) -> None:
        assert built.spec.n_outer == smoke_cfg.splits.n_outer
        assert built.spec.n_inner == smoke_cfg.splits.n_inner

    def test_is_deterministic(self, smoke_cfg: Config) -> None:
        rows = [
            (s, 1.0, 2.0, 300.0)
            for s in ["Cc1ccccc1", "Oc1ccccc1", "c1ccncc1", "Cc1ccncc1", "CCO", "CCCO"]
        ]
        first = build_nested_folds(_table(rows), smoke_cfg)
        second = build_nested_folds(_table(rows), smoke_cfg)
        for a, b in zip(first.spec.outer, second.spec.outer, strict=True):
            assert np.array_equal(a.test_idx, b.test_idx)

    def test_thin_folds_are_reported_not_hidden(self, smoke_cfg: Config) -> None:
        """A fold too thin to score must surface, so its interval can be widened."""
        rows: list[tuple[str, float | None, float | None, float | None]] = [
            (s, 1.0, None, 300.0)
            for s in ["Cc1ccccc1", "Oc1ccccc1", "c1ccncc1", "Cc1ccncc1", "CCO", "CCCO"]
        ]
        rows[0] = (rows[0][0], 1.0, 2.0, 300.0)  # a single logP label in total

        build = build_nested_folds(_table(rows), smoke_cfg)
        report = build.report(smoke_cfg.splits.min_test_labels_per_property)
        assert report["thin_folds"], "a fold with almost no logP labels went unreported"
        assert any(entry["property"] == "logP" for entry in report["thin_folds"])
