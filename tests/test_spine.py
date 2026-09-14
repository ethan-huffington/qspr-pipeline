"""The whole pipeline, walked end to end, asserted on structurally.

These are integration tests deliberately kept cheap enough to run on every change.
They check the shape of the pipeline rather than the quality of its numbers: that
no molecule crosses a fold boundary, that the cache is genuinely reused, that a bad
input cannot kill a batch. Those properties must survive every later swap of the
components underneath.
"""

from __future__ import annotations

import numpy as np
import pytest

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES, SMILES_COLUMN, targets_and_mask
from dupont_qspr.skeleton import (
    HashFeatureCache,
    NearestNeighbourDomain,
    RidgeQSPRModel,
    SplitConformalCalibrator,
    grouped_nested_folds,
    rank_correlation,
    scaffold_ids_from_smiles,
    score_molecules,
    synthetic_union_table,
)
from dupont_qspr.spine import label_availability, run_spine


@pytest.fixture
def spine_summary(smoke_cfg: Config) -> dict:
    """A full smoke run. Cheap enough (~0.05s) not to need caching across tests."""
    return run_spine(smoke_cfg)


class TestFolds:
    def test_no_molecule_appears_in_two_outer_test_folds(
        self, smoke_cfg: Config
    ) -> None:
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        spec = grouped_nested_folds(scaffold_ids_from_smiles(smiles), smoke_cfg)

        gathered = np.concatenate([s.test_idx for s in spec.outer])
        assert gathered.size == np.unique(gathered).size
        assert np.array_equal(np.sort(gathered), np.arange(len(smiles)))

    def test_folds_are_scaffold_disjoint_at_both_levels(
        self, smoke_cfg: Config
    ) -> None:
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        spec = grouped_nested_folds(scaffold_ids_from_smiles(smiles), smoke_cfg)

        spec.validate_disjoint()  # raises on leakage
        assert spec.n_outer == smoke_cfg.splits.n_outer
        assert spec.n_inner == smoke_cfg.splits.n_inner

    def test_inner_folds_stay_inside_their_outer_training_set(
        self, smoke_cfg: Config
    ) -> None:
        """An inner fold that reached into outer-test would leak the held-out data."""
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        spec = grouped_nested_folds(scaffold_ids_from_smiles(smiles), smoke_cfg)

        for outer, inner_group in zip(spec.outer, spec.inner, strict=True):
            allowed = set(outer.train_idx.tolist())
            for inner in inner_group:
                assert set(inner.train_idx.tolist()) <= allowed
                assert set(inner.test_idx.tolist()) <= allowed

    def test_splitting_is_deterministic_for_a_fixed_seed(
        self, smoke_cfg: Config
    ) -> None:
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        ids = scaffold_ids_from_smiles(smiles)

        first = grouped_nested_folds(ids, smoke_cfg)
        second = grouped_nested_folds(ids, smoke_cfg)
        for a, b in zip(first.outer, second.outer, strict=True):
            assert np.array_equal(a.test_idx, b.test_idx)


class TestFeatureCache:
    def test_second_pass_is_served_entirely_from_cache(self, smoke_cfg: Config) -> None:
        """The property that makes nested CV affordable: featurize once, reuse everywhere."""
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        cache = HashFeatureCache()

        first = cache.transform(smiles)
        misses_after_first = cache.misses
        second = cache.transform(smiles)

        assert misses_after_first == len(smiles)
        assert cache.misses == misses_after_first  # no further computation
        assert np.array_equal(first, second)

    def test_features_are_stable_across_cache_instances(
        self, smoke_cfg: Config
    ) -> None:
        """Content-addressing only works if the content hash does not move."""
        smiles = ["S0001_M000", "S0001_M001", "S0002_M000"]
        assert np.array_equal(
            HashFeatureCache().transform(smiles), HashFeatureCache().transform(smiles)
        )


class TestMaskedFitting:
    def test_unlabelled_cells_do_not_influence_the_fit(self, smoke_cfg: Config) -> None:
        """The masked-loss guarantee, checked at the level the skeleton can express it.

        Overwriting an unlabelled target with an absurd value must leave every
        coefficient untouched. If it does not, something is reading through the
        mask - the exact failure the multi-task track is exposed to at step 6.
        """
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        X = HashFeatureCache().transform(smiles)
        Y, M = targets_and_mask(table)

        baseline = RidgeQSPRModel().fit(X, Y, M)

        poisoned = Y.copy()
        poisoned[~M] = 1e9
        perturbed = RidgeQSPRModel().fit(X, poisoned, M)

        for prop in PROPERTIES:
            assert np.allclose(baseline._coef[prop], perturbed._coef[prop])

    def test_target_standardisation_uses_only_the_rows_it_was_given(
        self, smoke_cfg: Config
    ) -> None:
        """Statistics must not be computed across a fold boundary."""
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        X = HashFeatureCache().transform(smiles)
        Y, M = targets_and_mask(table)

        subset = np.arange(0, len(smiles), 2)
        model = RidgeQSPRModel().fit(X[subset], Y[subset], M[subset])

        for j, prop in enumerate(PROPERTIES):
            labelled = Y[subset][M[subset, j], j]
            mean, std = model._centre[prop]
            assert mean == pytest.approx(float(labelled.mean()))
            assert std == pytest.approx(float(labelled.std()))


class TestConformal:
    def test_split_conformal_reaches_nominal_coverage(self) -> None:
        """The guarantee is marginal and distribution-free; verify it holds here."""
        rng = np.random.default_rng(0)
        truth = rng.normal(size=4000)
        raw = np.column_stack(
            [truth - 1, truth + rng.normal(scale=0.5, size=4000), truth + 1]
        )

        calibrator = SplitConformalCalibrator(nominal_coverage=0.9).fit(
            raw[:2000], truth[:2000]
        )
        bounds = calibrator.transform(raw[2000:])
        held_out = truth[2000:]
        coverage = float(
            ((held_out >= bounds[:, 0]) & (held_out <= bounds[:, 1])).mean()
        )

        assert 0.87 <= coverage <= 0.94

    def test_wider_nominal_coverage_gives_wider_intervals(self) -> None:
        rng = np.random.default_rng(1)
        truth = rng.normal(size=2000)
        raw = np.column_stack(
            [truth - 1, truth + rng.normal(scale=0.5, size=2000), truth + 1]
        )

        narrow = SplitConformalCalibrator(nominal_coverage=0.8).fit(raw, truth)
        wide = SplitConformalCalibrator(nominal_coverage=0.99).fit(raw, truth)
        assert wide._radius > narrow._radius


class TestScoring:
    def test_batch_returns_one_record_per_input_in_order(
        self, smoke_cfg: Config
    ) -> None:
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        cache = HashFeatureCache()
        X = cache.transform(smiles)
        Y, M = targets_and_mask(table)

        model = RidgeQSPRModel().fit(X, Y, M)
        raw = model.predict_quantiles(X, smoke_cfg.uncertainty.quantiles)
        calibrators = {
            prop: SplitConformalCalibrator().fit(raw[M[:, j], j, :], Y[M[:, j], j])
            for j, prop in enumerate(PROPERTIES)
        }
        domain = NearestNeighbourDomain().fit(X)

        inputs = [smiles[0], "not a molecule", smiles[1], "", smiles[2]]
        records = score_molecules(
            inputs,
            model=model,
            cache=cache,
            calibrators=calibrators,
            domain=domain,
            cfg=smoke_cfg,
            model_version="test",
        )

        assert len(records) == len(inputs)
        assert records[0]["smiles_canonical"] == smiles[0]
        assert records[1]["predictions"] is None  # structured error, not an exception
        assert records[2]["smiles_canonical"] == smiles[1]
        assert records[3]["predictions"] is None
        assert records[4]["smiles_canonical"] == smiles[2]

    def test_novel_scaffold_lands_further_from_the_training_set(
        self, smoke_cfg: Config
    ) -> None:
        """The applicability-domain signal has to actually respond to novelty."""
        table = synthetic_union_table(smoke_cfg)
        smiles = table.get_column(SMILES_COLUMN).to_list()
        cache = HashFeatureCache()
        domain = NearestNeighbourDomain().fit(cache.transform(smiles))

        seen = domain.distance(cache.transform([smiles[0]]))
        novel = domain.distance(cache.transform(["S9999_M000"]))

        assert seen[0] < 1e-6
        assert novel[0] > seen[0]


class TestSpineRun:
    def test_completes_quickly(self, spine_summary: dict) -> None:
        assert spine_summary["elapsed_s"] < 60.0

    def test_reports_the_label_availability_matrix(self, spine_summary: dict) -> None:
        availability = spine_summary["label_availability"]
        assert availability["n_molecules"] == 500
        assert set(availability["per_property"]) == set(PROPERTIES)
        # The design premise: most rows carry one or two labels, few carry all three.
        assert availability["n_complete_rows"] < availability["n_molecules"] * 0.1

    def test_produces_a_nested_estimate_per_property(self, spine_summary: dict) -> None:
        nested = spine_summary["nested_estimate"]
        assert set(nested) == set(PROPERTIES)
        for prop in PROPERTIES:
            assert "rmse" in nested[prop]
            assert "picp" in nested[prop]
            # Coverage is never reported without width; width alone can buy coverage.
            assert "mean_interval_width" in nested[prop]

    def test_beats_predicting_the_mean(self, spine_summary: dict) -> None:
        """A ratio near 1 would mean the spine is scoring noise, not signal."""
        for prop in PROPERTIES:
            assert spine_summary["nested_estimate"][prop]["rmse_over_sd"]["mean"] < 0.95

    def test_every_outer_fold_selected_its_own_configuration(
        self, spine_summary: dict
    ) -> None:
        """One search per outer fold - folds are allowed to disagree, and that is the point."""
        folds = spine_summary["outer_folds"]
        assert len(folds) == 2
        assert all("alpha" in fold["best_params"] for fold in folds)

    def test_emits_scored_records_including_structured_errors(
        self, spine_summary: dict
    ) -> None:
        records = spine_summary["scored_records"]
        assert len(records) == 5
        assert sum(r.get("predictions") is None for r in records) == 2


def test_rank_correlation_matches_known_cases() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert rank_correlation(values, values) == pytest.approx(1.0)
    assert rank_correlation(values, -values) == pytest.approx(-1.0)
    assert np.isnan(rank_correlation(np.array([1.0]), np.array([1.0])))


def test_label_availability_counts_combinations() -> None:
    mask = np.array([[True, False, False], [True, True, False], [True, False, False]])
    result = label_availability(mask)

    assert result["per_property"] == {"logS": 3, "logP": 1, "mp_K": 0}
    assert result["per_combination"] == {"logS": 2, "logS+logP": 1}
    assert result["n_complete_rows"] == 0
