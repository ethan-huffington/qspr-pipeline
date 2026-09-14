"""Track 1, and the leak the nested structure exists to prevent.

The load-bearing claim of this whole step is that the outer test fold is touched
exactly once, after every choice has been made. Nothing enforces that at runtime -
a stray ``eval_set`` would leak it silently and every reported number would improve
- so the pieces that could leak are tested individually and the fold arithmetic is
checked directly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES
from dupont_qspr.dataset import Prepared, _fold_spec_from_frame
from dupont_qspr.metrics.point import mae, point_metrics, rmse, rmse_over_sd, spearman
from dupont_qspr.models.xgb import IndependentXGBModel, XGBPropertyModel
from dupont_qspr.tracking import MlflowRunLogger, RunLogger
from dupont_qspr.tuning.nested_run import _study_seed
from dupont_qspr.tuning.spaces import XGB_SPACE, suggest_xgb


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A learnable signal with sparse, unevenly distributed labels."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(160, 8)).astype(np.float32)
    weights = rng.normal(size=(8, 3))
    Y = X @ weights + rng.normal(scale=0.1, size=(160, 3))
    M = rng.random((160, 3)) < 0.7
    M[:, 0] = True  # keep one property fully labelled
    Y[~M] = np.nan
    return X, Y.astype(np.float64), M


class TestXGBPropertyModel:
    def test_early_stopping_records_a_tree_count(self, toy) -> None:
        X, Y, _ = toy
        model = XGBPropertyModel(max_rounds=200, early_stopping_rounds=10).fit(
            X[:120], Y[:120, 0], eval_set=(X[120:], Y[120:, 0])
        )
        assert model.best_iteration is not None
        assert 1 <= model.best_iteration <= 200

    def test_fixed_tree_count_needs_no_validation_set(self, toy) -> None:
        """How the outer refit works: no eval_set exists that would not leak."""
        X, Y, _ = toy
        model = XGBPropertyModel().fit(X, Y[:, 0], n_estimators=25)
        assert model.best_iteration == 25
        assert model.predict(X).shape == (X.shape[0],)

    def test_early_stopping_and_fixed_count_are_mutually_exclusive(self, toy) -> None:
        """Allowing both would make it easy to leak by accident."""
        X, Y, _ = toy
        with pytest.raises(ValueError, match="not both"):
            XGBPropertyModel().fit(X, Y[:, 0], eval_set=(X, Y[:, 0]), n_estimators=10)

    def test_predicting_before_fitting_raises(self, toy) -> None:
        X, _, _ = toy
        with pytest.raises(RuntimeError, match="not been fitted"):
            XGBPropertyModel().predict(X)

    def test_quantiles_point_at_the_step_that_implements_them(self, toy) -> None:
        X, Y, _ = toy
        model = XGBPropertyModel().fit(X, Y[:, 0], n_estimators=5)
        with pytest.raises(NotImplementedError, match="build step 8"):
            model.predict_quantiles(X, [0.05, 0.5, 0.95])

    def test_nan_features_are_handled_rather_than_rejected(self, toy) -> None:
        """1,371 real descriptor cells are nan; XGBoost learns a default branch."""
        X, Y, _ = toy
        holed = X.copy()
        holed[::5, 0] = np.nan
        model = XGBPropertyModel().fit(holed, Y[:, 0], n_estimators=10)
        assert np.isfinite(model.predict(holed)).all()


class TestIndependentXGBModel:
    def test_unlabelled_rows_do_not_influence_any_fit(self, toy) -> None:
        """The structural contrast with Track 2: rows are filtered, not masked.

        Overwriting a masked target with an absurd value must leave predictions
        untouched. If it does not, something is reading through the mask.
        """
        X, Y, M = toy
        baseline = IndependentXGBModel().fit(X, Y, M)

        poisoned = Y.copy()
        poisoned[~M] = 1e9
        perturbed = IndependentXGBModel().fit(X, poisoned, M)

        assert np.allclose(baseline.predict(X), perturbed.predict(X), equal_nan=True)

    def test_predicts_one_column_per_property(self, toy) -> None:
        X, Y, M = toy
        predicted = IndependentXGBModel().fit(X, Y, M).predict(X)
        assert predicted.shape == (X.shape[0], len(PROPERTIES))

    def test_a_property_with_no_labels_yields_nan_not_zero(self, toy) -> None:
        """Zero is a plausible prediction; absence must be distinguishable."""
        X, Y, M = toy
        M = M.copy()
        M[:, 1] = False
        Y = Y.copy()
        Y[:, 1] = np.nan

        predicted = IndependentXGBModel().fit(X, Y, M).predict(X)
        assert np.isnan(predicted[:, 1]).all()
        assert np.isfinite(predicted[:, 0]).all()


class TestFoldReconstruction:
    """Folds are rebuilt from parquet rather than re-derived, so verify the rebuild."""

    def _frame(self) -> pl.DataFrame:
        # 12 molecules, 2 outer folds, 2 inner folds within each outer-train
        outer = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        inner0 = [-1, -1, -1, -1, -1, -1, 0, 0, 0, 1, 1, 1]
        inner1 = [0, 0, 0, 1, 1, 1, -1, -1, -1, -1, -1, -1]
        return pl.DataFrame(
            {
                "smiles": [f"M{i}" for i in range(12)],
                "scaffold_id": list(range(12)),
                "outer_fold": outer,
                "inner_fold_0": inner0,
                "inner_fold_1": inner1,
            }
        )

    def test_outer_train_and_test_partition_the_dataset(self) -> None:
        spec = _fold_spec_from_frame(self._frame(), 12)
        for split in spec.outer:
            assert split.train_idx.size + split.test_idx.size == 12
            assert not np.intersect1d(split.train_idx, split.test_idx).size

    def test_inner_folds_never_reach_into_outer_test(self) -> None:
        """The leak that would inflate every hyperparameter score."""
        spec = _fold_spec_from_frame(self._frame(), 12)
        for outer, group in zip(spec.outer, spec.inner, strict=True):
            forbidden = set(outer.test_idx.tolist())
            for inner in group:
                assert not (set(inner.train_idx.tolist()) & forbidden)
                assert not (set(inner.test_idx.tolist()) & forbidden)

    def test_inner_folds_partition_outer_train(self) -> None:
        spec = _fold_spec_from_frame(self._frame(), 12)
        for outer, group in zip(spec.outer, spec.inner, strict=True):
            gathered = np.sort(np.concatenate([s.test_idx for s in group]))
            assert np.array_equal(gathered, np.sort(outer.train_idx))


class TestPreparedHelpers:
    def _prepared(self, toy) -> Prepared:
        X, Y, M = toy
        spec = _fold_spec_from_frame(
            pl.DataFrame(
                {
                    "smiles": [f"M{i}" for i in range(160)],
                    "scaffold_id": list(range(160)),
                    "outer_fold": [i % 2 for i in range(160)],
                    "inner_fold_0": [
                        -1 if i % 2 == 0 else (i // 2) % 2 for i in range(160)
                    ],
                    "inner_fold_1": [
                        -1 if i % 2 == 1 else (i // 2) % 2 for i in range(160)
                    ],
                }
            ),
            160,
        )
        return Prepared(
            smiles=[f"M{i}" for i in range(160)],
            X=X,
            Y=Y,
            M=M,
            folds=spec,
            feature_name="toy",
            featurizer_version="v0",
        )

    def test_labelled_filters_to_measured_rows_only(self, toy) -> None:
        data = self._prepared(toy)
        indices = np.arange(160, dtype=np.int64)
        for j in range(len(PROPERTIES)):
            rows = data.labelled(indices, j)
            assert data.M[rows, j].all()
            assert not np.isnan(data.Y[rows, j]).any()

    def test_train_sd_uses_only_the_rows_it_was_given(self, toy) -> None:
        """Taking spread from the whole dataset would leak the test scale."""
        data = self._prepared(toy)
        subset = np.arange(0, 80, dtype=np.int64)
        expected = float(np.std(data.Y[data.labelled(subset, 0), 0]))
        assert data.train_sd(subset, 0) == pytest.approx(expected)


class TestStudyIndependence:
    def test_each_property_and_fold_gets_its_own_seed(self) -> None:
        """Fifteen independent studies, not one shared across folds.

        A shared study would let a configuration selected using fold 3's data be
        scored on fold 3's test set, which is exactly what nesting prevents.
        """
        seeds = {
            (prop, fold): _study_seed(20260813, prop, fold)
            for prop in PROPERTIES
            for fold in range(5)
        }
        assert len(set(seeds.values())) == len(seeds)

    def test_seeds_are_reproducible(self) -> None:
        assert _study_seed(7, "logS", 2) == _study_seed(7, "logS", 2)


class TestSearchSpace:
    def test_tree_count_is_deliberately_not_tuned(self) -> None:
        """Early stopping finds it directly; searching for it wastes trials."""
        assert "not tuned" in XGB_SPACE["n_estimators"]

    def test_sampled_configs_stay_inside_the_declared_ranges(self) -> None:
        optuna = pytest.importorskip("optuna")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study()
        for _ in range(20):
            params = suggest_xgb(study.ask())
            assert 0.01 <= params["learning_rate"] <= 0.3
            assert 3 <= params["max_depth"] <= 10
            assert 0.5 <= params["subsample"] <= 1.0
            assert 0.3 <= params["colsample_bytree"] <= 1.0
            assert "n_estimators" not in params


class TestMetrics:
    def test_perfect_predictions_score_zero_error(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0])
        assert rmse(y, y) == 0.0
        assert mae(y, y) == 0.0
        assert spearman(y, y) == pytest.approx(1.0)

    def test_rmse_over_sd_near_one_means_no_better_than_the_mean(self) -> None:
        """The metric that makes an RMSE interpretable."""
        rng = np.random.default_rng(0)
        y = rng.normal(size=500)
        predicted = np.full_like(y, y.mean())
        assert rmse_over_sd(y, predicted, float(y.std())) == pytest.approx(
            1.0, abs=0.01
        )

    def test_rmse_exceeds_mae_when_errors_are_outlier_driven(self) -> None:
        """The gap is the diagnostic, which is why both are reported."""
        y = np.zeros(100)
        uniform = np.full(100, 1.0)
        spiky = np.zeros(100)
        spiky[0] = 10.0

        assert rmse(y, uniform) - mae(y, uniform) == pytest.approx(0.0)
        assert rmse(y, spiky) - mae(y, spiky) > 0.8

    def test_spearman_is_blind_to_monotone_bias(self) -> None:
        """A biased but correctly ranking model is still a good triage tool."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert spearman(y, y * 3 + 100) == pytest.approx(1.0)

    def test_too_few_points_gives_nan_rather_than_a_fake_number(self) -> None:
        assert np.isnan(spearman(np.array([1.0, 2.0]), np.array([2.0, 1.0])))

    def test_point_metrics_reports_the_count_it_used(self) -> None:
        y = np.array([1.0, 2.0, 3.0])
        result = point_metrics(y, y, train_sd=1.0)
        assert result["n"] == 3


def test_mlflow_logger_satisfies_the_same_protocol(tmp_path) -> None:
    """Swapping tracking backends must stay a config edit, not a refactor."""
    assert isinstance(MlflowRunLogger(run_id="r", experiment="e"), RunLogger)


def test_both_backends_are_selectable_from_config(smoke_cfg: Config) -> None:
    for backend in ("jsonl", "mlflow"):
        switched = smoke_cfg.model_copy(
            update={
                "tracking": smoke_cfg.tracking.model_copy(update={"backend": backend})
            }
        )
        assert switched.tracking.backend == backend
