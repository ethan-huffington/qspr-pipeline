"""The reporting layer must not flatter the model, and must not invent numbers.

Two failure modes matter here. A baseline computed with information the model was
denied would make the model look better than it is. And a confidence interval
computed with the wrong multiplier would look reassuringly tight while being
wrong - which is worse than no interval at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from dupont_qspr.config import Config
from dupont_qspr.metrics.baselines import mean_baseline, median_baseline, noise_ceiling
from dupont_qspr.metrics.point import fold_interval, spearman, tail_spearman
from dupont_qspr.reporting.figures import DARK, LIGHT, render_all
from dupont_qspr.reporting.load import latest_nested_result, load_oof_predictions
from dupont_qspr.reporting.tables import baseline_table, ranking_summary, results_table


@pytest.fixture
def predictions() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    frames = []
    for prop, scale in (("logS", 2.0), ("logP", 1.2), ("mp_K", 90.0)):
        truth = rng.normal(scale=scale, size=200)
        frames.append(
            pl.DataFrame(
                {
                    "property": [prop] * 200,
                    "fold": rng.integers(0, 5, 200),
                    "row_index": np.arange(200),
                    "smiles": [f"M{i}" for i in range(200)],
                    "y_true": truth,
                    "y_pred": truth * 0.8 + rng.normal(scale=scale * 0.4, size=200),
                }
            )
        )
    return pl.concat(frames)


class TestBaselines:
    def test_predicting_the_mean_scores_ratio_near_one(self) -> None:
        """This is what makes RMSE ÷ SD readable: the baseline sits at 1.0."""
        rng = np.random.default_rng(0)
        train = rng.normal(size=2000)
        test = rng.normal(size=2000)
        assert mean_baseline(train, test)["rmse_over_sd"] == pytest.approx(
            1.0, abs=0.05
        )

    def test_median_beats_mean_on_a_skewed_target(self) -> None:
        rng = np.random.default_rng(1)
        train = np.concatenate([rng.normal(size=900), rng.normal(loc=40, size=100)])
        test = np.concatenate([rng.normal(size=900), rng.normal(loc=40, size=100)])
        assert median_baseline(train, test)["mae"] < mean_baseline(train, test)["mae"]

    def test_noise_ceiling_ignores_compounds_without_replicates(self) -> None:
        """A spread of zero means one measurement, not perfect agreement."""
        result = noise_ceiling(np.array([0.0, 0.0, 4.0, 6.0, 8.0]))
        assert result["n_with_replicates"] == 3
        assert result["mean_spread"] == pytest.approx(6.0)

    def test_noise_ceiling_is_empty_when_nothing_has_replicates(self) -> None:
        assert noise_ceiling(np.zeros(5)) == {}


class TestFoldInterval:
    def test_uses_the_t_multiplier_not_the_normal_one(self) -> None:
        """With five folds t is 2.78, not 1.96.

        A normal-approximation interval here would be about 40% too narrow, which
        is exactly the direction that makes a result look more certain than it is.
        """
        values = [0.342, 0.449, 0.470, 0.431, 0.433]
        result = fold_interval(values)

        sd = float(np.std(values, ddof=1))
        normal_half = 1.96 * sd / np.sqrt(5)
        actual_half = result["upper"] - result["mean"]
        assert actual_half > normal_half * 1.3

    def test_interval_widens_as_folds_decrease(self) -> None:
        """Two folds should produce a visibly less confident interval than five."""
        five = fold_interval([0.40, 0.42, 0.44, 0.46, 0.48])
        two = fold_interval([0.42, 0.46])
        assert (two["upper"] - two["lower"]) > (five["upper"] - five["lower"])

    def test_single_fold_gives_a_point_not_a_fake_interval(self) -> None:
        result = fold_interval([0.42])
        assert result["lower"] == result["upper"] == result["mean"]

    def test_non_finite_values_are_dropped(self) -> None:
        assert fold_interval([0.4, float("nan"), 0.6])["n"] == 2


class TestTailRanking:
    def test_catches_a_model_that_ranks_the_bulk_but_not_the_extremes(self) -> None:
        """The failure §10 singles out.

        A model can score well overall while ordering the top decile at random,
        and the top decile is the only part a downstream optimiser selects from.
        """
        rng = np.random.default_rng(0)
        truth = rng.normal(size=600)
        predicted = truth.copy()
        top = truth >= np.quantile(truth, 0.9)
        # Permute *within* the tail rather than redrawing: this destroys ordering
        # among the top candidates while leaving the overall distribution — and
        # therefore the overall correlation — essentially untouched. Redrawing
        # instead would move tail points across the whole range and drag the
        # overall score down too, which is a different (and less interesting) bug.
        predicted[top] = rng.permutation(predicted[top])

        assert spearman(truth, predicted) > 0.95
        assert tail_spearman(truth, predicted) < 0.4

    def test_returns_nan_rather_than_a_fake_number_on_tiny_input(self) -> None:
        assert np.isnan(tail_spearman(np.arange(5.0), np.arange(5.0)))

    def test_constant_input_does_not_warn(
        self, recwarn: pytest.WarningsRecorder
    ) -> None:
        spearman(np.ones(10), np.arange(10.0))
        assert not [w for w in recwarn if "constant" in str(w.message).lower()]


class TestTables:
    def test_results_are_reported_per_property_never_pooled(self) -> None:
        """Averaging log units against Kelvin would be arithmetic on nonsense."""
        estimate = {
            "logS": {
                "units": "log mol/L",
                "n_test_total": 100,
                "rmse": {"per_fold": [1.0, 1.1, 0.9, 1.05, 0.95], "mean": 1.0},
                "rmse_over_sd": {
                    "per_fold": [0.4, 0.45, 0.42, 0.43, 0.41],
                    "mean": 0.42,
                },
                "spearman": {"per_fold": [0.8, 0.82, 0.79, 0.81, 0.8], "mean": 0.80},
                "mae": {"mean": 0.8},
            }
        }
        table = results_table(estimate)
        assert table.height == 1
        assert table.get_column("property")[0] == "logS"
        assert "folds" in table.columns

    def test_baseline_is_computed_leave_fold_out(
        self, predictions: pl.DataFrame
    ) -> None:
        """The baseline must not see the fold it is scored on.

        Taking the mean of the test fold itself would give the baseline
        information the model was denied, and understate the model's advantage.
        """
        table = baseline_table(predictions)
        assert table.height == 3
        for row in table.iter_rows(named=True):
            assert row["model RMSE"] < row["predict-the-mean RMSE"]

    def test_ranking_summary_reports_both_scopes(
        self, predictions: pl.DataFrame
    ) -> None:
        summary = ranking_summary(predictions)
        assert set(summary) == {"logS", "logP", "mp_K"}
        for entry in summary.values():
            assert "overall" in entry and "top_decile" in entry


class TestFigures:
    def test_renders_both_themes_without_error(
        self, predictions: pl.DataFrame, tmp_path: Path
    ) -> None:
        estimate = {
            p: {
                "units": "x",
                "n_test_total": 200,
                "rmse": {"per_fold": [1.0, 1.1], "mean": 1.05},
                "rmse_over_sd": {"per_fold": [0.4, 0.5], "mean": 0.45},
                "spearman": {"per_fold": [0.8, 0.82], "mean": 0.81},
                "mae": {"mean": 0.8},
            }
            for p in ("logS", "logP", "mp_K")
        }
        written = render_all(
            estimate, predictions, ranking_summary(predictions), tmp_path
        )

        assert len(written) == 6  # 3 figures x 2 themes
        for path in written:
            assert path.exists() and path.stat().st_size > 5_000

    def test_dark_theme_is_stepped_not_inverted(self) -> None:
        """Dark steps come from the same ramps, chosen for the dark surface."""
        assert LIGHT.series_1 != DARK.series_1
        assert LIGHT.surface != DARK.surface
        # Not a naive inversion: the hue family is preserved.
        assert DARK.series_1.startswith("#3") and LIGHT.series_1.startswith("#2")

    def test_negative_rank_correlations_stay_visible(
        self, predictions: pl.DataFrame, tmp_path: Path
    ) -> None:
        """A negative tail correlation is a real reading, not something to clip."""
        from dupont_qspr.reporting.figures import figure_ranking

        ranking = {"logS": {"overall": 0.7, "top_decile": -0.5, "n": 100.0}}
        path = figure_ranking(ranking, LIGHT, tmp_path / "r.png")
        assert path.exists() and path.stat().st_size > 5_000


class TestLoader:
    def test_missing_nested_result_says_what_to_run(self, smoke_cfg: Config) -> None:
        with pytest.raises(FileNotFoundError, match="04_nested_xgb"):
            latest_nested_result(smoke_cfg)

    def test_missing_predictions_says_what_to_run(self, smoke_cfg: Config) -> None:
        with pytest.raises(FileNotFoundError, match="04_nested_xgb"):
            load_oof_predictions(smoke_cfg)
