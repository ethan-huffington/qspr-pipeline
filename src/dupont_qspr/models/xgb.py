"""Track 1 - gradient-boosted trees, one independent model per property.

This is the baseline every other model has to beat, and the brief is emphatic that
it be built first and completely rather than treated as a straw man. Gradient
boosting on descriptors plus fingerprints is the strongest conventional approach
for QSPR at this data scale; if the multi-task neural track cannot beat it, that is
a real result and not a failure of effort here.

Three details matter.

**One model per property, each seeing only its own labelled rows.** No masking
trick is needed - the rows where a property is missing are simply not passed. This
is the structural contrast with Track 2, which shares one representation across
all three and therefore *does* need a mask.

**Missing feature values are handed to XGBoost, not imputed.** Around 1,371 cells
in the descriptor matrix are ``nan``, mostly partial-charge and BCUT2D descriptors
that are genuinely undefined for molecules containing elements without Gasteiger
parameters. XGBoost learns a default branch direction for missing values at each
split, which is strictly better information than a substituted zero - zero being a
plausible real value for most of these descriptors and therefore indistinguishable
from a measurement.

**Tree count comes from early stopping, never from the search space.** Tuning
``n_estimators`` directly wastes trials on a parameter that has an obvious optimum
given a validation curve.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt
from xgboost import XGBRegressor

from dupont_qspr.contracts import PROPERTIES, PropertyName

__all__ = [
    "IndependentXGBModel",
    "XGBPropertyModel",
    "XGBQuantileModel",
    "default_xgb_params",
]


def default_xgb_params() -> dict[str, Any]:
    """Sensible starting point, used when no tuning has happened yet."""
    return {
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    }


@dataclass(slots=True)
class XGBPropertyModel:
    """One gradient-boosted regressor. Satisfies ``PropertyModel``."""

    params: dict[str, Any] = field(default_factory=default_xgb_params)
    n_jobs: int = 10
    max_rounds: int = 2000
    early_stopping_rounds: int = 50
    seed: int = 0
    #: Set after fitting: the round early stopping settled on. Carried forward so
    #: the final refit can use a fixed tree count without needing a validation set.
    best_iteration: int | None = None
    _model: XGBRegressor | None = None

    def fit(
        self,
        X: npt.NDArray[np.float32],
        y: npt.NDArray[np.float64],
        *,
        eval_set: tuple[npt.NDArray[np.float32], npt.NDArray[np.float64]] | None = None,
        n_estimators: int | None = None,
    ) -> Self:
        """Fit on labelled rows.

        With ``eval_set``, tree count is chosen by early stopping and recorded in
        ``best_iteration``. With ``n_estimators`` instead, that many trees are
        built unconditionally - which is how the outer refit works, because using
        the outer test fold for early stopping would leak it.
        """
        if eval_set is not None and n_estimators is not None:
            raise ValueError(
                "pass eval_set (early stopping) or n_estimators (fixed), not both"
            )

        rounds = n_estimators if n_estimators is not None else self.max_rounds
        model = XGBRegressor(
            n_estimators=rounds,
            tree_method="hist",
            n_jobs=self.n_jobs,
            random_state=self.seed,
            objective="reg:squarederror",
            early_stopping_rounds=(
                self.early_stopping_rounds if eval_set is not None else None
            ),
            **self.params,
        )
        if eval_set is not None:
            model.fit(X, y, eval_set=[eval_set], verbose=False)
            # +1 because best_iteration is a zero-based index and n_estimators is
            # a count; off by one here would silently under-train every refit.
            self.best_iteration = int(model.best_iteration) + 1
        else:
            model.fit(X, y, verbose=False)
            self.best_iteration = rounds

        self._model = model
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        if self._model is None:
            raise RuntimeError("model has not been fitted")
        return np.asarray(self._model.predict(X), dtype=np.float64)

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError(
            "Quantile regression arrives with the uncertainty layer (build step 8), "
            "where reg:quantileerror models are fitted alongside the point model and "
            "then conformalized. Track 1 at step 4 reports point accuracy only."
        )


@dataclass(slots=True)
class IndependentXGBModel:
    """Three independent regressors behind the multi-property interface.

    Satisfies ``QSPRModel``, so the evaluation harness, the metrics layer and the
    conformal wrapper never need to know whether they hold this or the multi-task
    network. That is the whole reason the seam sits at ``QSPRModel``.
    """

    per_property: dict[PropertyName, XGBPropertyModel] = field(default_factory=dict)
    properties: tuple[PropertyName, ...] = PROPERTIES

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
        """Fit each property on the rows where it is measured, and only those."""
        for j, prop in enumerate(self.properties):
            rows = np.flatnonzero(M[:, j])
            if rows.size < 2:
                continue
            model = self.per_property.setdefault(prop, XGBPropertyModel())
            fold_eval = None
            if eval_set is not None:
                Xv, Yv, Mv = eval_set
                valid = np.flatnonzero(Mv[:, j])
                if valid.size:
                    fold_eval = (Xv[valid], Yv[valid, j])
            model.fit(X[rows], Y[rows, j], eval_set=fold_eval)
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        out = np.full((X.shape[0], len(self.properties)), np.nan, dtype=np.float64)
        for j, prop in enumerate(self.properties):
            model = self.per_property.get(prop)
            if model is not None and model._model is not None:
                out[:, j] = model.predict(X)
        return out

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        raise NotImplementedError("see XGBPropertyModel.predict_quantiles")


@dataclass(slots=True)
class XGBQuantileModel:
    """Three quantile regressors, giving Track 1 an adaptive uncertainty signal.

    ``reg:quantileerror`` optimises the pinball loss, so a model fitted at alpha
    0.05 learns the conditional 5th percentile rather than the conditional mean.
    Fitting at 0.05, 0.5 and 0.95 yields a per-molecule interval whose *width*
    varies with how uncertain the model is about that molecule - which is the
    property a constant-width residual interval can never have, and the one §10's
    conditional-coverage analysis is looking for.

    The quantile models deliberately reuse the point model's hyperparameters
    rather than being tuned separately. That is an approximation: the optimal
    depth for a 0.05-quantile objective need not equal the optimum for squared
    error. It is standard practice, it costs 15 extra fits per cell instead of a
    second full search, and it belongs in the limitations section rather than
    being passed over.
    """

    params: dict[str, Any] = field(default_factory=default_xgb_params)
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95)
    n_jobs: int = 10
    seed: int = 0
    _models: list[XGBRegressor] = field(default_factory=list)

    def fit(
        self,
        X: npt.NDArray[np.float32],
        y: npt.NDArray[np.float64],
        *,
        n_estimators: int = 300,
    ) -> Self:
        """Fit one regressor per quantile with a fixed tree count.

        No early stopping: this runs after the search, using the tree count the
        point model settled on, and the only unused data is the outer test fold.
        """
        self._models = []
        for alpha in self.quantiles:
            model = XGBRegressor(
                n_estimators=n_estimators,
                tree_method="hist",
                n_jobs=self.n_jobs,
                random_state=self.seed,
                objective="reg:quantileerror",
                quantile_alpha=alpha,
                **self.params,
            )
            model.fit(X, y, verbose=False)
            self._models.append(model)
        return self

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Sequence[float] | None = None
    ) -> npt.NDArray[np.float64]:
        """``(n_samples, n_quantiles)`` in the quantile order this was fitted with."""
        if not self._models:
            raise RuntimeError("model has not been fitted")
        if quantiles is not None and tuple(quantiles) != self.quantiles:
            raise ValueError(
                f"fitted for quantiles {self.quantiles}; refit to change them"
            )
        columns = [np.asarray(m.predict(X), dtype=np.float64) for m in self._models]
        raw = np.column_stack(columns)
        # Quantile crossing is a known artefact of fitting each level separately.
        # Sorting each row restores monotonicity without refitting.
        return np.sort(raw, axis=1)

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """The median quantile, for use as a point estimate."""
        return self.predict_quantiles(X)[:, len(self.quantiles) // 2]
