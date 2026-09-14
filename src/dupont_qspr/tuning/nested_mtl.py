"""Nested cross-validation for Track 2, the multi-task neural heads.

The control flow mirrors Track 1's exactly - search inside each outer fold, refit,
score once - but the geometry differs in one way that follows from the
architecture rather than from choice.

Track 1 fits three independent models, so it runs one study per
(property, outer fold): fifteen studies, each free to pick different
hyperparameters for its property. Track 2 fits **one network covering all three
properties**, so a single configuration has to serve all of them. That gives
**five studies per encoder**, and it is why the per-property loss weights are
tunable: they are the only mechanism by which one configuration can allocate
different amounts of shared capacity to different properties.

The objective averages the per-property normalized RMSE, so no single property
dominates the search purely by having a larger scale or more labels.

Unlike Track 1, the refit here *can* use early stopping honestly - the neural
model validates against an inner fold, not the outer test fold - but the inner
folds are consumed by the search, so the refit reuses one of them as its
validation split rather than touching held-out data.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES, PROPERTY_UNITS
from dupont_qspr.dataset import Prepared
from dupont_qspr.metrics import point_metrics
from dupont_qspr.metrics.point import rmse
from dupont_qspr.models.mtl import MultiTaskModel
from dupont_qspr.tracking import RunLogger
from dupont_qspr.tuning.spaces import MTL_SPACE, suggest_mtl

__all__ = ["MtlFoldResult", "MtlResult", "run_nested_mtl"]

optuna.logging.set_verbosity(optuna.logging.WARNING)


@dataclass(slots=True)
class MtlFoldResult:
    """One outer fold: one network, scored separately on each property's head."""

    fold: int
    best_params: dict[str, Any]
    best_inner_score: float
    best_epoch: int | None
    n_trials: int
    n_pruned: int
    metrics: dict[str, dict[str, Any]]
    seconds: float
    test_rows: npt.NDArray[np.int64] = field(
        default_factory=lambda: np.empty(0, np.int64)
    )
    predictions: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.empty((0, len(PROPERTIES)))
    )


@dataclass(slots=True)
class MtlResult:
    encoder: str
    feature_name: str
    results: list[MtlFoldResult]
    seconds: float

    def estimate(self) -> dict[str, dict[str, Any]]:
        summary: dict[str, dict[str, Any]] = {}
        for prop in PROPERTIES:
            collected: dict[str, list[float]] = {}
            counts = 0
            for entry in self.results:
                metrics = entry.metrics.get(prop, {})
                counts += int(metrics.get("n", 0))
                for key, value in metrics.items():
                    if (
                        key != "n"
                        and isinstance(value, (int, float))
                        and np.isfinite(value)
                    ):
                        collected.setdefault(key, []).append(float(value))
            if not collected:
                continue
            summary[prop] = {
                "units": PROPERTY_UNITS[prop],  # type: ignore[index]
                "n_folds": len(self.results),
                "n_test_total": counts,
                **{
                    key: {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=1))
                        if len(values) > 1
                        else 0.0,
                        "per_fold": [round(v, 4) for v in values],
                    }
                    for key, values in collected.items()
                },
            }
        return summary

    def predictions_frame(
        self, smiles: list[str], mask: npt.NDArray[np.bool_]
    ) -> pl.DataFrame:
        """Out-of-fold predictions, one row per (molecule, labelled property)."""
        frames = []
        for entry in self.results:
            for j, prop in enumerate(PROPERTIES):
                keep = mask[entry.test_rows, j]
                rows = entry.test_rows[keep]
                if not rows.size:
                    continue
                frames.append(
                    pl.DataFrame(
                        {
                            "property": [prop] * rows.size,
                            "fold": np.full(rows.size, entry.fold, dtype=np.int64),
                            "row_index": rows,
                            "smiles": [smiles[i] for i in rows],
                            "y_pred": entry.predictions[keep, j],
                        }
                    )
                )
        return pl.concat(frames, how="vertical") if frames else pl.DataFrame()


def _score_all_properties(
    data: Prepared,
    model: MultiTaskModel,
    rows: npt.NDArray[np.int64],
    train: npt.NDArray[np.int64],
) -> tuple[float, dict[str, dict[str, Any]]]:
    """Mean normalized RMSE across properties, plus the per-property detail."""
    predicted = model.predict(data.X[rows])
    per_property: dict[str, dict[str, Any]] = {}
    normalized: list[float] = []

    for j, prop in enumerate(PROPERTIES):
        keep = data.M[rows, j]
        if keep.sum() < 3:
            continue
        truth = data.Y[rows[keep], j]
        spread = data.train_sd(train, j)
        per_property[prop] = point_metrics(truth, predicted[keep, j], train_sd=spread)
        normalized.append(rmse(truth, predicted[keep, j]) / spread)

    return (float(np.mean(normalized)) if normalized else float("inf")), per_property


def _make_objective(
    data: Prepared,
    cfg: Config,
    inner_splits: tuple[Any, ...],
    device: str,
):
    """Build the Optuna objective with its fold binding fixed.

    Defining the closure inline inside the fold loop would capture the loop
    variable by reference rather than by value. It happens to be safe here because
    the study is optimised within the same iteration, but that is a property of
    the call order rather than of the code, and it silently stops being true the
    moment anything defers execution. Binding through arguments makes it not
    depend on that.
    """

    def objective(trial: optuna.Trial) -> float:
        params = suggest_mtl(trial)
        scores: list[float] = []
        for step, split in enumerate(inner_splits):
            model = MultiTaskModel(
                params=params,
                max_epochs=cfg.models.mtl_max_epochs,
                patience=cfg.models.mtl_patience,
                seed=cfg.seed,
                device=device,
            ).fit(
                data.X[split.train_idx],
                data.Y[split.train_idx],
                data.M[split.train_idx],
                eval_set=(
                    data.X[split.test_idx],
                    data.Y[split.test_idx],
                    data.M[split.test_idx],
                ),
            )
            score, _ = _score_all_properties(
                data, model, split.test_idx, split.train_idx
            )
            scores.append(score)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned
        return float(np.mean(scores))

    return objective


def run_nested_mtl(
    data: Prepared, cfg: Config, logger: RunLogger, *, encoder: str, device: str = "cpu"
) -> MtlResult:
    """Full nested run for Track 2 on one encoder's embeddings."""
    started = time.perf_counter()
    logger.log_params({"search_space": MTL_SPACE, "track": "mtl", "encoder": encoder})
    results: list[MtlFoldResult] = []

    for fold in range(data.folds.n_outer):
        cell_started = time.perf_counter()
        outer = data.folds.outer[fold]
        inner_splits = data.folds.inner[fold]

        objective = _make_objective(data, cfg, inner_splits, device)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=cfg.seed + 977 * fold),
            pruner=(
                optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
                if cfg.tuning.pruning
                else optuna.pruners.NopPruner()
            ),
        )
        study.optimize(
            objective, n_trials=cfg.tuning.n_trials, timeout=cfg.tuning.timeout_s
        )
        best = study.best_trial

        # Refit on the full outer-training set. Early stopping is honest here
        # because it validates against an inner fold, never the outer test fold.
        holdout = inner_splits[0]
        refit_train = np.setdiff1d(outer.train_idx, holdout.test_idx)
        final = MultiTaskModel(
            params=suggest_mtl(optuna.trial.FixedTrial(best.params)),
            max_epochs=cfg.models.mtl_max_epochs,
            patience=cfg.models.mtl_patience,
            seed=cfg.seed,
            device=device,
        ).fit(
            data.X[refit_train],
            data.Y[refit_train],
            data.M[refit_train],
            eval_set=(
                data.X[holdout.test_idx],
                data.Y[holdout.test_idx],
                data.M[holdout.test_idx],
            ),
        )

        _, metrics = _score_all_properties(data, final, outer.test_idx, outer.train_idx)
        entry = MtlFoldResult(
            fold=fold,
            best_params=dict(best.params),
            best_inner_score=float(best.value),
            best_epoch=final.best_epoch,
            n_trials=len(study.trials),
            n_pruned=sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            ),
            metrics=metrics,
            seconds=time.perf_counter() - cell_started,
            test_rows=outer.test_idx,
            predictions=final.predict(data.X[outer.test_idx]),
        )
        results.append(entry)

        summary = "  ".join(
            f"{p}: rmse/sd={entry.metrics[p]['rmse_over_sd']:.3f}"
            for p in PROPERTIES
            if p in entry.metrics
        )
        print(
            f"  fold {fold}  {summary}   trials={entry.n_trials} "
            f"({entry.n_pruned} pruned)  epochs={entry.best_epoch}  {entry.seconds:.0f}s",
            flush=True,
        )
        logger.log_metrics(
            {
                f"outer/{p}/{k}": v
                for p, m in entry.metrics.items()
                for k, v in m.items()
                if isinstance(v, (int, float))
            },
            step=fold,
        )

    result = MtlResult(
        encoder=encoder,
        feature_name=data.feature_name,
        results=results,
        seconds=time.perf_counter() - started,
    )
    logger.log_dict(
        {
            "encoder": encoder,
            "estimate": result.estimate(),
            "folds": [
                {
                    "fold": r.fold,
                    "best_params": r.best_params,
                    "best_inner_score": r.best_inner_score,
                    "best_epoch": r.best_epoch,
                    "n_trials": r.n_trials,
                    "n_pruned": r.n_pruned,
                    "metrics": r.metrics,
                    "seconds": round(r.seconds, 1),
                }
                for r in results
            ],
        },
        f"nested_mtl_{encoder}",
    )
    return result
