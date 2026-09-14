"""Step 10: the low-data ablation.

Hypothesis from the brief: the multi-task advantage is largest when labels are
scarce, and shrinks or reverses as data grows. The reasoning is that a shared
representation lets a property borrow strength from the others' labels, which
matters most when its own are few. The experiment confirms or refutes that, and
either result is reported.

The design keeps it cheap and clean:

* labels are subsampled to N per property *within each outer-training fold*, and
  scored on that fold's full held-out set, so the scaffold split still holds;
* hyperparameters are fixed defaults for both tracks, so the curves differ only
  by architecture and data, not by how lucky a search was at each size;
* each subsample reserves a small holdout for early stopping, so neither track
  validates against the outer test fold.

Each run function imports its model family inside the function, because XGBoost
and PyTorch cannot share a process.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES
from dupont_qspr.dataset import load_prepared
from dupont_qspr.metrics.point import rmse

__all__ = [
    "holdout_split",
    "run_mtl_ablation",
    "run_xgb_ablation",
    "subsample",
    "summarise_ablation",
]

Rows = npt.NDArray[np.int64]


def _rng(
    cfg: Config, *, fold: int, size: int, seed: int, prop_index: int = 0
) -> np.random.Generator:
    return np.random.default_rng(
        (cfg.seed + 7919 * seed + 131 * size + 17 * fold + prop_index) % (2**32 - 1)
    )


def subsample(pool: Rows, size: int, rng: np.random.Generator) -> Rows:
    """``size`` rows drawn without replacement, or ``None``-equivalent empty if too few."""
    if size > pool.size:
        return np.empty(0, dtype=np.int64)
    return np.sort(rng.choice(pool, size=size, replace=False)).astype(np.int64)


def holdout_split(
    rows: Rows, fraction: float, rng: np.random.Generator
) -> tuple[Rows, Rows]:
    """Split a subsample into (train, early-stopping holdout)."""
    shuffled = rng.permutation(rows)
    n_val = max(1, round(fraction * rows.size))
    return np.sort(shuffled[n_val:]), np.sort(shuffled[:n_val])


def run_xgb_ablation(cfg: Config) -> pl.DataFrame:
    from dupont_qspr.models.xgb import XGBPropertyModel, default_xgb_params

    data = load_prepared(cfg, path="a")
    records: list[dict[str, Any]] = []
    for fold, outer in enumerate(data.folds.outer):
        for j, prop in enumerate(PROPERTIES):
            pool = data.labelled(outer.train_idx, j)
            test = data.labelled(outer.test_idx, j)
            spread = data.train_sd(outer.train_idx, j)
            for size in cfg.ablation.sizes:
                for seed in range(cfg.ablation.seeds):
                    rng = _rng(cfg, fold=fold, size=size, seed=seed, prop_index=j)
                    sample = subsample(pool, size, rng)
                    if sample.size == 0 or test.size < 3:
                        continue
                    train, val = holdout_split(
                        sample, cfg.ablation.holdout_fraction, rng
                    )
                    model = XGBPropertyModel(
                        params=default_xgb_params(),
                        n_jobs=cfg.models.xgb_n_jobs,
                        max_rounds=cfg.models.xgb_max_rounds,
                        early_stopping_rounds=cfg.models.xgb_early_stopping_rounds,
                        seed=cfg.seed + seed,
                    ).fit(
                        data.X[train],
                        data.Y[train, j],
                        eval_set=(data.X[val], data.Y[val, j]),
                    )
                    error = rmse(data.Y[test, j], model.predict(data.X[test]))
                    records.append(
                        _record("xgb", prop, size, seed, fold, error, spread)
                    )
        print(f"  xgb fold {fold} done", flush=True)
    return pl.DataFrame(records)


def run_mtl_ablation(cfg: Config, encoder: str) -> pl.DataFrame:
    from dupont_qspr.models.mtl import MultiTaskModel, default_mtl_params

    data = load_prepared(cfg, path="b", encoder=encoder)
    records: list[dict[str, Any]] = []
    for fold, outer in enumerate(data.folds.outer):
        for size in cfg.ablation.sizes:
            for seed in range(cfg.ablation.seeds):
                mask = np.zeros_like(data.M)
                sampled: list[int] = []
                for j in range(len(PROPERTIES)):
                    rng = _rng(cfg, fold=fold, size=size, seed=seed, prop_index=j)
                    rows = subsample(data.labelled(outer.train_idx, j), size, rng)
                    if rows.size:
                        mask[rows, j] = True
                        sampled.append(j)
                if not sampled:
                    continue
                rows = np.flatnonzero(mask.any(axis=1)).astype(np.int64)
                train, val = holdout_split(
                    rows,
                    cfg.ablation.holdout_fraction,
                    _rng(cfg, fold=fold, size=size, seed=seed, prop_index=99),
                )
                model = MultiTaskModel(
                    params=default_mtl_params(),
                    max_epochs=cfg.models.mtl_max_epochs,
                    patience=cfg.models.mtl_patience,
                    seed=cfg.seed + seed,
                ).fit(
                    data.X[train],
                    data.Y[train],
                    mask[train],
                    eval_set=(data.X[val], data.Y[val], mask[val]),
                )
                for j in sampled:
                    test = data.labelled(outer.test_idx, j)
                    if test.size < 3:
                        continue
                    error = rmse(data.Y[test, j], model.predict(data.X[test])[:, j])
                    records.append(
                        _record(
                            f"mtl_{encoder}",
                            PROPERTIES[j],
                            size,
                            seed,
                            fold,
                            error,
                            data.train_sd(outer.train_idx, j),
                        )
                    )
        print(f"  mtl fold {fold} done", flush=True)
    return pl.DataFrame(records)


def _record(
    track: str, prop: str, size: int, seed: int, fold: int, error: float, spread: float
) -> dict[str, Any]:
    return {
        "track": track,
        "property": prop,
        "size": size,
        "seed": seed,
        "fold": fold,
        "rmse": error,
        "rmse_over_sd": error / (spread or 1.0),
    }


def summarise_ablation(frame: pl.DataFrame) -> pl.DataFrame:
    """Mean and spread of normalised error per track, property and size."""
    return (
        frame.group_by(["track", "property", "size"])
        .agg(
            pl.col("rmse_over_sd").mean().alias("mean_rmse_over_sd"),
            pl.col("rmse_over_sd").std().alias("sd_rmse_over_sd"),
            pl.len().alias("n_runs"),
        )
        .sort(["property", "size", "track"])
    )
