"""Build the deployable bundle from all the training data.

The nested runs evaluated the *procedure*; they did not produce this model. Brief
§8 is explicit that the shipped model comes from one final search over all the
data, refit once. This module does that for the family step 7 chose:

1. One Optuna study over all data, scored by cross-validation over the outer
   folds, which already exist and are scaffold-disjoint.
2. Conformal calibration from out-of-fold predictions over those same folds, so
   every molecule contributes a calibration residual from a model that never saw it
   and no dedicated calibration split is spent.
3. A refit on everything. There is no test set left and none is needed: its job was
   to say how the model would perform, and nested CV already answered that.

The same handoff as the nested runs keeps the refit leak-free without a validation
set: XGBoost carries the tree count its search settled on, and the ensemble carries
the epoch count its calibration fits settled on.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES
from dupont_qspr.dataset import Prepared, load_prepared
from dupont_qspr.features.descriptors import n_path_a_features
from dupont_qspr.metrics.point import rmse
from dupont_qspr.serving.scorer import MANIFEST, save_ad_reference
from dupont_qspr.tracking import capture_environment

__all__ = ["build_bundle"]


def build_bundle(
    cfg: Config,
    *,
    family: str,
    encoder: str | None,
    ad_threshold: float | None,
    directory: Path,
    n_trials: int | None = None,
) -> dict[str, Any]:
    import json

    trials = n_trials or cfg.tuning.n_trials
    directory.mkdir(parents=True, exist_ok=True)
    descriptors = load_prepared(cfg, path="a")

    start = n_path_a_features() - cfg.features.ecfp_bits
    save_ad_reference(descriptors.X[:, start:], directory)

    revision = capture_environment().get("git_revision") or "nogit"
    manifest: dict[str, Any] = {
        "family": family,
        "properties": list(PROPERTIES),
        "quantiles": list(cfg.uncertainty.quantiles),
        "nominal_coverage": cfg.uncertainty.nominal_coverage,
        "ad_threshold": ad_threshold,
        "fingerprint_start": start,
        "n_training_molecules": descriptors.n_molecules,
        "n_trials": trials,
        "profile": cfg.profile,
        "standardize": {
            "mixture_ratio": cfg.data.mixture_fragment_ratio,
            "require_carbon": cfg.data.require_carbon,
            "keep_stereochemistry": cfg.data.keep_stereochemistry,
        },
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_revision": revision,
    }

    if family == "xgb":
        _fit_xgb(cfg, descriptors, directory, manifest, trials)
        manifest["featurizer_version"] = f"{cfg.features.version}:rdkit-ecfp4"
    elif family == "mtl":
        name = encoder or cfg.features.primary_encoder
        _fit_mtl(cfg, name, directory, manifest, trials)
        manifest["featurizer_version"] = f"{cfg.features.version}:{name}"
    else:
        raise ValueError(f"family must be 'xgb' or 'mtl', got {family!r}")

    manifest["model_version"] = (
        f"{family}-{cfg.profile}-{revision[:7]}-{time.strftime('%Y%m%d')}"
    )
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return manifest


def _fit_xgb(
    cfg: Config, data: Prepared, directory: Path, manifest: dict[str, Any], trials: int
) -> None:
    import optuna

    from dupont_qspr.models.xgb import XGBPropertyModel, XGBQuantileModel
    from dupont_qspr.tuning.spaces import suggest_xgb
    from dupont_qspr.uncertainty.conformal import ConformalizedQR

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (directory / "xgb").mkdir(exist_ok=True)
    splits = data.folds.outer
    everything = np.arange(data.n_molecules, dtype=np.int64)
    quantiles = tuple(cfg.uncertainty.quantiles)
    manifest.update(
        cqr_offsets={}, calibration_n={}, hyperparameters={}, n_estimators={}
    )

    for j, prop in enumerate(PROPERTIES):

        def objective(trial: optuna.Trial, j: int = j) -> float:
            params = suggest_xgb(trial)
            scores, iterations = [], []
            for step, split in enumerate(splits):
                train = data.labelled(split.train_idx, j)
                test = data.labelled(split.test_idx, j)
                if train.size < 10 or test.size < 3:
                    continue
                model = XGBPropertyModel(
                    params=params,
                    n_jobs=cfg.models.xgb_n_jobs,
                    max_rounds=cfg.models.xgb_max_rounds,
                    early_stopping_rounds=cfg.models.xgb_early_stopping_rounds,
                    seed=cfg.seed,
                ).fit(
                    data.X[train],
                    data.Y[train, j],
                    eval_set=(data.X[test], data.Y[test, j]),
                )
                scores.append(
                    rmse(data.Y[test, j], model.predict(data.X[test]))
                    / data.train_sd(split.train_idx, j)
                )
                iterations.append(model.best_iteration or cfg.models.xgb_max_rounds)
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned
            if not scores:
                raise optuna.TrialPruned
            trial.set_user_attr("n_estimators", int(np.mean(iterations)))
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=cfg.seed + 31 * (j + 1)),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        )
        study.optimize(objective, n_trials=trials)
        best = study.best_trial
        trees = int(best.user_attrs["n_estimators"])

        calibration_raw, calibration_truth = [], []
        for split in splits:
            train = data.labelled(split.train_idx, j)
            test = data.labelled(split.test_idx, j)
            if train.size < 10 or test.size < 3:
                continue
            fold_model = XGBQuantileModel(
                params=best.params,
                quantiles=quantiles,
                n_jobs=cfg.models.xgb_n_jobs,
                seed=cfg.seed,
            ).fit(data.X[train], data.Y[train, j], n_estimators=trees)
            calibration_raw.append(fold_model.predict_quantiles(data.X[test]))
            calibration_truth.append(data.Y[test, j])
        calibrator = ConformalizedQR(
            nominal_coverage=cfg.uncertainty.nominal_coverage
        ).fit(np.vstack(calibration_raw), np.concatenate(calibration_truth))

        rows = data.labelled(everything, j)
        point = XGBPropertyModel(
            params=best.params, n_jobs=cfg.models.xgb_n_jobs, seed=cfg.seed
        ).fit(data.X[rows], data.Y[rows, j], n_estimators=trees)
        point._model.save_model(str(directory / "xgb" / f"{prop}_point.json"))
        quantile = XGBQuantileModel(
            params=best.params,
            quantiles=quantiles,
            n_jobs=cfg.models.xgb_n_jobs,
            seed=cfg.seed,
        ).fit(data.X[rows], data.Y[rows, j], n_estimators=trees)
        for index, regressor in enumerate(quantile._models):
            regressor.save_model(str(directory / "xgb" / f"{prop}_q{index}.json"))

        manifest["cqr_offsets"][prop] = calibrator.offset
        manifest["calibration_n"][prop] = calibrator.n_calibration
        manifest["hyperparameters"][prop] = dict(best.params)
        manifest["n_estimators"][prop] = trees
        print(
            f"  {prop:<6} trials={len(study.trials)} trees={trees} "
            f"cqr_offset={calibrator.offset:.4f} calibrated_on={calibrator.n_calibration:,}",
            flush=True,
        )


def _fit_mtl(
    cfg: Config, encoder: str, directory: Path, manifest: dict[str, Any], trials: int
) -> None:
    import optuna
    import torch

    from dupont_qspr.models.ensemble import DeepEnsemble
    from dupont_qspr.tuning.nested_mtl import _make_objective
    from dupont_qspr.tuning.spaces import suggest_mtl
    from dupont_qspr.uncertainty.conformal import ConformalizedQR

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = load_prepared(cfg, path="b", encoder=encoder)
    (directory / "mtl").mkdir(exist_ok=True)
    splits = data.folds.outer

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed + 4099),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0),
    )
    study.optimize(_make_objective(data, cfg, splits, "cpu"), n_trials=trials)
    params = suggest_mtl(optuna.trial.FixedTrial(study.best_trial.params))

    rng = np.random.default_rng(cfg.seed)
    raw_by_prop: dict[str, list[np.ndarray]] = {p: [] for p in PROPERTIES}
    truth_by_prop: dict[str, list[np.ndarray]] = {p: [] for p in PROPERTIES}
    epochs: list[int] = []
    for split in splits:
        shuffled = rng.permutation(split.train_idx)
        n_val = max(1, int(0.2 * shuffled.size))
        val, train = shuffled[:n_val], shuffled[n_val:]
        fold_ensemble = DeepEnsemble(
            params=params,
            n_members=cfg.models.ensemble_seeds,
            max_epochs=cfg.models.mtl_max_epochs,
            patience=cfg.models.mtl_patience,
            base_seed=cfg.seed,
        ).fit(
            data.X[train],
            data.Y[train],
            data.M[train],
            eval_set=(data.X[val], data.Y[val], data.M[val]),
        )
        epochs += [m.best_epoch for m in fold_ensemble._members if m.best_epoch]
        raw = fold_ensemble.predict_quantiles(data.X[split.test_idx])
        for j, prop in enumerate(PROPERTIES):
            keep = data.M[split.test_idx, j]
            if keep.any():
                raw_by_prop[prop].append(raw[keep, j, :])
                truth_by_prop[prop].append(data.Y[split.test_idx][keep, j])

    manifest.update(cqr_offsets={}, calibration_n={})
    for prop in PROPERTIES:
        calibrator = ConformalizedQR(
            nominal_coverage=cfg.uncertainty.nominal_coverage
        ).fit(np.vstack(raw_by_prop[prop]), np.concatenate(truth_by_prop[prop]))
        manifest["cqr_offsets"][prop] = calibrator.offset
        manifest["calibration_n"][prop] = calibrator.n_calibration

    n_epochs = round(float(np.mean(epochs))) if epochs else cfg.models.mtl_max_epochs
    final = DeepEnsemble(
        params=params,
        n_members=cfg.models.ensemble_seeds,
        max_epochs=n_epochs,
        patience=n_epochs,
        base_seed=cfg.seed,
    ).fit(data.X, data.Y, data.M)

    stats: dict[str, np.ndarray] = {}
    for member, model in enumerate(final._members):
        torch.save(model._net.state_dict(), directory / "mtl" / f"member_{member}.pt")
        stats[f"centre_{member}"] = model._centre
        stats[f"scale_{member}"] = model._scale
    np.savez(directory / "mtl" / "standardization.npz", **stats)

    manifest.update(
        encoder=encoder,
        encoder_model=cfg.features.encoders[encoder],
        encoder_batch_size=cfg.features.chemberta_batch_size,
        encoder_pooling=cfg.features.chemberta_pooling,
        embedding_width=data.n_features,
        mtl_params=params,
        n_members=cfg.models.ensemble_seeds,
        n_epochs=n_epochs,
        hyperparameters=dict(study.best_trial.params),
    )
    print(
        f"  ensemble of {cfg.models.ensemble_seeds}, {n_epochs} epochs, "
        f"offsets={ {p: round(v, 4) for p, v in manifest['cqr_offsets'].items()} }",
        flush=True,
    )
