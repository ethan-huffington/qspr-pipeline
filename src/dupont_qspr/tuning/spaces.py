"""What Optuna is allowed to propose.

The brief names the two parameters that matter most for boosted trees, then the
regularization group that matters disproportionately at a few thousand rows.

``learning_rate`` and ``max_depth`` are the primary pair: how big a step each tree
takes, and how much interaction a single tree can express. Everything else guards
against overfitting, which is the dominant risk here - the melting-point set is
20,076 rows against 1,241 features, and the lipophilicity set only 4,200.

``n_estimators`` is deliberately absent. It is settled by early stopping against
the inner validation fold, which finds its optimum directly instead of spending
trials searching for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    import optuna

__all__ = ["MTL_SPACE", "XGB_SPACE", "suggest_mtl", "suggest_xgb"]

#: Declarative record of the space, logged with every run so a result can be read
#: against what was actually searched.
XGB_SPACE: dict[str, str] = {
    "learning_rate": "log-uniform [0.01, 0.3]",
    "max_depth": "int [3, 10]",
    "min_child_weight": "log-uniform [1, 50]",
    "subsample": "uniform [0.5, 1.0]",
    "colsample_bytree": "uniform [0.3, 1.0]",
    "reg_lambda": "log-uniform [1e-3, 100]",
    "reg_alpha": "log-uniform [1e-4, 10]",
    "n_estimators": "not tuned - set by early stopping",
}


def suggest_xgb(trial: optuna.Trial) -> dict[str, Any]:
    """One configuration from the space above."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        # Log scale: the interesting behaviour is between 1 and 10, not 40 and 50.
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1.0, 50.0, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        # Goes lower than subsample on purpose: with 1,024 sparse fingerprint bits,
        # aggressive column sampling is a strong regularizer.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
    }


#: The neural track's space. Regularization dominates it, because the thing being
#: fitted is a few thousand labels against a 768-dimensional input.
MTL_SPACE: dict[str, str] = {
    "learning_rate": "log-uniform [1e-4, 1e-2]",
    "hidden_width": "categorical {128, 256, 512}",
    "n_layers": "int [1, 2]",
    "dropout": "uniform [0.0, 0.5]",
    "weight_decay": "log-uniform [1e-6, 1e-2]",
    "loss_weight_*": "log-uniform [0.2, 5.0] per property",
}


def suggest_mtl(trial: optuna.Trial) -> dict[str, Any]:
    """One configuration for the multi-task heads.

    The per-property loss weights are the parameter unique to this track, and the
    one the brief singles out. They decide how much of the shared representation
    each property gets to claim, so they are precisely the knob that determines
    whether the three tasks cooperate or compete. Only their *ratios* matter -
    scaling all three scales the whole loss - so the sampled values are normalized
    to a mean of one, which keeps the effective learning rate comparable across
    trials rather than confounded with the weights.
    """
    raw = {
        p: trial.suggest_float(f"loss_weight_{p}", 0.2, 5.0, log=True)
        for p in ("logS", "logP", "mp_K")
    }
    scale = sum(raw.values()) / len(raw)
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "hidden_width": trial.suggest_categorical("hidden_width", [128, 256, 512]),
        # The brief specifies small heads: 1-2 layers on a frozen trunk. Deeper
        # would be fitting capacity nobody has the labels to support.
        "n_layers": trial.suggest_int("n_layers", 1, 2),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "loss_weights": {p: v / scale for p, v in raw.items()},
    }
