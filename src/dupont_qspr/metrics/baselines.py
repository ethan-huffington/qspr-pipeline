"""The comparators every reported number has to be read against.

An RMSE of 41 K means nothing on its own. It means something once you know that
guessing the training mean for every molecule scores 88 K, and that the
measurement replicates themselves disagree by 6 K. The first is the floor a model
must beat to be worth anything; the second is the ceiling beyond which better
numbers would indicate leakage rather than skill.

Brief §10 requires the predict-the-mean baseline explicitly. It is deliberately
computed the same way as the model - trained on the outer-training rows, evaluated
on the held-out fold - so the comparison is like for like. Taking the mean of the
test fold instead would give the baseline information the model was denied.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from dupont_qspr.metrics.point import point_metrics

__all__ = ["mean_baseline", "median_baseline", "noise_ceiling"]

Array = npt.NDArray[np.float64]


def mean_baseline(y_train: Array, y_test: Array) -> dict[str, Any]:
    """Predict the training mean for every molecule.

    By construction this scores RMSE ÷ SD very close to 1.0, which is what makes
    that metric readable: it is the ratio to this baseline.
    """
    prediction = np.full_like(y_test, float(np.mean(y_train)))
    return point_metrics(y_test, prediction, train_sd=float(np.std(y_train)))


def median_baseline(y_train: Array, y_test: Array) -> dict[str, Any]:
    """Predict the training median. Beats the mean when the target is skewed."""
    prediction = np.full_like(y_test, float(np.median(y_train)))
    return point_metrics(y_test, prediction, train_sd=float(np.std(y_train)))


def noise_ceiling(replicate_spread: Array) -> dict[str, float]:
    """The other end: how self-consistent the measurements are.

    Derived from compounds with repeated measurements. A model reporting an RMSE
    materially below this should be treated as suspicious rather than excellent -
    it would mean predicting the target more precisely than the target is known,
    which usually indicates a leak.
    """
    finite = replicate_spread[np.isfinite(replicate_spread) & (replicate_spread > 0)]
    if finite.size == 0:
        return {}
    return {
        "n_with_replicates": int(finite.size),
        "median_spread": float(np.median(finite)),
        "mean_spread": float(np.mean(finite)),
        # A crude but standard conversion from range to a comparable sigma.
        "implied_sigma": float(np.mean(finite) / 2.0),
    }
