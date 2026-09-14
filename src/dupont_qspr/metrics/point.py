"""Point-accuracy metrics.

Four numbers, each answering a different question, and the brief is specific about
why all four are needed rather than just RMSE.

**RMSE in native units** is the chemist-facing "typical miss". It is the number
that decides whether a property is shippable at all.

**RMSE divided by the property's standard deviation** answers "is this better than
guessing the mean?". A ratio near 1 means the model has learned essentially
nothing; well below 1 means real signal. Without it, an RMSE of 40 K is
uninterpretable - you cannot tell whether that is good until you know the spread.

**MAE** alongside RMSE, because the *gap* between them is diagnostic. RMSE
penalizes large errors quadratically, so a wide gap means a few big misses rather
than uniform moderate error - which changes how the model should be trusted in
triage.

**Spearman rank correlation** is the decision-relevant one. The oracle's job
downstream is to *order* candidates, so a model with systematic bias but
near-perfect ranking is still an excellent triage tool, while one that is accurate
near the mean but scrambles the extremes will fail at exactly the task of finding
the best candidates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

__all__ = [
    "fold_interval",
    "mae",
    "point_metrics",
    "rmse",
    "rmse_over_sd",
    "spearman",
    "tail_spearman",
]

Array = npt.NDArray[np.float64]


def rmse(y_true: Array, y_pred: Array) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: Array, y_pred: Array) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_over_sd(y_true: Array, y_pred: Array, sd: float) -> float:
    """RMSE normalized by a reference spread, which must come from training rows.

    Using the test set's own standard deviation would make the metric depend on
    how heterogeneous that particular fold happened to be, and would leak the test
    distribution's scale into the reported number.
    """
    return rmse(y_true, y_pred) / (sd or 1.0)


def spearman(y_true: Array, y_pred: Array) -> float:
    """Rank correlation. ``nan`` when there are too few points to be meaningful."""
    if y_true.size < 3:
        return float("nan")
    # A constant slice has no defined rank correlation. This happens legitimately
    # on small tail subsets, so it returns nan rather than warning.
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float("nan")
    result = stats.spearmanr(y_true, y_pred)
    return float(result.statistic)


def point_metrics(y_true: Array, y_pred: Array, *, train_sd: float) -> dict[str, Any]:
    """All four metrics plus the count they were computed on."""
    error = rmse(y_true, y_pred)
    absolute = mae(y_true, y_pred)
    return {
        "n": int(y_true.size),
        "rmse": error,
        "mae": absolute,
        "rmse_over_sd": error / (train_sd or 1.0),
        # A large gap means outlier-driven error rather than uniform miss.
        "rmse_minus_mae": error - absolute,
        "spearman": spearman(y_true, y_pred),
    }


def tail_spearman(y_true: Array, y_pred: Array, *, quantile: float = 0.9) -> float:
    """Rank correlation restricted to the top slice of true values.

    Brief §10 singles this out: "pay particular attention to ranking quality in
    the high-value tail". The reason is that overall Spearman is dominated by the
    bulk, where most molecules sit and ranking is easy. A model can score 0.89
    overall while scrambling the extremes - and the extremes are the only part a
    downstream optimiser actually looks at, because it is selecting the best
    candidates, not the median ones.

    Which end is "high value" is property-dependent (most soluble, most lipophilic,
    highest melting) so this always takes the upper tail and the interpretation is
    left to the reader.
    """
    if y_true.size < 10:
        return float("nan")
    threshold = float(np.quantile(y_true, quantile))
    keep = y_true >= threshold
    if keep.sum() < 3:
        return float("nan")
    return spearman(y_true[keep], y_pred[keep])


def fold_interval(
    values: list[float] | Array, *, confidence: float = 0.95
) -> dict[str, float]:
    """Mean and a t-interval across outer folds.

    With five folds the t-multiplier is 2.78 rather than 1.96, so these intervals
    are wide - and that width is the honest answer rather than a presentational
    problem. Reporting a normal-approximation interval here would understate the
    uncertainty by about 40%.
    """
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {}
    mean = float(np.mean(array))
    if array.size == 1:
        return {"mean": mean, "std": 0.0, "lower": mean, "upper": mean, "n": 1}
    sd = float(np.std(array, ddof=1))
    half = (
        float(stats.t.ppf(0.5 + confidence / 2, array.size - 1))
        * sd
        / np.sqrt(array.size)
    )
    return {
        "mean": mean,
        "std": sd,
        "lower": mean - half,
        "upper": mean + half,
        "n": int(array.size),
        "confidence": confidence,
    }
