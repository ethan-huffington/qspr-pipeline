"""Turning a raw uncertainty signal into intervals that mean something.

Conformal prediction needs exactly one thing: a pile of ``(prediction, truth)``
pairs where the model did not train on those rows. From how wrong it was there,
you can bound how wrong it will be on new data - distribution-free, with no
assumption that errors are Gaussian or even symmetric.

**Where the pile comes from.** Not a dedicated calibration split, which at 4,200
lipophilicity labels would take rows straight out of training. Instead the inner
cross-validation folds are recycled: each inner model predicts its own held-out
fold, and those out-of-fold residuals are the calibration set. That is the
data-efficiency the brief asks for.

**What this is and is not.** Strict CV+ and jackknife+ construct the interval from
every fold-model's prediction on the *new* molecule, which requires keeping all K
models at prediction time. Ours keeps one model and takes the width from
out-of-fold residuals. Empirically the two behave very similarly; formally, only
strict CV+ carries the finite-sample guarantee, and ours is split conformal
applied to CV-recycled calibration data. The distinction is stated rather than
glossed, and step 9 measures the coverage that actually results.

Two score functions live here, and the difference between them is the whole
argument for quantile regression:

``SplitConformal``  score = |y - ŷ|              → constant width everywhere
``ConformalizedQR`` score = max(lo - y, y - hi)  → width varies per molecule

Both achieve marginal coverage. Only the second can be narrow on easy molecules
and wide on hard ones, which is what §10's conditional-coverage analysis is
looking for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

__all__ = [
    "ConformalizedQR",
    "EnsembleInterval",
    "SplitConformal",
    "conformal_quantile",
]

Array = npt.NDArray[np.float64]


def conformal_quantile(scores: Array, coverage: float) -> float:
    """The finite-sample-corrected quantile of the conformity scores.

    The correction is ``ceil((n + 1) * coverage) / n`` rather than plain
    ``coverage``, and it is not cosmetic. Without it, coverage is biased *low* on
    small calibration sets - exactly the regime this project lives in, where a
    per-property calibration set can be a few hundred rows. With it, marginal
    coverage is guaranteed at or above nominal for any exchangeable data.

    When the correction exceeds 1 there are too few calibration points to certify
    the requested coverage at all, and the honest answer is an infinite interval
    rather than a narrow one that quietly under-covers.
    """
    finite = scores[np.isfinite(scores)]
    n = finite.size
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * coverage) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(finite, level, method="higher"))


@dataclass(slots=True)
class SplitConformal:
    """Symmetric interval from absolute residuals. Satisfies ``IntervalCalibrator``.

    The simplest correct conformal procedure, and a genuine baseline rather than a
    placeholder: it really does deliver marginal coverage. What it cannot do is
    adapt - every molecule gets the same width, so it is systematically too wide
    for easy predictions and too narrow for hard ones. That failure is invisible
    in marginal coverage and obvious in conditional coverage, which is why §10
    insists on the latter.
    """

    nominal_coverage: float = 0.9
    radius: float = float("nan")
    n_calibration: int = 0

    def fit(self, raw: Array, y_true: Array) -> Self:
        """``raw`` is ``(n, n_quantiles)``; the median column is the point estimate."""
        point = raw[:, raw.shape[1] // 2]
        scores = np.abs(y_true - point)
        self.radius = conformal_quantile(scores, self.nominal_coverage)
        self.n_calibration = int(np.isfinite(scores).sum())
        return self

    def transform(self, raw: Array) -> Array:
        point = raw[:, raw.shape[1] // 2]
        return np.column_stack([point - self.radius, point + self.radius])


@dataclass(slots=True)
class ConformalizedQR:
    """Conformalized quantile regression - adaptive width with a coverage guarantee.

    Quantile regression already produces a per-molecule interval, but an
    uncalibrated one: a model's nominal 5th and 95th percentiles routinely cover
    far less or far more than 90% of reality. CQR keeps the *shape* the quantile
    model learned - wide where it is unsure, narrow where it is confident - and
    shifts both edges outward by a single calibrated amount until coverage is
    correct.

    The conformity score is the signed distance outside the predicted interval::

        score = max(lo(x) - y,  y - hi(x))

    Negative when the truth falls inside, positive by however much it falls
    outside. Taking the corrected quantile of that and adding it to both edges is
    the entire method. Note the offset can be *negative*, which tightens an
    over-wide interval - calibration narrows as readily as it widens.
    """

    nominal_coverage: float = 0.9
    offset: float = float("nan")
    n_calibration: int = 0
    _lower_index: int = 0
    _upper_index: int = -1

    def fit(self, raw: Array, y_true: Array) -> Self:
        lower = raw[:, self._lower_index]
        upper = raw[:, self._upper_index]
        scores = np.maximum(lower - y_true, y_true - upper)
        self.offset = conformal_quantile(scores, self.nominal_coverage)
        self.n_calibration = int(np.isfinite(scores).sum())
        return self

    def transform(self, raw: Array) -> Array:
        lower = raw[:, self._lower_index] - self.offset
        upper = raw[:, self._upper_index] + self.offset
        # A quantile model can cross its own quantiles on out-of-distribution
        # input. Sorting keeps the interval well-formed rather than negative-width.
        return np.column_stack([np.minimum(lower, upper), np.maximum(lower, upper)])


@dataclass(slots=True)
class EnsembleInterval:
    """Raw interval from a deep ensemble's spread, ready to be conformalized.

    The neural track has no quantile head; its uncertainty signal is the variance
    across independently seeded head-sets. This converts that spread into the same
    ``(n, 3)`` lower/median/upper shape the calibrators consume, so one conformal
    implementation serves both tracks.

    The multiplier controls only the *shape* handed to CQR - calibration fixes the
    absolute width afterwards - so its exact value matters much less than the fact
    that the spread varies per molecule, which is what buys adaptivity.
    """

    z: float = 1.645  # the 90% two-sided normal multiplier, as a starting shape
    _members: list[Array] = field(default_factory=list)

    def add(self, prediction: Array) -> Self:
        self._members.append(np.asarray(prediction, dtype=np.float64))
        return self

    def raw(self) -> Array:
        if not self._members:
            raise RuntimeError("no ensemble members were added")
        stacked = np.stack(self._members)
        mean = stacked.mean(axis=0)
        # ddof=1: this is a sample of seeds, not the population of them.
        spread = (
            stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros_like(mean)
        )
        return np.column_stack([mean - self.z * spread, mean, mean + self.z * spread])
