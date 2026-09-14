"""The uncertainty layer: raw signal, calibration, and the applicability domain.

Three mechanisms with three distinct jobs, kept apart on purpose. Nested CV
estimates how well the procedure performs and produces no per-molecule number at
all. Quantile regression and deep ensembles produce a raw per-molecule signal that
means nothing quantitative on its own. Conformal calibration — and only that —
turns the raw signal into an interval whose "90%" means the same thing for both
model families. The applicability domain answers a fourth question entirely:
whether this molecule is the kind of thing the model has any business predicting.
"""

from __future__ import annotations

from dupont_qspr.uncertainty.applicability import ApplicabilityDomainIndex
from dupont_qspr.uncertainty.conformal import (
    ConformalizedQR,
    EnsembleInterval,
    SplitConformal,
    conformal_quantile,
)

__all__ = [
    "ApplicabilityDomainIndex",
    "ConformalizedQR",
    "EnsembleInterval",
    "SplitConformal",
    "conformal_quantile",
]
