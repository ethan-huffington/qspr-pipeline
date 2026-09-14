"""Deep ensemble of multi-task heads - Track 2's raw uncertainty signal.

Train the same architecture from several random initialisations and the spread of
their predictions estimates *epistemic* uncertainty: disagreement about what the
function is, which is large exactly where training data was sparse. That is the
component which should spike for novel chemotypes, and therefore the one the
applicability-domain analysis at step 9 is checking against.

This is nearly free here, and the reason is the frozen trunk. The encoder runs
once at step 2 and its embeddings are cached, so an ensemble member is a couple of
small dense layers fitted to stored vectors - seconds, not a re-encoding of 29,404
molecules. An ensemble over a fine-tuned transformer would cost N full training
runs; over a frozen one it costs N trivial ones.

Only the seed varies between members. Same data, same hyperparameters, same
epochs: the disagreement comes from initialisation and batch order alone, which is
what makes it a measure of model ignorance rather than of data subsetting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from dupont_qspr.contracts import PROPERTIES, PropertyName
from dupont_qspr.models.mtl import MultiTaskModel, default_mtl_params

__all__ = ["DeepEnsemble"]


@dataclass(slots=True)
class DeepEnsemble:
    """N independently seeded multi-task networks, predicting as a group."""

    params: dict[str, Any] = field(default_factory=default_mtl_params)
    n_members: int = 5
    max_epochs: int = 200
    patience: int = 20
    base_seed: int = 0
    device: str = "cpu"
    properties: tuple[PropertyName, ...] = PROPERTIES
    _members: list[MultiTaskModel] = field(default_factory=list)

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
        self._members = [
            MultiTaskModel(
                params=self.params,
                max_epochs=self.max_epochs,
                patience=self.patience,
                # Distinct seeds are the entire source of ensemble diversity.
                seed=self.base_seed + 1000 * member,
                device=self.device,
            ).fit(X, Y, M, eval_set=eval_set)
            for member in range(self.n_members)
        ]
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Ensemble mean - the point prediction."""
        return self._stack(X).mean(axis=0)

    def spread(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Across-member standard deviation, per molecule per property."""
        stacked = self._stack(X)
        if stacked.shape[0] < 2:
            return np.zeros(stacked.shape[1:], dtype=np.float64)
        return stacked.std(axis=0, ddof=1)

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Any = None
    ) -> npt.NDArray[np.float64]:
        """``(n_samples, n_properties, 3)`` lower/mean/upper from the spread.

        The multiplier only sets the interval's *shape*; conformal calibration
        fixes its absolute width afterwards. What matters is that the width varies
        per molecule, which is what buys adaptivity.
        """
        mean, deviation = self.predict(X), self.spread(X)
        z = 1.645
        return np.stack([mean - z * deviation, mean, mean + z * deviation], axis=2)

    def _stack(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        if not self._members:
            raise RuntimeError("ensemble has not been fitted")
        return np.stack([m.predict(X) for m in self._members])
