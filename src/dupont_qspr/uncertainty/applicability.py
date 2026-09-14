"""How far a molecule is from anything the model was trained on.

This is the load-bearing honesty in the whole deliverable. The training data is
drug-molecule-flavoured - AqSolDB, an AstraZeneca assay, a literature melting-point
compilation - so a photoresist monomer or a fluorinated surfactant is extrapolation,
and the consumer has to be told so rather than handed a confident-looking number.

The measure is nearest-neighbour Tanimoto distance over ECFP4 fingerprints::

    similarity = |A ∩ B| / |A ∪ B|        over the set bits of two fingerprints
    distance   = 1 - max similarity       against every training molecule

Tanimoto rather than Euclidean because fingerprints are sparse binary sets, and
set overlap is the meaningful comparison; Euclidean distance on bit vectors is
dominated by how many bits each molecule happens to set. Nearest neighbour rather
than mean distance because what matters is whether *anything* similar was seen,
not the average of everything - a molecule sitting right beside one training
compound is well supported even if it is unlike the other 29,403.

The threshold is deliberately not set here. It comes from the error-versus-distance
curve at step 9, where the distance at which error starts climbing is read off the
data rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt

__all__ = ["ApplicabilityDomainIndex", "tanimoto_similarity_matrix"]


def tanimoto_similarity_matrix(
    query: npt.NDArray[np.float32], reference: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Pairwise Tanimoto over binary fingerprint blocks.

    Computed as matrix products rather than pairwise loops::

        intersection = Q · Rᵀ
        union        = |Q| + |R| - intersection

    which turns 29,404 × 29,404 comparisons into one BLAS call. The naive loop is
    minutes; this is under a second.
    """
    q = (query > 0).astype(np.float32)
    r = (reference > 0).astype(np.float32)

    intersection = q @ r.T
    q_bits = q.sum(axis=1, keepdims=True)
    r_bits = r.sum(axis=1, keepdims=True).T
    union = q_bits + r_bits - intersection

    # A fingerprint with no bits set has no meaningful overlap with anything.
    return np.divide(
        intersection, union, out=np.zeros_like(intersection), where=union > 0
    )


@dataclass(slots=True)
class ApplicabilityDomainIndex:
    """Nearest-neighbour Tanimoto distance against a fixed training set."""

    #: Column slice of the Path A feature matrix holding the ECFP bits. The first
    #: 217 columns are RDKit descriptors on wildly different scales and have no
    #: business in a set-overlap measure.
    fingerprint_start: int = 217
    threshold: float | None = None
    #: Chunk size for the query side, so a large batch does not allocate an
    #: n_query x n_train float matrix all at once.
    chunk: int = 2048
    _reference: npt.NDArray[np.float32] = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float32)
    )

    def fit(self, X_train: npt.NDArray[np.float32]) -> Self:
        """Record the training fingerprints this domain is defined against."""
        self._reference = (X_train[:, self.fingerprint_start :] > 0).astype(np.float32)
        return self

    def distance(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        """Distance to the closest training molecule, in ``[0, 1]``."""
        if self._reference.size == 0:
            raise RuntimeError("index has not been fitted")
        query = X[:, self.fingerprint_start :]
        out = np.empty(query.shape[0], dtype=np.float64)
        for start in range(0, query.shape[0], self.chunk):
            block = query[start : start + self.chunk]
            similarity = tanimoto_similarity_matrix(block, self._reference)
            out[start : start + self.chunk] = 1.0 - similarity.max(axis=1)
        return np.clip(out, 0.0, 1.0)

    def flag(self, distances: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """In-domain where distance is within the threshold.

        With no threshold set nothing is claimed to be in domain, because an
        unset threshold means step 9 has not yet been run - and silently defaulting
        to "everything is fine" is the exact failure this flag exists to prevent.
        """
        if self.threshold is None:
            return np.zeros(distances.shape[0], dtype=bool)
        return distances <= self.threshold

    def summary(self, distances: npt.NDArray[np.float64]) -> dict[str, Any]:
        return {
            "n": int(distances.size),
            "median": float(np.median(distances)),
            "p90": float(np.quantile(distances, 0.9)),
            "max": float(distances.max()) if distances.size else float("nan"),
            "threshold": self.threshold,
            "n_in_domain": int(self.flag(distances).sum()),
        }
