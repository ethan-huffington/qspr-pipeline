"""On-disk feature storage, content-addressed by ``(canonical SMILES, version)``.

This is the piece that makes nested cross-validation affordable. A feature vector
is computed once per molecule, ever, and then reused by every outer fold, every
inner fold, and every Optuna trial - roughly 4,500 model fits in the full run, all
reading the same array. Without it the ChemBERTa forward pass alone would dominate
the wall clock and the protocol in the brief would be impractical.

Three properties make that safe:

**Content-addressed, not position-addressed.** The key is the canonical SMILES
string, so a cache built before splitting stays valid after re-splitting,
subsampling, or switching profiles. Nothing about the cache knows what a fold is.

**Profile-independent.** Caches live in ``data/features/``, shared, rather than
under ``data/processed/<profile>/``. The same molecule yields the same vector
whoever asks, so the smoke profile reads whatever the full profile already
computed. This is the opposite of the union table, whose contents genuinely depend
on the profile that built it.

**Versioned by filename.** The recipe version is embedded in the file name, so
bumping it produces a new cache beside the old one rather than silently mixing
vectors computed under two different recipes. A model registered without its exact
featurizer version is not reproducible, which is why the version travels with the
array rather than living only in a config file.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

__all__ = ["CacheStats", "DiskFeatureCache"]


@dataclass(slots=True)
class CacheStats:
    """Observability for a cache, so reuse can be demonstrated rather than assumed."""

    hits: int = 0
    misses: int = 0
    compute_seconds: float = 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "compute_seconds": round(self.compute_seconds, 2),
        }


@dataclass(slots=True)
class DiskFeatureCache:
    """A persistent array keyed by canonical SMILES. Satisfies ``FeatureCache``.

    ``compute`` is the only part that differs between the two featurization paths;
    everything about storage, lookup, incremental extension, and persistence is
    shared. That is why the RDKit descriptor path and the frozen-encoder path are
    two small modules rather than two parallel caching implementations.
    """

    name: str
    version: str
    n_features: int
    compute: Callable[[Sequence[str]], npt.NDArray[np.float32]]
    directory: Path
    feature_names: tuple[str, ...] | None = None
    recipe: dict[str, Any] = field(default_factory=dict)

    _row_of: dict[str, int] = field(default_factory=dict, init=False)
    _values: npt.NDArray[np.float32] | None = field(default=None, init=False)
    stats: CacheStats = field(default_factory=CacheStats, init=False)

    # -- paths ------------------------------------------------------------- #

    @property
    def _stem(self) -> Path:
        return self.directory / f"{self.name}-{self.version}"

    @property
    def values_path(self) -> Path:
        return self._stem.with_suffix(".npy")

    @property
    def index_path(self) -> Path:
        return Path(f"{self._stem}.index.parquet")

    @property
    def metadata_path(self) -> Path:
        return Path(f"{self._stem}.meta.json")

    # -- lifecycle --------------------------------------------------------- #

    def load(self) -> None:
        """Read an existing cache from disk, or start an empty one."""
        if self._values is not None:
            return
        if self.values_path.exists() and self.index_path.exists():
            values = np.load(self.values_path)
            index = pl.read_parquet(self.index_path)
            if values.shape[1] != self.n_features:
                raise ValueError(
                    f"{self.values_path} has {values.shape[1]} features, expected "
                    f"{self.n_features}. The recipe changed without the version being "
                    "bumped; bump features.version rather than deleting this check."
                )
            self._values = values.astype(np.float32, copy=False)
            self._row_of = dict(
                zip(
                    index.get_column("smiles").to_list(),
                    index.get_column("row").to_list(),
                    strict=True,
                )
            )
        else:
            self._values = np.empty((0, self.n_features), dtype=np.float32)
            self._row_of = {}

    def persist(self) -> None:
        """Write the cache out atomically.

        Written to a temporary name and renamed, so an interrupted run leaves the
        previous cache intact rather than a half-written array that would load
        without complaint and produce wrong features.
        """
        if self._values is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)

        # np.save appends ".npy" to a path that lacks it, which would silently
        # write to a different file than the one we then rename. Passing an open
        # handle sidesteps that entirely.
        tmp_values = self.values_path.with_suffix(".npy.tmp")
        with tmp_values.open("wb") as handle:
            np.save(handle, self._values)
        tmp_values.replace(self.values_path)

        ordered = sorted(self._row_of.items(), key=lambda kv: kv[1])
        tmp_index = Path(f"{self.index_path}.tmp")
        pl.DataFrame(
            {"smiles": [s for s, _ in ordered], "row": [r for _, r in ordered]},
            schema={"smiles": pl.String, "row": pl.Int64},
        ).write_parquet(tmp_index)
        tmp_index.replace(self.index_path)

        self.metadata_path.write_text(
            json.dumps(
                {
                    "name": self.name,
                    "version": self.version,
                    "n_features": self.n_features,
                    "n_molecules": len(self._row_of),
                    "feature_names": list(self.feature_names)
                    if self.feature_names
                    else None,
                    "recipe": self.recipe,
                    "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            )
        )

    # -- the FeatureCache contract ----------------------------------------- #

    def transform(self, smiles: Sequence[str]) -> npt.NDArray[np.float32]:
        """Return ``(len(smiles), n_features)`` in input order, computing misses.

        Input order is part of the contract - callers index into the result
        positionally against their own labels, so returning rows in cache order
        would silently pair every molecule with the wrong target.
        """
        self.load()
        assert self._values is not None  # narrowed by load()

        # Deduplicate before computing: a batch may name the same molecule twice,
        # and the expensive path should see each one once.
        cached = self._row_of
        missing = list(dict.fromkeys(s for s in smiles if s not in cached))
        # Hits count requests served from cache; misses count molecules actually
        # computed, which is why a batch of 100 repeats of one new molecule is
        # one miss rather than a hundred.
        self.stats.hits += sum(1 for s in smiles if s in cached)
        self.stats.misses += len(missing)

        if missing:
            started = time.perf_counter()
            computed = self.compute(missing)
            self.stats.compute_seconds += time.perf_counter() - started

            computed = np.asarray(computed, dtype=np.float32)
            if computed.shape != (len(missing), self.n_features):
                raise ValueError(
                    f"{self.name} compute returned {computed.shape}, expected "
                    f"{(len(missing), self.n_features)}"
                )
            start_row = self._values.shape[0]
            self._values = np.vstack([self._values, computed])
            for offset, key in enumerate(missing):
                self._row_of[key] = start_row + offset
            self.persist()

        rows = [self._row_of[s] for s in smiles]
        return self._values[rows]

    def contains(self, smiles: Sequence[str]) -> int:
        """How many of these are already cached. Used to demonstrate reuse."""
        self.load()
        return sum(1 for s in smiles if s in self._row_of)

    @property
    def n_cached(self) -> int:
        self.load()
        return len(self._row_of)
