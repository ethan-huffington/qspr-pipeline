"""A deliberately dumb end-to-end walk of the whole pipeline.

Every stage the real system will have is present here in its simplest honest form:
a sparse union table, a content-addressed feature cache, scaffold-grouped nested
folds, a masked multi-property model, split-conformal intervals, a Tanimoto-shaped
applicability check, and a scored record. None of it is chemistry - the molecules
are synthetic strings - and none of it is meant to produce meaningful numbers.

Its job is to make the seams in :mod:`dupont_qspr.contracts` load-bearing from day
one, so that when the real curation, featurization, models, and conformal layers
arrive they slot into an interface that has already been exercised. The failure
modes this project is most exposed to - scaffold leakage, target statistics
computed across a fold boundary, a mask that does not actually mask - are all
structural, and structural bugs are cheapest to find in a run that takes seconds.

Each piece below is replaced by a real implementation at the build step named in
its docstring. Nothing here should survive to the final artifact.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Self

import numpy as np
import numpy.typing as npt
import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import (
    PROPERTIES,
    SMILES_COLUMN,
    ApplicabilityDomain,
    ErrorRecord,
    FoldSpec,
    PropertyName,
    PropertyPrediction,
    ScoredRecord,
    Split,
    source_column,
    validate_union_table,
)

__all__ = [
    "HashFeatureCache",
    "NearestNeighbourDomain",
    "RidgeQSPRModel",
    "SplitConformalCalibrator",
    "grouped_nested_folds",
    "rank_correlation",
    "rmse",
    "score_molecules",
    "synthetic_union_table",
]

_FEATURE_DIM = 64


def _rng_for(text: str, salt: str = "") -> np.random.Generator:
    """Deterministic per-string RNG.

    ``hash()`` is salted per process, so anything keyed off it would silently stop
    reproducing between runs. Hashing explicitly is the whole point.
    """
    digest = hashlib.blake2b(f"{salt}|{text}".encode(), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "big"))


# --------------------------------------------------------------------------- #
# Stand-in for build step 1 - data acquisition and the union table
# --------------------------------------------------------------------------- #

#: Roughly the label rates the real union table will have: melting point is the
#: largest source (~20k), solubility next (~10k), lipophilicity smallest (~4.2k),
#: over a union of ~30k unique molecules. Most rows carry one or two labels and
#: very few carry all three, which is exactly the structure the masked loss exists
#: to consume.
_LABEL_RATE: dict[PropertyName, float] = {"logS": 0.33, "logP": 0.14, "mp_K": 0.66}

_SOURCE_NAME: dict[PropertyName, str] = {
    "logS": "synthetic-aqsoldb",
    "logP": "synthetic-lipophilicity",
    "mp_K": "synthetic-bradley",
}


def synthetic_union_table(cfg: Config) -> pl.DataFrame:
    """Fabricate a union table with the shape of the real one.

    Molecules are named ``S<scaffold>_M<member>`` so a scaffold is recoverable
    without RDKit. Scaffold group sizes are skewed, as real Bemis-Murcko groups
    are: a few large families and a long tail of singletons.

    Crucially the latent signal has a per-scaffold component, so predicting a
    held-out scaffold is genuinely harder than predicting a held-out molecule. A
    skeleton where random and scaffold splits scored the same would not be
    exercising the thing the split protocol exists to protect against.

    Replaced at build step 1 by the real AqSolDB / Lipophilicity / Bradley union.
    """
    n_target = cfg.data.max_molecules or 500
    rng = np.random.default_rng(cfg.seed)

    smiles: list[str] = []
    scaffold_ids: list[int] = []
    scaffold_index = 0
    while len(smiles) < n_target:
        size = int(rng.integers(1, 13))
        for member in range(min(size, n_target - len(smiles))):
            smiles.append(f"S{scaffold_index:04d}_M{member:03d}")
            scaffold_ids.append(scaffold_index)
        scaffold_index += 1

    # Latent targets, built from the same features the cache will serve so there is
    # real signal to recover rather than noise.
    features = np.stack([_synthetic_features(s) for s in smiles])
    weights = rng.normal(size=(_FEATURE_DIM, len(PROPERTIES)))
    latent = features @ weights
    latent = (latent - latent.mean(axis=0)) / latent.std(axis=0)

    location = np.array([-2.5, 2.5, 400.0])
    scale = np.array([2.0, 1.5, 65.0])
    noise = rng.normal(size=latent.shape) * 0.35
    values = location + scale * (latent + noise)

    # Sparse, partially overlapping label coverage.
    mask = np.column_stack(
        [rng.random(len(smiles)) < _LABEL_RATE[p] for p in PROPERTIES]
    )
    unlabelled = ~mask.any(axis=1)
    if unlabelled.any():
        # The union table contract forbids a row with no label at all; in the real
        # pipeline such a row simply never enters the join.
        forced = rng.integers(0, len(PROPERTIES), size=int(unlabelled.sum()))
        mask[np.flatnonzero(unlabelled), forced] = True

    # Missing targets must be genuine nulls, not NaN: Polars treats the two
    # differently and only null round-trips as "no measurement was taken".
    columns: dict[str, Any] = {SMILES_COLUMN: smiles}
    for j, prop in enumerate(PROPERTIES):
        columns[prop] = [
            float(v) if m else None
            for v, m in zip(values[:, j], mask[:, j], strict=True)
        ]
    for j, prop in enumerate(PROPERTIES):
        columns[source_column(prop)] = [
            _SOURCE_NAME[prop] if m else None for m in mask[:, j]
        ]

    table = pl.DataFrame(columns).with_columns(
        [pl.col(p).cast(pl.Float64) for p in PROPERTIES]
    )
    validate_union_table(table)
    return table


def scaffold_ids_from_smiles(smiles: Sequence[str]) -> npt.NDArray[np.int64]:
    """Recover the synthetic scaffold from the molecule name.

    Replaced at build step 3 by real Bemis-Murcko scaffold assignment.
    """
    return np.array([int(s.split("_")[0][1:]) for s in smiles], dtype=np.int64)


# --------------------------------------------------------------------------- #
# Stand-in for build step 2 - featurization caches
# --------------------------------------------------------------------------- #


def _synthetic_features(smiles: str) -> npt.NDArray[np.float32]:
    """Scaffold-driven features plus a molecule-specific perturbation."""
    scaffold = smiles.split("_")[0]
    core = _rng_for(scaffold, salt="scaffold").normal(size=_FEATURE_DIM)
    decoration = _rng_for(smiles, salt="molecule").normal(size=_FEATURE_DIM)
    return (core + 0.55 * decoration).astype(np.float32)


@dataclass(slots=True)
class HashFeatureCache:
    """In-memory content-addressed cache satisfying :class:`FeatureCache`.

    The important property is not speed but identity: the same SMILES always
    yields the same vector, so a cache built before splitting stays valid across
    every fold and every tuning trial. That reuse is what the real ChemBERTa path
    depends on, and it is why the trunk is frozen.

    Replaced at build step 2 by the on-disk RDKit-descriptor and ChemBERTa caches.
    """

    version: str = "skeleton-v1"
    _store: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @property
    def n_features(self) -> int:
        return _FEATURE_DIM

    def transform(self, smiles: Sequence[str]) -> npt.NDArray[np.float32]:
        out = np.empty((len(smiles), self.n_features), dtype=np.float32)
        for i, key in enumerate(smiles):
            cached = self._store.get(key)
            if cached is None:
                self.misses += 1
                cached = _synthetic_features(key)
                self._store[key] = cached
            else:
                self.hits += 1
            out[i] = cached
        return out


# --------------------------------------------------------------------------- #
# Stand-in for build step 3 - scaffold splitting and the nested harness
# --------------------------------------------------------------------------- #


def grouped_nested_folds(scaffold_id: npt.NDArray[np.int64], cfg: Config) -> FoldSpec:
    """Nested folds that never split a scaffold group across a boundary.

    Scaffold groups are sorted largest-first and dealt to whichever fold is
    currently smallest, which keeps folds balanced without ever separating members
    of a group. Inner folds are produced by re-dealing the outer-train *groups*,
    not the outer-train molecules - re-splitting molecules would put analogues of
    an inner-test compound into inner-train and quietly inflate every tuning score.

    Replaced at build step 3 by a label-aware allocator that also guarantees each
    outer test fold clears ``splits.min_test_labels_per_property``.
    """
    groups = _deal_groups(scaffold_id, np.arange(len(scaffold_id)), cfg.splits.n_outer)
    outer: list[Split] = []
    inner: list[tuple[Split, ...]] = []
    for held_out in groups:
        train_idx = np.sort(np.concatenate([g for g in groups if g is not held_out]))
        outer.append(Split(train_idx=train_idx, test_idx=np.sort(held_out)))
        inner_groups = _deal_groups(scaffold_id, train_idx, cfg.splits.n_inner)
        inner.append(
            tuple(
                Split(
                    train_idx=np.sort(
                        np.concatenate([g for g in inner_groups if g is not held])
                    ),
                    test_idx=np.sort(held),
                )
                for held in inner_groups
            )
        )

    spec = FoldSpec(
        outer=tuple(outer),
        inner=tuple(inner),
        scaffold_id=scaffold_id,
        seed=cfg.seed,
    )
    spec.validate_disjoint()
    return spec


def _deal_groups(
    scaffold_id: npt.NDArray[np.int64],
    subset: npt.NDArray[np.int64],
    n_folds: int,
) -> list[npt.NDArray[np.int64]]:
    """Deal whole scaffold groups within ``subset`` into ``n_folds`` balanced bins."""
    by_scaffold: dict[int, list[int]] = {}
    for idx in subset.tolist():
        by_scaffold.setdefault(int(scaffold_id[idx]), []).append(idx)

    bins: list[list[int]] = [[] for _ in range(n_folds)]
    for members in sorted(by_scaffold.values(), key=len, reverse=True):
        target = min(range(n_folds), key=lambda b: len(bins[b]))
        bins[target].extend(members)

    empty = [i for i, b in enumerate(bins) if not b]
    if empty:
        raise ValueError(
            f"fold(s) {empty} received no molecules: {len(by_scaffold)} scaffold "
            f"groups cannot fill {n_folds} folds"
        )
    return [np.array(sorted(b), dtype=np.int64) for b in bins]


# --------------------------------------------------------------------------- #
# Stand-in for build steps 4 and 6 - the two model tracks
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RidgeQSPRModel:
    """Three independent ridge regressions behind the multi-property interface.

    Structurally this is the XGBoost track in miniature: per-property models that
    each see only their own labelled rows, presented through :class:`QSPRModel` so
    the harness above never needs to know whether it is holding an independent or a
    multi-task model.

    Two details are deliberate rather than incidental, because both are real
    leakage traps:

    * target standardization statistics come from the rows passed to ``fit`` and
      nowhere else, so a fold boundary is never crossed;
    * the mask is consumed, not imputed - an unlabelled cell contributes nothing.

    Replaced at build step 4 (XGBoost) and step 6 (multi-task heads).
    """

    alpha: float = 1.0
    properties: tuple[PropertyName, ...] = PROPERTIES
    _coef: dict[PropertyName, npt.NDArray[np.float64]] = field(default_factory=dict)
    _centre: dict[PropertyName, tuple[float, float]] = field(default_factory=dict)
    _residual_quantiles: dict[PropertyName, npt.NDArray[np.float64]] = field(
        default_factory=dict
    )

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
        design = _with_bias(X)
        for j, prop in enumerate(self.properties):
            rows = np.flatnonzero(M[:, j])
            if rows.size < 2:
                raise ValueError(f"{prop}: only {rows.size} labelled rows to fit on")
            target = Y[rows, j]
            mean, std = float(target.mean()), float(target.std() or 1.0)
            self._centre[prop] = (mean, std)

            a = design[rows]
            penalty = self.alpha * np.eye(a.shape[1])
            penalty[-1, -1] = 0.0  # never penalise the intercept
            coef = np.linalg.solve(a.T @ a + penalty, a.T @ ((target - mean) / std))
            self._coef[prop] = coef

            residuals = target - (mean + std * (a @ coef))
            self._residual_quantiles[prop] = np.quantile(residuals, [0.05, 0.5, 0.95])
        return self

    def predict(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        design = _with_bias(X)
        out = np.empty((X.shape[0], len(self.properties)), dtype=np.float64)
        for j, prop in enumerate(self.properties):
            mean, std = self._centre[prop]
            out[:, j] = mean + std * (design @ self._coef[prop])
        return out

    def predict_quantiles(
        self, X: npt.NDArray[np.float32], quantiles: Sequence[float]
    ) -> npt.NDArray[np.float64]:
        """Point prediction shifted by training-residual quantiles.

        A homoscedastic stand-in: the width is identical for every molecule, which
        is precisely what real quantile regression and deep ensembles will fix.
        Keeping it obviously crude here avoids anyone mistaking it for calibration.
        """
        point = self.predict(X)
        out = np.empty((X.shape[0], len(self.properties), len(quantiles)))
        for j, prop in enumerate(self.properties):
            offsets = np.quantile(self._residual_quantiles[prop], list(quantiles))
            out[:, j, :] = point[:, [j]] + offsets[None, :]
        return out


def _with_bias(X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    return np.column_stack([X.astype(np.float64), np.ones(X.shape[0])])


# --------------------------------------------------------------------------- #
# Stand-in for build step 8 - calibration and applicability domain
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class SplitConformalCalibrator:
    """Symmetric split conformal on absolute residuals.

    This one is not a fake: it is the simplest correct conformal procedure and it
    really does deliver marginal coverage. What it lacks is adaptivity - every
    interval comes out the same width - which is exactly why build step 8 replaces
    it with conformalized quantile regression, and why the brief insists coverage
    is always reported next to mean interval width. A calibrator can hit 90%
    coverage by being uselessly wide.

    Replaced at build step 8 by CQR plus CV+/Jackknife+.
    """

    nominal_coverage: float = 0.9
    _radius: float = float("nan")

    def fit(
        self, raw: npt.NDArray[np.float64], y_true: npt.NDArray[np.float64]
    ) -> Self:
        point = raw[:, raw.shape[1] // 2]
        scores = np.abs(y_true - point)
        n = scores.size
        # The finite-sample correction. Without it coverage is biased low on small
        # calibration sets, which is the regime this project lives in.
        level = min(1.0, np.ceil((n + 1) * self.nominal_coverage) / n)
        self._radius = float(np.quantile(scores, level, method="higher"))
        return self

    def transform(self, raw: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        point = raw[:, raw.shape[1] // 2]
        return np.column_stack([point - self._radius, point + self._radius])


@dataclass(slots=True)
class NearestNeighbourDomain:
    """Distance from a molecule to its closest training neighbour, in [0, 1].

    The real version is nearest-neighbour Tanimoto over ECFP bit vectors; this uses
    cosine distance over the synthetic feature space, which has the same shape and
    the same role. The threshold is left unset by default because the brief is
    explicit that it comes off the error-versus-distance curve rather than a guess;
    the skeleton falls back to a placeholder so the record is well-formed.

    Replaced at build step 8 by ECFP Tanimoto, with the threshold chosen at step 9.
    """

    threshold: float = 0.5
    _reference: npt.NDArray[np.float64] = field(
        default_factory=lambda: np.zeros((0, 0))
    )

    def fit(self, X: npt.NDArray[np.float32]) -> Self:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        self._reference = (X / np.where(norms == 0, 1.0, norms)).astype(np.float64)
        return self

    def distance(self, X: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        query = (X / np.where(norms == 0, 1.0, norms)).astype(np.float64)
        similarity = (query @ self._reference.T).max(axis=1)
        return np.clip(1.0 - similarity, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Stand-in for build step 5 - metrics
# --------------------------------------------------------------------------- #


def rmse(y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def rank_correlation(
    y_true: npt.NDArray[np.float64], y_pred: npt.NDArray[np.float64]
) -> float:
    """Spearman, as Pearson over ranks.

    Written out rather than imported so the skeleton stays dependency-free; step 5
    swaps in ``scipy.stats.spearmanr`` and adds confidence intervals across folds.
    """
    if y_true.size < 3:
        return float("nan")
    a, b = _ranks(y_true), _ranks(y_pred)
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a**2).sum() * (b**2).sum())
    return float(a @ b / denominator) if denominator else float("nan")


def _ranks(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    order = np.argsort(values, kind="stable")
    out = np.empty(values.size, dtype=np.float64)
    out[order] = np.arange(values.size, dtype=np.float64)
    return out


# --------------------------------------------------------------------------- #
# Stand-in for build step 11 - the scored record
# --------------------------------------------------------------------------- #


def score_molecules(
    smiles: Sequence[str],
    *,
    model: RidgeQSPRModel,
    cache: HashFeatureCache,
    calibrators: dict[PropertyName, SplitConformalCalibrator],
    domain: NearestNeighbourDomain,
    cfg: Config,
    model_version: str,
) -> list[dict[str, Any]]:
    """Score a batch, returning one record per input in input order.

    Two requirements from brief §13 are structural and so are honoured here rather
    than bolted on later: the interface takes a list, and an input that cannot be
    parsed yields an :class:`ErrorRecord` instead of an exception. One bad SMILES
    in a batch of ten thousand must not cost the other 9,999.

    Replaced at build step 11 by the MLflow pyfunc.
    """
    valid: list[tuple[int, str]] = []
    records: list[dict[str, Any] | None] = [None] * len(smiles)
    for i, raw in enumerate(smiles):
        canonical = _canonicalize_or_none(raw)
        if canonical is None:
            records[i] = ErrorRecord(
                smiles_input=raw, error="could not be parsed as a molecule"
            ).to_dict()
        else:
            valid.append((i, canonical))

    if valid:
        keys = [s for _, s in valid]
        X = cache.transform(keys)
        point = model.predict(X)
        raw_q = model.predict_quantiles(X, cfg.uncertainty.quantiles)
        distances = domain.distance(X)

        bounds = {
            prop: calibrators[prop].transform(raw_q[:, j, :])
            for j, prop in enumerate(PROPERTIES)
        }
        threshold = (
            cfg.uncertainty.ad_threshold
            if cfg.uncertainty.ad_threshold is not None
            else domain.threshold
        )
        for k, (position, canonical) in enumerate(valid):
            records[position] = ScoredRecord(
                smiles_canonical=canonical,
                predictions={
                    prop: PropertyPrediction(
                        value=float(point[k, j]),
                        lower=float(bounds[prop][k, 0]),
                        upper=float(bounds[prop][k, 1]),
                        nominal_coverage=cfg.uncertainty.nominal_coverage,
                    )
                    for j, prop in enumerate(PROPERTIES)
                },
                applicability_domain=ApplicabilityDomain(
                    nn_tanimoto_distance=float(distances[k]),
                    in_domain=bool(distances[k] <= threshold),
                ),
                model_version=model_version,
                featurizer_version=cache.version,
            ).to_dict()

    return [r for r in records if r is not None]


def _canonicalize_or_none(smiles: str) -> str | None:
    """Placeholder for RDKit canonicalization, arriving at build step 1."""
    text = smiles.strip()
    if not text or "_" not in text or not text.startswith("S"):
        return None
    return text
