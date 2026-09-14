"""Assemble the three persisted artifacts into the arrays a model run needs.

By this point every expensive thing has already been done and stored: the union
table from step 1, the feature caches from step 2, the folds from step 3. This
module does no computation - it reads, aligns, and hands back arrays. That is the
payoff of the earlier steps, and it is why a nested run with thousands of fits is
affordable.

Alignment is the one thing that can go wrong here, and it would go wrong silently:
if the folds file and the union table disagreed about row order, every molecule
would be paired with another molecule's labels and nothing would raise. So the
join is explicit and verified rather than positional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import (
    PROPERTIES,
    SMILES_COLUMN,
    FoldSpec,
    Split,
    targets_and_mask,
)
from dupont_qspr.features import build_descriptor_cache, build_encoder_cache

__all__ = ["Prepared", "load_prepared"]

FeaturePath = Literal["a", "b"]


@dataclass(frozen=True, slots=True)
class Prepared:
    """Everything a nested run consumes, already aligned."""

    smiles: list[str]
    X: npt.NDArray[np.float32]
    Y: npt.NDArray[np.float64]
    M: npt.NDArray[np.bool_]
    folds: FoldSpec
    feature_name: str
    featurizer_version: str

    @property
    def n_molecules(self) -> int:
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def labelled(
        self, indices: npt.NDArray[np.int64], prop_index: int
    ) -> npt.NDArray[np.int64]:
        """Restrict ``indices`` to rows where this property is actually measured.

        Every fit in the project goes through this. A model for one property must
        never see a row where that property is missing, and the masked target
        would be ``nan``, so failing to filter would poison the fit rather than
        merely waste time.
        """
        return indices[self.M[indices, prop_index]]

    def train_sd(self, indices: npt.NDArray[np.int64], prop_index: int) -> float:
        """Standard deviation of a property over the given rows.

        Used to normalize RMSE. It must be computed on *training* rows only -
        taking it from the whole dataset would leak the test distribution's scale
        into the number being reported.
        """
        rows = self.labelled(indices, prop_index)
        if rows.size < 2:
            return 1.0
        return float(np.std(self.Y[rows, prop_index])) or 1.0


def _fold_spec_from_frame(folds: pl.DataFrame, n_rows: int) -> FoldSpec:
    """Rebuild the FoldSpec recorded at step 3 rather than re-splitting.

    Re-deriving folds here would risk a different answer from a config drift, and
    then the numbers reported would not correspond to the folds that were audited.
    """
    outer_of = folds.get_column("outer_fold").to_numpy()
    n_outer = int(outer_of.max()) + 1
    all_indices = np.arange(n_rows, dtype=np.int64)

    outer: list[Split] = []
    inner: list[tuple[Split, ...]] = []
    for f in range(n_outer):
        test_idx = all_indices[outer_of == f]
        train_idx = all_indices[outer_of != f]
        outer.append(Split(train_idx=train_idx, test_idx=test_idx))

        column = f"inner_fold_{f}"
        inner_of = folds.get_column(column).to_numpy()
        n_inner = int(inner_of[train_idx].max()) + 1
        inner.append(
            tuple(
                Split(
                    train_idx=train_idx[inner_of[train_idx] != k],
                    test_idx=train_idx[inner_of[train_idx] == k],
                )
                for k in range(n_inner)
            )
        )

    return FoldSpec(
        outer=tuple(outer),
        inner=tuple(inner),
        scaffold_id=folds.get_column("scaffold_id").to_numpy(),
        seed=0,
    )


def load_prepared(
    cfg: Config, *, path: FeaturePath = "a", encoder: str | None = None
) -> Prepared:
    """Load table, folds and features for one featurization path.

    ``path="a"`` gives the descriptor representation feeding the XGBoost track;
    ``path="b"`` gives a frozen-encoder embedding, defaulting to
    ``features.primary_encoder``.
    """
    table_path = cfg.processed_dir / "union_table.parquet"
    folds_path = cfg.processed_dir / "folds.parquet"
    for required in (table_path, folds_path):
        if not required.exists():
            raise FileNotFoundError(
                f"{required} not found. Run experiments/01_build_dataset.py and "
                f"experiments/03_build_folds.py for profile {cfg.profile!r} first."
            )

    table = pl.read_parquet(table_path)
    folds = pl.read_parquet(folds_path)

    # Verified alignment, not assumed. A silent mismatch here would pair every
    # molecule with another molecule's labels.
    if folds.height != table.height:
        raise ValueError(
            f"folds file has {folds.height} rows but the union table has "
            f"{table.height}; they were built from different data"
        )
    if not (
        folds.get_column("smiles").to_list()
        == table.get_column(SMILES_COLUMN).to_list()
    ):
        raise ValueError(
            "folds file and union table disagree on SMILES order. Rebuild folds "
            "from the current union table rather than reordering either file."
        )

    cache = (
        build_descriptor_cache(cfg)
        if path == "a"
        else build_encoder_cache(cfg, encoder)
    )
    smiles = table.get_column(SMILES_COLUMN).to_list()
    already = cache.contains(smiles)
    if already < len(smiles):
        raise FileNotFoundError(
            f"{cache.name} has {already:,} of {len(smiles):,} molecules cached. Run "
            f"experiments/02_featurize.py --profile {cfg.profile} first; a nested "
            "run should never pay featurization cost."
        )

    X = cache.transform(smiles)
    Y, M = targets_and_mask(table)
    spec = _fold_spec_from_frame(folds, table.height)
    spec.validate_disjoint()

    return Prepared(
        smiles=smiles,
        X=X,
        Y=Y,
        M=M,
        folds=spec,
        feature_name=cache.name,
        featurizer_version=cache.version,
    )


def property_index(name: str) -> int:
    return PROPERTIES.index(name)  # type: ignore[arg-type]
