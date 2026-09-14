"""Nested, scaffold-disjoint cross-validation folds.

Two constraints pull against each other here, and the whole module is about
reconciling them.

**Scaffold groups are atomic.** Every molecule sharing a scaffold must land in the
same fold, or analogues leak across the train/test boundary and every downstream
number is quietly inflated. This is non-negotiable and is asserted, not assumed.

**Folds still have to be scoreable.** Each outer test fold needs enough labelled
molecules *per property* to compute a meaningful metric. That is not automatic,
because label coverage is wildly uneven across scaffolds: in this dataset 15% of
molecules are acyclic and they carry 27% of the solubility labels but 0.1% of the
lipophilicity labels. Dealing groups to balance *molecule counts* would therefore
produce folds badly skewed in *label* counts.

So the allocator balances labels, not molecules. Groups are dealt largest-first
into the fold where they do least damage to per-property balance, measured against
each property's ideal per-fold share. Largest-first matters: the biggest group in
this dataset is the 6,137 molecules whose scaffold is a lone benzene ring, which is
larger than an entire fold's fair share. Placing it last would leave no room to
compensate; placing it first lets every subsequent decision work around it.

Inner folds are produced by re-splitting the outer-training *groups*, never the
outer-training molecules. Re-splitting molecules would put analogues of an
inner-test compound into inner-train and inflate every hyperparameter score, which
is the same leak one level down.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
from dupont_qspr.splits.scaffold import ScaffoldAssignment, assign_scaffolds

__all__ = ["FoldBuild", "allocate_groups", "build_nested_folds"]


@dataclass(slots=True)
class FoldBuild:
    """Folds plus everything needed to judge whether they are usable."""

    spec: FoldSpec
    assignment: ScaffoldAssignment
    label_counts: npt.NDArray[np.int64]  # (n_outer, n_properties) on the test side
    thin_folds: list[dict[str, Any]]

    def report(self, minimum: int) -> dict[str, Any]:
        return {
            "n_outer": self.spec.n_outer,
            "n_inner": self.spec.n_inner,
            "scaffolds": self.assignment.summary(),
            "test_sizes": [int(s.test_idx.size) for s in self.spec.outer],
            "test_labels_per_fold": [
                {p: int(self.label_counts[f, j]) for j, p in enumerate(PROPERTIES)}
                for f in range(self.spec.n_outer)
            ],
            "min_required": minimum,
            "thin_folds": self.thin_folds,
        }


def allocate_groups(
    group_members: list[npt.NDArray[np.int64]],
    labels: npt.NDArray[np.bool_],
    n_folds: int,
) -> list[npt.NDArray[np.int64]]:
    """Deal whole scaffold groups into ``n_folds`` bins, balancing label counts.

    ``labels`` is the ``(n_molecules, n_properties)`` mask. Groups are visited
    largest-first and each is placed in the fold minimising a cost that grows
    quadratically with per-property overshoot::

        cost(fold) = sum_p ((count[fold][p] + group[p]) / target[p]) ** 2

    The square is what makes this work. A linear cost would be indifferent between
    "one fold at 200% and one at 0%" and "both at 100%", since both sum the same;
    squaring makes concentration expensive and spreads labels out. Dividing by the
    per-property target puts the three properties on comparable footing, so
    melting point - with five times as many labels as lipophilicity - cannot
    dominate the objective.

    Ties break toward the fold holding fewest molecules, which keeps fold *sizes*
    reasonable without letting size override the label objective.
    """
    if n_folds < 2:
        raise ValueError(f"need at least 2 folds, got {n_folds}")

    n_properties = labels.shape[1]
    totals = labels.sum(axis=0).astype(np.float64)
    # A property with no labels at all must not divide by zero, and contributes
    # nothing to the objective either way.
    targets = np.maximum(totals / n_folds, 1.0)

    counts = np.zeros((n_folds, n_properties), dtype=np.float64)
    sizes = np.zeros(n_folds, dtype=np.int64)
    bins: list[list[int]] = [[] for _ in range(n_folds)]

    order = sorted(range(len(group_members)), key=lambda g: -group_members[g].size)
    for g in order:
        members = group_members[g]
        contribution = labels[members].sum(axis=0).astype(np.float64)

        projected = (counts + contribution) / targets
        cost = (projected**2).sum(axis=1)
        # Tie-break on fold size: tiny, so it only decides genuine ties.
        cost = cost + 1e-9 * sizes

        target_fold = int(np.argmin(cost))
        bins[target_fold].extend(members.tolist())
        counts[target_fold] += contribution
        sizes[target_fold] += members.size

    empty = [i for i, b in enumerate(bins) if not b]
    if empty:
        raise ValueError(
            f"fold(s) {empty} received no molecules: {len(group_members)} scaffold "
            f"groups cannot fill {n_folds} folds"
        )
    return [np.array(sorted(b), dtype=np.int64) for b in bins]


def _groups_within(
    subset: npt.NDArray[np.int64], scaffold_id: npt.NDArray[np.int64]
) -> list[npt.NDArray[np.int64]]:
    """Scaffold groups restricted to ``subset``, as global index arrays."""
    by_scaffold: dict[int, list[int]] = {}
    for index in subset.tolist():
        by_scaffold.setdefault(int(scaffold_id[index]), []).append(index)
    return [np.array(v, dtype=np.int64) for v in by_scaffold.values()]


def build_nested_folds(table: pl.DataFrame, cfg: Config) -> FoldBuild:
    """Build the full nested fold structure over a union table."""
    smiles = table.get_column(SMILES_COLUMN).to_list()
    _, mask = targets_and_mask(table)

    assignment = assign_scaffolds(
        smiles,
        generic=cfg.splits.generic_scaffolds,
        acyclic_policy=cfg.splits.acyclic_policy,
    )
    scaffold_id = assignment.scaffold_id
    all_indices = np.arange(len(smiles), dtype=np.int64)

    outer_bins = allocate_groups(
        _groups_within(all_indices, scaffold_id), mask, cfg.splits.n_outer
    )

    outer: list[Split] = []
    inner: list[tuple[Split, ...]] = []
    for held_out in outer_bins:
        train_idx = np.sort(
            np.concatenate([b for b in outer_bins if b is not held_out])
        )
        outer.append(Split(train_idx=train_idx, test_idx=np.sort(held_out)))

        # Re-split the outer-training GROUPS, not its molecules.
        inner_bins = allocate_groups(
            _groups_within(train_idx, scaffold_id), mask, cfg.splits.n_inner
        )
        inner.append(
            tuple(
                Split(
                    train_idx=np.sort(
                        np.concatenate([b for b in inner_bins if b is not held])
                    ),
                    test_idx=np.sort(held),
                )
                for held in inner_bins
            )
        )

    spec = FoldSpec(
        outer=tuple(outer),
        inner=tuple(inner),
        scaffold_id=scaffold_id,
        seed=cfg.seed,
    )
    # Asserted rather than trusted: this is the invariant the whole protocol rests
    # on, and its failure mode is silent.
    spec.validate_disjoint()

    label_counts = np.stack([mask[s.test_idx].sum(axis=0) for s in spec.outer]).astype(
        np.int64
    )
    minimum = cfg.splits.min_test_labels_per_property
    thin = [
        {
            "fold": f,
            "property": p,
            "n_labels": int(label_counts[f, j]),
            "required": minimum,
        }
        for f in range(spec.n_outer)
        for j, p in enumerate(PROPERTIES)
        if label_counts[f, j] < minimum
    ]

    return FoldBuild(
        spec=spec, assignment=assignment, label_counts=label_counts, thin_folds=thin
    )
