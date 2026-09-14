"""Bemis-Murcko scaffold assignment - the grouping that folds are built from.

A Bemis-Murcko scaffold is what remains of a molecule after every side chain is
stripped away: the ring systems plus the linkers joining them. Two molecules
sharing a scaffold are analogues of each other, and putting one in training and
the other in test is the leak that scaffold splitting exists to prevent. A random
split would scatter analogues across the boundary and report a number that
collapses the moment a genuinely novel chemotype arrives - which is precisely the
job this model is being built for.

Two decisions here are not obvious and are both configurable.

**Acyclic molecules have no scaffold at all.** Strip the side chains from ethanol
and nothing is left. In this dataset that is 4,471 molecules, 15.2% of the total,
and they are distributed very unevenly across properties: 26.7% of the solubility
set but 0.1% of the lipophilicity set. The convention inherited from DeepChem puts
them all in one bucket keyed by the empty string, which makes a single
4,471-member group that must move between folds as one piece and drags its skewed
label mix with it. The default here instead gives each acyclic molecule its own
group, on the grounds that "no scaffold" means there is no shared core to leak.
The cost is that genuine acyclic analogues - hexane and heptane - may land on
opposite sides of a split, which is a real if smaller leak. Set
``acyclic_policy="shared"`` to get the conventional behaviour.

**Generic frameworks are available but off.** ``MakeScaffoldGeneric`` collapses
every atom to carbon and every bond to single, so pyridine and benzene become one
group. It produces a harder, more pessimistic split. The brief specifies plain
Bemis-Murcko, so that is the default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

__all__ = [
    "ScaffoldAssignment",
    "assign_scaffolds",
    "scaffold_for_smiles",
]

RDLogger.DisableLog("rdApp.*")

AcyclicPolicy = Literal["singleton", "shared"]


@dataclass(slots=True)
class ScaffoldAssignment:
    """Scaffold group membership for a set of molecules, plus reporting detail."""

    #: ``scaffold_id[i]`` is the group index of molecule ``i``. This is what
    #: :class:`~dupont_qspr.contracts.FoldSpec` validates disjointness against.
    scaffold_id: npt.NDArray[np.int64]
    #: Human-readable scaffold SMILES per group index, for the fold report.
    scaffold_smiles: list[str]
    #: Molecules RDKit could not parse, which should be none by this stage.
    n_unparseable: int
    #: Molecules with no ring system at all.
    n_acyclic: int
    policy: AcyclicPolicy

    @property
    def n_groups(self) -> int:
        return len(self.scaffold_smiles)

    def group_sizes(self) -> npt.NDArray[np.int64]:
        return np.bincount(self.scaffold_id, minlength=self.n_groups)

    def summary(self, top: int = 8) -> dict[str, Any]:
        sizes = self.group_sizes()
        order = np.argsort(sizes)[::-1]
        return {
            "n_molecules": int(self.scaffold_id.size),
            "n_groups": self.n_groups,
            "n_singletons": int((sizes == 1).sum()),
            "n_acyclic": self.n_acyclic,
            "acyclic_policy": self.policy,
            "n_unparseable": self.n_unparseable,
            "largest_group": int(sizes.max()) if sizes.size else 0,
            "largest_groups": [
                {"scaffold": self.scaffold_smiles[i] or "<acyclic>", "n": int(sizes[i])}
                for i in order[:top]
            ],
        }


def scaffold_for_smiles(smiles: str, *, generic: bool = False) -> str | None:
    """Bemis-Murcko scaffold as canonical SMILES, or ``None`` if unparseable.

    An acyclic molecule yields the empty string rather than ``None`` - it parsed
    fine, it simply has no ring system.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(core)
    except (
        Chem.KekulizeException,
        Chem.AtomValenceException,
        RuntimeError,
        ValueError,
    ):
        return None


def assign_scaffolds(
    smiles: Sequence[str],
    *,
    generic: bool = False,
    acyclic_policy: AcyclicPolicy = "singleton",
) -> ScaffoldAssignment:
    """Group molecules by scaffold, returning integer group ids.

    Group ids are assigned in order of first appearance, so the result is
    deterministic for a given input ordering and does not depend on hashing.
    """
    group_of: dict[str, int] = {}
    scaffold_smiles: list[str] = []
    ids = np.empty(len(smiles), dtype=np.int64)

    n_unparseable = 0
    n_acyclic = 0

    for i, entry in enumerate(smiles):
        scaffold = scaffold_for_smiles(entry, generic=generic)

        if scaffold is None:
            # Should not happen after step 1's round-trip guard. Give it a private
            # group rather than silently merging failures into one bucket, which
            # would create a fake "unparseable" scaffold shared by unrelated
            # molecules and leak across folds through pure accident.
            n_unparseable += 1
            ids[i] = len(scaffold_smiles)
            scaffold_smiles.append(f"<unparseable:{i}>")
            continue

        if scaffold == "":
            n_acyclic += 1
            if acyclic_policy == "singleton":
                ids[i] = len(scaffold_smiles)
                scaffold_smiles.append("")
                continue

        existing = group_of.get(scaffold)
        if existing is None:
            existing = len(scaffold_smiles)
            group_of[scaffold] = existing
            scaffold_smiles.append(scaffold)
        ids[i] = existing

    return ScaffoldAssignment(
        scaffold_id=ids,
        scaffold_smiles=scaffold_smiles,
        n_unparseable=n_unparseable,
        n_acyclic=n_acyclic,
        policy=acyclic_policy,
    )
