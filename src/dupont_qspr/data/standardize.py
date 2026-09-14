"""Turn whatever a source calls a structure into one canonical, comparable key.

Every downstream stage depends on this being right, because canonical SMILES is
the join key. Joining on strings that were never canonicalized is the single most
common way a table like this fails: the same molecule written two ways becomes two
rows, the union table gains duplicate keys, and one compound ends up in two
cross-validation folds with no metric anywhere showing it.

Nothing is dropped silently. Every rejected row is written to a ledger with a
reason, and the counts are reported as a first-class artifact - a curation step
whose losses are invisible cannot be audited.

The pipeline, in order:

1. parse - anything RDKit cannot read is rejected, not coerced
2. fragments - strip counterions and solvate; reject genuine mixtures
3. neutralize - undo protonation where it is unambiguous
4. canonicalize - the join key

Range screening is separate (:func:`screen_range`) because it is per-property and
runs after the value columns have been harmonized into their storage units.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import cache

import polars as pl
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

from dupont_qspr.contracts import PropertyName

__all__ = [
    "LEDGER_SCHEMA",
    "RejectReason",
    "Standardized",
    "resolve_duplicate_values",
    "screen_range",
    "standardize_frame",
    "standardize_smiles",
]

# RDKit narrates every parse failure to stderr. The return value already tells us
# what happened and the ledger records it properly, so the noise is pure cost.
RDLogger.DisableLog("rdApp.*")


class RejectReason(StrEnum):
    EMPTY = "empty_smiles"
    UNPARSEABLE = "unparseable"
    MIXTURE = "mixture"
    NO_HEAVY_ATOMS = "no_heavy_atoms"
    NO_CARBON = "no_carbon"
    SANITIZE_FAILED = "sanitize_failed"
    ROUND_TRIP_FAILED = "round_trip_failed"
    OUT_OF_RANGE = "out_of_range"
    MISSING_VALUE = "missing_value"


#: The curation ledger. One row per rejected input, everywhere in the pipeline.
LEDGER_SCHEMA: dict[str, pl.DataType] = {
    "source": pl.String(),
    "stage": pl.String(),
    "reason": pl.String(),
    "detail": pl.String(),
    "smiles_in": pl.String(),
    "value": pl.Float64(),
}


@dataclass(frozen=True, slots=True)
class Standardized:
    """Outcome for one input structure."""

    canonical: str | None
    reason: RejectReason | None
    detail: str = ""
    #: Fragments discarded as counterions or solvate.
    stripped_fragments: int = 0
    #: True when neutralization actually changed the structure.
    neutralized: bool = False
    #: True when the structure only parsed after aromatic-nitrogen repair.
    repaired: bool = False
    #: True when repeated copies of one fragment were collapsed to a single copy.
    deduplicated_fragments: bool = False

    @property
    def ok(self) -> bool:
        return self.canonical is not None


# The standardizer objects carry setup cost, so build them once. They are not
# thread-safe; this pipeline is single-threaded by design.
@cache
def _largest_fragment_chooser() -> rdMolStandardize.LargestFragmentChooser:
    return rdMolStandardize.LargestFragmentChooser()


@cache
def _uncharger() -> rdMolStandardize.Uncharger:
    return rdMolStandardize.Uncharger()


def _repair_aromatic_nitrogen(smiles: str) -> Chem.Mol | None:
    """Recover structures whose only fault is an aromatic N missing its hydrogen.

    Pyrrole-type nitrogens contribute a lone pair to the aromatic sextet and must
    carry an explicit hydrogen; pyridine-type nitrogens must not. Plenty of
    literature-compiled SMILES write the first as the second - ``c1cncn1`` for
    imidazole, ``c1cccn1`` for pyrrole - which RDKit correctly refuses to kekulize
    and returns as a parse failure.

    Discarding those would be a real loss. In the Bradley set it accounts for
    around 290 compounds, and they are ordinary heterocycles, not exotic edge
    cases, so losing them would quietly bias the melting-point training set away
    from an entire structural class.

    The repair adds one hydrogen to each candidate nitrogen in turn and keeps the
    first assignment that sanitizes; failing that, it tries setting all candidates
    at once, which covers fused systems needing more than one. Anything still
    failing is rejected normally.
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
    except Chem.AtomValenceException:
        return None

    candidates = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "N"
        and atom.GetIsAromatic()
        and atom.GetFormalCharge() == 0
        and atom.GetTotalNumHs() == 0
        and atom.GetDegree() == 2
    ]
    if not candidates:
        return None

    for subset in ([index] for index in candidates):
        if (repaired := _try_with_explicit_hs(smiles, subset)) is not None:
            return repaired
    return _try_with_explicit_hs(smiles, candidates)


def _try_with_explicit_hs(smiles: str, indices: list[int]) -> Chem.Mol | None:
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        mol.UpdatePropertyCache(strict=False)
        for index in indices:
            mol.GetAtomWithIdx(index).SetNumExplicitHs(1)
        Chem.SanitizeMol(mol)
    except (Chem.KekulizeException, Chem.AtomValenceException, RuntimeError):
        return None
    return mol


def _distinct_fragments(mol: Chem.Mol) -> tuple[list[Chem.Mol], bool]:
    """Fragments with exact repeats collapsed to one copy.

    A zinc salt carrying two identical ligands is written as three fragments, two
    of them tied for largest. Judged on size alone that looks exactly like a
    two-component mixture, and the entry would be thrown away even though the
    molecule being measured is completely unambiguous. Deduplicating first, then
    applying the size rule to what remains, separates stoichiometry from genuine
    ambiguity.
    """
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    seen: dict[str, Chem.Mol] = {}
    for fragment in fragments:
        try:
            key = Chem.MolToSmiles(fragment)
        except (Chem.KekulizeException, RuntimeError):  # pragma: no cover
            key = f"unhashable-{len(seen)}"
        seen.setdefault(key, fragment)
    distinct = list(seen.values())
    return distinct, len(distinct) < len(fragments)


def standardize_smiles(
    smiles: str | None,
    *,
    mixture_ratio: float = 0.5,
    require_carbon: bool = False,
    keep_stereochemistry: bool = True,
) -> Standardized:
    """Standardize one SMILES string.

    ``mixture_ratio`` decides where salt-stripping ends and mixture-rejection
    begins. Both look identical structurally - several disconnected fragments -
    and the difference is whether one of them is clearly *the* molecule. When the
    second-largest fragment has at least this share of the largest one's heavy
    atoms, no fragment is obviously the subject and the row is rejected. At the
    default of 0.5 a sodium salt (1 heavy atom against 13) is stripped, while
    ``c1ccccc1.CCO`` (3 against 6) is rejected rather than silently becoming
    benzene with ethanol's measured property attached to it.

    ``keep_stereochemistry`` is a genuine trade. Keeping it is correct - two
    enantiomers can have different measured properties - but it costs joins,
    because sources disagree about whether to record stereo at all, and the same
    compound then appears under two keys. Discarding it would instead merge those
    into one key with conflicting values, which the conflict machinery in
    :mod:`dupont_qspr.data.curate_mp` is built to record. The default keeps stereo.
    """
    if smiles is None or not smiles.strip():
        return Standardized(None, RejectReason.EMPTY)

    text = smiles.strip()
    repaired = False
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        mol = _repair_aromatic_nitrogen(text)
        if mol is None:
            return Standardized(None, RejectReason.UNPARSEABLE, detail=text[:120])
        repaired = True

    if mol.GetNumHeavyAtoms() == 0:
        return Standardized(None, RejectReason.NO_HEAVY_ATOMS)

    stripped = 0
    # Repeats are collapsed before the size test, so that stoichiometric copies of
    # one ligand are not mistaken for a multi-component mixture.
    distinct, deduplicated = _distinct_fragments(mol)
    if len(distinct) > 1:
        sizes = sorted((f.GetNumHeavyAtoms() for f in distinct), reverse=True)
        if sizes[1] >= mixture_ratio * sizes[0]:
            return Standardized(
                None,
                RejectReason.MIXTURE,
                detail=f"distinct fragment heavy-atom counts {sizes}",
            )
        mol = max(distinct, key=lambda f: f.GetNumHeavyAtoms())
        try:
            Chem.SanitizeMol(mol)
        except (Chem.KekulizeException, Chem.AtomValenceException) as error:
            return Standardized(
                None, RejectReason.SANITIZE_FAILED, detail=str(error)[:120]
            )
        stripped = len(distinct) - 1
    elif deduplicated:
        mol = distinct[0]
        try:
            Chem.SanitizeMol(mol)
        except (Chem.KekulizeException, Chem.AtomValenceException) as error:
            return Standardized(
                None, RejectReason.SANITIZE_FAILED, detail=str(error)[:120]
            )

    try:
        neutral = _uncharger().uncharge(mol)
    except (
        Chem.KekulizeException,
        Chem.AtomValenceException,
    ) as error:  # pragma: no cover
        return Standardized(None, RejectReason.SANITIZE_FAILED, detail=str(error)[:120])

    charged = mol
    neutralized = Chem.MolToSmiles(neutral) != Chem.MolToSmiles(charged)
    mol = neutral

    if require_carbon and not any(atom.GetAtomicNum() == 6 for atom in mol.GetAtoms()):
        return Standardized(None, RejectReason.NO_CARBON)

    canonical = Chem.MolToSmiles(mol, isomericSmiles=keep_stereochemistry)
    if not canonical:
        return Standardized(
            None, RejectReason.SANITIZE_FAILED, detail="empty canonical output"
        )

    # Round-trip guard. This function's output is the join key for the whole
    # project, so "RDKit wrote it" is not sufficient - it also has to be something
    # RDKit can read back. Without this check the failure surfaces two stages
    # later, in featurization, where the molecule silently becomes a row of NaN.
    if Chem.MolFromSmiles(canonical) is None:
        # Almost always caused by neutralizing a charge that was structural rather
        # than a protonation state. A cyclopentadienyl anion is aromatic *because*
        # of its negative charge; removing it leaves an all-carbon five-membered
        # ring that cannot be aromatic and will not parse. Keep the charged form.
        if neutralized:
            fallback = Chem.MolToSmiles(charged, isomericSmiles=keep_stereochemistry)
            if fallback and Chem.MolFromSmiles(fallback) is not None:
                return Standardized(
                    canonical=fallback,
                    reason=None,
                    stripped_fragments=stripped,
                    neutralized=False,
                    repaired=repaired,
                    deduplicated_fragments=deduplicated,
                )
        return Standardized(
            None, RejectReason.ROUND_TRIP_FAILED, detail=canonical[:120]
        )

    return Standardized(
        canonical=canonical,
        reason=None,
        stripped_fragments=stripped,
        neutralized=neutralized,
        repaired=repaired,
        deduplicated_fragments=deduplicated,
    )


def standardize_frame(
    frame: pl.DataFrame,
    *,
    smiles_column: str,
    source: str,
    value_column: str | None = None,
    mixture_ratio: float = 0.5,
    require_carbon: bool = False,
    keep_stereochemistry: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Standardize a whole table, returning ``(kept, ledger)``.

    ``kept`` gains a ``smiles`` column holding the canonical key, plus
    ``n_stripped_fragments`` and ``was_neutralized`` for the curation report. Rows
    that failed appear only in ``ledger``, never dropped without a trace.
    """
    originals = frame.get_column(smiles_column).to_list()
    values = (
        frame.get_column(value_column).cast(pl.Float64, strict=False).to_list()
        if value_column is not None
        else [None] * frame.height
    )

    canonical: list[str | None] = []
    stripped: list[int] = []
    neutralized: list[bool] = []
    repaired: list[bool] = []
    rejects: list[dict[str, object]] = []

    for index, raw in enumerate(originals):
        result = standardize_smiles(
            raw,
            mixture_ratio=mixture_ratio,
            require_carbon=require_carbon,
            keep_stereochemistry=keep_stereochemistry,
        )
        canonical.append(result.canonical)
        stripped.append(result.stripped_fragments)
        neutralized.append(result.neutralized)
        repaired.append(result.repaired)
        if result.reason is not None:
            rejects.append(
                {
                    "source": source,
                    "stage": "standardize",
                    "reason": str(result.reason),
                    "detail": result.detail,
                    "smiles_in": raw,
                    "value": values[index],
                }
            )

    annotated = frame.with_columns(
        pl.Series("smiles", canonical, dtype=pl.String),
        pl.Series("n_stripped_fragments", stripped, dtype=pl.Int32),
        pl.Series("was_neutralized", neutralized, dtype=pl.Boolean),
        pl.Series("was_repaired", repaired, dtype=pl.Boolean),
    )
    kept = annotated.filter(pl.col("smiles").is_not_null())
    return kept, _ledger(rejects)


def screen_range(
    frame: pl.DataFrame,
    *,
    column: str,
    prop: PropertyName,
    bounds: tuple[float, float],
    source: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Drop physically implausible values, recording each one.

    Runs after unit harmonization so the bounds are stated in storage units. A
    melting point of 1,200 K for an organic solid is a transcription error, not a
    measurement, and letting it through would distort both the target scale and
    every standardization statistic computed from it.
    """
    low, high = bounds
    values = frame.get_column(column).cast(pl.Float64, strict=False)

    missing = values.is_null() | values.is_nan()
    out_of_range = ~missing & ((values < low) | (values > high))

    rejects: list[dict[str, object]] = []
    for row in frame.filter(missing).iter_rows(named=True):
        rejects.append(
            {
                "source": source,
                "stage": f"range:{prop}",
                "reason": str(RejectReason.MISSING_VALUE),
                "detail": "",
                "smiles_in": row.get("smiles"),
                "value": None,
            }
        )
    for row in frame.filter(out_of_range).iter_rows(named=True):
        value = row.get(column)
        rejects.append(
            {
                "source": source,
                "stage": f"range:{prop}",
                "reason": str(RejectReason.OUT_OF_RANGE),
                "detail": f"outside [{low}, {high}]",
                "smiles_in": row.get("smiles"),
                "value": float(value) if value is not None else None,
            }
        )

    return frame.filter(~missing & ~out_of_range), _ledger(rejects)


def resolve_duplicate_values(
    frame: pl.DataFrame,
    *,
    value_column: str,
    source: str,
    keep: list[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Collapse rows that standardization mapped onto the same canonical key.

    Even a source published with one row per compound can produce duplicates here,
    because standardization merges structures the publisher treated as distinct -
    a free acid and its sodium salt reduce to the same parent, as do two entries
    that differ only in how a charge was written.

    Leaving them would put duplicate keys in the union table, which is exactly the
    condition :func:`~dupont_qspr.contracts.validate_union_table` refuses. Merging
    them silently would hide a real disagreement, so the median is kept and every
    collapse is written to the ledger with the spread it concealed.
    """
    counts = frame.group_by("smiles").len()
    duplicated = counts.filter(pl.col("len") > 1)

    rejects: list[dict[str, object]] = []
    if duplicated.height:
        detail = (
            frame.join(duplicated.select("smiles"), on="smiles", how="semi")
            .group_by("smiles")
            .agg(
                pl.col(value_column).min().alias("lo"),
                pl.col(value_column).max().alias("hi"),
                pl.len().alias("n"),
            )
        )
        for row in detail.iter_rows(named=True):
            spread = (row["hi"] or 0.0) - (row["lo"] or 0.0)
            rejects.append(
                {
                    "source": source,
                    "stage": "duplicate_canonical_key",
                    "reason": "collapsed_to_median",
                    "detail": (
                        f"{row['n']} rows shared this canonical SMILES; "
                        f"values spanned {spread:.3f}"
                    ),
                    "smiles_in": row["smiles"],
                    "value": row["hi"],
                }
            )

    aggregated = (
        frame.group_by("smiles")
        .agg(
            pl.col(value_column).median().alias(value_column),
            *[pl.col(c).first().alias(c) for c in keep if c in frame.columns],
        )
        .sort("smiles")
    )
    return aggregated, _ledger(rejects)


def _ledger(rows: list[dict[str, object]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=LEDGER_SCHEMA)
    return pl.DataFrame(rows, schema=LEDGER_SCHEMA)
