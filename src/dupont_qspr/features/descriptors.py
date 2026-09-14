"""Path A - classical descriptors, the representation the XGBoost track sees.

Two very different things concatenated into one vector:

**RDKit 2D descriptors** (217 of them) are named physical and topological
quantities - molecular weight, TPSA, ring counts, estimated logP. They are
interpretable, comparable across molecules, and on wildly different scales.

**ECFP4 fingerprint bits** (1,024) are the opposite: a circular substructure
fingerprint folded by hashing, so bit 512 means "some substructure hashed here"
and nothing more. Individually meaningless, collectively a precise description of
what fragments a molecule contains.

Together they cover both what a molecule *is* like and what it *contains*, which
is why this pairing is the standard baseline representation for QSPR and why the
brief makes it the number every other model has to beat.

**On non-finite values.** Some descriptors overflow or divide by zero on real
molecules - ``Ipc`` grows factorially with size and exceeds float64 on large
structures. Those become ``nan`` rather than being clipped or zeroed, because
XGBoost handles missing values natively and learns a default split direction for
them. Substituting zero would be a lie: zero is a plausible value for many of
these descriptors, so it would be indistinguishable from a real measurement.
Counts are recorded per descriptor so the damage is visible.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Sequence
from functools import cache

import numpy as np
import numpy.typing as npt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator

from dupont_qspr.config import Config
from dupont_qspr.features.cache import DiskFeatureCache

__all__ = ["build_descriptor_cache", "compute_path_a", "descriptor_feature_names"]

RDLogger.DisableLog("rdApp.*")

# Module-level so multiprocessing workers rebuild them once per process rather
# than once per molecule; the generator is not picklable, so it cannot be passed.
_ECFP_RADIUS = 2
_ECFP_BITS = 1024


@cache
def _descriptor_functions() -> tuple[tuple[str, object], ...]:
    return tuple(Descriptors.descList)


@cache
def _fingerprint_generator(radius: int, bits: int):
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)


def descriptor_feature_names(
    radius: int = _ECFP_RADIUS, bits: int = _ECFP_BITS
) -> tuple[str, ...]:
    """Names for every column, so a fitted model can be interrogated later.

    Descriptor columns carry their real names; fingerprint columns are numbered,
    because a folded hash bucket has no meaningful name to give it.
    """
    descriptors = tuple(name for name, _ in _descriptor_functions())
    return descriptors + tuple(f"ecfp{radius * 2}_{i}" for i in range(bits))


def n_path_a_features(radius: int = _ECFP_RADIUS, bits: int = _ECFP_BITS) -> int:
    return len(_descriptor_functions()) + bits


def _featurize_one(smiles: str) -> npt.NDArray[np.float32]:
    """One molecule to one vector. Must stay top-level to be picklable."""
    width = n_path_a_features()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # Should not happen: callers pass already-standardized SMILES. Returning
        # all-nan rather than raising keeps one bad row from killing a pool.
        return np.full(width, np.nan, dtype=np.float32)

    functions = _descriptor_functions()
    values = np.empty(len(functions), dtype=np.float64)
    try:
        # The fast path: compute the whole block at once.
        for i, (_, function) in enumerate(functions):
            values[i] = function(mol)
    except (ValueError, OverflowError, ZeroDivisionError, RuntimeError):
        # The slow path, taken only for molecules that broke something: isolate
        # the failure so one bad descriptor does not discard the other 216.
        for i, (_, function) in enumerate(functions):
            try:
                values[i] = function(mol)
            except (ValueError, OverflowError, ZeroDivisionError, RuntimeError):
                values[i] = np.nan

    fingerprint = _fingerprint_generator(
        _ECFP_RADIUS, _ECFP_BITS
    ).GetFingerprintAsNumPy(mol)
    combined = np.concatenate([values, fingerprint.astype(np.float64)])

    # Narrow first, then test. Some descriptors are finite in float64 but exceed
    # float32 - Ipc reaches 2.8e54 on a 164-atom molecule against a float32 limit
    # of 3.4e38 - so testing before the cast lets an inf through into storage.
    with np.errstate(over="ignore"):
        narrowed = combined.astype(np.float32)
    # inf is not something a tree can split on sensibly; fold it into nan, which
    # XGBoost already has defined behaviour for.
    return np.where(np.isfinite(narrowed), narrowed, np.float32(np.nan))


def compute_path_a(
    smiles: Sequence[str], *, n_jobs: int = 1, chunksize: int = 64
) -> npt.NDArray[np.float32]:
    """Featurize a batch, optionally across processes.

    RDKit releases nothing useful to threads, so parallelism has to be by process.
    Only SMILES strings cross the process boundary - ``Mol`` objects are expensive
    to pickle and the workers rebuild them anyway.
    """
    if not smiles:
        return np.empty((0, n_path_a_features()), dtype=np.float32)

    if n_jobs <= 1 or len(smiles) < chunksize * 2:
        return np.stack([_featurize_one(s) for s in smiles])

    with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
        rows = pool.map(_featurize_one, smiles, chunksize=chunksize)
    return np.stack(rows)


def build_descriptor_cache(
    cfg: Config, *, n_jobs: int | None = None
) -> DiskFeatureCache:
    """Wire the Path A featurizer into a persistent cache."""
    workers = n_jobs if n_jobs is not None else cfg.models.xgb_n_jobs
    bits = cfg.features.ecfp_bits
    radius = cfg.features.ecfp_radius

    if (radius, bits) != (_ECFP_RADIUS, _ECFP_BITS):
        raise NotImplementedError(
            f"configs request ECFP radius {radius} / {bits} bits, but the worker "
            "functions are pinned to the module defaults so they stay picklable. "
            "Change _ECFP_RADIUS and _ECFP_BITS together with the config."
        )

    return DiskFeatureCache(
        name="path_a_descriptors",
        version=cfg.features.version,
        n_features=n_path_a_features(radius, bits),
        compute=lambda batch: compute_path_a(batch, n_jobs=workers),
        directory=cfg.features_dir,
        feature_names=descriptor_feature_names(radius, bits),
        recipe={
            "rdkit_descriptors": len(_descriptor_functions()),
            "ecfp_radius": radius,
            "ecfp_bits": bits,
            "non_finite_policy": "nan",
        },
    )
