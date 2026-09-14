"""Typed configuration, loaded from ``configs/`` YAML and selected by profile.

Every knob that trades runtime against thoroughness lives behind ``--profile``.
``smoke`` runs the entire pipeline end to end in seconds and is what the test suite
uses; ``dev`` is the working loop; ``full`` produces the numbers that get reported.
Keeping the profile as the only speed dial means the smoke path exercises the same
code the full run does, rather than a parallel toy implementation that drifts.

Models are frozen and forbid unknown fields, so a typo in a YAML file fails at load
rather than being silently ignored halfway through a three-hour run.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dupont_qspr.contracts import PROPERTIES, PropertyName

__all__ = [
    "Config",
    "DataConfig",
    "FeatureConfig",
    "ModelConfig",
    "SplitConfig",
    "TrackingConfig",
    "TuningConfig",
    "UncertaintyConfig",
    "configs_dir",
    "load_config",
    "project_root",
]

Profile = Literal["smoke", "dev", "full"]


def project_root() -> Path:
    """Repository root, overridable with ``DUPONT_QSPR_ROOT`` for out-of-tree runs."""
    if (override := os.environ.get("DUPONT_QSPR_ROOT")) is not None:
        return Path(override).resolve()
    # src/dupont_qspr/config.py -> src/dupont_qspr -> src -> root
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    return project_root() / "configs"


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------- #


class DataConfig(_Base):
    """Acquisition, curation, and the physical sanity screens."""

    #: The high-purity Bradley subset keeps only entries whose replicate
    #: measurements agree within a few degrees (~3k rows); the full curated set is
    #: roughly 20k. Default to full, per the brief.
    bradley_subset: Literal["full", "double_plus_good"] = "full"

    #: Replicate melting points disagreeing by more than this are recorded as a
    #: conflict rather than averaged. The spread bounds achievable RMSE and is
    #: reported as a limitation, so it must not be quietly smoothed away.
    mp_conflict_tolerance_k: float = 5.0

    #: Physical plausibility windows, in each property's storage units. Values
    #: outside are excluded and logged to the curation ledger.
    ranges: dict[PropertyName, tuple[float, float]] = Field(
        default_factory=lambda: {
            "logS": (-13.0, 2.0),
            "logP": (-5.0, 10.0),
            "mp_K": (100.0, 750.0),
        }
    )

    #: Where salt-stripping ends and mixture-rejection begins. When the
    #: second-largest fragment has at least this share of the largest one's heavy
    #: atoms, no fragment is obviously the subject of the measurement and the row
    #: is rejected instead of being silently reduced to its biggest piece.
    mixture_fragment_ratio: float = 0.5

    #: Keeping stereochemistry is correct - enantiomers can differ - but costs
    #: joins, because sources disagree about whether to record it. Discarding it
    #: would instead merge those entries into one key with conflicting values.
    keep_stereochemistry: bool = True

    #: Reject structures with no carbon. Off by default: AqSolDB legitimately
    #: contains inorganic salts and their solubilities are real measurements.
    require_carbon: bool = False

    #: Cap on molecules carried through the pipeline. ``None`` means all of them;
    #: the smoke profile sets a small number.
    max_molecules: int | None = None

    @model_validator(mode="after")
    def _check_ranges(self) -> DataConfig:
        for prop in PROPERTIES:
            if prop not in self.ranges:
                raise ValueError(f"data.ranges is missing a window for {prop!r}")
            low, high = self.ranges[prop]
            if not low < high:
                raise ValueError(f"data.ranges[{prop!r}] is not an increasing interval")
        return self


class FeatureConfig(_Base):
    """Both featurization paths. Path A feeds XGBoost, Path B feeds the neural heads."""

    # Path A - classical descriptors
    use_rdkit_descriptors: bool = True
    ecfp_radius: int = 2
    ecfp_bits: int = 1024

    # Path B - frozen pretrained encoders.
    #
    # Two are cached rather than one, because ChemBERTa-2's published tokenizer
    # has zero BPE merge rules and therefore cannot emit its own multi-character
    # tokens: `Cl` is read as `C`, `Br` as `B`, and stereochemistry is dropped.
    # The model was pretrained that way - the multi-character embeddings sit at
    # their initialization - so it cannot be repaired by fixing the tokenizer.
    #
    # 31% of our molecules carry a halogen, rising to 43% of the logP set, and
    # the descriptor track sees halogens perfectly well. Running only ChemBERTa-2
    # would confound the step-7 comparison: a neural-track loss could be negative
    # transfer or could just be missing chlorine, with no way to tell. Caching
    # both turns that confound into a measurement.
    encoders: dict[str, str] = Field(
        default_factory=lambda: {
            # The brief's specified encoder. Largest pretraining corpus (77M
            # PubChem SMILES), broken tokenizer.
            "chemberta2": "DeepChem/ChemBERTa-77M-MLM",
            # ChemBERTa v1. Smaller corpus (10M) but a working BPE tokenizer that
            # keeps halogens and stereochemistry.
            "pubchem10m": "seyonec/PubChem10M_SMILES_BPE_450k",
        }
    )
    #: Which encoder feeds Track 2 unless a run says otherwise.
    primary_encoder: str = "pubchem10m"

    chemberta_batch_size: int = 256
    chemberta_pooling: Literal["cls", "mean"] = "mean"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"

    @model_validator(mode="after")
    def _check_primary(self) -> FeatureConfig:
        if self.primary_encoder not in self.encoders:
            raise ValueError(
                f"features.primary_encoder {self.primary_encoder!r} is not one of "
                f"features.encoders {sorted(self.encoders)}"
            )
        return self

    #: Bumped by hand whenever the recipe above changes in a way that invalidates
    #: cached vectors. Travels with the registered model.
    version: str = "v1"


class SplitConfig(_Base):
    """Nested scaffold-disjoint cross-validation geometry."""

    n_outer: int = 5
    n_inner: int = 3

    #: Fold assignment is label-aware so no outer test fold ends up too thin to
    #: score. Folds falling below this get reported with wide intervals, not hidden.
    min_test_labels_per_property: int = 30

    #: Collapse every atom to carbon and every bond to single before grouping.
    #: Produces a harsher, more pessimistic split; the brief specifies plain
    #: Bemis-Murcko, so this stays off.
    generic_scaffolds: bool = False

    #: What to do with the 15% of molecules that have no ring system and therefore
    #: no scaffold at all. "singleton" gives each its own group, on the grounds
    #: that no shared core means nothing to leak; "shared" follows the DeepChem
    #: convention of one bucket, which in this dataset would be a single
    #: 4,471-member group carrying a badly skewed label mix.
    acyclic_policy: Literal["singleton", "shared"] = "singleton"


class TuningConfig(_Base):
    """Optuna, running inside the inner loop as the search strategy."""

    n_trials: int = 50
    pruning: bool = True
    timeout_s: float | None = None


class ModelConfig(_Base):
    xgb_n_jobs: int = 10  # M4 has 10 cores
    xgb_max_rounds: int = 2000  # ceiling; early stopping picks the real number
    xgb_early_stopping_rounds: int = 50

    #: Deep ensemble size for the neural track. Nearly free because the trunk is
    #: frozen and its embeddings are cached.
    ensemble_seeds: int = 5
    mtl_max_epochs: int = 200
    mtl_patience: int = 20


class UncertaintyConfig(_Base):
    nominal_coverage: float = 0.9
    quantiles: tuple[float, float, float] = (0.05, 0.5, 0.95)

    #: Nearest-neighbour Tanimoto distance above which a molecule is flagged
    #: out-of-domain. ``None`` means "not yet chosen"; it is read off the
    #: error-versus-distance curve in step 9, never guessed.
    ad_threshold: float | None = None

    #: Distance bands used for the conditional-coverage diagnostic.
    ad_bands: tuple[float, ...] = (0.2, 0.35, 0.5, 0.65, 1.0)

    @model_validator(mode="after")
    def _check_coverage(self) -> UncertaintyConfig:
        if not 0.0 < self.nominal_coverage < 1.0:
            raise ValueError("uncertainty.nominal_coverage must lie in (0, 1)")
        lo, mid, hi = self.quantiles
        if not lo < mid < hi:
            raise ValueError("uncertainty.quantiles must be strictly increasing")
        expected = round((1.0 - self.nominal_coverage) / 2.0, 6)
        if round(lo, 6) != expected or round(hi, 6) != round(1.0 - expected, 6):
            raise ValueError(
                f"uncertainty.quantiles {self.quantiles} do not bracket "
                f"nominal_coverage {self.nominal_coverage}; expected outer quantiles "
                f"({expected}, {round(1.0 - expected, 6)})"
            )
        return self


class TrackingConfig(_Base):
    """Experiment tracking backend.

    ``jsonl`` is a dependency-free local logger used until the MLflow backend lands
    with the first real nested run. Both satisfy the same ``RunLogger`` protocol, so
    swapping is a config change rather than a code change.
    """

    backend: Literal["jsonl", "mlflow"] = "jsonl"
    experiment: str = "dupont-qspr"
    uri: str | None = None  # mlflow tracking URI; None -> local ./mlruns


# --------------------------------------------------------------------------- #


class Config(_Base):
    profile: Profile = "dev"

    #: Single root seed. Everything stochastic derives from it so a run is
    #: reproducible to the same numbers.
    seed: int = 20260813

    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    splits: SplitConfig = Field(default_factory=SplitConfig)
    tuning: TuningConfig = Field(default_factory=TuningConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    # Paths are derived, not configured, so the layout stays consistent.
    @property
    def root(self) -> Path:
        return project_root()

    @property
    def raw_dir(self) -> Path:
        """Shared across profiles - the downloaded files are identical either way.

        Scoping this by profile would mean re-downloading the same verified bytes
        once per profile for no benefit.
        """
        return self.root / "data" / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.root / "data" / "interim" / self.profile

    @property
    def processed_dir(self) -> Path:
        """Profile-scoped: derived data depends on which profile produced it.

        This must not be shared. The smoke profile subsamples to 500 molecules, so
        a shared directory means running smoke after dev silently replaces the full
        union table with a toy one, and the next training run reads the toy without
        any indication that it did.
        """
        return self.root / "data" / "processed" / self.profile

    @property
    def artifacts_dir(self) -> Path:
        """Profile-scoped, so a smoke run can never overwrite full-run results."""
        return self.root / "artifacts" / self.profile

    @property
    def features_dir(self) -> Path:
        """Shared across profiles, like raw/ and unlike processed/.

        Feature caches are content-addressed by (canonical SMILES, version), so the
        same molecule yields the same vector whoever asks. Scoping them by profile
        would recompute identical vectors for no benefit; sharing them means the
        smoke profile reads whatever the full profile already computed.
        """
        return self.root / "data" / "features"

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.features_dir,
            self.artifacts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    loaded = yaml.safe_load(path.read_text()) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"{path} must contain a YAML mapping at the top level")
    return loaded


def load_config(
    profile: Profile = "dev",
    *,
    overrides: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> Config:
    """Load ``base.yaml``, overlay ``profiles/<profile>.yaml``, then ``overrides``.

    The merge is recursive, so a profile only states what it changes.
    """
    directory = directory or configs_dir()
    merged = _deep_merge(
        _read_yaml(directory / "base.yaml"),
        _read_yaml(directory / "profiles" / f"{profile}.yaml"),
    )
    if overrides:
        merged = _deep_merge(merged, overrides)
    merged["profile"] = profile
    return Config.model_validate(merged)
