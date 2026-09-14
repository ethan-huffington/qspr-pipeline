"""Configuration must fail loudly at load, not quietly three hours into a run."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from dupont_qspr.config import Config, load_config


@pytest.mark.parametrize("profile", ["smoke", "dev", "full"])
def test_every_shipped_profile_loads(profile: str, configs_path: Path) -> None:
    cfg = load_config(profile, directory=configs_path)
    assert cfg.profile == profile


def test_profile_overlays_base_recursively(configs_path: Path) -> None:
    """A profile states only what it changes; everything else comes from base."""
    base = load_config("dev", directory=configs_path)
    smoke = load_config("smoke", directory=configs_path)

    assert smoke.data.max_molecules == 500
    assert smoke.splits.n_outer == 2
    # Untouched by the smoke overlay, so inherited from base.yaml.
    assert smoke.data.bradley_subset == base.data.bradley_subset
    assert smoke.features.ecfp_bits == base.features.ecfp_bits


def test_unknown_key_is_rejected(configs_path: Path) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        load_config(
            "smoke",
            directory=configs_path,
            overrides={"splits": {"n_outter": 5}},
        )


def test_quantiles_must_bracket_the_nominal_coverage(configs_path: Path) -> None:
    """Reporting 90% intervals built from 0.1/0.9 quantiles would be a quiet lie."""
    with pytest.raises(ValidationError, match="do not bracket"):
        load_config(
            "smoke",
            directory=configs_path,
            overrides={"uncertainty": {"nominal_coverage": 0.8}},
        )


def test_property_range_must_be_an_increasing_interval(configs_path: Path) -> None:
    with pytest.raises(ValidationError, match="increasing interval"):
        load_config(
            "smoke",
            directory=configs_path,
            overrides={"data": {"ranges": {"logS": [2.0, -13.0]}}},
        )


def test_config_is_immutable(configs_path: Path) -> None:
    cfg = load_config("smoke", directory=configs_path)
    with pytest.raises(ValidationError):
        cfg.splits.n_outer = 7  # type: ignore[misc]


def test_artifacts_are_scoped_by_profile(
    isolated_root: Path, configs_path: Path
) -> None:
    """A smoke run must never be able to overwrite full-run results."""
    smoke = load_config("smoke", directory=configs_path)
    full = load_config("full", directory=configs_path)

    assert smoke.artifacts_dir != full.artifacts_dir
    assert smoke.artifacts_dir.parent == isolated_root / "artifacts"


def test_derived_data_is_scoped_by_profile(configs_path: Path) -> None:
    """The smoke profile subsamples to 500 molecules.

    Sharing data/processed/ across profiles means running smoke after full
    replaces the real union table with a toy one, and the next training run reads
    the toy with nothing to indicate it happened.
    """
    smoke = load_config("smoke", directory=configs_path)
    full = load_config("full", directory=configs_path)

    assert smoke.processed_dir != full.processed_dir
    assert smoke.interim_dir != full.interim_dir


def test_raw_downloads_are_shared_across_profiles(configs_path: Path) -> None:
    """Raw files are profile-independent; re-downloading them per profile is waste."""
    smoke = load_config("smoke", directory=configs_path)
    full = load_config("full", directory=configs_path)

    assert smoke.raw_dir == full.raw_dir


def test_ensure_dirs_creates_the_layout(
    isolated_root: Path, configs_path: Path
) -> None:
    cfg: Config = load_config("smoke", directory=configs_path)
    cfg.ensure_dirs()

    for path in (cfg.raw_dir, cfg.interim_dir, cfg.processed_dir, cfg.artifacts_dir):
        assert path.is_dir()
