"""Shared fixtures.

Tests run against the real ``configs/`` files - a test suite that validated a
hand-built config object would not notice the day a YAML key was renamed - but
write their artifacts into a temporary root, so running the suite never disturbs
``artifacts/`` or ``data/``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from dupont_qspr.config import Config, configs_dir, load_config

#: Resolved once, at import, before any test redirects the project root.
_REAL_CONFIGS_DIR = configs_dir()


@pytest.fixture(scope="session")
def configs_path() -> Path:
    """The repository's real ``configs/`` directory, unaffected by root redirection."""
    return _REAL_CONFIGS_DIR


@pytest.fixture
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point derived output paths at a throwaway directory."""
    monkeypatch.setenv("DUPONT_QSPR_ROOT", str(tmp_path))
    yield tmp_path


@pytest.fixture
def smoke_cfg(isolated_root: Path, configs_path: Path) -> Config:
    """The real smoke profile, writing into a temporary root."""
    return load_config("smoke", directory=configs_path)
