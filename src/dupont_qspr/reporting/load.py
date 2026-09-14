"""Find and load what a nested run persisted, without re-running anything."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl

from dupont_qspr.config import Config

__all__ = ["latest_nested_result", "load_oof_predictions"]


def latest_nested_result(
    cfg: Config, name: str = "nested_xgb.json"
) -> tuple[dict[str, Any], Path]:
    """The most recent run directory containing a nested result.

    Runs are timestamped directories, so "most recent" is a real ordering rather
    than a guess, and the path is returned alongside the payload so a report can
    say which run it describes.
    """
    candidates = sorted(
        (cfg.artifacts_dir / "runs").glob(f"*/{name}"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no {name} under {cfg.artifacts_dir / 'runs'}. Run "
            f"experiments/04_nested_xgb.py --profile {cfg.profile} first."
        )
    path = candidates[-1]
    return json.loads(path.read_text()), path


def load_oof_predictions(cfg: Config, track: str = "xgb") -> pl.DataFrame:
    """Out-of-fold predictions written by the nested run."""
    path = cfg.processed_dir / f"oof_predictions_{track}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run experiments/04_nested_xgb.py --profile "
            f"{cfg.profile} first; the report does not refit anything."
        )
    return pl.read_parquet(path)
