"""Step 7: compare the tracks per property, then choose what ships.

Every track is scored on the *same* held-out rows. The folds are identical across
tracks by construction, so the comparison is paired: for each outer fold, the
difference in RMSE between two tracks is computed on the same molecules, and a
t-interval over the five differences decides the winner. A win requires that
interval to exclude zero; otherwise the property is a tie.

**Why one family ships, not a per-property hybrid.** XGBoost and the transformer
encoder link separate OpenMP runtimes and cannot share a process on this machine,
even when the encoder is loaded first. An in-process hybrid artifact is therefore
not buildable, so the artifact carries the family that wins the most properties,
and the per-property winners are reported regardless. Ties go to XGBoost: it is
the baseline the brief says everything must beat, and it needs no transformer at
inference time.

**A caveat worth stating in the report.** Choosing the family (and the encoder)
from outer-fold scores is a mild form of selection on held-out data. It is two
options per property rather than fifty hyperparameter configurations, so the
optimism is small, but it is not zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from dupont_qspr.config import Config
from dupont_qspr.contracts import PROPERTIES
from dupont_qspr.metrics.point import (
    enrichment_factor,
    fold_interval,
    rmse,
    spearman,
    tail_spearman,
)

__all__ = [
    "FamilyDecision",
    "comparison_table",
    "decide_family",
    "load_track_predictions",
    "paired_difference",
]

_KEYS = ["property", "row_index"]


def _truth_frame(cfg: Config) -> pl.DataFrame:
    """One row per (property, labelled molecule) with its measured value."""
    table = pl.read_parquet(cfg.processed_dir / "union_table.parquet")
    frames = []
    for prop in PROPERTIES:
        values = table.get_column(prop).to_numpy().astype(np.float64)
        rows = np.flatnonzero(~np.isnan(values)).astype(np.int64)
        frames.append(
            pl.DataFrame(
                {
                    "property": [prop] * rows.size,
                    "row_index": rows,
                    "y_true": values[rows],
                }
            )
        )
    return pl.concat(frames)


def load_track_predictions(cfg: Config) -> dict[str, pl.DataFrame]:
    """Out-of-fold predictions for every track that has been run, with truth joined.

    Keys are ``"xgb"`` and ``"mtl_<encoder>"``. The neural track's file does not
    carry the measured value, so truth is joined from the union table for every
    track alike rather than trusting two different sources.
    """
    found: dict[str, pl.DataFrame] = {}
    xgb = cfg.processed_dir / "oof_predictions_xgb.parquet"
    if xgb.exists():
        found["xgb"] = pl.read_parquet(xgb)
    for encoder in cfg.features.encoders:
        path = cfg.processed_dir / f"oof_predictions_mtl_{encoder}.parquet"
        if path.exists():
            found[f"mtl_{encoder}"] = pl.read_parquet(path)
    if not found:
        raise FileNotFoundError(
            f"no out-of-fold predictions under {cfg.processed_dir}. Run "
            "experiments/04_nested_xgb.py and/or 06_nested_mtl.py first."
        )
    truth = _truth_frame(cfg)
    return {
        name: frame.select("property", "fold", "row_index", "y_pred").join(
            truth, on=_KEYS, how="inner"
        )
        for name, frame in found.items()
    }


def _common_rows(tracks: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Restrict every track to the rows all of them predicted."""
    keys: pl.DataFrame | None = None
    for frame in tracks.values():
        these = frame.select(_KEYS)
        keys = these if keys is None else keys.join(these, on=_KEYS, how="inner")
    assert keys is not None
    return {n: f.join(keys, on=_KEYS, how="semi") for n, f in tracks.items()}


def comparison_table(tracks: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Accuracy, ranking and enrichment per track per property, on common rows."""
    rows: list[dict[str, Any]] = []
    for name, frame in _common_rows(tracks).items():
        for prop in PROPERTIES:
            sub = frame.filter(pl.col("property") == prop)
            if sub.height < 10:
                continue
            y = sub.get_column("y_true").to_numpy()
            p = sub.get_column("y_pred").to_numpy()
            per_fold = [
                rmse(
                    g.get_column("y_true").to_numpy(), g.get_column("y_pred").to_numpy()
                )
                for _, g in sub.group_by("fold")
            ]
            interval = fold_interval(per_fold)
            error = rmse(y, p)
            rows.append(
                {
                    "property": prop,
                    "track": name,
                    "n": sub.height,
                    "rmse": error,
                    "rmse_ci_low": interval.get("lower", float("nan")),
                    "rmse_ci_high": interval.get("upper", float("nan")),
                    "rmse_over_sd": error / (float(np.std(y)) or 1.0),
                    "spearman": spearman(y, p),
                    "tail_rho_upper": tail_spearman(y, p, tail="upper"),
                    "tail_rho_lower": tail_spearman(y, p, tail="lower"),
                    "ef10_upper": enrichment_factor(y, p, tail="upper"),
                    "ef10_lower": enrichment_factor(y, p, tail="lower"),
                }
            )
    return pl.DataFrame(rows)


def paired_difference(
    tracks: dict[str, pl.DataFrame], a: str, b: str, prop: str
) -> dict[str, float]:
    """t-interval over folds of ``RMSE(a) - RMSE(b)`` on the same molecules.

    Negative means ``a`` is more accurate. Pairing by fold removes the large
    fold-to-fold variation that both tracks share (the benzene fold is easier for
    everyone), which is what makes a real difference detectable with five folds.
    """
    left = tracks[a].filter(pl.col("property") == prop)
    right = (
        tracks[b]
        .filter(pl.col("property") == prop)
        .select("row_index", pl.col("y_pred").alias("y_pred_b"))
    )
    joined = left.join(right, on="row_index", how="inner")
    differences = []
    for _, group in joined.group_by("fold"):
        y = group.get_column("y_true").to_numpy()
        differences.append(
            rmse(y, group.get_column("y_pred").to_numpy())
            - rmse(y, group.get_column("y_pred_b").to_numpy())
        )
    return fold_interval(differences)


@dataclass(slots=True)
class FamilyDecision:
    family: str
    encoder: str | None
    mtl_candidate: str | None
    winners: dict[str, str] = field(default_factory=dict)
    differences: dict[str, dict[str, float]] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "encoder": self.encoder,
            "mtl_candidate": self.mtl_candidate,
            "winners": self.winners,
            "rmse_difference_xgb_minus_mtl": self.differences,
            "rationale": self.rationale,
        }


def decide_family(tracks: dict[str, pl.DataFrame]) -> FamilyDecision:
    """Choose the shipped family from paired per-property comparisons."""
    mtl_names = [n for n in tracks if n.startswith("mtl_")]
    if "xgb" not in tracks and not mtl_names:
        raise ValueError("no tracks to compare")
    if not mtl_names:
        return FamilyDecision(
            "xgb",
            None,
            None,
            dict.fromkeys(PROPERTIES, "xgb"),
            rationale="only the XGBoost track has results",
        )

    table = comparison_table(tracks)
    means = {
        name: float(
            table.filter(pl.col("track") == name).get_column("rmse_over_sd").mean()
        )
        for name in mtl_names
    }
    candidate = min(means, key=means.__getitem__)
    encoder = candidate.removeprefix("mtl_")

    if "xgb" not in tracks:
        return FamilyDecision(
            "mtl",
            encoder,
            candidate,
            dict.fromkeys(PROPERTIES, "mtl"),
            rationale="only neural-track results are present",
        )

    common = _common_rows(tracks)
    winners: dict[str, str] = {}
    differences: dict[str, dict[str, float]] = {}
    for prop in PROPERTIES:
        interval = paired_difference(common, "xgb", candidate, prop)
        if not interval:
            continue
        differences[prop] = interval
        if interval["upper"] < 0:
            winners[prop] = "xgb"
        elif interval["lower"] > 0:
            winners[prop] = "mtl"
        else:
            winners[prop] = "tie"

    xgb_wins = sum(w == "xgb" for w in winners.values())
    mtl_wins = sum(w == "mtl" for w in winners.values())
    family = "mtl" if mtl_wins > xgb_wins else "xgb"
    rationale = (
        f"XGBoost wins {xgb_wins}, {candidate} wins {mtl_wins}, "
        f"{len(winners) - xgb_wins - mtl_wins} tied (paired fold interval excludes zero "
        "to count as a win). Ties go to XGBoost, the baseline, which also needs no "
        "transformer at inference."
    )
    return FamilyDecision(
        family,
        encoder if family == "mtl" else None,
        candidate,
        winners,
        differences,
        rationale,
    )
