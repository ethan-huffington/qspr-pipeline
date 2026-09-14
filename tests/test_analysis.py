"""Steps 7, 9 and 11 on synthetic inputs whose right answers are known by construction."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dupont_qspr.analysis import (
    choose_ad_threshold,
    conditional_coverage,
    decide_family,
    error_vs_distance,
)
from dupont_qspr.contracts import PROPERTIES
from dupont_qspr.metrics.point import enrichment_factor, tail_spearman
from dupont_qspr.serving.pyfunc import as_smiles_list

# --------------------------------------------------------------------------- #
# Enrichment factor and two-tailed ranking


def test_enrichment_factor_perfect_ranking_hits_ceiling() -> None:
    y = np.arange(100, dtype=np.float64)
    assert enrichment_factor(y, y, fraction=0.1) == pytest.approx(10.0)
    assert enrichment_factor(y, y, fraction=0.1, tail="lower") == pytest.approx(10.0)


def test_enrichment_factor_reversed_ranking_finds_nothing() -> None:
    y = np.arange(100, dtype=np.float64)
    assert enrichment_factor(y, -y, fraction=0.1) == 0.0


def test_enrichment_factor_random_ranking_is_near_one() -> None:
    rng = np.random.default_rng(0)
    y = rng.normal(size=20_000)
    assert enrichment_factor(y, rng.normal(size=y.size)) == pytest.approx(1.0, abs=0.15)


def test_tail_spearman_rejects_unknown_tail() -> None:
    y = np.arange(50, dtype=np.float64)
    with pytest.raises(ValueError, match="tail"):
        tail_spearman(y, y, tail="middle")


def test_lower_tail_only_sees_the_lower_tail() -> None:
    # Perfect in the bottom decile, scrambled everywhere else.
    rng = np.random.default_rng(1)
    y = np.arange(200, dtype=np.float64)
    p = y.copy()
    p[20:] = rng.permutation(p[20:])
    assert tail_spearman(y, p, tail="lower") == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Step 7: family decision


def _track(rng: np.random.Generator, noise: dict[str, float]) -> pl.DataFrame:
    frames = []
    for prop in PROPERTIES:
        n = 400
        y = rng.normal(size=n)
        frames.append(
            pl.DataFrame(
                {
                    "property": [prop] * n,
                    "fold": np.repeat(np.arange(5), n // 5),
                    "row_index": np.arange(n),
                    "y_true": y,
                    "y_pred": y + rng.normal(scale=noise[prop], size=n),
                }
            )
        )
    return pl.concat(frames)


def _paired_tracks(
    xgb_noise: dict[str, float], mtl_noise: dict[str, float]
) -> dict[str, pl.DataFrame]:
    """Two tracks scored on the same rows with the same truth, as the real runs are."""
    base = _track(np.random.default_rng(2), dict.fromkeys(PROPERTIES, 0.0))
    rng = np.random.default_rng(3)
    out = {}
    for name, noise in (("xgb", xgb_noise), ("mtl_pubchem10m", mtl_noise)):
        scale = base.get_column("property").replace_strict(
            noise, return_dtype=pl.Float64
        )
        out[name] = base.with_columns(
            (pl.col("y_true") + pl.Series(rng.normal(size=base.height)) * scale).alias(
                "y_pred"
            )
        )
    return out


def test_family_follows_clear_per_property_wins() -> None:
    tracks = _paired_tracks(
        {"logS": 1.0, "logP": 1.0, "mp_K": 0.2},
        {"logS": 0.2, "logP": 0.2, "mp_K": 1.0},
    )
    decision = decide_family(tracks)
    assert decision.winners == {"logS": "mtl", "logP": "mtl", "mp_K": "xgb"}
    assert decision.family == "mtl"
    assert decision.encoder == "pubchem10m"


def test_ties_go_to_the_baseline() -> None:
    # Identical predictions, so every paired difference is exactly zero. Equal
    # *noise levels* would not do: independent draws at the same scale still
    # produce a spurious "win" about 5% of the time per property, which is the
    # interval behaving correctly, not a tie.
    tracks = _paired_tracks(
        dict.fromkeys(PROPERTIES, 0.5), dict.fromkeys(PROPERTIES, 0.5)
    )
    tracks["mtl_pubchem10m"] = tracks["xgb"]
    decision = decide_family(tracks)
    assert set(decision.winners.values()) == {"tie"}
    assert decision.family == "xgb"
    assert decision.encoder is None


# --------------------------------------------------------------------------- #
# Step 9: conditional coverage and the AD threshold


def test_conditional_coverage_bins_by_distance() -> None:
    frame = pl.DataFrame(
        {
            "property": ["logS"] * 4,
            "method": ["cqr"] * 4,
            "nn_distance": [0.1, 0.15, 0.9, 0.95],
            "covered": [True, True, False, True],
            "width": [1.0, 1.0, 2.0, 2.0],
        }
    )
    table = conditional_coverage(frame, (0.5, 1.0))
    assert table.get_column("picp").to_list() == [1.0, 0.5]
    assert table.get_column("n").to_list() == [2, 2]


def test_ad_threshold_sits_where_error_climbs() -> None:
    rng = np.random.default_rng(4)
    n = 2_000
    distance = rng.uniform(0, 1, size=n)
    y = rng.normal(size=n)
    # Error is flat to 0.7, then triples.
    noise = np.where(distance < 0.7, 0.2, 0.6)
    predictions = pl.DataFrame(
        {
            "property": ["logS"] * n,
            "fold": [0] * n,
            "row_index": np.arange(n),
            "y_true": y,
            "y_pred": y + rng.normal(size=n) * noise,
        }
    )
    distances = predictions.select("property", "fold", "row_index").with_columns(
        pl.Series("nn_distance", distance)
    )
    threshold, per_property = choose_ad_threshold(
        error_vs_distance(predictions, distances), tolerance=1.5
    )
    assert threshold == pytest.approx(0.7, abs=0.1)
    assert set(per_property) == {"logS"}


def test_flat_error_puts_everything_in_domain() -> None:
    curve = pl.DataFrame(
        {
            "property": ["logP"] * 10,
            "bin": list(range(10)),
            "d_high": np.linspace(0.1, 1.0, 10),
            "norm_abs_error": [0.5] * 10,
            "n": [50] * 10,
        }
    )
    threshold, _ = choose_ad_threshold(curve)
    assert threshold == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Step 11: the pyfunc input adapter


@pytest.mark.parametrize(
    "value",
    [
        "CCO",
        ["CCO"],
        np.array(["CCO"]),
        {"smiles": ["CCO"]},
        pl.DataFrame({"smiles": ["CCO"]}).to_pandas(),
    ],
)
def test_pyfunc_accepts_common_input_shapes(value: object) -> None:
    assert as_smiles_list(value) == ["CCO"]
