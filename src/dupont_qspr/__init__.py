"""Multi-property small-molecule QSPR oracle.

Predicts aqueous solubility, lipophilicity and melting point from SMILES, each with
a calibrated prediction interval and an applicability-domain flag.

See ``qspr_project_brief.md`` for the design, ``dupont_qspr.contracts`` for the
interfaces the pipeline stages meet at, and ``experiments/run_pipeline.sh`` for the
order the stages run in.
"""

from __future__ import annotations

import argparse
import json

from dupont_qspr.config import Config, load_config

__all__ = ["Config", "load_config", "main"]


def main() -> None:
    """``uv run dupont-qspr CCO c1ccccc1O`` scores SMILES with the built bundle."""
    parser = argparse.ArgumentParser(
        prog="dupont-qspr", description="Score SMILES with the final model bundle."
    )
    parser.add_argument("smiles", nargs="*", help="SMILES strings to score")
    parser.add_argument("--profile", default="full", choices=("smoke", "dev", "full"))
    args = parser.parse_args()

    if not args.smiles:
        parser.print_help()
        print("\nBuild everything first with: experiments/run_pipeline.sh <profile>")
        return

    bundle = load_config(args.profile).artifacts_dir / "final" / "bundle"
    if not bundle.exists():
        raise SystemExit(
            f"no bundle at {bundle}; run experiments/11_final_fit.py --profile {args.profile}"
        )

    from dupont_qspr.serving.scorer import QSPRScorer

    for record in QSPRScorer.load(bundle).score(args.smiles):
        print(json.dumps(record, indent=2))
