"""Multi-property small-molecule QSPR oracle.

Predicts aqueous solubility, lipophilicity, and melting point from SMILES, each
with a calibrated uncertainty interval and an applicability-domain flag.

See ``qspr_project_brief.md`` for the design and ``dupont_qspr.contracts`` for the
interfaces the pipeline stages meet at.
"""

from __future__ import annotations

import argparse

from dupont_qspr.config import Config, load_config

__all__ = ["Config", "load_config", "main"]


def main() -> None:
    """Entry point for ``uv run dupont-qspr``.

    Until the real pipeline stages land this runs the skeleton spine, which walks
    every stage end to end on synthetic data.
    """
    parser = argparse.ArgumentParser(prog="dupont-qspr", description=__doc__)
    parser.add_argument(
        "--profile",
        default="smoke",
        choices=("smoke", "dev", "full"),
        help="runtime profile: data size, fold counts, tuning budget",
    )
    args = parser.parse_args()

    from dupont_qspr.spine import run_spine, summary_table

    cfg = load_config(args.profile)
    summary = run_spine(cfg)

    print(f"profile={cfg.profile}  elapsed={summary['elapsed_s']:.2f}s")
    print(f"labels: {summary['label_availability']['per_property']}")
    print(summary_table(summary))
