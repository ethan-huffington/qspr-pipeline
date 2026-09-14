"""Walk every pipeline stage end to end and print the result.

    uv run python experiments/00_smoke_spine.py --profile smoke

This exists to be run constantly. It is the cheapest way to notice that a change
broke the shape of the pipeline rather than just the quality of a number.
"""

from __future__ import annotations

import argparse
import json

from dupont_qspr.config import load_config
from dupont_qspr.spine import run_spine, summary_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke", choices=("smoke", "dev", "full"))
    parser.add_argument("--json", action="store_true", help="dump the full summary")
    args = parser.parse_args()

    cfg = load_config(args.profile)
    summary = run_spine(cfg)

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
        return

    print(f"profile   : {cfg.profile}")
    print(f"elapsed   : {summary['elapsed_s']:.2f}s")
    print(f"molecules : {summary['label_availability']['n_molecules']}")
    print(f"labels    : {summary['label_availability']['per_property']}")
    print(f"combos    : {summary['label_availability']['per_combination']}")
    print(
        f"complete  : {summary['label_availability']['n_complete_rows']} rows with all three"
    )
    print()
    print("nested estimate (5 outer folds, scaffold-disjoint):")
    print(summary_table(summary))
    print()
    print(f"selected  : {summary['final_params']}")
    print(f"records   : {len(summary['scored_records'])} returned for 5 inputs")
    for record in summary["scored_records"]:
        if record.get("predictions") is None:
            print(f"  ERROR  {record['smiles_input']!r}: {record['error']}")
        else:
            ad = record["applicability_domain"]
            print(
                f"  {record['smiles_canonical']:<14} "
                f"logS={record['predictions']['logS']['value']:+.2f}  "
                f"d_nn={ad['nn_tanimoto_distance']:.3f}  "
                f"in_domain={ad['in_domain']}"
            )


if __name__ == "__main__":
    main()
