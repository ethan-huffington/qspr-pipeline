"""Step 11: final fit over all data, then register the MLflow pyfunc.

    uv run python experiments/11_final_fit.py --profile full

Reads artifacts/<profile>/decision.json from experiments/07_analyze.py for the
family and the applicability-domain threshold; either can be overridden. Loads the
registered model back and scores a demo batch, including deliberately bad inputs,
as an end-to-end check of the deliverable.
"""

from __future__ import annotations

import argparse
import json
import shutil

from dupont_qspr.config import load_config
from dupont_qspr.contracts import PROPERTIES

DEMO = [
    "CCO",
    "Oc1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "FC(F)(F)C(F)(F)C(F)(F)S(=O)(=O)O",
    "not a molecule",
    "",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="full", choices=("smoke", "dev", "full"))
    parser.add_argument("--family", default="auto", choices=("auto", "xgb", "mtl"))
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--ad-threshold", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.profile)

    decision_path = cfg.artifacts_dir / "decision.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
    family = decision.get("family", "xgb") if args.family == "auto" else args.family
    encoder = args.encoder or decision.get("encoder")
    threshold = (
        args.ad_threshold
        if args.ad_threshold is not None
        else decision.get("ad_threshold")
    )
    if threshold is None:
        print(
            "  WARNING: no AD threshold — every molecule will be flagged out of domain. Run 07_analyze.py."
        )

    bundle = cfg.artifacts_dir / "final" / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)

    from dupont_qspr.serving.final_fit import build_bundle

    print(f"family={family} encoder={encoder} ad_threshold={threshold}")
    manifest = build_bundle(
        cfg,
        family=family,
        encoder=encoder,
        ad_threshold=threshold,
        directory=bundle,
        n_trials=args.trials,
    )

    import mlflow.pyfunc

    from dupont_qspr.serving.pyfunc import log_pyfunc

    info = log_pyfunc(cfg, bundle, manifest)
    loaded = mlflow.pyfunc.load_model(info.model_uri)
    records = loaded.predict(DEMO)

    print(f"\nREGISTERED  {info.model_uri}   version={manifest['model_version']}")
    print("\nDEMO BATCH (scored through the reloaded MLflow model)")
    for record in records:
        if record.get("predictions") is None:
            print(f"  ERROR  {record['smiles_input']!r}: {record['error']}")
            continue
        ad = record["applicability_domain"]
        cells = "  ".join(
            f"{p}={record['predictions'][p]['value']:.2f} "
            f"[{record['predictions'][p]['lower']:.2f}, {record['predictions'][p]['upper']:.2f}]"
            for p in PROPERTIES
        )
        print(
            f"  {record['smiles_canonical'][:28]:<28} {cells}  "
            f"d={ad['nn_tanimoto_distance']:.2f} in_domain={ad['in_domain']}"
        )


if __name__ == "__main__":
    main()
