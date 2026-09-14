"""Step 10: the low-data ablation.

    uv run python experiments/10_ablation.py run --track xgb --profile full
    uv run python experiments/10_ablation.py run --track mtl --profile full
    uv run python experiments/10_ablation.py plot --profile full

The two tracks run as separate invocations because XGBoost and PyTorch cannot
share a process; ``plot`` imports neither. The neural track uses the primary
encoder only, which is enough to answer the question the experiment asks.
"""

from __future__ import annotations

import argparse

import polars as pl

from dupont_qspr.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "plot"))
    parser.add_argument("--track", choices=("xgb", "mtl"))
    parser.add_argument("--profile", default="full", choices=("smoke", "dev", "full"))
    args = parser.parse_args()
    cfg = load_config(args.profile)
    cfg.ensure_dirs()

    if args.action == "run":
        from dupont_qspr.analysis.ablation import run_mtl_ablation, run_xgb_ablation

        if args.track == "xgb":
            frame = run_xgb_ablation(cfg)
        elif args.track == "mtl":
            frame = run_mtl_ablation(cfg, cfg.features.primary_encoder)
        else:
            parser.error("run needs --track xgb or --track mtl")
        path = cfg.processed_dir / f"ablation_{args.track}.parquet"
        frame.write_parquet(path)
        print(f"  {frame.height:,} runs -> {path}")
        return

    from dupont_qspr.analysis.ablation import summarise_ablation
    from dupont_qspr.analysis.figures import render_ablation

    frames = [
        pl.read_parquet(p) for p in sorted(cfg.processed_dir.glob("ablation_*.parquet"))
    ]
    if not frames:
        raise SystemExit("no ablation results; run the xgb and mtl tracks first")
    summary = summarise_ablation(pl.concat(frames))
    with pl.Config(tbl_rows=60, float_precision=3, tbl_hide_dataframe_shape=True):
        print(summary)
    tracks = summary.get_column("track").unique().to_list()
    if len(tracks) == 2:
        wide = summary.pivot(
            on="track", index=["property", "size"], values="mean_rmse_over_sd"
        )
        xgb = next(t for t in tracks if t == "xgb")
        mtl = next(t for t in tracks if t != "xgb")
        print("\n  multi-task advantage (positive = multi-task more accurate):")
        with pl.Config(tbl_rows=60, float_precision=3, tbl_hide_dataframe_shape=True):
            print(
                wide.with_columns((pl.col(xgb) - pl.col(mtl)).alias("advantage")).sort(
                    ["property", "size"]
                )
            )
    for path in render_ablation(summary, cfg.artifacts_dir / "figures"):
        print(f"  {path}")


if __name__ == "__main__":
    main()
