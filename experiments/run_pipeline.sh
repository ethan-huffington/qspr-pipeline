#!/usr/bin/env bash
# The whole pipeline, in dependency order, for one profile.
#
#   experiments/run_pipeline.sh smoke   # seconds; proves every stage end to end
#   experiments/run_pipeline.sh full    # ~8–10 hours on the M4 — the reported numbers
#
# Each stage is its own process. That is required, not stylistic: XGBoost and
# PyTorch link separate OpenMP runtimes and cannot share one.
set -euo pipefail
cd "$(dirname "$0")/.."
PROFILE="${1:-smoke}"
run() { echo; echo "=== $* ==="; uv run python "$@" --profile "$PROFILE"; }

run experiments/01_build_dataset.py
run experiments/02_featurize.py
run experiments/03_build_folds.py
run experiments/04_nested_xgb.py            # track 1 nested search        ~3 h at full
run experiments/05_report.py
run experiments/06_nested_mtl.py            # track 2, every encoder       ~2 h at full
run experiments/08_uncertainty.py --track xgb
for ENCODER in $(uv run python -c "from dupont_qspr.config import load_config; print(' '.join(load_config('$PROFILE').features.encoders))"); do
  run experiments/08_uncertainty.py --track mtl --encoder "$ENCODER"
done
run experiments/07_analyze.py               # step 7 + 9: choose family, set AD threshold
run experiments/10_ablation.py run --track xgb
run experiments/10_ablation.py run --track mtl
run experiments/10_ablation.py plot
run experiments/11_final_fit.py             # final study, calibrate, register pyfunc
echo; echo "done: $PROFILE"
