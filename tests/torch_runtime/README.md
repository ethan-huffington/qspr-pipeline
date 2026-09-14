# Tests that load PyTorch

These run in a **separate pytest process** from the rest of the suite.

XGBoost and PyTorch each link their own OpenMP runtime, and a process holding both
hangs or segfaults on macOS/arm64. Pytest imports every collected module during
collection, so marker-based deselection does not help — merely importing xgboost
anywhere in the run is enough to poison it. Isolation has to be by collection
path, which is what this directory provides.

    uv run pytest --ignore=tests/torch_runtime   # everything else, incl. xgboost
    uv run pytest tests/torch_runtime            # this directory

Put any test that imports torch or transformers here. The two model tracks are
separate processes in production anyway, so this costs nothing.
