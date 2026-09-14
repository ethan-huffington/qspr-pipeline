"""A run has to record enough to explain why a rerun did or did not reproduce."""

from __future__ import annotations

import json
from pathlib import Path

from dupont_qspr.config import Config
from dupont_qspr.tracking import (
    JsonlRunLogger,
    MlflowRunLogger,
    RunLogger,
    capture_environment,
    start_run,
)


def test_jsonl_logger_satisfies_the_protocol(tmp_path: Path) -> None:
    """The seam matters more than the backend: step 4 swaps in MLflow behind it."""
    assert isinstance(JsonlRunLogger(run_dir=tmp_path / "r", run_id="r"), RunLogger)


def test_params_are_flattened_and_merged(tmp_path: Path) -> None:
    logger = JsonlRunLogger(run_dir=tmp_path / "run", run_id="run")
    logger.log_params({"splits": {"n_outer": 5, "n_inner": 3}})
    logger.log_params({"tuning": {"n_trials": 50}})

    written = json.loads((tmp_path / "run" / "params.json").read_text())
    assert written == {"splits.n_outer": 5, "splits.n_inner": 3, "tuning.n_trials": 50}


def test_metrics_append_one_row_per_call(tmp_path: Path) -> None:
    logger = JsonlRunLogger(run_dir=tmp_path / "run", run_id="run")
    logger.log_metrics({"rmse": 1.0}, step=0)
    logger.log_metrics({"rmse": 0.9}, step=1)

    rows = [
        json.loads(line)
        for line in (tmp_path / "run" / "metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in rows] == [0, 1]
    assert [r["rmse"] for r in rows] == [1.0, 0.9]


def test_log_dict_writes_a_named_artifact(tmp_path: Path) -> None:
    logger = JsonlRunLogger(run_dir=tmp_path / "run", run_id="run")
    logger.log_dict({"logS": 9982, "logP": 4200}, "label_availability")

    assert json.loads((tmp_path / "run" / "label_availability.json").read_text()) == {
        "logS": 9982,
        "logP": 4200,
    }


def test_environment_capture_records_what_moves_numbers() -> None:
    environment = capture_environment()

    assert environment["python"].startswith("3.13")
    assert "numpy" in environment["packages"]
    # Packages not yet installed are recorded as absent rather than omitted, so a
    # run logged before and after a dependency lands stays comparable.
    assert "torch" in environment["packages"]


def test_start_run_seeds_the_record_with_config_and_environment(
    smoke_cfg: Config,
) -> None:
    with start_run(smoke_cfg, "unit") as logger:
        run_dir = logger.run_dir  # type: ignore[attr-defined]

    params = json.loads((run_dir / "params.json").read_text())
    assert params["run.profile"] == "smoke"
    assert params["run.seed"] == smoke_cfg.seed
    assert params["splits.n_outer"] == smoke_cfg.splits.n_outer
    assert (run_dir / "environment.json").exists()


def test_mlflow_backend_is_selectable(smoke_cfg: Config) -> None:
    """Both backends satisfy RunLogger, so switching is a config edit.

    This replaced an earlier test asserting the MLflow backend raised
    NotImplementedError; it landed with the first nested run at build step 4.
    """
    switched = smoke_cfg.model_copy(
        update={"tracking": smoke_cfg.tracking.model_copy(update={"backend": "mlflow"})}
    )
    assert switched.tracking.backend == "mlflow"
    assert isinstance(MlflowRunLogger(run_id="r", experiment="e"), RunLogger)
