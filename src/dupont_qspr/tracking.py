"""Experiment tracking behind a single narrow protocol.

Brief §12 wants MLflow throughout, but MLflow is a heavy dependency that earns its
place only once there are real nested runs and a model registry to fill. So the
*seam* is frozen here and the backend is swappable: ``JsonlRunLogger`` is the
dependency-free default, and an MLflow-backed implementation of the same protocol
lands alongside the first nested run. Call sites never learn which one they have.

Whatever the backend, a run records enough to be re-derived: the resolved config,
the seed, the git revision, and the versions of the packages that actually shape
numeric output.
"""

from __future__ import annotations

import json
import math
import platform
import subprocess
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from dupont_qspr.config import Config, project_root

__all__ = [
    "JsonlRunLogger",
    "MlflowRunLogger",
    "RunLogger",
    "capture_environment",
    "start_run",
]

#: Packages whose version can change a number. Logged with every run.
_TRACKED_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "polars",
    "pandas",
    "rdkit",
    "xgboost",
    "optuna",
    "torch",
    "transformers",
    "mlflow",
)


@runtime_checkable
class RunLogger(Protocol):
    """Minimal surface both backends provide."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record inputs that define the run. Flattened with dotted keys."""
        ...

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        """Record numeric outputs. ``step`` distinguishes repeated observations."""
        ...

    def log_dict(self, obj: Any, name: str) -> None:
        """Record a structured artifact, e.g. the label-availability matrix."""
        ...

    def log_artifact(self, path: Path) -> None:
        """Record a file already written to disk, e.g. a figure."""
        ...


def _git_revision() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _git_is_dirty() -> bool | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(project_root()), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.stdout.strip()) if out.returncode == 0 else None


def capture_environment() -> dict[str, Any]:
    """Everything needed to explain why a rerun might not reproduce."""
    packages: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            packages[name] = pkg_version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_revision": _git_revision(),
        "git_dirty": _git_is_dirty(),
        "packages": packages,
    }


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(obj, Mapping):
        out: dict[str, Any] = {}
        for key, value in obj.items():
            out |= _flatten(value, f"{prefix}{key}.")
        return out
    return {prefix.rstrip("."): obj}


@dataclass(slots=True)
class JsonlRunLogger:
    """Local, append-only run directory. No service, no schema migration.

    Layout under ``artifacts/<profile>/runs/<run_id>/``::

        params.json      resolved inputs, flattened
        metrics.jsonl    one JSON object per log_metrics call
        environment.json versions and git revision
        <name>.json      whatever log_dict was given
    """

    run_dir: Path
    run_id: str
    _metrics_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.run_dir / "metrics.jsonl"

    def log_params(self, params: Mapping[str, Any]) -> None:
        path = self.run_dir / "params.json"
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing |= _flatten(params)
        path.write_text(json.dumps(existing, indent=2, default=str))

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        row = {
            "step": step,
            "time": time.time(),
            **{k: float(v) for k, v in metrics.items()},
        }
        with self._metrics_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")

    def log_dict(self, obj: Any, name: str) -> None:
        target = self.run_dir / (name if name.endswith(".json") else f"{name}.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(obj, indent=2, default=str))

    def log_artifact(self, path: Path) -> None:
        destination = self.run_dir / "artifacts" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(Path(path).read_bytes())


@dataclass(slots=True)
class MlflowRunLogger:
    """MLflow-backed logger, satisfying the same protocol as the JSONL one.

    Call sites are unchanged, which is the point of having frozen the seam at step
    0: swapping tracking backends is a config edit, not a refactor.

    Two MLflow behaviours are smoothed over here. It rejects re-logging a parameter
    with a different value, which our merge-friendly ``log_params`` would otherwise
    trip on; and it silently truncates long parameter values, so anything oversized
    is diverted to an artifact where it stays readable.
    """

    run_id: str
    experiment: str
    _logged_params: set[str] = field(default_factory=set)

    def log_params(self, params: Mapping[str, Any]) -> None:
        import mlflow

        flat = _flatten(params)
        fresh = {
            k: str(v)[:_MLFLOW_PARAM_LIMIT]
            for k, v in flat.items()
            if k not in self._logged_params
        }
        oversized = {k: v for k, v in flat.items() if len(str(v)) > _MLFLOW_PARAM_LIMIT}
        if fresh:
            mlflow.log_params(fresh)
            self._logged_params.update(fresh)
        if oversized:
            # Truncated values are worse than useless - they look complete.
            mlflow.log_dict(
                {k: str(v) for k, v in oversized.items()}, "oversized_params.json"
            )

    def log_metrics(
        self, metrics: Mapping[str, float], *, step: int | None = None
    ) -> None:
        import mlflow

        finite = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and math.isfinite(float(v))
        }
        if finite:
            mlflow.log_metrics(finite, step=step)

    def log_dict(self, obj: Any, name: str) -> None:
        import mlflow

        mlflow.log_dict(
            json.loads(json.dumps(obj, default=str)),
            name if name.endswith(".json") else f"{name}.json",
        )

    def log_artifact(self, path: Path) -> None:
        import mlflow

        mlflow.log_artifact(str(path))


_MLFLOW_PARAM_LIMIT = 500


@contextmanager
def start_run(cfg: Config, name: str) -> Iterator[RunLogger]:
    """Open a run, pre-loaded with the resolved config, seed, and environment."""
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{name}"

    if cfg.tracking.backend == "mlflow":
        import mlflow

        mlflow.set_tracking_uri(cfg.tracking.uri or f"file://{cfg.root / 'mlruns'}")
        mlflow.set_experiment(cfg.tracking.experiment)
        with mlflow.start_run(run_name=run_id):
            logger: RunLogger = MlflowRunLogger(
                run_id=run_id, experiment=cfg.tracking.experiment
            )
            _seed_run(logger, cfg, name)
            yield logger
        return

    logger = JsonlRunLogger(run_dir=cfg.artifacts_dir / "runs" / run_id, run_id=run_id)
    _seed_run(logger, cfg, name)
    yield logger


def _seed_run(logger: RunLogger, cfg: Config, name: str) -> None:
    logger.log_params({"run": {"name": name, "profile": cfg.profile, "seed": cfg.seed}})
    logger.log_params(cfg.model_dump(mode="json"))
    logger.log_dict(capture_environment(), "environment")
