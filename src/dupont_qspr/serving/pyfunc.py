"""MLflow wrapper around the bundle - the registered, versioned deliverable.

The wrapper is thin on purpose. All scoring logic lives in :class:`QSPRScorer`,
which is testable without MLflow; this file only adapts MLflow's input conventions
to a list of SMILES and records the bundle as a model artifact, together with the
package source (``code_paths``) so the artifact is self-contained.

Tracking uses a SQLite store under ``artifacts/<profile>/`` because the model
registry needs a database-backed store, and keeping it there leaves the project
root clean.
"""

from __future__ import annotations

from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc

from dupont_qspr.config import Config

__all__ = ["QSPRPyfunc", "as_smiles_list", "log_pyfunc"]


def as_smiles_list(model_input: Any) -> list[Any]:
    """Accept a string, a list, an array, a Series, a DataFrame or a dict."""
    if isinstance(model_input, str):
        return [model_input]
    if isinstance(model_input, dict):
        value = model_input.get("smiles", next(iter(model_input.values()), []))
        return as_smiles_list(value)
    columns = getattr(model_input, "columns", None)
    if columns is not None:
        column = "smiles" if "smiles" in columns else columns[0]
        return list(model_input[column])
    if hasattr(model_input, "tolist"):
        return list(model_input.tolist())
    return list(model_input)


class QSPRPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context: Any) -> None:
        from dupont_qspr.serving.scorer import QSPRScorer

        self._scorer = QSPRScorer.load(Path(context.artifacts["bundle"]))

    # Deliberately unannotated: MLflow reads a type hint on ``model_input`` as a
    # schema and rejects anything not wrapped in list[...], but this model accepts
    # a string, list, Series or DataFrame and normalises them itself.
    def predict(self, context, model_input, params=None):
        return self._scorer.score(as_smiles_list(model_input))


def _requirements(family: str) -> list[str]:
    names = ["numpy", "pandas", "polars", "rdkit", "scipy", "mlflow"]
    names += ["xgboost"] if family == "xgb" else ["torch", "transformers"]
    return [f"{name}=={version(name)}" for name in names]


def log_pyfunc(
    cfg: Config,
    bundle: Path,
    manifest: dict[str, Any],
    *,
    registered_name: str = "dupont-qspr-oracle",
) -> Any:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{cfg.artifacts_dir / 'mlflow.db'}")
    experiment = f"{cfg.tracking.experiment}-final"
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(
            experiment,
            artifact_location=(cfg.artifacts_dir / "mlflow-artifacts").as_uri(),
        )
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=manifest["model_version"]):
        mlflow.log_params(
            {
                "family": manifest["family"],
                "profile": manifest["profile"],
                "n_trials": manifest["n_trials"],
                "ad_threshold": manifest["ad_threshold"],
                "featurizer_version": manifest["featurizer_version"],
            }
        )
        mlflow.log_dict(manifest, "manifest.json")
        return mlflow.pyfunc.log_model(
            name="qspr_oracle",
            python_model=QSPRPyfunc(),
            artifacts={"bundle": str(bundle)},
            code_paths=[str(cfg.root / "src" / "dupont_qspr")],
            pip_requirements=_requirements(manifest["family"]),
            registered_model_name=registered_name,
        )
