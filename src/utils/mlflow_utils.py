"""MLflow helpers — experiment setup, run logging, model registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

from utils.logging import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "flight-fare-prediction"


def _ensure_local_artifact_location(artifact_location: str | None) -> str | None:
    if not artifact_location:
        return None

    parsed = urlparse(artifact_location)
    if parsed.scheme in {"", "file"}:
        path = Path(parsed.path if parsed.scheme == "file" else artifact_location)
        path.mkdir(parents=True, exist_ok=True)
        return path.resolve().as_uri()

    return artifact_location


def setup_experiment(
    tracking_uri: str = "sqlite:///mlflow.db",
    artifact_location: str | None = "mlartifacts",
) -> str:
    """Configure MLflow tracking URI and create experiment if missing. Returns experiment_id."""
    mlflow.set_tracking_uri(tracking_uri)
    artifact_uri = _ensure_local_artifact_location(artifact_location)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
        logger.info("Created MLflow experiment '%s' (id=%s)", EXPERIMENT_NAME, experiment_id)
    else:
        experiment_id = experiment.experiment_id
        logger.info("Using MLflow experiment '%s' (id=%s)", EXPERIMENT_NAME, experiment_id)
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def log_model_run(
    model_name: str,
    model,
    params: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    feature_names: list[str] | None = None,
) -> str:
    """Log a single model run to MLflow. Returns the run_id.

    Logs:
    - params: hyperparameters used
    - metrics: r2/mae/rmse/mape for train/val/test splits
    - model artifact: the fitted sklearn/xgboost model
    """
    with mlflow.start_run(run_name=model_name) as run:
        mlflow.set_tag("model_type", model_name)

        # log hyperparameters
        for k, v in params.items():
            mlflow.log_param(k, v)

        # log metrics — flat names like "test_r2", "val_mae"
        for split, split_metrics in metrics.items():
            for metric_name, value in split_metrics.items():
                mlflow.log_metric(f"{split}_{metric_name}", value)

        # log model artifact
        _log_model(model, model_name, feature_names)

        run_id = run.info.run_id
        logger.info("MLflow run logged: %s  run_id=%s", model_name, run_id)
        return run_id


def _log_model(model, model_name: str, feature_names: list[str] | None) -> None:
    """Log model artifact — XGBoost gets native flavor, sklearn gets sklearn flavor."""
    from xgboost import XGBRegressor

    if isinstance(model, XGBRegressor):
        mlflow.xgboost.log_model(model, artifact_path="model")
    else:
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=None,
        )


def register_best_model(
    metric: str = "test_r2",
    registry_name: str = "FarePredictor",
) -> str | None:
    """Find the run with highest test_r2 in the experiment and register it.

    Returns the model version URI, or None if no runs found.

    WHY: After training all models we pick the best by test R² and promote
    it to the registry so inference code loads by name, not by file path.
    """
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        logger.warning("Experiment '%s' not found — skipping registration.", EXPERIMENT_NAME)
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="",
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        logger.warning("No runs found in experiment — skipping registration.")
        return None

    best_run = runs[0]
    best_r2 = best_run.data.metrics.get(metric, 0)
    model_type = best_run.data.tags.get("model_type", "unknown")
    run_id = best_run.info.run_id

    logger.info(
        "Best model: %s  %s=%.4f  run_id=%s",
        model_type,
        metric,
        best_r2,
        run_id,
    )

    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=registry_name)

    # alias the version as "champion" so inference code loads by alias
    client.set_registered_model_alias(registry_name, "champion", mv.version)

    logger.info(
        "Registered '%s' version %s as @champion  (%s=%.4f)",
        registry_name,
        mv.version,
        metric,
        best_r2,
    )
    return f"models:/{registry_name}@champion"
