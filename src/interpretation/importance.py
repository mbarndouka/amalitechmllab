"""Step 6 — Feature importance extraction for all trained models.

WHY THIS MODULE EXISTS
----------------------
After training, we need to answer: "which features actually drive predictions?"

Linear models  → use *coefficients* (signed, unit-aware: BDT per unit of feature)
Tree models    → use *feature_importances_* (Gini/MSE impurity reduction, always >= 0)

These are fundamentally different quantities — we handle them separately.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from models.trainer import load_features
from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _coef_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract signed coefficients from linear models (LinearRegression, Ridge, Lasso).

    WHY SIGNED: A positive coefficient means "as this feature increases, fare increases."
    A negative coefficient means the opposite. Magnitude tells us HOW MUCH.
    We sort by absolute value so the most influential features appear first regardless of direction.
    """
    coef = model.coef_
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": coef,  # signed — direction matters
                "abs_importance": np.abs(coef),  # magnitude — for ranking
            }
        )
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )


def _tree_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature_importances_ from tree-based models.

    WHY NOT SIGNED: Trees split on features to reduce prediction error (MSE).
    Each feature gets a score = total MSE reduction it caused across all splits.
    Higher = more useful for prediction. There's no direction here.
    """
    imp = model.feature_importances_
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": imp,
                "abs_importance": imp,
            }
        )
        .sort_values("abs_importance", ascending=False)
        .reset_index(drop=True)
    )


# Maps model file name → extraction function
_EXTRACTORS = {
    "linear_regression": _coef_importance,
    "ridge": _coef_importance,
    "lasso": _coef_importance,
    "decision_tree": _tree_importance,
    "random_forest": _tree_importance,
    "gradient_boosting": _tree_importance,
    "xgboost": _tree_importance,
}


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_all(
    models_dir: str | Path,
    feature_names: list[str],
    reports_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Load every saved model and extract its feature importances.

    Returns dict mapping model_name → importance DataFrame.
    Also saves each DataFrame to reports/ as CSV for later use.
    """
    models_dir = Path(models_dir)
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for model_name, extractor in _EXTRACTORS.items():
        model_path = models_dir / f"{model_name}.pkl"
        if not model_path.exists():
            logger.warning("Model not found: %s — skipping", model_path)
            continue

        model = joblib.load(model_path)
        logger.info("Extracting importance: %s", model_name)

        df = extractor(model, feature_names)
        results[model_name] = df

        out_path = reports_dir / f"importance_{model_name}.csv"
        df.to_csv(out_path, index=False)
        logger.info(
            "  Saved → %s  (top feature: %s = %.4f)", out_path, df.iloc[0]["feature"], df.iloc[0]["abs_importance"]
        )

    # Cross-model summary: top-10 features per model side by side
    _save_cross_model_summary(results, reports_dir)

    return results


def _save_cross_model_summary(
    results: dict[str, pd.DataFrame],
    reports_dir: Path,
    top_n: int = 10,
) -> None:
    """Save a JSON summary of top-N features for every model.

    WHY: Having one file with all models' top features makes it easy to spot
    which features are universally important vs model-specific.
    """
    summary: dict[str, list[dict]] = {}
    for model_name, df in results.items():
        summary[model_name] = (
            df.head(top_n)[["feature", "importance", "abs_importance"]].round(6).to_dict(orient="records")
        )

    out_path = reports_dir / "importance_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    logger.info("Cross-model importance summary saved → %s", out_path)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    features_dir = data_cfg.get("features_dir", "data/features")

    logger.info("━━━━━━  Step 6a: Feature Importance Extraction  ━━━━━━")

    X_train, *_ = load_features(features_dir)
    feature_names = X_train.columns.tolist()
    logger.info("Feature count: %d", len(feature_names))

    extract_all("models", feature_names, "reports")

    logger.info("━━━━━━  Importance extraction complete  ━━━━━━")
