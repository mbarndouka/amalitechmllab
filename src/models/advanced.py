"""Step 5 — Advanced models: Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from evaluation.metrics import compute_metrics, log_metrics
from models.trainer import load_features
from utils.logging import get_logger

logger = get_logger(__name__)

_CV_SCORING = "neg_root_mean_squared_error"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_search(estimator, param_grid: dict, X, y, cv: int) -> GridSearchCV:
    gs = GridSearchCV(
        estimator, param_grid,
        cv=cv, scoring=_CV_SCORING,
        n_jobs=-1, refit=True, verbose=0,
    )
    gs.fit(X, y)
    return gs


def _random_search(estimator, param_dist: dict, X, y, cv: int, n_iter: int, seed: int) -> RandomizedSearchCV:
    rs = RandomizedSearchCV(
        estimator, param_dist,
        n_iter=n_iter, cv=cv, scoring=_CV_SCORING,
        n_jobs=-1, refit=True, verbose=0, random_state=seed,
    )
    rs.fit(X, y)
    return rs


def _evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test, name: str,
              log_target: bool = False) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        preds = model.predict(X)
        if log_target:
            preds = np.expm1(preds)
            y     = np.expm1(y)
        m = compute_metrics(y, preds)
        log_metrics(m, f"{name}/{split}")
        results[split] = m
    return results


# ---------------------------------------------------------------------------
# Individual trainers
# ---------------------------------------------------------------------------

def train_ridge(X_train, y_train, cfg: dict, cv: int) -> tuple[GridSearchCV, dict]:
    alphas = cfg.get("models", {}).get("ridge", {}).get("alpha", [0.1, 1.0, 10.0, 100.0])
    logger.info("Ridge GridSearchCV  alphas=%s  cv=%d", alphas, cv)
    gs = _grid_search(Ridge(), {"alpha": alphas}, X_train, y_train, cv)
    logger.info("Best alpha=%.4f  cv_rmse=%.0f", gs.best_params_["alpha"], -gs.best_score_)
    return gs.best_estimator_, gs.best_params_


def train_lasso(X_train, y_train, cfg: dict, cv: int) -> tuple[GridSearchCV, dict]:
    lasso_cfg = cfg.get("models", {}).get("lasso", {})
    alphas   = lasso_cfg.get("alpha", [0.01, 0.1, 1.0, 10.0])
    max_iter = lasso_cfg.get("max_iter", 5000)
    logger.info("Lasso GridSearchCV  alphas=%s  cv=%d", alphas, cv)
    gs = _grid_search(Lasso(max_iter=max_iter), {"alpha": alphas}, X_train, y_train, cv)
    logger.info("Best alpha=%.4f  cv_rmse=%.0f", gs.best_params_["alpha"], -gs.best_score_)
    return gs.best_estimator_, gs.best_params_


def train_decision_tree(X_train, y_train, cfg: dict, cv: int) -> tuple[DecisionTreeRegressor, dict]:
    dt_cfg = cfg.get("models", {}).get("decision_tree", {})
    param_grid = {
        "max_depth":         dt_cfg.get("max_depth", [3, 5, 10, 15, 20]),
        "min_samples_split": dt_cfg.get("min_samples_split", [2, 5, 10]),
        "min_samples_leaf":  dt_cfg.get("min_samples_leaf", [1, 2, 4]),
    }
    seed = dt_cfg.get("random_state", 42)
    logger.info("DecisionTree GridSearchCV  grid_size=%d  cv=%d",
                len(param_grid["max_depth"]) * len(param_grid["min_samples_split"]) * len(param_grid["min_samples_leaf"]), cv)
    gs = _grid_search(DecisionTreeRegressor(random_state=seed), param_grid, X_train, y_train, cv)
    logger.info("Best params=%s  cv_rmse=%.0f", gs.best_params_, -gs.best_score_)
    return gs.best_estimator_, gs.best_params_


def train_random_forest(X_train, y_train, cfg: dict, cv: int) -> tuple[RandomForestRegressor, dict]:
    rf_cfg = cfg.get("models", {}).get("random_forest", {})
    seed = rf_cfg.get("random_state", 42)
    param_dist = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [8, 10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", 0.5, 0.7],
    }
    n_iter = 20
    logger.info("RandomForest RandomizedSearchCV  n_iter=%d  cv=%d", n_iter, cv)
    rs = _random_search(
        RandomForestRegressor(random_state=seed, n_jobs=-1),
        param_dist, X_train, y_train, cv, n_iter=n_iter, seed=seed,
    )
    logger.info("Best params=%s  cv_rmse=%.4f", rs.best_params_, -rs.best_score_)
    return rs.best_estimator_, rs.best_params_


def train_xgboost(X_train, y_train, X_val, y_val, cfg: dict) -> tuple[XGBRegressor, dict]:
    xgb_cfg = cfg.get("models", {}).get("xgboost", {})
    params = {
        "n_estimators":     xgb_cfg.get("n_estimators", 500),
        "learning_rate":    xgb_cfg.get("learning_rate", 0.03),
        "max_depth":        xgb_cfg.get("max_depth", 7),
        "subsample":        xgb_cfg.get("subsample", 0.8),
        "colsample_bytree": xgb_cfg.get("colsample_bytree", 0.8),
        "random_state":     xgb_cfg.get("random_state", 42),
        "n_jobs":           -1,
        "verbosity":        0,
    }
    logger.info("XGBoost  params=%s  early_stopping_rounds=50", params)
    model = XGBRegressor(**params, early_stopping_rounds=50, eval_metric="rmse")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iter = model.best_iteration
    logger.info("XGBoost best_iteration=%d", best_iter)
    return model, {**params, "best_iteration": best_iter}


def train_gradient_boosting(X_train, y_train, cfg: dict) -> tuple[GradientBoostingRegressor, dict]:
    gb_cfg = cfg.get("models", {}).get("gradient_boosting", {})
    params = {
        "n_estimators":  gb_cfg.get("n_estimators", 300),
        "learning_rate": gb_cfg.get("learning_rate", 0.05),
        "max_depth":     gb_cfg.get("max_depth", 6),
        "random_state":  gb_cfg.get("random_state", 42),
        "subsample":     0.8,
    }
    logger.info("GradientBoosting  params=%s", params)
    model = GradientBoostingRegressor(**params)
    model.fit(X_train, y_train)
    logger.info("GradientBoosting training complete.")
    return model, params


def train_stacking(X_train, y_train, cfg: dict) -> tuple[StackingRegressor, dict]:
    """Stack RF + GB + XGBoost with Ridge meta-learner (cv=3 for speed).

    WHY: Each base model captures different patterns. Ridge meta-learner learns
    optimal blend weights. passthrough=False keeps meta-learner input clean.
    """
    seed = 42
    base_estimators = [
        ("rf",  RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=4,
            max_features=0.7, random_state=seed, n_jobs=-1,
        )),
        ("gb",  GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=seed,
        )),
        ("xgb", XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed, n_jobs=-1, verbosity=0,
        )),
    ]
    meta = Ridge(alpha=1.0)
    stacker = StackingRegressor(
        estimators=base_estimators,
        final_estimator=meta,
        cv=3,
        n_jobs=-1,
        passthrough=False,
    )
    logger.info("Stacking: RF + GB + XGB → Ridge meta-learner  cv=3")
    stacker.fit(X_train, y_train)
    logger.info("Stacking training complete.")
    return stacker, {"base_models": ["rf", "gb", "xgb"], "meta": "ridge", "cv": 3}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_comparison_table(
    all_results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Build a flat DataFrame: model × split × metric."""
    rows = []
    for model_name, entry in all_results.items():
        for split, m in entry["metrics"].items():
            rows.append({
                "model":  model_name,
                "split":  split,
                "r2":     m["r2"],
                "mae":    m["mae"],
                "rmse":   m["rmse"],
                "mape":   m["mape"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(cfg: dict[str, Any]) -> None:
    data_cfg     = cfg.get("data", {})
    eval_cfg     = cfg.get("evaluation", {})
    features_cfg = cfg.get("features", {})
    features_dir = data_cfg.get("features_dir", "data/features")
    cv           = eval_cfg.get("cv_folds", 5)
    log_target   = bool(features_cfg.get("log_target", False))
    models_dir   = Path("models")
    reports_dir  = Path("reports")
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    logger.info("━━━━━━  Step 5: Advanced Modeling & Optimization  ━━━━━━")
    if log_target:
        logger.info("log_target=True — metrics computed in original BDT scale via expm1")

    X_train, X_val, X_test, y_train, y_val, y_test = load_features(features_dir)

    all_results: dict[str, dict] = {}

    # Load baseline linear regression metrics for comparison
    baseline_path = reports_dir / "metrics_linear_regression.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        all_results["linear_regression"] = {
            "metrics": baseline["metrics"],
            "best_params": {},
        }
        logger.info("Loaded baseline LinearRegression metrics.")

    trainers = [
        ("ridge",              lambda: train_ridge(X_train, y_train, cfg, cv)),
        ("lasso",              lambda: train_lasso(X_train, y_train, cfg, cv)),
        ("decision_tree",      lambda: train_decision_tree(X_train, y_train, cfg, cv)),
        ("random_forest",      lambda: train_random_forest(X_train, y_train, cfg, cv)),
        ("gradient_boosting",  lambda: train_gradient_boosting(X_train, y_train, cfg)),
        ("xgboost",            lambda: train_xgboost(X_train, y_train, X_val, y_val, cfg)),
        ("stacking",           lambda: train_stacking(X_train, y_train, cfg)),
    ]

    for name, train_fn in trainers:
        logger.info("── Training: %s ──", name)
        model, best_params = train_fn()
        metrics = _evaluate(model, X_train, X_val, X_test, y_train, y_val, y_test, name,
                            log_target=log_target)
        joblib.dump(model, models_dir / f"{name}.pkl")
        all_results[name] = {"metrics": metrics, "best_params": best_params}
        logger.info("Saved → models/%s.pkl", name)

    # Save comparison table
    comparison_df = build_comparison_table(all_results)
    comparison_path = reports_dir / "model_comparison.json"
    comparison_path.write_text(
        json.dumps({k: v for k, v in all_results.items()}, indent=2, default=str)
    )
    logger.info("Comparison saved → %s", comparison_path)

    # Summary — val split ranked by R²
    val_summary = (
        comparison_df[comparison_df["split"] == "val"]
        .sort_values("r2", ascending=False)
        [["model", "r2", "mae", "rmse"]]
        .reset_index(drop=True)
    )
    logger.info("━━  Validation Leaderboard  ━━")
    for _, row in val_summary.iterrows():
        logger.info("  %-22s  R²=%.4f  MAE=%s  RMSE=%s",
                    row["model"], row["r2"],
                    f"{row['mae']:,.0f}", f"{row['rmse']:,.0f}")

    logger.info("━━━━━━  Advanced modeling complete  ━━━━━━")
