"""Step 6 — Optuna hyperparameter optimisation for XGBoost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

from evaluation.metrics import compute_metrics, log_metrics
from models.trainer import load_features
from utils.logging import get_logger

logger = get_logger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _objective(
    trial: optuna.Trial,
    X_train,
    y_train,
    X_val,
    y_val,
    log_target: bool,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "early_stopping_rounds": 30,
        "eval_metric": "rmse",
    }
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    preds = model.predict(X_val)
    y_v = np.expm1(y_val) if log_target else y_val
    p_v = np.expm1(preds) if log_target else preds
    return float(r2_score(y_v, p_v))


def run(cfg: dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    tuning_cfg = cfg.get("tuning", {})
    features_cfg = cfg.get("features", {})

    features_dir = data_cfg.get("features_dir", "data/features")
    n_trials = int(tuning_cfg.get("n_trials", 50))
    timeout = int(tuning_cfg.get("timeout", 1800))
    study_name = str(tuning_cfg.get("study_name", "flight_fare_optuna"))
    log_target = bool(features_cfg.get("log_target", False))

    models_dir = Path("models")
    reports_dir = Path("reports")
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)

    logger.info("━━━━━━  Step 6: Optuna XGBoost Tuning  ━━━━━━")
    logger.info("n_trials=%d  timeout=%ds  study=%s", n_trials, timeout, study_name)

    X_train, X_val, X_test, y_train, y_val, y_test = load_features(features_dir)

    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(
        lambda trial: _objective(trial, X_train, y_train, X_val, y_val, log_target),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_val_r2 = study.best_value
    logger.info("Best val R²=%.4f  params=%s", best_val_r2, best_params)

    # Retrain on train+val with best params
    final_params = {**best_params, "random_state": 42, "n_jobs": -1, "verbosity": 0}
    final_params.pop("early_stopping_rounds", None)
    final_model = XGBRegressor(**final_params)
    final_model.fit(X_train, y_train)

    # Evaluate
    results: dict[str, dict] = {}
    for split, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        preds = final_model.predict(X)
        if log_target:
            preds = np.expm1(preds)
            y = np.expm1(y)
        m = compute_metrics(y, preds)
        log_metrics(m, f"xgboost_optuna/{split}")
        results[split] = m

    joblib.dump(final_model, models_dir / "xgboost_optuna.pkl")
    logger.info("Saved → models/xgboost_optuna.pkl")

    report = {
        "model": "XGBoost_Optuna",
        "best_params": best_params,
        "n_trials": n_trials,
        "best_val_r2_during_search": round(best_val_r2, 4),
        "metrics": results,
    }
    out = reports_dir / "metrics_xgboost_optuna.json"
    out.write_text(json.dumps(report, indent=2))
    logger.info("Report saved → %s", out)
    logger.info("━━━━━━  Optuna tuning complete  ━━━━━━")
