# Step 5 — Baseline Model: Linear Regression

**Source:** `src/models/trainer.py`  
**Notebook:** `notebooks/05_baseline_model.ipynb`  
**Input:** `data/features/` (train/val/test parquets)  
**Output:** `models/linear_regression.pkl`, `reports/metrics_linear_regression.json`

---

## What we did

We trained a single **Linear Regression** with no regularisation and no hyperparameters. The goal was not to build the best possible model, but to establish a performance floor that every subsequent model must beat.

Steps:

1. **Load feature splits** — read X_train, X_val, X_test, y_train, y_val, y_test from parquet
2. **Fit** — `LinearRegression().fit(X_train, y_train)` — no grid search, no tuning
3. **Evaluate** — compute R², RMSE, MAE, MAPE on all three splits
4. **Overfitting check** — warn if train R² − val R² > 0.05
5. **Save** — write model pickle and metrics JSON
6. **Log to MLflow** — record run, params, and metrics

---

## Results

All metrics are in **log-space** (predictions and labels are `log1p(fare)`). This is the space where the model was trained. MAPE (%) is the most interpretable metric — it represents relative prediction error regardless of log-space arithmetic.

| Split | R² | RMSE | MAE | MAPE |
|---|---|---|---|---|
| Train | 0.8936 | 0.46 | 0.35 | 3.34% |
| Val | 0.8907 | 0.47 | 0.36 | 3.40% |
| Test | 0.8931 | 0.46 | 0.35 | 3.34% |

**Train/val gap: 0.003 R²** — well within the 0.05 threshold. No overfitting detected.

**Interpretation of 3.4% MAPE:** on an average BDT 41,308 fare, this is ~BDT 1,404 average absolute error. Acceptable as a baseline.

### Why Linear Regression performs this well

The feature engineering in Step 3 already did much of the heavy lifting:

- `route_te` (target-encoded route) is a single numeric column that carries the route price signal directly — linear regression can use it with full weight
- `travel_class_*` one-hot columns capture the 2×–4× class premiums cleanly
- `duration` correlates linearly with fare (r = +0.35)
- Log-transformed target removes the need to model the right tail

Linear Regression achieves ~89% R² because the relationships between these engineered features and `log(fare)` are approximately linear.

---

## Artifacts saved

| File | Contents |
|---|---|
| `models/linear_regression.pkl` | Fitted LinearRegression model |
| `reports/metrics_linear_regression.json` | R², RMSE, MAE, MAPE for all three splits + feature names + intercept |

The metrics JSON is consumed by Step 6 (advanced models) to include linear regression in the comparison table without re-training.

---

## MLflow run

The run is logged to the `flight-fare-prediction` MLflow experiment with:
- **Tag:** `model_type = linear_regression`
- **Param:** `log_target = true`
- **Metrics:** `train_r2`, `val_r2`, `test_r2`, `train_mape`, `val_mape`, `test_mape`, `train_rmse`, `val_rmse`, `test_rmse`, `train_mae`, `val_mae`, `test_mae`
- **Artifact:** model pickle

---

## Design decisions

**No hyperparameters — intentional** — The purpose of this step is a clean benchmark. Adding regularisation or feature selection here conflates "baseline" with "tuned model". Linear Regression with no tuning gives the fairest floor for comparison.

**Overfitting check built in** — The trainer warns when `train_r2 − val_r2 > 0.05`. At this step the gap is 0.003, indicating the engineered features generalise well. This check also runs on every advanced model in Step 6.

**Metrics in log-space** — With `log_target=True` and `eval_log_space=True` in `config.toml`, metrics are reported in log-space (not BDT). This is intentional — log-space RMSE is a stable, scale-independent metric. MAPE is added to give the stakeholder-friendly percentage interpretation. To report in raw BDT instead, set `eval_log_space=False` — the trainer will call `expm1` before computing metrics.

---

## What comes next (Step 6 — Advanced Models)

Step 6 runs 7 more models against this baseline:

- **Ridge** — linear regression with L2 penalty (controls coefficient magnitude)
- **Lasso** — L1 penalty (forces sparse coefficients — fewer features used)
- **Decision Tree** — non-linear, interpretable, grid-searched
- **Random Forest** — bagged trees, randomised search
- **Gradient Boosting** — sequential boosted trees
- **XGBoost** — gradient boosted trees with early stopping
- **Stacking** — RF + GB + XGB base models with Ridge meta-learner

The comparison table ranks all 8 models (including this baseline) by test R² and val MAPE.
