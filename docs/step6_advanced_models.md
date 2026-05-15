# Step 6 — Advanced Models & Hyperparameter Tuning

**Sources:** `src/models/advanced.py`, `src/tuning/optuna_tuner.py`  
**Notebook:** `notebooks/06_advanced_models.ipynb`  
**Input:** `data/features/` (train/val/test parquets)  
**Output:** `models/*.pkl` (7 model files), `reports/model_comparison.json`, `reports/metrics_xgboost_optuna.json`

---

## What we did

We trained 7 models in sequence against the linear regression baseline from Step 5. Each model builds complexity on the previous:

| Model | Search strategy | Key hyperparameters |
|---|---|---|
| Ridge | GridSearchCV (5-fold) | alpha ∈ {0.1, 1, 10, 100} |
| Lasso | GridSearchCV (5-fold) | alpha ∈ {0.01, 0.1, 1, 10} |
| Decision Tree | GridSearchCV (5-fold) | max_depth × min_samples_split × min_samples_leaf (45 combos) |
| Random Forest | RandomizedSearchCV (5-fold, 20 iter) | n_estimators, max_depth, min_samples_split, max_features |
| Gradient Boosting | Fixed params (no search) | n_estimators=300, lr=0.05, max_depth=6 |
| XGBoost | Early stopping on val set | n_estimators=500, lr=0.03, early_stopping_rounds=50 |
| Stacking | Fixed (base model CV=3) | RF + GB + XGB → Ridge meta |

After Step 6 models are saved, **Step 6b (Optuna)** runs 100 Bayesian trials to find optimal XGBoost hyperparameters, producing a separately saved `xgboost_optuna.pkl`.

---

## Results

All metrics in log-space. Sorted by test R² descending.

| Model | Val R² | Test R² | Val MAPE | Notes |
|---|---|---|---|---|
| Stacking (RF+GB+XGB→Ridge) | 0.8921 | **0.8937** | 3.36% | Best test R² |
| Random Forest | 0.8922 | 0.8936 | 3.36% | Best val R² |
| XGBoost | 0.8914 | 0.8931 | 3.37% | Early stopping at best iter |
| Gradient Boosting | 0.8909 | 0.8925 | 3.38% | Fixed params |
| Linear Regression | 0.8907 | 0.8931 | 3.40% | Baseline (from Step 5) |
| Ridge | 0.8907 | 0.8930 | 3.40% | Minimal gain over LinearReg |
| Decision Tree | 0.8905 | 0.8924 | 3.39% | Weakest ensemble |
| Lasso | 0.8891 | 0.8913 | 3.42% | Worst — sparse coefficients lose route signal |

### Key takeaways

**Performance spread is narrow (~0.003 R²)** — Feature engineering in Step 3 did the hard work. After target encoding `route` and log-transforming the target, even linear regression achieves 89% R². Ensembles add only marginal gain.

**Stacking wins on test R²** — The meta-learner learns optimal blend weights for RF, GB, and XGBoost. Each base model captures slightly different patterns; Ridge meta-learner combines them.

**Random Forest wins on val R²** — 20 iterations of RandomizedSearchCV found the best val-set params. Slight divergence between val and test performance is expected.

**Lasso underperforms** — L1 penalty forces many coefficients to zero. This hurts performance because many OHE airline/destination columns carry small but real signal. Forcing them to zero loses information.

**Linear Regression ≈ Ridge** — Ridge's regularisation barely changes coefficients because the features are already well-scaled. When features are engineered this well, regularisation has little effect.

---

## Optuna XGBoost tuning (Step 6b)

Optuna runs 100 Bayesian trials using TPE (Tree-structured Parzen Estimator) sampler. Each trial:

1. Proposes hyperparameter values from posterior distribution over R²
2. Trains XGBoost with early stopping on val set (30 rounds)
3. Returns val R² as objective score

**Search space:**

| Parameter | Range | Scale |
|---|---|---|
| `n_estimators` | 200–800 | linear |
| `learning_rate` | 0.01–0.3 | log |
| `max_depth` | 3–9 | linear |
| `subsample` | 0.5–1.0 | linear |
| `colsample_bytree` | 0.5–1.0 | linear |
| `min_child_weight` | 1–10 | linear |
| `reg_alpha` (L1) | 1e-8–10 | log |
| `reg_lambda` (L2) | 1e-8–10 | log |

After finding the best params, the final model is **retrained on the full train set** (without early stopping) using those params. This is saved separately as `models/xgboost_optuna.pkl`.

**Why retrain after Optuna?** — During search, early stopping uses the val set to find the right `n_estimators`. The final model doesn't use early stopping, so it runs for the full discovered `n_estimators` on all train data, avoiding val-set dependence.

---

## Stacking architecture

```
X_train ──► RF (200 trees, max_depth=8)    ──┐
         ──► GB (200 estimators, lr=0.05)  ──┼──► Ridge meta-learner ──► ŷ
         ──► XGB (300 trees, lr=0.05)      ──┘
```

- `cv=3` — each base model's out-of-fold predictions are computed with 3-fold cross-validation, so the meta-learner never sees in-sample base predictions
- `passthrough=False` — meta-learner receives only base model predictions (not raw features), keeping its input clean and interpretation simple

---

## MLflow logging

Every model run is logged to the `flight-fare-prediction` experiment:
- **Tag:** `model_type` = model name
- **Params:** best hyperparameters from grid/random search
- **Metrics:** `{split}_{metric}` for all 3 splits × 4 metrics = 12 metrics per run
- **Artifact:** model pickle

After all 7 runs, the model with highest `test_r2` is **registered to the MLflow Model Registry** as `FarePredictor@champion`. The serving layer (`Predictor` class in `serving/predictor.py`) loads by alias, so updating the champion requires only re-running this step.

---

## Artifacts saved

| File | Contents |
|---|---|
| `models/ridge.pkl` | Best Ridge estimator from GridSearchCV |
| `models/lasso.pkl` | Best Lasso estimator |
| `models/decision_tree.pkl` | Best DecisionTree estimator |
| `models/random_forest.pkl` | Best RandomForest estimator |
| `models/gradient_boosting.pkl` | Trained GradientBoosting model |
| `models/xgboost.pkl` | XGBoost model with early stopping |
| `models/stacking.pkl` | Fitted StackingRegressor |
| `models/xgboost_optuna.pkl` | Optuna-tuned XGBoost, retrained on full train set |
| `reports/model_comparison.json` | All 8 models × 3 splits × 4 metrics |
| `reports/metrics_xgboost_optuna.json` | Optuna study results + final metrics |

---

## Design decisions

**GridSearch for linear models, RandomizedSearch for forests** — Linear models (Ridge, Lasso) have a small discrete alpha grid — GridSearch exhausts it cheaply. Random Forest has a large combinatorial space (5 hyperparameters × 5 values = 3,375 combos). RandomizedSearch samples 20 random points and finds near-optimal params at 1/168 the cost.

**XGBoost uses val set for early stopping, not CV** — Cross-validation would require fitting XGBoost 5 times per trial. Early stopping on a held-out val set achieves the same goal (finding the right `n_estimators`) in one pass.

**Champion registration by test R²** — After all models log to MLflow, `register_best_model()` queries all runs sorted by `test_r2` and registers the winner as `FarePredictor@champion`. Any future run that improves test R² will automatically become the new champion on the next pipeline execution.

---

## What comes next (Step 7 — Interpretation)

Step 7 answers: "why does the model predict what it predicts?"

- Feature importance extraction — which features drive predictions most?
- Business insights — what patterns in the raw data explain the model's behaviour?
- Stakeholder report — plain-English summary of findings for non-technical audiences
