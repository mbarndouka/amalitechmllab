# Step 7 — Model Interpretation & Business Insights

**Sources:** `src/interpretation/importance.py`, `src/interpretation/insights.py`  
**Notebook:** `notebooks/07_interpretation.ipynb`  
**Input:** `models/*.pkl`, `data/features/X_train.parquet`, `data/processed/Flight_Price_Dataset_of_Bangladesh.parquet`  
**Output:** `reports/importance_*.csv`, `reports/importance_summary.json`, `reports/insights.json`, `reports/stakeholder_report.txt`

---

## What we did

Interpretation runs two parallel analyses:

**6a — Feature importance** (`importance.py`)  
Load every saved model and extract which features it relied on most. Linear models and tree models report importance differently — each is handled correctly.

**6b — Business insights** (`insights.py`)  
Aggregate the raw processed data to answer business questions: which airlines charge the most? Does Eid really spike prices? How much does booking early save?

These are complementary, not duplicate. Feature importance tells you what the **model** learned; business insights tell you what the **data** actually shows.

---

## Part A — Feature Importance

### How each model type reports importance

**Linear models (LinearRegression, Ridge, Lasso)**  
Use raw model coefficients. A coefficient of +0.3 on `travel_class_First_Class` means: "holding all else equal, First Class adds 0.3 log-units of fare (~35% premium)." Coefficients are **signed** — direction matters.

**Tree models (DecisionTree, RandomForest, GradientBoosting, XGBoost)**  
Use `feature_importances_` — the total reduction in MSE (mean squared error) caused by splitting on each feature across all trees. Values are always non-negative. Higher = more useful for prediction. There is no direction.

This distinction is enforced in code via two separate extractor functions mapped by model name — no shared logic between the two types.

### Top features (Random Forest, representative)

| Rank | Feature | Importance | Notes |
|---|---|---|---|
| 1 | `route_te` | 0.603 | Target-encoded route — dominant predictor |
| 2 | `duration` | 0.129 | Longer flight = higher cost |
| 3 | `travel_class_Economy` | 0.106 | Class premium |
| 4 | `aircraft_type_Airbus A320` | 0.063 | Aircraft as proxy for route type |
| 5–10 | Various OHE columns | 0.01–0.04 | Airline, destination, season |

`route_te` alone explains 60% of the model's predictions. Route is the primary driver — a DAC→LHR fare is fundamentally different from a DAC→CGP fare regardless of other factors.

### Cross-model summary

`importance_summary.json` contains the top-10 features for every model side by side. This makes it easy to spot:
- Features universally important across all models (route, duration, travel class)
- Features that only certain models rely on (some OHE columns in linear models that trees ignore)

---

## Part B — Business Insights

### Airline pricing

Cathay Pacific charges the highest median fare (BDT 46,282). Singapore Airlines is the cheapest full-service international carrier (BDT 38,441) — a 17% difference. US-Bangla Airlines and other domestic carriers cluster around BDT 25,000–30,000.

The high standard deviation within each airline (up to BDT 83,000 for Cathay) shows that class and route matter more than airline brand alone.

### Seasonal pricing

| Season | Median fare (BDT) | Premium over Regular |
|---|---|---|
| Eid | 56,621 | +42% |
| Hajj | ~50,000 | +25% |
| Winter Holidays | ~44,000 | +10% |
| Regular | ~40,000 | — |

Eid is the single strongest demand shock. Diaspora returning home for the holiday creates a supply constraint that drives fares up 42% above regular season. This pattern is captured by the `seasonality` OHE columns in the model.

### Travel class premium

First Class costs 3.9× Economy median fare (BDT 94,191 vs BDT 24,123). Business is 2.1× Economy. These multipliers are large and consistent — travel class is the second most important predictor after route.

### Booking window (days until departure)

| Booking window | Median fare (BDT) |
|---|---|
| 0–7 days | 50,923 |
| 8–14 days | 43,999 |
| 15–30 days | ~41,000 |
| 31–60 days | ~39,000 |
| 61–90 days | ~38,000 |

Last-minute bookings (0–7 days) pay 34% more than far-advance bookings (61–90 days). The negative correlation between `days_left` and `fare` (r = −0.18) is consistent across all routes and classes.

### Routes

**Most expensive routes** — Long-haul international routes from Sylhet (ZYL) and Chittagong (CXB) to Bangkok (BKK) and Kuala Lumpur (KUL) rank highest in median fare. These routes have fewer flights per day, driving premiums.

**Cheapest routes** — Domestic short-haul routes (Barisal↔Jessore, Sylhet↔Chittagong) with BDT 5,000–8,000 median fares. High frequency and competition keeps these cheap.

---

## Outputs

| File | Contents |
|---|---|
| `reports/importance_linear_regression.csv` | Feature × coefficient × abs_coefficient |
| `reports/importance_ridge.csv` | Ridge coefficients |
| `reports/importance_lasso.csv` | Lasso coefficients (many near zero) |
| `reports/importance_decision_tree.csv` | Tree feature importances |
| `reports/importance_random_forest.csv` | RF feature importances |
| `reports/importance_gradient_boosting.csv` | GB feature importances |
| `reports/importance_xgboost.csv` | XGBoost feature importances |
| `reports/importance_summary.json` | Top-10 features per model, all models in one file |
| `reports/insights.json` | Structured pricing breakdowns (airline, season, class, route, booking_window) |
| `reports/stakeholder_report.txt` | Plain-English narrative for non-technical readers |

---

## Design decisions

**Separate extractors for linear vs tree models** — Coefficients and `feature_importances_` are not the same quantity. Mixing them in one function (e.g., checking `hasattr(model, 'coef_')`) creates fragile duck-typing. Instead, a lookup dict maps model name → extraction function explicitly. Adding a new model type requires adding one entry to the dict.

**Median, not mean, for business insights** — All group-level fare aggregations use median. Flight fares are right-skewed — a single first-class or business ticket on a route can double the mean without representing a typical price. Median is what a typical traveller would experience.

**Business insights read raw data, not model predictions** — Insights computed from model predictions would conflate "what the model learned" with "what's actually in the data." Reading directly from the processed parquet ensures insights reflect ground truth, independent of model choices.

**`importance_summary.json` for cross-model comparison** — Saving each model's top-10 features to a single JSON (rather than requiring consumers to read 7 separate CSVs) makes it easy for downstream tools (Streamlit, notebooks) to display a side-by-side feature importance view with one file read.

---

## How to use these outputs

**For model debugging:** if a model is performing poorly, compare its `importance_*.csv` to a well-performing model. Features with near-zero importance might indicate a data issue or encoding problem.

**For feature selection:** any feature with near-zero importance across all models can be dropped to reduce prediction latency. `importance_summary.json` makes this multi-model check easy.

**For stakeholder communication:** `stakeholder_report.txt` is written for non-technical audiences — no log-space, no R², just BDT premiums and plain-English conclusions. Share directly without modification.

**For the Streamlit dashboard:** the `Market Insights` tab reads `insights.json` directly to render airline pricing charts, seasonal fare bars, and booking window charts. No re-computation needed.
