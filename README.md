# Flight Fare Prediction — Bangladesh Airlines

An end-to-end machine learning pipeline that predicts domestic and international flight ticket prices (in BDT) for flights departing from Bangladesh. Covers data cleaning, feature engineering, model training, hyperparameter tuning, model interpretation, and a live prediction API.

---

## What This Project Does

1. **Cleans raw flight data** — handles missing values, removes leakage columns, fixes types
2. **Engineers features** — encodes airlines, routes, travel class, seasonality; applies log transforms
3. **Trains and compares 8 models** — from Linear Regression to XGBoost with Optuna tuning
4. **Tracks all experiments** with MLflow (metrics, parameters, model artifacts)
5. **Extracts business insights** — seasonal surges, airline pricing tiers, booking timing effects
6. **Serves predictions** via a FastAPI REST API, containerized with Docker

---

## Project Structure

```
.
├── configs/config.toml          # All pipeline settings (one file to rule them all)
├── data/
│   ├── raw/                     # Original CSV dataset
│   ├── processed/               # Cleaned parquet
│   └── features/                # Train/val/test splits + scaler
├── models/                      # Saved model files (.pkl)
├── notebooks/                   # Step-by-step Jupyter notebooks (01–07)
├── reports/                     # Metrics JSON, feature importance CSVs, stakeholder report
├── src/
│   ├── features/                # Cleaning + feature engineering logic
│   ├── models/                  # Baseline and advanced model training
│   ├── evaluation/              # Metrics computation
│   ├── tuning/                  # Optuna hyperparameter search
│   ├── interpretation/          # Feature importance + business insights
│   ├── serving/                 # FastAPI prediction server
│   ├── pipeline/                # Stage runner (wires everything together)
│   └── utils/                   # Config loader, logger
├── tests/                       # Unit and integration tests
├── main.py                      # CLI entry point
├── Dockerfile
└── docker-compose.yml
```

---

## Quick Start

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) **or** pip

### Install

```bash
git clone <repo-url>
cd amalitechmllab

# with uv
uv sync

# or with pip
pip install -e .
```

### Run the full pipeline

```bash
uv run python main.py
```

### Run a single stage

```bash
uv run python main.py --stage clean       # data cleaning
uv run python main.py --stage features    # feature engineering
uv run python main.py --stage train       # baseline model (Linear Regression)
uv run python main.py --stage advanced    # Ridge, Lasso, Tree, RF, GBT, XGBoost, Stacking
uv run python main.py --stage insights    # feature importance + business report
```

---

## Notebooks (follow in order)

| Notebook | What it covers |
|---|---|
| `01_data_exploration.ipynb` | Dataset overview, distributions, missing values |
| `02_data_cleaning.ipynb` | Outliers, imputation, leakage removal |
| `03_feature_engineering.ipynb` | Encoding, log transforms, train/val/test split |
| `04_eda.ipynb` | Correlations, price by airline/route/season |
| `05_baseline_model.ipynb` | Linear Regression — baseline benchmark |
| `06_advanced_models.ipynb` | Ensemble models + XGBoost with Optuna |
| `07_interpretation.ipynb` | SHAP values, feature importance, business insights |

---

## Models Trained

All models train and evaluate on `log1p`-transformed fare (right-skewed target). Metrics are in log space — MAPE (%) is the most interpretable: ~3.4% average relative error.

| Model | Val R² | Test R² | Val MAPE | Notes |
|---|---|---|---|---|
| Random Forest | 0.8922 | 0.8936 | 3.36% | RandomizedSearchCV |
| Stacking (RF+GB+XGB→Ridge) | 0.8921 | **0.8937** | 3.36% | Best test R² |
| XGBoost | 0.8914 | 0.8931 | 3.37% | Optuna (100 trials) |
| Gradient Boosting | 0.8909 | 0.8925 | 3.38% | Fixed params |
| Linear Regression | 0.8907 | 0.8931 | 3.40% | Baseline |
| Ridge Regression | 0.8907 | 0.8930 | 3.40% | Grid search alpha |
| Decision Tree | 0.8905 | 0.8924 | 3.39% | Grid search depth |
| Lasso Regression | 0.8891 | 0.8913 | 3.42% | Grid search alpha |

All experiments tracked in MLflow. View the UI with:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## Key Business Findings

- **Eid season** drives a **42% fare premium** over regular season
- **Cathay Pacific** is the most expensive carrier (median BDT 46,282); **Singapore Airlines** the cheapest (BDT 38,441)
- **First Class** costs **3.9×** more than Economy
- **Booking earlier** consistently lowers fares — negative correlation between `days_left` and price
- **Flight duration** is the strongest continuous predictor (longer flight = higher price, r = +0.35)

Full report: `reports/stakeholder_report.txt`

---

## Prediction API

### Start with Docker

```bash
docker-compose up api
```

API will be live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airline": "Emirates",
    "source": "DAC",
    "destination": "DXB",
    "travel_class": "Economy",
    "aircraft_type": "Boeing 777",
    "booking_source": "Online Website",
    "seasonality": "Regular",
    "stopovers": 0,
    "duration": 4.5,
    "days_left": 30,
    "departure_hour": 8,
    "departure_day_of_week": 2,
    "departure_month": 6,
    "arrival_hour": 12,
    "arrival_day_of_week": 2,
    "arrival_month": 6
  }'
```

### Example response

```json
{
  "predicted_fare_bdt": 42500.0,
  "model_source": "mlflow-registry"
}
```

---

## Run Tests

```bash
uv run pytest

# with coverage report
uv run pytest --cov=src
```

---

## Configuration

Everything is controlled from `configs/config.toml`. Key settings:

| Setting | Default | Effect |
|---|---|---|
| `features.log_target` | `true` | Apply log1p to fare before training |
| `tuning.n_trials` | `100` | Number of Optuna trials for XGBoost |
| `data.test_size` | `0.2` | Hold-out test split ratio |
| `mlflow.enabled` | `true` | Toggle MLflow experiment tracking |

---

## Dataset

**Flight Price Dataset of Bangladesh** — ~35,000 flight records with features including airline, source/destination airports, travel class, booking source, seasonality, stopovers, duration, and days until departure.

Target variable: **Total Fare (BDT)**
