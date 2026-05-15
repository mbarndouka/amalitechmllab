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
uv run python main.py --stage engineer    # feature engineering
uv run python main.py --stage train       # baseline model (Linear Regression)
uv run python main.py --stage advanced    # Ridge, Lasso, Tree, RF, GBT, XGBoost, Stacking
uv run python main.py --stage tune        # Optuna hyperparameter search (XGBoost, 100 trials)
uv run python main.py --stage interpret   # feature importance + business report
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

## MLflow UI

Browse experiments, compare runs, and inspect artifacts:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000`. Port conflicts? Use `--port`:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

What you'll find:
- All runs grouped under `flight_fare_prediction` experiment
- Metrics per run: R², RMSE, MAE, MAPE
- Hyperparameters logged (Optuna trials, grid search values)
- Registered model versions under **Models → flight_fare_predictor**

---

## Streamlit App

Interactive UI for fare prediction with model comparison dashboard:

```bash
uv run streamlit run app.py
```

Opens at `http://localhost:8501`. Features:
- Select model (MLflow registry, local pkl, or fallback)
- Input flight details via sidebar form
- Predicted fare in BDT with model source shown
- Model comparison table (R², MAPE across all trained models)

Custom port:

```bash
uv run streamlit run app.py --server.port 8502
```

> **Note:** Run the full pipeline (`uv run python main.py`) before launching the app — it needs trained models and `reports/model_comparison.json`.

---

## Airflow Orchestration

The full pipeline runs automatically every Monday at 02:00 UTC via an Airflow DAG (`ml_training_pipeline`) defined in `amalitechairflowdbtlab/dags/ml_training_pipeline.py`. Each stage runs as an isolated Docker container sharing host volumes so artifacts persist between stages.

### DAG stages (in order)

| Task | Stage | What it does |
|---|---|---|
| `stage_clean` | clean | Remove leakage columns, fix types, handle missing values |
| `stage_engineer` | engineer | Encode features, log-transform target, train/val/test split |
| `stage_train` | train | Linear Regression baseline |
| `stage_advanced` | advanced | Ridge, Lasso, Tree, RF, GBT, XGBoost, Stacking |
| `stage_tune` | tune | Optuna hyperparameter search for XGBoost (100 trials) |
| `stage_interpret` | interpret | SHAP values, feature importance, business insights report |

### Trigger manually

```bash
# from amalitechairflowdbtlab/
docker exec <airflow-apiserver-container> airflow dags trigger ml_training_pipeline
```

### Artifacts written (host paths)

- `amalitechmllab/models/` — trained `.pkl` files
- `amalitechmllab/mlflow.db` — experiment tracking DB
- `amalitechmllab/mlartifacts/` — MLflow model artifacts
- `amalitechmllab/reports/` — metrics JSON, stakeholder report

> **Note:** The Airflow worker container requires access to `/var/run/docker.sock` with the host Docker group GID. This is configured via `group_add` in `amalitechairflowdbtlab/docker-compose.yaml`.

---

## Docker — Full Stack (ML lab + Airflow lab)

All services run in Docker and communicate over a shared bridge network (`amalitech-net`). This lets Airflow DAGs call the ML prediction API by container name.

### Port map

| Service | URL |
|---|---|
| FastAPI prediction API | `http://localhost:8000` |
| Streamlit UI | `http://localhost:8501` |
| MLflow UI | `http://localhost:5000` |
| Airflow UI | `http://localhost:8080` |

### 1. Create shared network (once)

```bash
docker network create amalitech-net
```

### 2. Start ML lab services

```bash
# from amalitechmllab/
docker compose up api mlflow-ui streamlit
```

### 3. Start Airflow lab

```bash
# from amalitechairflowdbtlab/
docker compose up
```

### 4. Call ML API from Airflow DAG

Inside any Airflow DAG or operator, use the container name as hostname:

```python
import requests

response = requests.post(
    "http://mllab-api:8000/predict",   # resolved via amalitech-net
    json={...}
)
```

> **Note:** The `mllab-api` hostname is the Docker Compose service name `api` — prefix with the compose project name if needed (`amalitechmllab-api-1`). Use `docker network inspect amalitech-net` to verify container names.

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
