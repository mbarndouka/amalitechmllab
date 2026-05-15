# Step 3 — Feature Engineering

**Source:** `src/features/engineering.py`  
**Notebook:** `notebooks/03_feature_engineering.ipynb`  
**Input:** `data/processed/Flight_Price_Dataset_of_Bangladesh.parquet`  
**Output:** `data/features/` (X_train, X_val, X_test, y_train, y_val, y_test parquets + scaler.pkl + route_te_map.pkl)

---

## What we did

We took the cleaned 19-column dataset and transformed it into model-ready feature matrices. Seven steps ran in order:

1. **Drop redundant columns** — remove `source_name` and `destination_name` (full airport names, redundant once IATA code columns exist)
2. **Add route feature** — create `route = source + '_' + destination` before encoding (captures the source-destination interaction)
3. **One-hot encode categoricals** — expand 7 categorical columns into binary indicator columns
4. **Split features from target** — separate `fare` column from feature matrix
5. **Log-transform target** — apply `log1p` to `fare` to reduce right-skew (skewness: 1.58 → −0.17)
6. **Train / val / test split** — 70% / 10% / 20% random split
7. **Log-transform skewed numericals** — apply `log1p` to `duration` and `days_left` before scaling
8. **Target-encode `route`** — replace route strings with mean log-fare per route (train stats only — no leakage)
9. **Fit and apply StandardScaler** — fit on train set only, then apply to val and test

---

## What we found

### Dataset shape change

| Stage | Rows | Columns |
|---|---|---|
| Input (processed) | 57,000 | 19 |
| After drop redundant | 57,000 | 17 |
| After route feature | 57,000 | 18 |
| After OHE | 57,000 | ~130 |
| After target encoding route | 57,000 | ~95 |

One-hot encoding creates many columns from 7 categoricals (airlines, aircraft types, destinations, etc.). Target encoding route then replaces 30+ route indicators with a single numeric column.

### Split sizes

| Split | Rows | Purpose |
|---|---|---|
| Train | ~39,900 (70%) | Fit models and scalers |
| Val | ~5,700 (10%) | Early stopping, hyperparameter selection |
| Test | ~11,400 (20%) | Final unbiased evaluation |

### Log-transforming the target

Fare has a right-skewed distribution — most flights cost BDT 20,000–60,000 but some reach BDT 500,000+. Training a regression on raw BDT values causes the model to optimise heavily for those rare expensive outliers at the expense of the majority.

`log1p` transformation compresses the tail:

| | Raw fare | Log1p fare |
|---|---|---|
| Skewness | 1.58 | −0.17 |
| Min | BDT 1,801 | 7.50 |
| Median | BDT 41,308 | 10.63 |
| Max | BDT 558,987 | 13.23 |

Models train and predict in log-space. Predictions are inverse-transformed with `expm1` before reporting BDT metrics.

### Route target encoding

`route` (source + destination) had 100+ unique values. One-hot encoding would create 100+ sparse binary columns that tree models find hard to split on. Instead we replaced each route string with the mean log-fare for that route computed on the **train set only**:

- Known routes → mapped to their mean fare
- Unseen routes (val/test not in train) → fall back to global train mean

This captures the route price signal in one dense numeric column with no leakage.

### StandardScaler — train only

`StandardScaler` is fit on the **train set only**, then applied to val and test. Fitting on all data would leak test-set mean and variance into the scaler, giving the model optimistically scaled test inputs.

Scaled columns: `duration`, `days_left`, `stopovers`, `departure_hour`, `departure_day_of_week`, `departure_month`, `arrival_hour`, `arrival_day_of_week`, `arrival_month`.

---

## Artifacts saved

| File | Contents |
|---|---|
| `data/features/X_train.parquet` | Training feature matrix |
| `data/features/X_val.parquet` | Validation feature matrix |
| `data/features/X_test.parquet` | Test feature matrix |
| `data/features/y_train.parquet` | Training labels (log-transformed fare) |
| `data/features/y_val.parquet` | Validation labels |
| `data/features/y_test.parquet` | Test labels |
| `data/features/scaler.pkl` | Fitted StandardScaler |
| `data/features/route_te_map.pkl` | Route → mean log-fare mapping (for inference) |

---

## Design decisions

**Route as interaction feature** — Source and destination encoded separately miss the pair interaction. DAC→LHR is expensive; DAC→DEL is cheap. A combined route column captures this directly. Tree models can split on the route signal in one step.

**Target encoding after split** — Target encoding uses the mean of `y` per group. Computing this on the full dataset before splitting would leak label information into the test set. We compute means only from `y_train`, then apply the same mapping to val and test.

**Functional pipeline, immutable output** — Every transform is a pure function `(df) → df`. The final output is a frozen `FeatureSet` dataclass — no mutation after construction. This makes the engineering pipeline easy to test, reproduce, and extend without side effects.

**Column order determinism** — After one-hot encoding, columns are sorted alphabetically. This ensures the same column order across different runs regardless of pandas version or category order.

---

## What comes next (Step 4 — Baseline Model)

Step 4 loads these feature matrices and trains Linear Regression as a benchmark:

- Load `X_train`, `X_val`, `X_test`, `y_train`, `y_val`, `y_test` from parquet
- Fit `LinearRegression` with no hyperparameters
- Evaluate on all three splits (R², RMSE, MAE, MAPE)
- Save model and metrics as baseline for comparison in Step 5
