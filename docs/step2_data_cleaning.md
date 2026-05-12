# Step 2 — Data Cleaning, Preprocessing & Initial Feature Engineering

**Source:** `src/features/cleaning.py`  
**Notebook:** `notebooks/02_data_cleaning.ipynb`  
**Input:** `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`  
**Output:** `data/processed/Flight_Price_Dataset_of_Bangladesh.parquet`

---

## What we did

We loaded the raw dataset and applied eight steps in order, using a functional pipeline where every step returns a new DataFrame — nothing is modified in place.

This step covers both **cleaning** (steps 1–5) and **initial feature engineering** (steps 6–8). The line between the two is intentional: temporal extraction and ordinal encoding depend on the raw column values, so they belong here before anything is split or scaled.

**Cleaning**
1. **Drop unnamed columns** — remove any index-artifact columns pandas adds when a CSV is saved with `index=True`
2. **Rename columns** — convert all column names to `snake_case` (e.g. `Total Fare (BDT)` → `fare`)
3. **Drop leakage columns** — remove `base_fare` and `tax_surcharge` before any modelling
4. **Drop high-missingness rows** — remove rows where more than 50% of columns are empty
5. **Impute missing values** — fill numerical gaps with the median, categorical gaps with `"Unknown"`

**Feature Engineering**
6. **Parse datetime columns** — convert `departure_datetime` and `arrival_datetime` from strings to real datetime objects
7. **Extract temporal features** — pull hour, day-of-week, and month from each datetime column, then drop the raw datetime columns
8. **Ordinal-encode Stopovers** — convert `Direct / 1 Stop / 2 Stops` to `0 / 1 / 2`

---

## What we found

### Shape change
- **Before:** 57,000 rows × 17 columns  
- **After:** 57,000 rows × 19 columns

Net +2 columns: leakage columns removed (−2), datetime columns replaced by 6 temporal features (−2 +6).

### Data leakage — why we dropped base_fare and tax_surcharge
`fare = base_fare + tax_surcharge`. If we kept these two columns, any model would simply learn to add them together instead of learning real flight-price patterns. They were dropped immediately after renaming.

### Missing values and duplicates — nothing to fix
The dataset had zero missing values and zero duplicate rows (confirmed in Step 1). The imputation and row-drop steps ran but made no changes. They exist to make the pipeline safe on future data.

### Datetime parsing
Both datetime columns were plain strings in the raw data. After parsing, we extracted:

| New column | What it means |
|---|---|
| `departure_hour` | Hour of departure (0–23) |
| `departure_day_of_week` | Day of week (0=Monday … 6=Sunday) |
| `departure_month` | Month (1–12) |
| `arrival_hour` | Hour of arrival (0–23) |
| `arrival_day_of_week` | Day of week of arrival |
| `arrival_month` | Month of arrival |

All six are stored as `int8` to save memory.

### Stopovers ordinal encoding
`Stopovers` had three string values. Order matters (more stops = longer trip), so we encoded it as an integer instead of one-hot:

| Original value | Encoded value |
|---|---|
| Direct | 0 |
| 1 Stop | 1 |
| 2 Stops | 2 |

### Final column list (19 columns)

| Column | Type | Notes |
|---|---|---|
| `airline` | string | carrier name |
| `source` | string | departure airport IATA code |
| `source_name` | string | full airport name (dropped in Step 3) |
| `destination` | string | arrival airport IATA code |
| `destination_name` | string | full airport name (dropped in Step 3) |
| `duration` | float64 | flight duration in hours |
| `stopovers` | int8 | ordinal: 0=Direct, 1=1 Stop, 2=2 Stops |
| `aircraft_type` | string | e.g. Airbus A320 |
| `travel_class` | string | Economy / Business / First Class |
| `booking_source` | string | Online / Travel Agency / Direct |
| `fare` | float64 | target — total fare in BDT |
| `seasonality` | string | Regular / Summer / Winter Holidays / Eid |
| `days_left` | int64 | days between booking and departure |
| `departure_hour` | int8 | extracted from departure datetime |
| `departure_day_of_week` | int8 | extracted from departure datetime |
| `departure_month` | int8 | extracted from departure datetime |
| `arrival_hour` | int8 | extracted from arrival datetime |
| `arrival_day_of_week` | int8 | extracted from arrival datetime |
| `arrival_month` | int8 | extracted from arrival datetime |

---

## Design decisions

**Functional pipeline** — each transform is a pure function `(df) → df` with no side effects. All transforms are composed into one function using `functools.reduce` and run in a single pass. This makes the pipeline easy to test, reorder, or extend.

**Config-driven** — leakage columns and stopovers ordinal order live in `configs/config.toml`, not hardcoded. Change them in one place and the pipeline adapts.

**Scaler not applied here** — `StandardScaler` is fit and applied in Step 3, after the train/test split. Scaling before splitting would leak test-set statistics into training.

---

## What comes next (Step 3 — Encoding, Scaling & Splitting)

Step 3 handles the remaining transformations that require a train/test split to be done safely:

- Drop `source_name` and `destination_name` (verbose, redundant with IATA codes)
- One-hot encode the 7 remaining categorical columns (`airline`, `source`, `destination`, etc.)
- Split into train (70%) / val (10%) / test (20%)
- Fit `StandardScaler` on the **train set only**, then apply to val and test — scaling before the split would leak test-set statistics into training
