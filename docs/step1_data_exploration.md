# Step 1 — Data Exploration

**Source:** `src/features/exploration.py`  
**Notebook:** `notebooks/01_data_exploration.ipynb`  
**Dataset:** `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`

---

## What we did

We loaded the raw dataset and inspected it before touching any values. Four checks were run in order:

1. **Load & inspect** — shape, column types, first rows, numerical summary
2. **Quality audit** — missing values, duplicate rows
3. **Column profiling** — classify every column, check cardinality and outliers
4. **Initial observations** — summarise findings and flag risks for modelling

---

## What we found

### The dataset
- **57,000 rows, 17 columns.** No missing values, no duplicate rows. Clean start.
- Covers flights from Bangladesh-origin airports to international destinations.

### Target variable
- **`Total Fare (BDT)`** is what we want to predict.
- Range: BDT 1,801 → 558,987. Mean BDT 71,030, median BDT 41,308.
- Right-skewed — a few very expensive flights pull the mean up.

### Column types

| Type | Columns |
|------|---------|
| Numerical (3 usable) | `Duration (hrs)`, `Days Before Departure` |
| Numerical (leakage — drop) | `Base Fare (BDT)`, `Tax & Surcharge (BDT)` |
| Categorical (10) | `Airline`, `Source`, `Source Name`, `Destination`, `Destination Name`, `Stopovers`, `Aircraft Type`, `Class`, `Booking Source`, `Seasonality` |
| Datetime (2 — need parsing) | `Departure Date & Time`, `Arrival Date & Time` |

### Notable findings

**Data leakage risk**  
`Base Fare (BDT)` + `Tax & Surcharge (BDT)` add up to `Total Fare (BDT)`. They must be dropped before training — the model would learn a trivial sum, not a real pattern.

**Datetime columns not yet converted**  
Both date columns are stored as strings (`"2025-11-17 06:25:00"`). They need to be parsed with `pd.to_datetime()` in the next step so we can extract features like departure hour, day of week, and month.

**Outliers in fare and duration**  
- `Duration (hrs)`: 10.5% IQR outliers (long-haul routes)
- `Total Fare (BDT)`: 6.0% IQR outliers (business/first class, long-haul)
- `Days Before Departure`: perfectly uniform 1–90, zero outliers

**`Stopovers` is ordinal**  
Three values: `Direct`, `1 Stop`, `2+ Stops`. Order matters — treat as ordinal, not nominal.

**Most common values**
- Most flights: Economy class, Direct routing, US-Bangla Airlines
- Top departure airport: Chittagong (CGP), top destination: Jeddah (JED)
- 78% of flights tagged as `Regular` season

---

## What comes next (Step 2 — Feature Engineering)

- Parse `Departure Date & Time` and `Arrival Date & Time` → extract hour, day-of-week, month
- Drop `Base Fare (BDT)` and `Tax & Surcharge (BDT)` (data leakage)
- Encode `Stopovers` as ordinal integer (0, 1, 2)
- One-hot or target-encode remaining categorical columns
