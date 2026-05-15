# Step 4 — Exploratory Data Analysis (EDA)

**Source:** `src/features/eda.py`  
**Notebook:** `notebooks/04_eda.ipynb`  
**Input:** `data/processed/Flight_Price_Dataset_of_Bangladesh.parquet`

---

## What we did

EDA runs on the **cleaned, processed dataset** (after Step 2, before model training). It answers three questions:

1. **How does fare vary by categorical segment?** — airline, route, season, travel class, booking source, stopovers
2. **Which numerical features correlate with fare?** — Pearson correlation matrix + bar charts
3. **What are the headline business insights?** — the key patterns that stakeholders need to understand

This step is **read-only** — it produces reports and plots but does not transform data.

---

## What we found

### Fare by airline

| Airline | Median fare (BDT) | Notes |
|---|---|---|
| Cathay Pacific | 46,282 | Most expensive |
| Biman Bangladesh | ~42,000 | National carrier, premium pricing |
| US-Bangla Airlines | ~28,000 | Budget domestic/regional carrier |
| Singapore Airlines | 38,441 | Cheapest among full-service international |

Airlines form two tiers: full-service international carriers (Cathay, Singapore, Emirates) with premiums of 20–50% over budget carriers.

### Fare by season

| Season | Median fare (BDT) | vs Regular |
|---|---|---|
| Eid | 56,621 | +42% premium |
| Hajj | ~50,000 | +25% |
| Winter Holidays | ~44,000 | +10% |
| Regular | ~40,000 | baseline |

Eid is the peak season. Fares spike 42% above regular — demand from diaspora travel home creates a supply constraint.

### Fare by travel class

| Class | Median fare (BDT) | Ratio to Economy |
|---|---|---|
| First Class | 94,191 | 3.9× |
| Business | 51,062 | 2.1× |
| Economy | 24,123 | 1.0× (baseline) |

### Fare by stopovers

Direct flights are cheaper on average, but the relationship is non-linear. A single stop often reduces price (lower-demand routing), but 2+ stops with layovers can increase total fare on certain routes.

### Fare by booking source

| Booking source | Median fare (BDT) |
|---|---|
| Direct booking | Slightly higher |
| Online travel agency | Mid-range |
| Travel agency | Lowest |

Difference is small — booking source is a weak predictor compared to route and class.

### Numerical correlations with fare

| Feature | Pearson r with fare |
|---|---|
| `duration` | +0.35 (strongest continuous predictor) |
| `days_left` | −0.18 (book earlier → cheaper) |
| `stopovers` | −0.08 (weak negative) |
| `departure_month` | ~0.05 (very weak) |

**Duration** is the strongest continuous driver — longer flights inherently cost more (fuel, distance). **Days left** shows a consistent negative correlation — last-minute bookings cost more on average.

### Top 10 most expensive routes

Routes from Chittagong (CXB) and Sylhet (ZYL) to Southeast Asian hubs (Bangkok, Kuala Lumpur) appear at the top. International long-haul routes (DAC→LHR, DAC→JFK) also rank high.

### Top 10 cheapest routes

Domestic routes dominate: Barisal (BZL)↔Jessore (JSR), Sylhet (ZYL)↔Chittagong (CXB). These are short regional hops with high competition and low demand.

---

## Design decisions

**Median, not mean, for group comparisons** — Flight fares are right-skewed. A few first-class or business fares pull the mean upward, making it a misleading representative of "typical" price. We use median throughout EDA for honest comparisons.

**Seasons ordered by peak intensity** — Displaying seasons alphabetically (Eid, Hajj, Regular, Winter) hides the story. We order them peak→off-peak (Eid, Hajj, Winter Holidays, Regular) so the seasonal premium pattern is immediately visible.

**Correlation computed on processed data** — EDA uses the processed parquet (after renaming, leakage removal, temporal extraction). This ensures we're computing correlations on the actual features the model will see, not on raw string columns.

**EDA is non-destructive** — All outputs are written to `reports/`. No data files are modified. EDA can be re-run at any point without affecting the pipeline.

---

## Outputs

| File | Contents |
|---|---|
| `reports/insights.json` | Structured price breakdowns by airline, season, class, route, booking source, days_left buckets |
| `reports/stakeholder_report.txt` | Human-readable business insights summary |

---

## What comes next (Step 5 — Baseline Model)

The EDA findings inform model selection and evaluation strategy:

- Right-skewed fare → use `log1p` target transform (already done in Step 3)
- Route and travel class are strong predictors → include as features
- Duration is the strongest continuous predictor → keep as-is, do not discard
- Seasonality shows clear signal → include; tree models will learn the Eid premium
