"""Step 1 — Data Understanding: load, inspect, and profile the raw dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


def load_data(raw_path: str | Path) -> pd.DataFrame:
    """Load the raw data from *raw_path* and return as a DataFrame."""
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data not found at {path}")
    return pd.read_csv(path)


def inspect_structure(df: pd.DataFrame) -> None:
    """Print a summary of the DataFrame structure."""
    logger.info("=== HEAD (5 rows) ===")
    logger.info("\n%s", df.head(5).to_string())

    logger.info("=== SHAPE ===")
    logger.info("Rows: %d  |  Columns: %d", *df.shape)

    logger.info("=== DATA TYPES & NON-NULL COUNTS ===")
    buf = []
    for col in df.columns:
        buf.append(f"  {col:<45} dtype={str(df[col].dtype):<12} non-null={df[col].notna().sum()}")
    logger.info("\n%s", "\n".join(buf))

    logger.info("=== NUMERICAL SUMMARY (.describe()) ===")
    logger.info("\n%s", df.describe(include="number").round(2).to_string())


def audit_quality(df: pd.DataFrame) -> None:
    """Perform basic quality checks on the DataFrame."""
    logger.info("=== MISSING VALUES ===")
    missing_values = df.isna().sum()
    missing_pct = (missing_values / len(df) * 100).round(2)
    summary = pd.DataFrame({"missing_count": missing_values, "missing_%": missing_pct})
    summary = summary[summary["missing_count"] > 0]
    if summary.empty:
        logger.info("  No missing values found.")
    else:
        logger.info("\n%s", summary.to_string())

    dupes = int(df.duplicated().sum())
    logger.info("=== DUPLICATE ROWS ===")
    logger.info("  Duplicate rows: %d  (%.2f%%)", dupes, dupes / len(df) * 100)


def _is_datetime(series: pd.Series) -> bool:
    sample = series.dropna().head(10)
    try:
        pd.to_datetime(sample, format="mixed")
        return True
    except (ValueError, TypeError):
        return False


def profile_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    numerical: list[str] = df.select_dtypes(include="number").columns.tolist()
    object_cols: list[str] = df.select_dtypes(include="object").columns.tolist()
    datetime_cols: list[str] = [c for c in object_cols if _is_datetime(df[c])]
    categorical: list[str] = [c for c in object_cols if c not in datetime_cols]

    classification: dict[str, list[str]] = {
        "numerical": numerical,
        "categorical": categorical,
        "datetime": datetime_cols,
    }

    logger.info("=== COLUMN CLASSIFICATION ===")
    for kind, cols in classification.items():
        logger.info("  %-15s %s", kind + ":", cols)

    logger.info("=== CATEGORICAL COLUMNS — cardinality & top value ===")
    for col in categorical:
        vc = df[col].value_counts()
        logger.info(
            "  %-42s  unique=%-5d  top='%s' (%d rows)",
            col, len(vc), vc.index[0], vc.iloc[0],
        )

    logger.info("=== DATETIME-LIKE COLUMNS — sample value ===")
    for col in datetime_cols:
        logger.info("  %-42s  e.g. '%s'", col, df[col].iloc[0])

    logger.info("=== NUMERICAL COLUMNS — min / max / skew / IQR outliers ===")
    for col in numerical:
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        logger.info(
            "  %-42s  min=%-12.2f  max=%-12.2f  skew=%-8.2f  outliers=%d (%.1f%%)",
            col, float(s.min()), float(s.max()), float(s.skew()),
            outliers, outliers / len(s) * 100,
        )

    return classification


def summarise_observations(
    df: pd.DataFrame,
    classification: dict[str, list[str]],
) -> None:
    """Print the structured Initial Observations summary block."""
    target = "Total Fare (BDT)"
    logger.info("=== INITIAL OBSERVATIONS ===")
    logger.info("  Rows x Columns      : %d x %d", *df.shape)
    logger.info("  Target variable     : %s", target)
    if target in df.columns:
        t = df[target]
        logger.info(
            "  Target range        : %.2f – %.2f  |  mean=%.2f  median=%.2f",
            t.min(), t.max(), t.mean(), t.median(),
        )
    logger.info("  Numerical features  : %s", classification["numerical"])
    logger.info("  Categorical features: %s", classification["categorical"])
    logger.info("  Datetime columns    : %s", classification["datetime"])
    logger.info("  Total missing cells : %d", int(df.isnull().sum().sum()))
    logger.info("  Duplicate rows      : %d", int(df.duplicated().sum()))
    logger.info("")
    logger.info("  Assumptions & Limitations")
    logger.info("  ──────────────────────────────────────────────────────────")
    logger.info("  • Target is 'Total Fare (BDT)' = Base Fare + Tax & Surcharge.")
    logger.info("  • 'Base Fare' and 'Tax & Surcharge' are sub-components of the")
    logger.info("    target and must be dropped before training (data leakage).")
    logger.info("  • Datetime columns must be parsed to extract temporal features")
    logger.info("    (departure hour, day-of-week, month, days-until-departure).")
    logger.info("  • 'Stopovers' is ordinal: Direct < 1 Stop < 2+ Stops.")
    logger.info("  • Dataset covers Bangladesh-origin routes only.")
    logger.info("  • All monetary values are in BDT (Bangladeshi Taka).")


def run(cfg: dict[str, Any]) -> None:
    """Stage entry point — orchestrates all exploration sub-tasks."""
    raw_path: str = cfg.get("data", {}).get(
        "raw_path", "data/raw/Flight_Price_Dataset_of_Bangladesh.csv"
    )
    logger.info("Loading raw dataset from: %s", raw_path)
    df = load_data(raw_path)

    inspect_structure(df)
    audit_quality(df)
    classification = profile_columns(df)
    summarise_observations(df, classification)
