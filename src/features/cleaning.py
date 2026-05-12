"""Step 2 — Data Cleaning & Preprocessing (functional style)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial, reduce
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

LEAKAGE_COLUMNS: frozenset[str] = frozenset({"Base Fare (BDT)", "Tax & Surcharge (BDT)"})

COLUMN_RENAME_MAP: dict[str, str] = {
    "Airline":                "airline",
    "Source":                 "source",
    "Source Name":            "source_name",
    "Destination":            "destination",
    "Destination Name":       "destination_name",
    "Stopovers":              "stopovers",
    "Aircraft Type":          "aircraft_type",
    "Class":                  "travel_class",
    "Booking Source":         "booking_source",
    "Seasonality":            "seasonality",
    "Duration (hrs)":         "duration",
    "Days Before Departure":  "days_left",
    "Total Fare (BDT)":       "fare",
    "Base Fare (BDT)":        "base_fare",
    "Tax & Surcharge (BDT)":  "tax_surcharge",
    "Departure Date & Time":  "departure_datetime",
    "Arrival Date & Time":    "arrival_datetime",
}

STOPOVER_ORDINAL: dict[str, int] = {
    "Direct":    0,
    "1 Stop":    1,
    "2 Stops":  2,
}

_DATETIME_COLUMNS: tuple[str, ...] = ("departure_datetime", "arrival_datetime")
_TEMPORAL_ATTRS: tuple[tuple[str, str], ...] = (
    ("hour",        "hour"),
    ("day_of_week", "dayofweek"),
    ("month",       "month"),
)

_DEFAULT_MISSINGNESS_THRESHOLD: float = 0.5
_DEFAULT_NUMERICAL_STRATEGY:    str   = "median"
_DEFAULT_CATEGORICAL_FILL:      str   = "Unknown"


# ── Report (pure value object) ─────────────────────────────────────────────────

@dataclass(frozen=True)
class CleaningReport:
    initial_shape:           tuple[int, int]
    final_shape:             tuple[int, int]
    rows_removed:            int
    cols_removed:            int
    temporal_features_added: tuple[str, ...]
    ordinal_encoded:         tuple[str, ...]


# ── Pure transform functions  (df → df, no mutation) ──────────────────────────

def drop_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    unnamed = [c for c in df.columns if re.match(r"^unnamed", c, re.IGNORECASE)]
    if unnamed:
        logger.info("Dropping unnamed columns: %s", unnamed)
    return df.drop(columns=unnamed)


def rename_columns(df: pd.DataFrame, rename_map: dict[str, str]) -> pd.DataFrame:
    applicable = {k: v for k, v in rename_map.items() if k in df.columns}
    logger.info("Renaming %d column(s) to snake_case", len(applicable))
    return df.rename(columns=applicable)


def drop_leakage_columns(
    df: pd.DataFrame, leakage: frozenset[str]
) -> pd.DataFrame:
    to_drop = [c for c in df.columns if c in leakage]
    logger.info("Dropping leakage columns: %s", to_drop)
    return df.drop(columns=to_drop)


def drop_high_missingness_rows(
    df: pd.DataFrame, threshold: float
) -> pd.DataFrame:
    min_valid = int(len(df.columns) * (1.0 - threshold))
    result = df.dropna(thresh=min_valid)
    removed = len(df) - len(result)
    if removed:
        logger.info(
            "Dropped %d row(s) exceeding %.0f%% column missingness",
            removed, threshold * 100,
        )
    return result


def impute_missing(
    df: pd.DataFrame,
    numerical_strategy: str,
    categorical_fill: str,
) -> pd.DataFrame:
    fill_values = {
        col: (
            df[col].median() if numerical_strategy == "median" else df[col].mean()
            if pd.api.types.is_numeric_dtype(df[col])
            else categorical_fill
        )
        for col in df.columns
        if df[col].isna().any()
    }
    if fill_values:
        logger.info("Imputing missing values in: %s", list(fill_values))
        return df.fillna(fill_values)
    logger.info("No missing values — imputation skipped")
    return df


def parse_datetime_columns(
    df: pd.DataFrame, cols: tuple[str, ...]
) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    absent  = [c for c in cols if c not in df.columns]
    if absent:
        logger.warning("Datetime column(s) not found — skipping: %s", absent)
    parsed = {col: pd.to_datetime(df[col], format="mixed") for col in present}
    logger.info("Parsed datetime column(s): %s", present)
    return df.assign(**parsed)


def extract_temporal_features(
    df: pd.DataFrame, cols: tuple[str, ...]
) -> pd.DataFrame:
    datetime_cols = [
        c for c in cols
        if c in df.columns and pd.api.types.is_datetime64_any_dtype(df[c])
    ]
    new_features = {
        f"{col.replace('_datetime', '')}_{feat_name}": (
            getattr(df[col].dt, attr).astype("int8")
        )
        for col in datetime_cols
        for feat_name, attr in _TEMPORAL_ATTRS
    }
    logger.info("Adding temporal features: %s", list(new_features))
    return df.assign(**new_features).drop(columns=datetime_cols)


def encode_stopovers(
    df: pd.DataFrame, col: str, ordinal_map: dict[str, int]
) -> pd.DataFrame:
    if col not in df.columns:
        return df
    unknown = set(df[col].unique()) - set(ordinal_map)
    if unknown:
        raise ValueError(
            f"Unexpected values in '{col}': {unknown}. "
            f"Expected one of {set(ordinal_map)}. Update STOPOVER_ORDINAL."
        )
    logger.info("Ordinal-encoding '%s' → %s", col, ordinal_map)
    return df.assign(**{col: df[col].map(ordinal_map).astype("int8")})


# ── Pipeline composition ───────────────────────────────────────────────────────

Transform = Callable[[pd.DataFrame], pd.DataFrame]


def compose(*transforms: Transform) -> Transform:
    """Compose N transforms left-to-right: f1 → f2 → ... → fN."""
    return lambda df: reduce(lambda acc, fn: fn(acc), transforms, df)


def build_pipeline(cfg: dict[str, Any]) -> Transform:
    cleaning_cfg  = cfg.get("cleaning", {})
    features_cfg  = cfg.get("features", {})

    # leakage columns — config is authoritative, module constant is fallback
    raw_leakage: list[str] = cleaning_cfg.get("leakage_columns", list(LEAKAGE_COLUMNS))
    leakage_renamed = frozenset(COLUMN_RENAME_MAP.get(c, c) for c in raw_leakage)

    # stopovers ordinal — ordered list in config → {value: index} map
    stopovers_order: list[str] = features_cfg.get("ordinal", {}).get(
        "stopovers", list(STOPOVER_ORDINAL)
    )
    stopovers_map: dict[str, int] = {v: i for i, v in enumerate(stopovers_order)}

    return compose(
        drop_unnamed_columns,
        partial(rename_columns,              rename_map=COLUMN_RENAME_MAP),
        partial(drop_leakage_columns,        leakage=leakage_renamed),
        partial(drop_high_missingness_rows,  threshold=cleaning_cfg.get(
            "missingness_drop_threshold", _DEFAULT_MISSINGNESS_THRESHOLD
        )),
        partial(impute_missing,
            numerical_strategy=cleaning_cfg.get(
                "numerical_impute_strategy", _DEFAULT_NUMERICAL_STRATEGY
            ),
            categorical_fill=cleaning_cfg.get(
                "categorical_impute_value", _DEFAULT_CATEGORICAL_FILL
            ),
        ),
        partial(parse_datetime_columns,      cols=_DATETIME_COLUMNS),
        partial(extract_temporal_features,   cols=_DATETIME_COLUMNS),
        partial(encode_stopovers,            col="stopovers", ordinal_map=stopovers_map),
    )


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, CleaningReport]:
    initial_shape = df.shape

    temporal_names = tuple(
        f"{col.replace('_datetime', '')}_{feat}"
        for col in _DATETIME_COLUMNS
        for feat, _ in _TEMPORAL_ATTRS
    )

    pipeline = build_pipeline(cfg)
    result   = pipeline(df)

    report = CleaningReport(
        initial_shape           = initial_shape,
        final_shape             = result.shape,
        rows_removed            = initial_shape[0] - result.shape[0],
        cols_removed            = initial_shape[1] - result.shape[1],
        temporal_features_added = temporal_names,
        ordinal_encoded         = ("stopovers",),
    )
    return result, report


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved → %s  (%d rows × %d cols)", path, *df.shape)


def log_report(report: CleaningReport) -> None:
    logger.info("=== CLEANING REPORT ===")
    logger.info("  Initial shape      : %d × %d", *report.initial_shape)
    logger.info("  Final shape        : %d × %d", *report.final_shape)
    logger.info("  Rows removed       : %d",        report.rows_removed)
    logger.info("  Cols removed       : %d",        report.cols_removed)
    logger.info("  Temporal features  : %s",        list(report.temporal_features_added))
    logger.info("  Ordinal encoded    : %s",        list(report.ordinal_encoded))


# ── Stage entry point ──────────────────────────────────────────────────────────

def run(cfg: dict[str, Any]) -> None:
    from features.exploration import load_data

    raw_path:       str = cfg.get("data", {}).get(
        "raw_path",       "data/raw/Flight_Price_Dataset_of_Bangladesh.csv"
    )
    processed_path: str = cfg.get("data", {}).get(
        "processed_path", "data/processed/Flight_Price_Dataset_of_Bangladesh.parquet"
    )

    logger.info("Loading raw dataset from: %s", raw_path)
    df = load_data(raw_path)

    df, report = clean(df, cfg)
    log_report(report)
    save_processed(df, processed_path)
