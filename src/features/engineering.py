"""Step 3 — Feature Engineering (functional style)."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.logging import get_logger

logger = get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

TARGET: str = "fare"

# Full airport name columns — redundant once we have the IATA code column
REDUNDANT_COLUMNS: tuple[str, ...] = ("source_name", "destination_name")

CATEGORICAL_COLUMNS: tuple[str, ...] = (
    "airline",
    "source",
    "destination",
    "aircraft_type",
    "travel_class",
    "booking_source",
    "seasonality",
)

NUMERICAL_COLUMNS: tuple[str, ...] = (
    "duration",
    "days_left",
    "stopovers",
    "departure_hour",
    "departure_day_of_week",
    "departure_month",
    "arrival_hour",
    "arrival_day_of_week",
    "arrival_month",
)

_DEFAULT_TEST_SIZE:    float = 0.2
_DEFAULT_VAL_SIZE:     float = 0.1
_DEFAULT_RANDOM_STATE: int   = 42


# ── Feature set (pure value object) ───────────────────────────────────────────

@dataclass(frozen=True)
class FeatureSet:
    X_train:       pd.DataFrame
    X_val:         pd.DataFrame
    X_test:        pd.DataFrame
    y_train:       pd.Series
    y_val:         pd.Series
    y_test:        pd.Series
    scaler:        StandardScaler
    feature_names: tuple[str, ...]


# ── Pure transforms (df → df, no mutation) ────────────────────────────────────

def drop_redundant_columns(
    df: pd.DataFrame, cols: tuple[str, ...]
) -> pd.DataFrame:
    to_drop = [c for c in cols if c in df.columns]
    if to_drop:
        logger.info("Dropping redundant columns: %s", to_drop)
    return df.drop(columns=to_drop)


def one_hot_encode(
    df: pd.DataFrame,
    cols: tuple[str, ...],
    drop_first: bool = False,
) -> pd.DataFrame:
    present = [c for c in cols if c in df.columns]
    absent  = [c for c in cols if c not in df.columns]
    if absent:
        logger.warning("Categorical columns not found — skipping: %s", absent)
    result = pd.get_dummies(df, columns=present, drop_first=drop_first, dtype="uint8")
    # Sort columns for deterministic order across runs
    non_target = sorted(c for c in result.columns if c != TARGET)
    result = result[[*non_target, TARGET]] if TARGET in result.columns else result[non_target]
    new_indicator_cols = len(result.columns) - (len(df.columns) - len(present))
    logger.info(
        "One-hot encoded %d column(s) → %d indicator columns added",
        len(present), new_indicator_cols,
    )
    return result


# ── Split helpers ──────────────────────────────────────────────────────────────

def split_features_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found.")
    return df.drop(columns=[target]), df[target]


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,    pd.Series,    pd.Series]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # val_size is fraction of the *full* dataset, so adjust for the reduced pool
    val_frac_of_trainval = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_frac_of_trainval,
        random_state=random_state,
    )
    logger.info(
        "Split → train=%d  val=%d  test=%d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ── Scaler — fit on train only, apply to all splits ───────────────────────────

def fit_and_scale(
    X_train: pd.DataFrame,
    X_val:   pd.DataFrame,
    X_test:  pd.DataFrame,
    num_cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    present = [c for c in num_cols if c in X_train.columns]
    scaler  = StandardScaler()
    scaler.fit(X_train[present])

    def _apply(X: pd.DataFrame) -> pd.DataFrame:
        scaled = scaler.transform(X[present])
        return X.assign(**{col: scaled[:, i] for i, col in enumerate(present)})

    logger.info(
        "StandardScaler fitted on train set — %d numerical column(s): %s",
        len(present), present,
    )
    return _apply(X_train), _apply(X_val), _apply(X_test), scaler


# ── Orchestrator ───────────────────────────────────────────────────────────────

def engineer(df: pd.DataFrame, cfg: dict[str, Any]) -> FeatureSet:
    """Run full feature engineering pipeline. Returns immutable FeatureSet."""
    data_cfg     = cfg.get("data",     {})
    features_cfg = cfg.get("features", {})

    # all column lists are config-authoritative; module constants are fallbacks
    target       = str(features_cfg.get("target",     TARGET))
    redundant    = tuple(features_cfg.get("redundant",    list(REDUNDANT_COLUMNS)))
    categorical  = tuple(features_cfg.get("categorical",  list(CATEGORICAL_COLUMNS)))
    numerical    = tuple(features_cfg.get("numerical",    list(NUMERICAL_COLUMNS)))
    test_size    = float(data_cfg.get("test_size",    _DEFAULT_TEST_SIZE))
    val_size     = float(data_cfg.get("val_size",     _DEFAULT_VAL_SIZE))
    random_state = int(data_cfg.get("random_state",   _DEFAULT_RANDOM_STATE))

    # 1 — drop verbose text columns redundant with IATA codes
    df = drop_redundant_columns(df, redundant)

    # 2 — one-hot encode categorical columns
    df = one_hot_encode(df, categorical)

    # 3 — separate features from target
    X, y = split_features_target(df, target)
    feature_names = tuple(X.columns)
    logger.info("Feature matrix: %d rows × %d features", *X.shape)

    # 4 — stratified-free random split into train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, test_size, val_size, random_state
    )

    # 5 — fit scaler on train, apply to all three splits (no leakage)
    X_train, X_val, X_test, scaler = fit_and_scale(
        X_train, X_val, X_test, numerical
    )

    return FeatureSet(
        X_train=X_train, X_val=X_val,   X_test=X_test,
        y_train=y_train, y_val=y_val,   y_test=y_test,
        scaler=scaler,
        feature_names=feature_names,
    )


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_feature_set(fset: FeatureSet, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    fset.X_train.to_parquet(out / "X_train.parquet", index=False)
    fset.X_val.to_parquet(  out / "X_val.parquet",   index=False)
    fset.X_test.to_parquet( out / "X_test.parquet",  index=False)
    pd.DataFrame({"fare": fset.y_train}).to_parquet(out / "y_train.parquet", index=False)
    pd.DataFrame({"fare": fset.y_val}).to_parquet(  out / "y_val.parquet",   index=False)
    pd.DataFrame({"fare": fset.y_test}).to_parquet( out / "y_test.parquet",  index=False)

    with open(out / "scaler.pkl", "wb") as fh:
        pickle.dump(fset.scaler, fh)

    logger.info(
        "Saved feature set → %s  (train=%s  val=%s  test=%s)",
        out, fset.X_train.shape, fset.X_val.shape, fset.X_test.shape,
    )


def log_feature_set(fset: FeatureSet) -> None:
    logger.info("=== FEATURE ENGINEERING REPORT ===")
    logger.info("  Total features    : %d", len(fset.feature_names))
    logger.info("  Train             : X=%s  y=%s", fset.X_train.shape, fset.y_train.shape)
    logger.info("  Val               : X=%s  y=%s", fset.X_val.shape,   fset.y_val.shape)
    logger.info("  Test              : X=%s  y=%s", fset.X_test.shape,  fset.y_test.shape)
    logger.info(
        "  Target (train)    : min=%.2f  median=%.2f  max=%.2f",
        fset.y_train.min(), fset.y_train.median(), fset.y_train.max(),
    )
    logger.info(
        "  Scaler columns    : %s",
        fset.scaler.feature_names_in_.tolist(),
    )


# ── Stage entry point ──────────────────────────────────────────────────────────

def run(cfg: dict[str, Any]) -> None:
    processed_path: str = cfg.get("data", {}).get(
        "processed_path", "data/processed/Flight_Price_Dataset_of_Bangladesh.parquet"
    )
    features_dir: str = cfg.get("data", {}).get(
        "features_dir", "data/features"
    )

    logger.info("Loading processed dataset from: %s", processed_path)
    df = pd.read_parquet(processed_path)

    fset = engineer(df, cfg)
    log_feature_set(fset)
    save_feature_set(fset, features_dir)
