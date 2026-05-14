"""Step 3 — Descriptive Statistics: fare summaries by group and feature correlations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import seaborn as sns

from utils.logging import get_logger

_THEME = {"style": "whitegrid", "palette": "muted", "font_scale": 1.05}

logger = get_logger(__name__)

_BAR_WIDTH = 20  # character width of the inline correlation bar


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_processed(processed_path: str | Path) -> pd.DataFrame:
    path = Path(processed_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded processed data  shape=%d×%d  path=%s", *df.shape, path)
    return df


# ---------------------------------------------------------------------------
# Fare summaries by categorical group
# ---------------------------------------------------------------------------


def _fare_summary_by_group(
    df: pd.DataFrame,
    group_col: str,
    target: str,
) -> pd.DataFrame:
    """Compute per-group fare statistics for *group_col*."""
    overall_mean = df[target].mean()
    summary = (
        df.groupby(group_col, observed=True)[target]
        .agg(
            count="count",
            mean="mean",
            median="median",
            std="std",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            min="min",
            max="max",
        )
        .round(2)
    )
    summary["pct_vs_avg"] = ((summary["mean"] - overall_mean) / overall_mean * 100).round(1)
    return summary.sort_values("mean", ascending=False)


def _log_group_summary(summary: pd.DataFrame, group_col: str) -> None:
    width = max(len(str(idx)) for idx in summary.index)
    header = (
        f"  {'Group':<{width}}  {'N':>6}  {'Mean':>10}  {'Median':>10}"
        f"  {'Std':>9}  {'P25':>10}  {'P75':>10}  {'Min':>10}  {'Max':>10}  {'%vsAvg':>7}"
    )
    separator = "  " + "-" * (len(header) - 2)

    logger.info("=== FARE SUMMARY BY %s ===", group_col.upper())
    logger.info(header)
    logger.info(separator)
    for group, row in summary.iterrows():
        sign = "+" if row["pct_vs_avg"] >= 0 else ""
        logger.info(
            "  %-*s  %6d  %10.2f  %10.2f  %9.2f  %10.2f  %10.2f  %10.2f  %10.2f  %s%.1f%%",
            width,
            group,
            int(row["count"]),
            row["mean"],
            row["median"],
            row["std"],
            row["p25"],
            row["p75"],
            row["min"],
            row["max"],
            sign,
            row["pct_vs_avg"],
        )


def summarise_fares_by_group(
    df: pd.DataFrame,
    group_cols: list[str],
    target: str = "fare",
) -> dict[str, pd.DataFrame]:
    """Compute and log fare summaries for each group column in *group_cols*."""
    results: dict[str, pd.DataFrame] = {}
    missing = [c for c in group_cols if c not in df.columns]
    if missing:
        logger.warning("Columns not found in DataFrame — skipping: %s", missing)

    for col in group_cols:
        if col not in df.columns:
            continue
        summary = _fare_summary_by_group(df, col, target)
        _log_group_summary(summary, col)
        results[col] = summary

    return results


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def _corr_bar(r: float) -> str:
    filled = round(abs(r) * _BAR_WIDTH)
    return ("+" if r >= 0 else "-") + "█" * filled + "░" * (_BAR_WIDTH - filled)


def correlation_matrix(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target: str = "fare",
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute and log the correlation matrix for *numerical_cols* + *target*."""
    cols = [c for c in [*numerical_cols, target] if c in df.columns]
    missing = set(numerical_cols) - set(df.columns)
    if missing:
        logger.warning("Numerical columns not in DataFrame — skipping: %s", sorted(missing))

    corr = df[cols].corr(method=method).round(3)

    logger.info("=== CORRELATION MATRIX (%s) — %d features ===", method.upper(), len(cols))
    logger.info("\n%s", corr.to_string())

    if target in corr.columns:
        target_corr = corr[target].drop(target, errors="ignore").sort_values(key=abs, ascending=False)
        logger.info(
            "=== FEATURE → '%s' CORRELATIONS  (method=%s, sorted by |r|) ===",
            target,
            method,
        )
        logger.info(
            "  %-32s  %7s  %s",
            "Feature",
            "r",
            f"{'bar':^{_BAR_WIDTH + 1}}",
        )
        logger.info("  " + "-" * (32 + 2 + 7 + 2 + _BAR_WIDTH + 1))
        for feat, r in target_corr.items():
            logger.info("  %-32s  %+7.3f  %s", feat, r, _corr_bar(r))

    return corr


def correlation_heatmap_data(corr: pd.DataFrame) -> dict[str, Any]:
    """Return corr matrix as a JSON-serialisable dict (for downstream reporting)."""
    return {
        "columns": corr.columns.tolist(),
        "index": corr.index.tolist(),
        "values": corr.values.tolist(),
    }


# ---------------------------------------------------------------------------
# Visual analysis — plotting functions (notebook use; not called by run())
# ---------------------------------------------------------------------------

_MONTH_NAMES: dict[int, str] = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def plot_fare_distributions(
    raw_path: str | Path,
    fare_cols: dict[str, str] | None = None,
) -> None:
    """Histogram + KDE and log-scale histogram for each fare component in *raw_path*."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sns.set_theme(**_THEME)

    if fare_cols is None:
        fare_cols = {
            "Total Fare (BDT)": "#2ecc71",
            "Base Fare (BDT)": "#3498db",
            "Tax & Surcharge (BDT)": "#e74c3c",
        }

    df_raw = pd.read_csv(raw_path)
    n = len(fare_cols)
    palette = sns.color_palette("husl", n)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    fig.patch.set_facecolor("white")

    for row_idx, (col, _) in enumerate(fare_cols.items()):
        s = df_raw[col].dropna()
        color = palette[row_idx]
        fmt = mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k")

        ax_hist = axes[row_idx, 0]
        sns.histplot(s, bins=80, kde=True, color=color, alpha=0.75, ax=ax_hist, stat="density")
        ax_hist.set_title(f"{col} — distribution")
        ax_hist.set_xlabel("BDT")
        ax_hist.set_ylabel("Density")
        ax_hist.xaxis.set_major_formatter(fmt)
        ax_hist.text(
            0.97,
            0.95,
            f"mean={s.mean() / 1000:.1f}k\nmedian={s.median() / 1000:.1f}k\nskew={s.skew():.2f}",
            transform=ax_hist.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

        ax_log = axes[row_idx, 1]
        sns.histplot(s, bins=80, kde=True, color=color, alpha=0.75, ax=ax_log)
        ax_log.set_yscale("log")
        ax_log.set_title(f"{col} — log-scale count")
        ax_log.set_xlabel("BDT")
        ax_log.set_ylabel("Count (log)")
        ax_log.xaxis.set_major_formatter(fmt)

    sns.despine()
    plt.suptitle("Fare Component Distributions (raw data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_airline_boxplot(
    df: pd.DataFrame,
    target: str = "fare",
    group_col: str = "airline",
) -> None:
    """Horizontal boxplot of fare across airlines, sorted by median."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sns.set_theme(**_THEME)

    order = df.groupby(group_col, observed=True)[target].median().sort_values().index.tolist()

    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=df,
        y=group_col,
        x=target,
        order=order,
        ax=ax,
        orient="h",
        palette="husl",
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        linewidth=0.8,
    )
    ax.axvline(df[target].mean(), color="red", linestyle="--", linewidth=1.2, label="Overall mean")
    ax.axvline(df[target].median(), color="steelblue", linestyle=":", linewidth=1.2, label="Overall median")
    ax.set_xlabel("Fare (BDT)")
    ax.set_ylabel("")
    ax.set_title(f"Fare Variation Across {group_col.capitalize()}s (sorted by median)", fontsize=13)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_avg_fare_by_time(
    df: pd.DataFrame,
    target: str = "fare",
    month_col: str = "departure_month",
    season_col: str = "seasonality",
) -> None:
    """Line chart (avg fare by month) + bar chart (avg fare by season)."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sns.set_theme(**_THEME)

    fmt = mticker.FuncFormatter(lambda y, _: f"{y / 1000:.0f}k")

    monthly = df.groupby(month_col, observed=True)[target].agg(["mean", "median", "std"]).rename(index=_MONTH_NAMES)
    seasonal = (
        df.groupby(season_col, observed=True)[target]
        .agg(["mean", "median", "std"])
        .sort_values("mean", ascending=False)
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor("white")

    monthly_plot = monthly.reset_index().rename(columns={"index": "month", month_col: "month"})
    monthly_plot["x"] = range(len(monthly_plot))

    sns.lineplot(
        x=monthly_plot["x"],
        y=monthly_plot["mean"],
        marker="o",
        color="#e74c3c",
        label="Mean",
        linewidth=2,
        ax=axes[0],
    )
    sns.lineplot(
        x=monthly_plot["x"],
        y=monthly_plot["median"],
        marker="s",
        color="#4C72B0",
        label="Median",
        linewidth=1.5,
        linestyle="--",
        ax=axes[0],
    )
    axes[0].fill_between(
        monthly_plot["x"],
        monthly_plot["mean"] - monthly_plot["std"],
        monthly_plot["mean"] + monthly_plot["std"],
        alpha=0.15,
        color="#e74c3c",
        label="±1 std",
    )
    axes[0].set_xticks(list(monthly_plot["x"]))
    axes[0].set_xticklabels(monthly.index)
    axes[0].set_ylabel("Fare (BDT)")
    axes[0].set_title("Average Fare by Departure Month")
    axes[0].yaxis.set_major_formatter(fmt)
    axes[0].legend(fontsize=9)

    seasonal_reset = seasonal.reset_index()
    n_seasons = len(seasonal_reset)
    season_palette = sns.color_palette("husl", n_seasons)
    sns.barplot(
        x=seasonal_reset[season_col],
        y=seasonal_reset["mean"],
        palette=season_palette,
        alpha=0.85,
        ax=axes[1],
    )
    x_sea = range(n_seasons)
    axes[1].errorbar(
        x=list(x_sea),
        y=seasonal_reset["mean"],
        yerr=seasonal_reset["std"],
        fmt="none",
        elinewidth=1,
        capsize=5,
        capthick=1,
        color="black",
    )
    axes[1].plot(
        x_sea,
        seasonal_reset["median"],
        marker="D",
        color="black",
        linestyle="none",
        markersize=7,
        label="Median",
        zorder=5,
    )
    axes[1].set_xticklabels(seasonal_reset[season_col], rotation=15, ha="right")
    axes[1].set_ylabel("Fare (BDT)")
    axes[1].set_title("Average Fare by Season (mean ± std, median ◆)")
    axes[1].yaxis.set_major_formatter(fmt)
    axes[1].legend(fontsize=9)

    sns.despine()
    plt.suptitle("Average Fare by Month and Season", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_multicollinearity_heatmap(
    df: pd.DataFrame,
    numerical_cols: list[str],
    method: str = "pearson",
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Feature-only correlation heatmap. Returns DataFrame of high-correlation pairs."""
    import matplotlib.pyplot as plt
    import numpy as np

    sns.set_theme(**_THEME)

    feat_cols = [c for c in numerical_cols if c in df.columns]
    feat_corr = df[feat_cols].corr(method=method).round(3)

    mask = np.triu(np.ones_like(feat_corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor("white")
    sns.heatmap(
        feat_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        square=True,
        ax=ax,
        annot_kws={"size": 9},
    )
    ax.set_title(
        f"Feature–Feature Correlation ({method.capitalize()})\n"
        f"Pairs with |r| > {threshold} indicate potential multicollinearity",
        fontsize=12,
    )
    sns.despine()
    plt.tight_layout()
    plt.show()

    high_corr = feat_corr.where(np.tril(np.ones_like(feat_corr, dtype=bool), k=-1)).stack().reset_index()
    high_corr.columns = ["feature_a", "feature_b", "r"]
    high_corr = (
        high_corr[high_corr["r"].abs() > threshold].sort_values("r", key=abs, ascending=False).reset_index(drop=True)
    )

    if high_corr.empty:
        logger.info("No feature pairs with |r| > %.2f — multicollinearity not detected.", threshold)
    else:
        logger.info("High-correlation pairs (|r| > %.2f):", threshold)
        for _, row in high_corr.iterrows():
            logger.info("  %s  ↔  %s  r=%+.3f", row["feature_a"], row["feature_b"], row["r"])

    return high_corr


# ---------------------------------------------------------------------------
# KPI exploration
# ---------------------------------------------------------------------------


def avg_fare_per_airline(
    df: pd.DataFrame,
    target: str = "fare",
    airline_col: str = "airline",
) -> pd.DataFrame:
    """Mean, median, std, and count of fare per airline, sorted by mean descending."""
    overall_mean = df[target].mean()
    result = (
        df.groupby(airline_col, observed=True)[target]
        .agg(count="count", mean="mean", median="median", std="std")
        .round(2)
    )
    result["pct_vs_avg"] = ((result["mean"] - overall_mean) / overall_mean * 100).round(1)
    result = result.sort_values("mean", ascending=False)

    logger.info("=== KPI: AVERAGE FARE PER AIRLINE ===")
    for airline, row in result.iterrows():
        sign = "+" if row["pct_vs_avg"] >= 0 else ""
        logger.info(
            "  %-28s  mean=%9.0f  median=%9.0f  std=%9.0f  n=%5d  %s%.1f%%",
            airline,
            row["mean"],
            row["median"],
            row["std"],
            int(row["count"]),
            sign,
            row["pct_vs_avg"],
        )
    return result


def most_popular_routes(
    df: pd.DataFrame,
    source_col: str = "source",
    destination_col: str = "destination",
    target: str = "fare",
    top_n: int = 10,
) -> pd.DataFrame:
    """Top routes by flight frequency with mean fare per route."""
    result = (
        df.groupby([source_col, destination_col], observed=True)
        .agg(flight_count=(target, "count"), mean_fare=(target, "mean"), median_fare=(target, "median"))
        .round(2)
        .sort_values("flight_count", ascending=False)
        .head(top_n)
        .reset_index()
    )
    result["route"] = result[source_col] + " → " + result[destination_col]

    logger.info("=== KPI: TOP %d MOST POPULAR ROUTES ===", top_n)
    for _, row in result.iterrows():
        logger.info(
            "  %-12s  flights=%5d  mean_fare=%9.0f  median_fare=%9.0f",
            row["route"],
            int(row["flight_count"]),
            row["mean_fare"],
            row["median_fare"],
        )
    return result


def seasonal_fare_variation(
    df: pd.DataFrame,
    target: str = "fare",
    season_col: str = "seasonality",
    baseline: str = "Regular",
) -> pd.DataFrame:
    """Fare stats per season with % premium vs baseline season."""
    result = (
        df.groupby(season_col, observed=True)[target]
        .agg(count="count", mean="mean", median="median", std="std", min="min", max="max")
        .round(2)
    )
    if baseline in result.index:
        base_mean = result.loc[baseline, "mean"]
        result["pct_vs_regular"] = ((result["mean"] - base_mean) / base_mean * 100).round(1)
    else:
        overall_mean = df[target].mean()
        result["pct_vs_regular"] = ((result["mean"] - overall_mean) / overall_mean * 100).round(1)
    result = result.sort_values("mean", ascending=False)

    logger.info("=== KPI: SEASONAL FARE VARIATION ===")
    for season, row in result.iterrows():
        sign = "+" if row["pct_vs_regular"] >= 0 else ""
        logger.info(
            "  %-18s  mean=%9.0f  median=%9.0f  min=%8.0f  max=%9.0f  n=%6d  %s%.1f%% vs %s",
            season,
            row["mean"],
            row["median"],
            row["min"],
            row["max"],
            int(row["count"]),
            sign,
            row["pct_vs_regular"],
            baseline,
        )
    return result


def top_expensive_routes(
    df: pd.DataFrame,
    source_col: str = "source",
    destination_col: str = "destination",
    target: str = "fare",
    top_n: int = 5,
) -> pd.DataFrame:
    """Top N routes by mean fare with full fare statistics."""
    result = (
        df.groupby([source_col, destination_col], observed=True)[target]
        .agg(count="count", mean="mean", median="median", std="std", max="max")
        .round(2)
        .sort_values("mean", ascending=False)
        .head(top_n)
        .reset_index()
    )
    result["route"] = result[source_col] + " → " + result[destination_col]

    logger.info("=== KPI: TOP %d MOST EXPENSIVE ROUTES ===", top_n)
    for _, row in result.iterrows():
        logger.info(
            "  %-12s  mean=%9.0f  median=%9.0f  max=%9.0f  n=%5d",
            row["route"],
            row["mean"],
            row["median"],
            row["max"],
            int(row["count"]),
        )
    return result


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> None:
    """Stage entry point — orchestrates all EDA sub-tasks."""
    data_cfg = cfg.get("data", {})
    features_cfg = cfg.get("features", {})
    eda_cfg = cfg.get("eda", {})

    processed_path: str = data_cfg.get(
        "processed_path",
        "data/processed/Flight_Price_Dataset_of_Bangladesh.parquet",
    )
    target: str = features_cfg.get("target", "fare")
    numerical_cols: list[str] = features_cfg.get("numerical", [])
    group_cols: list[str] = eda_cfg.get("group_cols", ["airline", "source", "destination", "seasonality"])
    correlation_method: str = eda_cfg.get("correlation_method", "pearson")

    logger.info("━━━━━━  EDA: Descriptive Statistics  ━━━━━━")

    df = load_processed(processed_path)

    # 1. Fare summaries by categorical group
    logger.info("── Part 1: Fare summaries by group ──")
    summarise_fares_by_group(df, group_cols, target)

    # 2. Correlation matrix
    logger.info("── Part 2: Correlation analysis ──")
    correlation_matrix(df, numerical_cols, target, method=correlation_method)

    # 3. KPI exploration
    logger.info("── Part 3: KPI exploration ──")
    avg_fare_per_airline(df, target)
    most_popular_routes(df, target=target)
    seasonal_fare_variation(df, target)
    top_expensive_routes(df, target=target)

    logger.info("━━━━━━  EDA complete  ━━━━━━")
