"""Interpretation plots for regression models — notebook use only."""
from __future__ import annotations

import numpy as np
import pandas as pd


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Fare",
    sample_n: int = 3000,
) -> None:
    """Scatter of actual vs predicted with identity line. Samples for readability."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(sample_n, len(y_true)), replace=False)
    yt, yp = y_true[idx], y_pred[idx]

    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(yt, yp, alpha=0.25, s=8, color="#2980b9", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Fare (BDT)")
    ax.set_ylabel("Predicted Fare (BDT)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str = "",
) -> None:
    """Residuals vs predicted + histogram of residuals."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    residuals = y_true - y_pred
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    # 1. Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.2, s=6, color="#e74c3c", rasterized=True)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Predicted Fare (BDT)")
    axes[0].set_ylabel("Residual (Actual − Predicted)")
    axes[0].set_title(f"{title_prefix}Residuals vs Predicted")
    axes[0].xaxis.set_major_formatter(fmt)
    axes[0].yaxis.set_major_formatter(fmt)

    # 2. Residual histogram
    axes[1].hist(residuals, bins=80, color="#3498db", alpha=0.75, edgecolor="none")
    axes[1].axvline(0, color="red", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Residual (BDT)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title_prefix}Residual Distribution")
    axes[1].xaxis.set_major_formatter(fmt)

    # 3. Absolute residuals vs actual (error magnitude)
    axes[2].scatter(y_true, np.abs(residuals), alpha=0.2, s=6, color="#8e44ad", rasterized=True)
    axes[2].set_xlabel("Actual Fare (BDT)")
    axes[2].set_ylabel("|Residual| (BDT)")
    axes[2].set_title(f"{title_prefix}Error Magnitude vs Actual")
    axes[2].xaxis.set_major_formatter(fmt)
    axes[2].yaxis.set_major_formatter(fmt)

    plt.suptitle(
        f"Residual Analysis  |  mean={residuals.mean():+,.0f}  std={residuals.std():,.0f}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def plot_coefficients(
    coef: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Top Feature Coefficients (Linear Regression)",
) -> None:
    """Horizontal bar chart of top_n largest-magnitude coefficients."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    coef_series = pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
    top = coef_series.head(top_n).sort_values()

    colors = ["#e74c3c" if v >= 0 else "#2980b9" for v in top.values]
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (BDT per unit)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(fmt)
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(
    metrics_by_split: dict[str, dict[str, float]],
) -> None:
    """Bar chart comparing R², MAE, RMSE across train/val/test splits."""
    import matplotlib.pyplot as plt

    splits = list(metrics_by_split.keys())
    metric_names = ["r2", "mae", "rmse", "mape"]
    titles = ["R² (higher = better)", "MAE (BDT)", "RMSE (BDT)", "MAPE (%)"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, metric, title in zip(axes, metric_names, titles):
        vals = [metrics_by_split[s][metric] for s in splits]
        ax.bar(splits, vals, color=colors[: len(splits)], alpha=0.85)
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f"{v:,.2f}", ha="center", fontsize=9)

    plt.suptitle("Linear Regression — Train / Val / Test Metrics", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()
