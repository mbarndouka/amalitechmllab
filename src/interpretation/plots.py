"""Interpretation plots for regression models — notebook use only.

Plot inventory
--------------
Existing (Step 4–5):
  plot_actual_vs_predicted   — scatter of y_true vs y_pred
  plot_residuals             — residual analysis (3-panel)
  plot_coefficients          — linear model coefficient bar chart
  plot_metrics_comparison    — bar chart of metrics across splits
  plot_regularization_path   — Ridge/Lasso alpha sweep
  plot_learning_curve        — bias-variance learning curve
  plot_model_comparison      — all models vs R²/MAE/RMSE

New (Step 6 — interpretation):
  plot_feature_importance    — tree-model Gini importances
  plot_airline_pricing       — boxplot fare by airline
  plot_seasonal_pricing      — bar chart fare by season
  plot_route_heatmap         — heatmap median fare by source→destination
  plot_days_left_fare        — line chart booking window vs fare
"""
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


def plot_regularization_path(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    alphas: list[float] | None = None,
) -> None:
    """Train/val R² vs alpha for Ridge and Lasso side-by-side."""
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1_000, 10_000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, ModelCls, name in [
        (axes[0], Ridge, "Ridge"),
        (axes[1], Lasso, "Lasso"),
    ]:
        train_r2, val_r2 = [], []
        for a in alphas:
            m = ModelCls(alpha=a, max_iter=10_000)
            m.fit(X_train, y_train)
            train_r2.append(r2_score(y_train, m.predict(X_train)))
            val_r2.append(r2_score(y_val,   m.predict(X_val)))

        ax.semilogx(alphas, train_r2, "o-", color="#2980b9", label="Train R²", linewidth=2)
        ax.semilogx(alphas, val_r2,   "s--", color="#e74c3c", label="Val R²",   linewidth=2)
        ax.set_xlabel("alpha (log scale)")
        ax.set_ylabel("R²")
        ax.set_title(f"{name} — Regularization Path")
        ax.legend(fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    plt.suptitle("Effect of Regularization Strength on Bias–Variance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_learning_curve(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = 5,
    train_sizes: list[float] | None = None,
) -> None:
    """Classic bias–variance learning curve: train/cv score vs training set size."""
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt

    if train_sizes is None:
        train_sizes = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

    sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=train_sizes,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sizes, train_mean, "o-", color="#2980b9", label="Train R²",       linewidth=2)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="#2980b9")
    ax.plot(sizes, val_mean,   "s--", color="#e74c3c", label="CV Val R²",     linewidth=2)
    ax.fill_between(sizes, val_mean - val_std,   val_mean + val_std,   alpha=0.15, color="#e74c3c")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R²")
    ax.set_title("Learning Curve — Bias–Variance Tradeoff")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    comparison_df: "pd.DataFrame",
    split: str = "val",
) -> None:
    """Bar chart comparing all models on R², MAE, RMSE for a given split."""
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    df = comparison_df[comparison_df["split"] == split].sort_values("r2", ascending=False)
    models = df["model"].tolist()
    x = range(len(models))
    fmt_k = mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    palette = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6", "#e74c3c", "#1abc9c"]

    for ax, metric, title, fmt in [
        (axes[0], "r2",   "R² (higher = better)",   None),
        (axes[1], "mae",  "MAE — BDT (lower = better)", fmt_k),
        (axes[2], "rmse", "RMSE — BDT (lower = better)", fmt_k),
    ]:
        bars = ax.bar(x, df[metric].values, color=palette[:len(models)], alpha=0.85)
        ax.set_xticks(list(x))
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
        ax.set_title(title)
        if fmt:
            ax.yaxis.set_major_formatter(fmt)
        for bar, val in zip(bars, df[metric].values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.3f}" if metric == "r2" else f"{val/1000:.1f}k",
                ha="center", va="bottom", fontsize=8,
            )

    plt.suptitle(f"Model Comparison — {split.capitalize()} Split", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Step 6 — Interpretation plots
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: "pd.DataFrame",
    model_name: str,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of tree-model feature importances.

    WHY HORIZONTAL BARS: Feature names are long strings. Horizontal layout
    lets them be readable without rotation. Sorted ascending so the longest
    bar is at the top (natural reading order: most important first).

    WHY ONLY TREE MODELS: Linear model coefficients are signed (can be
    negative), so use plot_coefficients() for those instead.
    """
    import matplotlib.pyplot as plt

    top = importance_df.head(top_n).sort_values("abs_importance")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(top["feature"], top["abs_importance"], color="#2ecc71", alpha=0.85)
    ax.set_xlabel("Feature Importance (total MSE reduction, normalised)")
    ax.set_title(f"{model_name.replace('_', ' ').title()} — Top {top_n} Feature Importances")
    ax.axvline(0, color="black", linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_airline_pricing(df: "pd.DataFrame") -> None:
    """Box-and-whisker plot of fare distribution per airline, sorted by median.

    WHY BOXPLOT not bar chart: A bar chart of median loses distribution info.
    Boxplots show median (orange line), IQR (box), and whiskers (1.5×IQR).
    showfliers=False removes extreme outliers so the box scaling is readable.

    WHY SORTED BY MEDIAN: Puts premium airlines on the left, budget on right —
    the viewer immediately sees the pricing tier structure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    order  = df.groupby("airline")["fare"].median().sort_values(ascending=False).index.tolist()
    data   = [df[df["airline"] == a]["fare"].values for a in order]
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(order)))  # type: ignore[attr-defined]
    fmt    = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

    fig, ax = plt.subplots(figsize=(16, 6))
    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False,
                    medianprops={"color": "black", "linewidth": 2})

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)

    ax.set_xlabel("Airline")
    ax.set_ylabel("Fare (BDT)")
    ax.set_title("Fare Distribution by Airline — sorted by median (no outliers shown)")
    ax.yaxis.set_major_formatter(fmt)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_seasonal_pricing(df: "pd.DataFrame") -> None:
    """Bar chart of median fare per season with percentage-above-Regular labels.

    WHY SHOW PREMIUM %: Raw BDT numbers are less actionable than "Eid is X% more
    expensive than Regular." The annotation makes the insight self-explanatory.

    WHY THIS COLOR SCHEME: Red for peak, blue for off-peak visually encodes
    urgency — standard in pricing/heat dashboards.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    order   = ["Eid", "Hajj", "Winter Holidays", "Regular"]
    palette = {"Eid": "#e74c3c", "Hajj": "#e67e22",
               "Winter Holidays": "#3498db", "Regular": "#2ecc71"}

    seasonal = (
        df.groupby("seasonality")["fare"]
        .median()
        .reindex(order)
    )
    base = seasonal["Regular"]
    fmt  = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        seasonal.index,
        seasonal.values,
        color=[palette[s] for s in seasonal.index],
        alpha=0.85,
        width=0.55,
    )
    for bar, (season, val) in zip(bars, seasonal.items()):
        pct = (val / base - 1) * 100
        label = f"BDT {val/1000:.1f}k\n({pct:+.0f}%)" if season != "Regular" else f"BDT {val/1000:.1f}k\n(baseline)"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Season")
    ax.set_ylabel("Median Fare (BDT)")
    ax.set_title("Median Fare by Seasonality — % premium over Regular season")
    ax.yaxis.set_major_formatter(fmt)
    ax.set_ylim(0, seasonal.max() * 1.18)
    plt.tight_layout()
    plt.show()


def plot_route_heatmap(df: "pd.DataFrame") -> None:
    """Heatmap of median fare: rows = source airports, cols = destination airports.

    WHY HEATMAP: A 2D matrix lets us see all source→destination combinations at
    once. Colour instantly reveals expensive routes (dark) vs cheap ones (light).

    NaN cells = no flights on that route in the dataset (not every combination
    exists). We mask them white so they don't mislead.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    pivot = df.pivot_table(values="fare", index="source", columns="destination", aggfunc="median")

    fig, ax = plt.subplots(figsize=(18, 6))
    masked = np.ma.masked_invalid(pivot.values.astype(float))
    cmap = plt.cm.YlOrRd  # type: ignore[attr-defined]
    cmap.set_bad("whitesmoke")
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Destination")
    ax.set_ylabel("Source")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                text_color = "white" if val > np.nanpercentile(pivot.values, 70) else "black"
                ax.text(j, i, f"{val/1000:.0f}k",
                        ha="center", va="center", fontsize=6, color=text_color)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Median Fare (BDT)")
    cbar.formatter = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")
    cbar.update_ticks()

    ax.set_title("Median Fare Heatmap — Source Airport × Destination Airport\n"
                 "(white cells = no flights on that route)")
    plt.tight_layout()
    plt.show()


def plot_days_left_fare(df: "pd.DataFrame") -> None:
    """Line chart of median fare vs booking window (days before departure).

    WHY BUCKETS not scatter: Raw scatter of days_left vs fare is a blob —
    too many points. Bucketing into windows (e.g., 0-7, 8-14 days) and
    plotting medians reveals the actual trend clearly.

    KEY INSIGHT THIS REVEALS: Is there a U-shaped curve?
    (last-minute expensive | 30-60 days cheapest | very early also rises)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    bins   = [0,   7,  14,  30,  60,  90, 180, 365, 9999]
    labels = ["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181-365", "365+"]

    df = df.copy()
    df["booking_window"] = pd.cut(df["days_left"], bins=bins, labels=labels, right=True)

    stats = (
        df.groupby("booking_window", observed=True)["fare"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )

    x    = range(len(stats))
    fmt  = mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, stats["median"], "o-", color="#2980b9", linewidth=2.5, markersize=7, label="Median fare")
    ax.fill_between(x, stats["q25"], stats["q75"], alpha=0.2, color="#2980b9", label="IQR (25th–75th %ile)")

    ax.set_xticks(list(x))
    ax.set_xticklabels(stats["booking_window"], fontsize=9)
    ax.set_xlabel("Days Before Departure (booking window)")
    ax.set_ylabel("Fare (BDT)")
    ax.set_title("How Booking Timing Affects Fare\n"
                 "Earlier booking = lower fare? (Look for U-shape or monotone drop)")
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.show()
