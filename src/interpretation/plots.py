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
import seaborn as sns

_THEME = {"style": "whitegrid", "palette": "muted", "font_scale": 1.05}


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Fare",
    sample_n: int = 3000,
) -> None:
    """Scatter of actual vs predicted with identity line. Samples for readability."""
    import matplotlib.pyplot as plt

    sns.set_theme(**_THEME)

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y_true), size=min(sample_n, len(y_true)), replace=False)
    yt, yp = y_true[idx], y_pred[idx]

    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("white")
    sns.scatterplot(x=yt, y=yp, alpha=0.3, s=8, color="#4C72B0", rasterized=True, ax=ax)
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual log(Fare)")
    ax.set_ylabel("Predicted log(Fare)")
    ax.set_title(title)
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_prefix: str = "",
) -> None:
    """Residuals vs predicted + histogram of residuals."""
    import matplotlib.pyplot as plt

    sns.set_theme(**_THEME)

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.patch.set_facecolor("white")

    # 1. Residuals vs predicted (log-space)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.2, s=6, color="#e74c3c",
                    rasterized=True, ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xlabel("Predicted log(Fare) (log-space)")
    axes[0].set_ylabel("Residual (log-space)")
    axes[0].set_title(f"{title_prefix}Residuals vs Predicted")

    # 2. Residual histogram (log-space)
    sns.histplot(residuals, bins=80, kde=True, color="#4C72B0", alpha=0.75, ax=axes[1])
    axes[1].axvline(0, color="red", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("Residual (log-space)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{title_prefix}Residual Distribution")

    # 3. Absolute residuals vs actual (log-space error magnitude)
    sns.scatterplot(x=y_true, y=np.abs(residuals), alpha=0.2, s=6, color="#8e44ad",
                    rasterized=True, ax=axes[2])
    axes[2].set_xlabel("Actual log(Fare) (log-space)")
    axes[2].set_ylabel("|Residual| (log-space)")
    axes[2].set_title(f"{title_prefix}Error Magnitude vs Actual")

    plt.suptitle(
        f"Residual Analysis  |  mean={residuals.mean():+.4f}  std={residuals.std():.4f}",
        fontsize=12, fontweight="bold",
    )
    sns.despine()
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

    sns.set_theme(**_THEME)

    coef_series = pd.Series(coef, index=feature_names).sort_values(key=abs, ascending=False)
    top = coef_series.head(top_n).sort_values()

    colors = ["#4C72B0" if v >= 0 else "#DD8452" for v in top.values]
    plot_df = pd.DataFrame({"feature": top.index, "coef": top.values})

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("white")
    sns.barplot(
        data=plot_df, y="feature", x="coef",
        palette=colors, orient="h", ax=ax,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (log-BDT units)")
    ax.set_title(title)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_metrics_comparison(
    metrics_by_split: dict[str, dict[str, float]],
) -> None:
    """Bar chart comparing R², MAE, RMSE across train/val/test splits."""
    import matplotlib.pyplot as plt

    sns.set_theme(**_THEME)

    splits = list(metrics_by_split.keys())
    metric_names = ["r2", "mae", "rmse", "mape"]
    titles = ["R² (higher = better)", "MAE", "RMSE", "MAPE (%)"]
    palette = sns.color_palette("muted", len(splits))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    for ax, metric, title in zip(axes, metric_names, titles, strict=False):
        vals = [metrics_by_split[s].get(metric, 0.0) for s in splits]
        plot_df = pd.DataFrame({"split": splits, "value": vals})
        sns.barplot(data=plot_df, x="split", y="value", palette=palette, alpha=0.85, ax=ax)
        ax.set_title(title)
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("")
        for i, v in enumerate(vals):
            ax.text(i, v * 1.01, f"{v:,.4f}", ha="center", fontsize=9)

    plt.suptitle("Linear Regression — Train / Val / Test Metrics", fontsize=13, fontweight="bold")
    sns.despine()
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
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import r2_score

    sns.set_theme(**_THEME)

    if alphas is None:
        alphas = [0.001, 0.01, 0.1, 1, 10, 100, 1_000, 10_000]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

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

        curve_df = pd.DataFrame({
            "alpha": alphas * 2,
            "R²": train_r2 + val_r2,
            "split": ["Train R²"] * len(alphas) + ["Val R²"] * len(alphas),
        })
        sns.lineplot(
            data=curve_df, x="alpha", y="R²", hue="split",
            marker="o", palette=["#4C72B0", "#DD8452"],
            linewidth=2, ax=ax,
        )
        ax.set_xscale("log")
        ax.set_xlabel("alpha (log scale)")
        ax.set_ylabel("R²")
        ax.set_title(f"{name} — Regularization Path")
        ax.legend(fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")

    plt.suptitle("Effect of Regularization Strength on Bias–Variance", fontsize=13, fontweight="bold")
    sns.despine()
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
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve

    sns.set_theme(**_THEME)

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
    fig.patch.set_facecolor("white")
    sns.lineplot(x=sizes, y=train_mean, marker="o", color="#4C72B0",
                 label="Train R²", linewidth=2, ax=ax)
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color="#4C72B0")
    sns.lineplot(x=sizes, y=val_mean, marker="s", color="#DD8452",
                 label="CV Val R²", linewidth=2, linestyle="--", ax=ax)
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.15, color="#DD8452")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R²")
    ax.set_title("Learning Curve — Bias–Variance Tradeoff")
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    split: str = "val",
) -> None:
    """Bar chart comparing all models on R², MAE, RMSE for a given split."""
    import matplotlib.pyplot as plt

    sns.set_theme(**_THEME)

    df = comparison_df[comparison_df["split"] == split].sort_values("r2", ascending=False)
    models = df["model"].tolist()
    n_models = len(models)
    palette = sns.color_palette("husl", n_models)

    # Smart formatter: if MAE < 10 assume log-space, else BDT
    sample_mae = df["mae"].median() if len(df) > 0 else 1000.0

    def _fmt_value(v: float, metric: str) -> str:
        if metric == "r2":
            return f"{v:.4f}"
        if sample_mae < 10:
            return f"{v:.4f}"
        return f"{v/1000:.1f}k"

    def _yformatter(v: float, metric: str) -> str:
        if metric == "r2":
            return f"{v:.4f}"
        if sample_mae < 10:
            return f"{v:.4f}"
        return f"{v/1000:.0f}k"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("white")

    for ax, metric, title in [
        (axes[0], "r2",   "R² (higher = better)"),
        (axes[1], "mae",  "MAE (lower = better)"),
        (axes[2], "rmse", "RMSE (lower = better)"),
    ]:
        plot_df = pd.DataFrame({"model": models, "value": df[metric].values})
        sns.barplot(
            data=plot_df, x="model", y="value",
            palette=palette, alpha=0.85, ax=ax,
        )
        ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        import matplotlib.ticker as mticker
        if metric != "r2" and sample_mae >= 10:
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")
            )
        elif metric != "r2":
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f"{v:.4f}")
            )
        for i, val in enumerate(df[metric].values):
            ax.text(
                i,
                val * 1.01,
                _fmt_value(val, metric),
                ha="center", va="bottom", fontsize=8,
            )

    plt.suptitle(f"Model Comparison — {split.capitalize()} Split", fontsize=13, fontweight="bold")
    sns.despine()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Step 6 — Interpretation plots
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
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

    sns.set_theme(**_THEME)

    top = importance_df.head(top_n).sort_values("abs_importance")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    fig.patch.set_facecolor("white")
    sns.barplot(
        data=top, y="feature", x="abs_importance",
        palette="YlOrRd", orient="h", ax=ax,
    )
    ax.set_xlabel("Feature Importance (total MSE reduction, normalised)")
    ax.set_title(f"{model_name.replace('_', ' ').title()} — Top {top_n} Feature Importances")
    ax.axvline(0, color="black", linewidth=0.5)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_airline_pricing(df: pd.DataFrame) -> None:
    """Box-and-whisker plot of fare distribution per airline, sorted by median.

    WHY BOXPLOT not bar chart: A bar chart of median loses distribution info.
    Boxplots show median (orange line), IQR (box), and whiskers (1.5×IQR).
    showfliers=False removes extreme outliers so the box scaling is readable.

    WHY SORTED BY MEDIAN: Puts premium airlines on the left, budget on right —
    the viewer immediately sees the pricing tier structure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sns.set_theme(**_THEME)

    order = df.groupby("airline")["fare"].median().sort_values(ascending=False).index.tolist()
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

    fig, ax = plt.subplots(figsize=(16, 6))
    fig.patch.set_facecolor("white")
    sns.boxplot(
        data=df, x="airline", y="fare",
        order=order, orient="v",
        palette="husl",
        showfliers=False,
        linewidth=0.9,
        ax=ax,
    )
    ax.set_xlabel("Airline")
    ax.set_ylabel("Fare (BDT)")
    ax.set_title("Fare Distribution by Airline — sorted by median (no outliers shown)")
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_seasonal_pricing(df: pd.DataFrame) -> None:
    """Bar chart of median fare per season with percentage-above-Regular labels.

    WHY SHOW PREMIUM %: Raw BDT numbers are less actionable than "Eid is X% more
    expensive than Regular." The annotation makes the insight self-explanatory.

    WHY THIS COLOR SCHEME: Red for peak, blue for off-peak visually encodes
    urgency — standard in pricing/heat dashboards.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sns.set_theme(**_THEME)

    order = ["Eid", "Hajj", "Winter Holidays", "Regular"]
    palette = {
        "Eid": "#E74C3C",
        "Hajj": "#E67E22",
        "Winter Holidays": "#3498DB",
        "Regular": "#2ECC71",
    }

    seasonal = (
        df.groupby("seasonality")["fare"]
        .median()
        .reindex(order)
    )
    base = seasonal["Regular"]
    fmt = mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")

    plot_df = seasonal.reset_index()
    plot_df.columns = ["season", "fare"]
    colors = [palette.get(s, "#4C72B0") for s in plot_df["season"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("white")
    sns.barplot(
        data=plot_df, x="season", y="fare",
        palette=colors, alpha=0.85, ax=ax,
    )
    for i, (season, val) in enumerate(zip(plot_df["season"], plot_df["fare"], strict=False)):
        pct = (val / base - 1) * 100
        label = (
            f"BDT {val/1000:.1f}k\n({pct:+.0f}%)"
            if season != "Regular"
            else f"BDT {val/1000:.1f}k\n(baseline)"
        )
        ax.text(
            i,
            val * 1.01,
            label,
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_xlabel("Season")
    ax.set_ylabel("Median Fare (BDT)")
    ax.set_title("Median Fare by Seasonality — % premium over Regular season")
    ax.yaxis.set_major_formatter(fmt)
    ax.set_ylim(0, seasonal.max() * 1.18)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_route_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of median fare: rows = source airports, cols = destination airports.

    WHY HEATMAP: A 2D matrix lets us see all source→destination combinations at
    once. Colour instantly reveals expensive routes (dark) vs cheap ones (light).

    NaN cells = no flights on that route in the dataset (not every combination
    exists). We mask them white so they don't mislead.
    """
    import matplotlib.pyplot as plt

    sns.set_theme(**_THEME)

    pivot = df.pivot_table(values="fare", index="source", columns="destination", aggfunc="median")

    annot_labels = pivot.applymap(
        lambda v: f"{v/1000:.0f}k" if not np.isnan(v) else ""
    )

    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor("white")
    sns.heatmap(
        pivot,
        annot=annot_labels,
        fmt="",
        cmap="YlOrRd",
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Median Fare (BDT)"},
    )
    ax.set_xlabel("Destination")
    ax.set_ylabel("Source")
    ax.set_title(
        "Median Fare Heatmap — Source Airport × Destination Airport\n"
        "(white cells = no flights on that route)"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_days_left_fare(df: pd.DataFrame) -> None:
    """Line chart of median fare vs booking window (days before departure).

    WHY BUCKETS not scatter: Raw scatter of days_left vs fare is a blob —
    too many points. Bucketing into windows (e.g., 0-7, 8-14 days) and
    plotting medians reveals the actual trend clearly.

    KEY INSIGHT THIS REVEALS: Is there a U-shaped curve?
    (last-minute expensive | 30-60 days cheapest | very early also rises)
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    sns.set_theme(**_THEME)

    bins   = [0,   7,  14,  30,  60,  90, 180, 365, 9999]
    labels = ["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181-365", "365+"]

    df = df.copy()
    df["booking_window"] = pd.cut(df["days_left"], bins=bins, labels=labels, right=True)

    stats = (
        df.groupby("booking_window", observed=True)["fare"]
        .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75))
        .reset_index()
    )

    x   = list(range(len(stats)))
    fmt = mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k")

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("white")
    sns.lineplot(
        x=x, y=stats["median"],
        marker="o", color="#4C72B0", linewidth=2.5,
        label="Median fare", ax=ax,
    )
    ax.fill_between(x, stats["q25"], stats["q75"],
                    alpha=0.2, color="#4C72B0", label="IQR (25th–75th %ile)")

    ax.set_xticks(x)
    ax.set_xticklabels(stats["booking_window"], fontsize=9)
    ax.set_xlabel("Days Before Departure (booking window)")
    ax.set_ylabel("Fare (BDT)")
    ax.set_title(
        "How Booking Timing Affects Fare\n"
        "Earlier booking = lower fare? (Look for U-shape or monotone drop)"
    )
    ax.yaxis.set_major_formatter(fmt)
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()
    plt.show()
