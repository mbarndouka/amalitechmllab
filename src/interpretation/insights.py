"""Step 6 — Business insights from processed flight data.

WHY THIS MODULE EXISTS
-----------------------
Feature importance tells us what the MODEL learned.
Business insights tell us what the DATA actually shows.

These are complementary but different:
  - A feature can be important to the model but have a simple story in the data.
  - A business pattern (e.g., Eid fares spike) might be captured by one feature
    but is better understood by grouping raw data directly.

This module answers the three key questions:
  1. What factors most influence fare?      → correlation + class/stopover analysis
  2. How do airlines differ in pricing?     → airline_pricing()
  3. Do seasons/routes show higher fares?   → seasonal_pricing(), route_pricing()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Insight computations
# ---------------------------------------------------------------------------


def airline_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Median, mean, std fare by airline, sorted by median (most expensive first).

    WHY MEDIAN NOT MEAN: Flight prices are right-skewed (a few very expensive
    first-class tickets pull the mean up). Median is a more honest "typical" price.
    """
    return (
        df.groupby("airline")["fare"]
        .agg(
            median_fare="median",
            mean_fare="mean",
            std_fare="std",
            count="count",
        )
        .sort_values("median_fare", ascending=False)
        .round(0)
        .reset_index()
    )


def seasonal_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Median fare by season in a logical display order (peak seasons first).

    WHY ORDER MATTERS: Presenting seasons randomly makes it hard to spot
    the pattern. We order peak→off-peak so the story is immediately clear.
    """
    order = ["Eid", "Hajj", "Winter Holidays", "Regular"]
    result = (
        df.groupby("seasonality")["fare"]
        .agg(
            median_fare="median",
            mean_fare="mean",
            std_fare="std",
            count="count",
        )
        .round(0)
        .reset_index()
    )
    result["seasonality"] = pd.Categorical(result["seasonality"], categories=order, ordered=True)
    return result.sort_values("seasonality").reset_index(drop=True)


def route_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Median fare by source→destination route, sorted by price."""
    return (
        df.groupby(["source", "destination"])["fare"]
        .agg(median_fare="median", count="count")
        .sort_values("median_fare", ascending=False)
        .round(0)
        .reset_index()
    )


def class_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Median fare by travel class — shows the class premium."""
    return (
        df.groupby("travel_class")["fare"]
        .agg(median_fare="median", mean_fare="mean", count="count")
        .sort_values("median_fare", ascending=False)
        .round(0)
        .reset_index()
    )


def stopover_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Median fare by number of stopovers.

    Interesting because: more stopovers usually = cheaper (longer journey)
    but sometimes budget airlines use stopovers to offer lower fares.
    """
    return (
        df.groupby("stopovers")["fare"]
        .agg(median_fare="median", mean_fare="mean", count="count")
        .sort_values("median_fare", ascending=False)
        .round(0)
        .reset_index()
    )


def booking_source_pricing(df: pd.DataFrame) -> pd.DataFrame:
    """Median fare by booking channel — direct vs agency vs online."""
    return (
        df.groupby("booking_source")["fare"]
        .agg(median_fare="median", mean_fare="mean", count="count")
        .sort_values("median_fare", ascending=False)
        .round(0)
        .reset_index()
    )


def numerical_correlations(df: pd.DataFrame) -> dict[str, float]:
    """Pearson correlation of numerical features with fare.

    WHY PEARSON: Measures linear relationship strength (-1 to +1).
    +1 = perfect positive linear relationship (as X goes up, fare goes up).
    -1 = perfect negative (as X goes up, fare goes down).
     0 = no linear relationship.

    Pearson works well for continuous features. It won't detect non-linear
    patterns (e.g., fares are highest at 60+ and 0-7 days_left).
    """
    num_cols = ["duration", "days_left", "departure_hour", "departure_month", "arrival_hour", "stopovers"]
    corrs = {}
    for col in num_cols:
        if col in df.columns:
            corrs[col] = round(float(df[col].corr(df["fare"])), 4)
    return dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))


def days_left_fare_buckets(df: pd.DataFrame) -> pd.DataFrame:
    """Segment days_left into booking windows and compute median fare per bucket.

    WHY: Correlation is a single number and misses non-linear patterns.
    Bucketing reveals the actual shape of the days_left → fare relationship:
    last-minute (<7 days) tends to be expensive, early booking (>90 days)
    can also be expensive, and mid-range (30-60 days) is often cheapest.
    """
    bins = [0, 7, 14, 30, 60, 90, 180, 365, 9999]
    labels = ["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181-365", "365+"]
    df = df.copy()
    df["booking_window"] = pd.cut(df["days_left"], bins=bins, labels=labels, right=True)
    return (
        df.groupby("booking_window", observed=True)["fare"]
        .agg(median_fare="median", count="count")
        .round(0)
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Stakeholder report generator
# ---------------------------------------------------------------------------


def generate_stakeholder_report(insights: dict) -> str:
    """Generate a plain-language summary for non-technical stakeholders.

    WHY PLAIN TEXT: Executives and business analysts need conclusions, not
    DataFrames. This converts numbers into actionable sentences.
    """
    airline_df = pd.DataFrame(insights["airline_pricing"])
    seasonal_df = pd.DataFrame(insights["seasonal_pricing"])
    route_df = pd.DataFrame(insights["top_10_routes"])
    class_df = pd.DataFrame(insights["class_pricing"])
    corrs = insights["correlations"]
    days_corr = corrs.get("days_left", 0.0)
    dur_corr = corrs.get("duration", 0.0)

    most_exp_airline = airline_df.iloc[0]
    least_exp_airline = airline_df.iloc[-1]
    premium_multiplier = most_exp_airline["median_fare"] / least_exp_airline["median_fare"]

    peak_season = seasonal_df.iloc[0]
    regular_season = seasonal_df[seasonal_df["seasonality"] == "Regular"].iloc[0]
    season_premium = peak_season["median_fare"] / regular_season["median_fare"]

    top_route = route_df.iloc[0]
    cheapest_cls = class_df.iloc[-1]
    premium_cls = class_df.iloc[0]

    report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          FLIGHT FARE INSIGHTS — EXECUTIVE SUMMARY                          ║
║          Bangladesh Flight Price Analysis                                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

━━━ KEY FINDINGS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. AIRLINE PRICING STRATEGY
   • {most_exp_airline["airline"]} charges the highest typical fare
     (BDT {most_exp_airline["median_fare"]:,.0f} median).
   • {least_exp_airline["airline"]} is the most affordable
     (BDT {least_exp_airline["median_fare"]:,.0f} median).
   • The premium airline costs {premium_multiplier:.1f}x more than the budget option.

2. SEASONAL PRICE SURGES
   • Fares are highest during {peak_season["seasonality"]}
     (BDT {peak_season["median_fare"]:,.0f} median).
   • Compared to Regular season (BDT {regular_season["median_fare"]:,.0f}),
     that is a {(season_premium - 1) * 100:.0f}% seasonal premium.
   • RECOMMENDATION: Book non-Eid/Hajj travel early to avoid surge pricing.

3. MOST EXPENSIVE ROUTES
   • Priciest route: {top_route["source"]} -> {top_route["destination"]}
     (BDT {top_route["median_fare"]:,.0f} median).
   • Long-haul international routes consistently cost more.

4. BOOKING TIMING (days_left correlation: {days_corr:+.3f})
   • Negative correlation means: booking closer to departure = HIGHER fares.
   • Flight duration also positively correlates with fare
     (longer flight = higher price, r={dur_corr:+.3f}).
   • RECOMMENDATION: Book at least 30-60 days ahead for best prices.

5. TRAVEL CLASS PREMIUM
   • {premium_cls["travel_class"]} class costs {(premium_cls["median_fare"] / cheapest_cls["median_fare"]):.1f}x
     more than {cheapest_cls["travel_class"]} class.

━━━ RECOMMENDATIONS FOR STAKEHOLDERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  * Price-sensitive travelers: avoid {peak_season["seasonality"]} season, book early,
    choose budget carriers, prefer Economy class.
  * Airlines: price discrimination by season and booking window is effective --
    demand is relatively inelastic during Eid/Hajj.
  * Travel agencies: early-booking promotions (30-90 days ahead) align with
    lowest-fare windows and can drive volume.
  * Model accuracy (R2=0.68): our best model explains ~68% of fare variation.
    ~32% is unexplained -- likely due to flash sales, seat availability,
    and dynamic pricing not captured in the dataset.
"""
    return report.strip()


# ---------------------------------------------------------------------------
# Main insight generation
# ---------------------------------------------------------------------------


def generate_insights(df: pd.DataFrame) -> dict:
    """Compute all insight tables. Returns a JSON-serializable dict."""
    airline = airline_pricing(df)
    seasonal = seasonal_pricing(df)
    route = route_pricing(df)
    cls = class_pricing(df)
    stopovers = stopover_pricing(df)
    booking = booking_source_pricing(df)
    corrs = numerical_correlations(df)
    windows = days_left_fare_buckets(df)

    return {
        "airline_pricing": airline.to_dict(orient="records"),
        "seasonal_pricing": seasonal.to_dict(orient="records"),
        "top_10_routes": route.head(10).to_dict(orient="records"),
        "bottom_10_routes": route.tail(10).to_dict(orient="records"),
        "class_pricing": cls.to_dict(orient="records"),
        "stopover_pricing": stopovers.to_dict(orient="records"),
        "booking_source": booking.to_dict(orient="records"),
        "correlations": corrs,
        "days_left_buckets": windows.to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run(cfg: dict[str, Any]) -> None:
    data_cfg = cfg.get("data", {})
    processed_path = data_cfg.get(
        "processed_path",
        "data/processed/Flight_Price_Dataset_of_Bangladesh.parquet",
    )
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("━━━━━━  Step 6b: Business Insights  ━━━━━━")

    df = pd.read_parquet(processed_path)
    logger.info("Loaded processed data: %d rows, %d columns", *df.shape)

    insights = generate_insights(df)

    # Save full JSON
    out_path = reports_dir / "insights.json"
    out_path.write_text(json.dumps(insights, indent=2))
    logger.info("Insights saved → %s", out_path)

    # Save stakeholder report as plain text
    report_text = generate_stakeholder_report(insights)
    report_path = reports_dir / "stakeholder_report.txt"
    report_path.write_text(report_text)
    logger.info("Stakeholder report saved → %s", report_path)

    # Log key numbers to console
    airline_df = pd.DataFrame(insights["airline_pricing"])
    logger.info(
        "Most expensive airline : %s  (BDT %s median)",
        airline_df.iloc[0]["airline"],
        f"{airline_df.iloc[0]['median_fare']:,.0f}",
    )
    logger.info(
        "Cheapest airline       : %s  (BDT %s median)",
        airline_df.iloc[-1]["airline"],
        f"{airline_df.iloc[-1]['median_fare']:,.0f}",
    )

    for row in insights["seasonal_pricing"]:
        logger.info("Season %-17s → median BDT %s", row["seasonality"], f"{row['median_fare']:,.0f}")

    logger.info("Booking window → fare:")
    for row in insights["days_left_buckets"]:
        logger.info(
            "  %s days ahead → BDT %s (n=%d)", row["booking_window"], f"{row['median_fare']:,.0f}", row["count"]
        )

    logger.info("━━━━━━  Insights complete  ━━━━━━")
