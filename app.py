"""Streamlit flight fare prediction app."""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from serving.predictor import AVAILABLE_MODELS, Predictor  # noqa: E402

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Flight Fare Predictor",
    page_icon="✈",
    layout="wide",
)

# ── Load metrics ──────────────────────────────────────────────────────────────

_METRICS_PATH = Path("reports/model_comparison.json")


@st.cache_data
def load_metrics() -> dict:
    if _METRICS_PATH.exists():
        return json.loads(_METRICS_PATH.read_text())
    return {}


# ── Load predictor (cached per model name) ────────────────────────────────────


@st.cache_resource
def load_predictor(model_name: str) -> Predictor:
    return Predictor(model_name)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Model")
    available = [m for m in AVAILABLE_MODELS if (Path("models") / f"{m}.pkl").exists()]
    model_name = st.selectbox("Select model", available, index=0)

    metrics = load_metrics()
    if model_name in metrics:
        m = metrics[model_name]
        st.markdown("---")
        st.markdown("**Validation performance**")
        col_a, col_b = st.columns(2)
        col_a.metric("R²", f"{m['metrics']['val']['r2']:.4f}")
        col_b.metric("MAPE", f"{m['metrics']['val']['mape']:.2f}%")
        col_a2, col_b2 = st.columns(2)
        col_a2.metric("MAE (log)", f"{m['metrics']['val']['mae']:.3f}")
        col_b2.metric("RMSE (log)", f"{m['metrics']['val']['rmse']:.3f}")
        if m.get("best_params"):
            with st.expander("Best params"):
                st.json(m["best_params"])
    else:
        st.info("No metrics available for this model.")

    st.markdown("---")
    st.caption("MAE/RMSE in log-space (not BDT). Prediction is inverse-transformed to BDT.")

# ── Main ──────────────────────────────────────────────────────────────────────

st.title("✈ Bangladesh Flight Fare Predictor")
st.caption("Top drivers: route, travel class, aircraft type, flight duration.")

predictor = load_predictor(model_name)
opts = predictor.options

# ── Primary inputs ────────────────────────────────────────────────────────────

st.subheader("Route")

def airport_label(code: str, name_map: dict[str, str]) -> str:
    name = name_map.get(code, "")
    return f"{code} — {name}" if name else code

src_codes = opts.get("source", [])
dst_codes = opts.get("destination", [])
src_labels = [airport_label(c, predictor.source_names) for c in src_codes]
dst_labels = [airport_label(c, predictor.dest_names) for c in dst_codes]

c1, c2 = st.columns(2)
src_label = c1.selectbox("From (source airport)", src_labels)
dst_label = c2.selectbox("To (destination airport)", dst_labels)

source = src_codes[src_labels.index(src_label)]
destination = dst_codes[dst_labels.index(dst_label)]

if source == destination:
    st.warning("Source and destination are the same.")

st.subheader("Flight Details")
c3, c4 = st.columns(2)
travel_class = c3.radio(
    "Travel class",
    opts.get("travel_class", ["Economy", "Business", "First Class"]),
    horizontal=True,
)
aircraft_type = c4.selectbox("Aircraft type", opts.get("aircraft_type", []))

st.subheader("Schedule")
today = datetime.date.today()

dc1, dc2 = st.columns(2)
dep_date = dc1.date_input("Departure date", value=today + datetime.timedelta(days=30), min_value=today)
dep_time = dc2.time_input("Departure time", value=datetime.time(8, 0), step=1800)

ac1, ac2 = st.columns(2)
arr_date = ac1.date_input("Arrival date", value=today + datetime.timedelta(days=30), min_value=today)
arr_time = ac2.time_input("Arrival time", value=datetime.time(10, 30), step=1800)

dep_dt = datetime.datetime.combine(dep_date, dep_time)
arr_dt = datetime.datetime.combine(arr_date, arr_time)

duration_hrs = (arr_dt - dep_dt).total_seconds() / 3600
days_left = (dep_date - today).days

if duration_hrs <= 0:
    st.warning("Arrival must be after departure.")
    duration_hrs = 1.0

sc1, sc2, sc3 = st.columns([1, 1, 2])
stopovers_label = sc1.radio("Stopovers", ["Direct (0)", "1 Stop", "2 Stops"], horizontal=False)
stopovers = {"Direct (0)": 0, "1 Stop": 1, "2 Stops": 2}[stopovers_label]

sc2.metric("Duration", f"{duration_hrs:.1f} hrs")
sc3.metric("Days until departure", f"{days_left} days")

# ── Advanced inputs (low impact) ──────────────────────────────────────────────

with st.expander("Advanced options (low impact on prediction)"):
    adv1, adv2, adv3 = st.columns(3)
    airline = adv1.selectbox("Airline", opts.get("airline", []))
    booking_source = adv2.selectbox("Booking source", opts.get("booking_source", []))
    seasonality = adv3.selectbox("Seasonality", opts.get("seasonality", []))

# ── Predict ───────────────────────────────────────────────────────────────────

st.markdown("---")
predict_btn = st.button("Predict Fare", type="primary", use_container_width=True)

if predict_btn:
    if source == destination:
        st.error("Source and destination must differ.")
    elif duration_hrs <= 0:
        st.error("Arrival must be after departure.")
    elif days_left < 0:
        st.error("Departure date must be today or in the future.")
    else:
        inputs = {
            "airline": airline,
            "source": source,
            "destination": destination,
            "aircraft_type": aircraft_type,
            "travel_class": travel_class,
            "booking_source": booking_source,
            "seasonality": seasonality,
            "stopovers": stopovers,
            "duration": duration_hrs,
            "days_left": max(days_left, 1),
            "departure_hour": dep_dt.hour,
            "departure_day_of_week": dep_dt.weekday(),
            "departure_month": dep_dt.month,
            "arrival_hour": arr_dt.hour,
            "arrival_day_of_week": arr_dt.weekday(),
            "arrival_month": arr_dt.month,
        }

        with st.spinner("Predicting..."):
            try:
                fare = predictor.predict(inputs)
                st.success("Prediction complete")
                st.markdown(
                    f"""
                    <div style="background:#0e1117;border:1px solid #21262d;border-radius:12px;
                                padding:32px;text-align:center;margin-top:16px;">
                        <p style="color:#8b949e;font-size:14px;margin:0;">Estimated fare ({model_name})</p>
                        <p style="color:#58a6ff;font-size:52px;font-weight:700;margin:8px 0;">
                            ৳ {fare:,.0f}
                        </p>
                        <p style="color:#8b949e;font-size:13px;margin:0;">Bangladeshi Taka (BDT)</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("Input summary"):
                    st.json(inputs)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
