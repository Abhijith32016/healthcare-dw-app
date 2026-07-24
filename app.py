import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Healthcare Outbreak Surveillance", layout="wide", page_icon="🏥")

st.title("🏥 Healthcare Data Warehouse — Disease Surveillance & Outbreak Prediction")
st.caption(
    "Big data ETL → Azure Synapse data warehouse → ML forecasting → "
    "business decision support — deployed live via Streamlit."
)

tab1, tab2, tab3 = st.tabs(
    ["📊 Surveillance Dashboard", "💰 Business Decision Support", "🔮 Outbreak Prediction"]
)

# ──────────────────────────────────────────────────────────────────
# TAB 1 — SURVEILLANCE DASHBOARD
# ──────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    daily = pd.read_csv("data/daily_global_cases.csv", parse_dates=["report_date"])
    fig1 = px.area(daily, x="report_date", y="new_cases", title="Global Daily New Cases")
    fig1.update_traces(line_color="#185FA5", fillcolor="rgba(24,95,165,0.15)")
    col1.plotly_chart(fig1, use_container_width=True)

    top10 = pd.read_csv("data/top10_countries.csv")
    fig2 = px.bar(
        top10, x="new_cases", y="country_name", orientation="h",
        title="Top 10 Countries by Total Cases", color="new_cases",
        color_continuous_scale="Blues"
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    col2.plotly_chart(fig2, use_container_width=True)

    severity = pd.read_csv("data/severity_distribution.csv")
    fig3 = px.pie(
        severity, names="severity_level", values="count",
        title="Severity Distribution", color="severity_level",
        color_discrete_map={"critical": "#A32D2D", "high": "#E24B4A",
                             "medium": "#EF9F27", "low": "#1D9E75"}
    )
    st.plotly_chart(fig3, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# TAB 2 — BUSINESS DECISION SUPPORT
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Capacity Planning")
    capacity = pd.read_csv("data/capacity_planning.csv", parse_dates=["date"])

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=capacity["date"], y=capacity["projected_bed_demand"],
        name="Projected bed demand", line=dict(color="#185FA5", width=2)
    ))
    fig4.add_trace(go.Scatter(
        x=capacity["date"], y=capacity["baseline_capacity"],
        name="Baseline capacity", line=dict(color="gray", dash="dash")
    ))
    fig4.update_layout(title="Projected Bed Demand vs. Available Capacity")
    st.plotly_chart(fig4, use_container_width=True)

    shortfall_days = int((capacity["capacity_gap"] > 0).sum())
    peak_demand = capacity["projected_bed_demand"].max()
    m1, m2 = st.columns(2)
    m1.metric("Projected days with bed shortfall", shortfall_days)
    m2.metric("Peak projected bed demand", f"{peak_demand:.0f}")

    st.divider()
    st.subheader("Budget Allocation (Linear Optimization)")
    budget = pd.read_csv("data/budget_allocation.csv")

    fig5 = px.bar(
        budget.sort_values("allocated_budget", ascending=False),
        x="country_name", y="allocated_budget",
        title="Response Budget Allocation by Country",
        color="risk_score", color_continuous_scale="Reds"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(budget, use_container_width=True)

# ──────────────────────────────────────────────────────────────────
# TAB 3 — OUTBREAK PREDICTION TOOL
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Predict Outbreak Probability")
    st.write(
        "Enter today's figures for a region to get an outbreak risk assessment "
        "from the trained Random Forest + XGBoost ensemble."
    )

    c1, c2, c3 = st.columns(3)
    new_cases_today = c1.number_input("New cases today", min_value=0, value=5000, step=100)
    new_deaths_today = c2.number_input("New deaths today", min_value=0, value=20, step=1)
    rolling_7d = c3.number_input("7-day rolling average cases", min_value=0, value=4000, step=100)

    if st.button("Run Prediction", type="primary"):
        rf_model = joblib.load("models/rf_outbreak_model.pkl")
        xgb_model = xgb.Booster()
        xgb_model.load_model("models/xgb_outbreak_model.json")

        FEATURES = [
            "new_cases", "new_deaths", "rolling_avg_7d", "rolling_avg_14d",
            "lag_1", "lag_3", "lag_7", "lag_14", "lag_21",
            "roll_mean_7", "roll_mean_14", "roll_std_7",
            "roll_max_7", "roll_min_7",
            "case_acceleration", "trend_7_14",
            "death_ratio", "lag_deaths_7",
            "case_growth_rate", "cfr_rolling",
            "day_of_year", "month_num", "year",
            "week_of_year", "day_of_week", "is_weekend",
            "country_encoded"
        ]

        sample = pd.DataFrame([{
            "new_cases": new_cases_today,
            "new_deaths": new_deaths_today,
            "rolling_avg_7d": rolling_7d,
            "rolling_avg_14d": rolling_7d * 0.9,
            "lag_1": new_cases_today * 0.95,
            "lag_3": new_cases_today * 0.85,
            "lag_7": rolling_7d,
            "lag_14": rolling_7d * 0.8,
            "lag_21": rolling_7d * 0.7,
            "roll_mean_7": rolling_7d,
            "roll_mean_14": rolling_7d * 0.9,
            "roll_std_7": rolling_7d * 0.1,
            "roll_max_7": new_cases_today * 1.1,
            "roll_min_7": new_cases_today * 0.8,
            "case_acceleration": 0,
            "trend_7_14": rolling_7d * 0.1,
            "death_ratio": new_deaths_today / max(rolling_7d, 1),
            "lag_deaths_7": new_deaths_today * 0.9,
            "case_growth_rate": 0,
            "cfr_rolling": new_deaths_today / max(new_cases_today, 1) * 100,
            "day_of_year": 180, "month_num": 6, "year": 2026,
            "week_of_year": 26, "day_of_week": 1, "is_weekend": 0,
            "country_encoded": 0
        }])[FEATURES]

        rf_prob = rf_model.predict_proba(sample)[0][1]
        dsample = xgb.DMatrix(sample, feature_names=FEATURES)
        xgb_prob = float(xgb_model.predict(dsample)[0])
        ensemble_prob = (rf_prob + xgb_prob) / 2

        st.metric("Ensemble Outbreak Probability", f"{ensemble_prob:.1%}")
        if ensemble_prob > 0.5:
            st.error("⚠️ OUTBREAK ALERT")
        else:
            st.success("Normal — no outbreak signal")

        colA, colB = st.columns(2)
        colA.metric("Random Forest", f"{rf_prob:.1%}")
        colB.metric("XGBoost", f"{xgb_prob:.1%}")

st.divider()
st.caption(
    "Healthcare Data Warehouse project — PySpark ETL, Azure Synapse star schema, "
    "ARIMA/Random Forest/XGBoost models, OCR clinical-note fusion, and "
    "linear-programming budget allocation, deployed via Streamlit Community Cloud."
)
