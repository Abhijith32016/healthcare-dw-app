import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Outbreak Intelligence Platform", layout="wide")

# ──────────────────────────────────────────────────────────────────
# STYLING
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #FAFBFC; }
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #14213D; }
    .subtitle { color: #5A6472; font-size: 1.05rem; margin-top: -0.6rem; margin-bottom: 1.5rem; }
    .section-note { color: #5A6472; font-size: 0.92rem; line-height: 1.5; }
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E6E9EF;
        border-radius: 10px;
        padding: 14px 18px;
    }
    .model-card {
        background-color: #FFFFFF;
        border: 1px solid #E6E9EF;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }
    .model-card-title { font-weight: 600; font-size: 1.05rem; color: #14213D; }
    .model-card-sub { color: #8A93A2; font-size: 0.82rem; margin-top: -4px; }
    .alert-box {
        border-radius: 10px;
        padding: 16px 20px;
        font-weight: 600;
        margin-top: 10px;
    }
    .alert-high { background-color: #FDECEC; color: #B3261E; border: 1px solid #F5C2C0; }
    .alert-normal { background-color: #EAF6EE; color: #1E6B3A; border: 1px solid #C3E6CE; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────
st.title("Outbreak Intelligence Platform")
st.markdown(
    '<div class="subtitle">Live disease surveillance, resource-allocation planning, '
    'and outbreak risk assessment — built on a cloud data pipeline processing 326,000+ '
    'epidemiological records.</div>',
    unsafe_allow_html=True
)

tab1, tab2, tab3 = st.tabs(["Surveillance Overview", "Decision Support", "Risk Assessment Tool"])

# ──────────────────────────────────────────────────────────────────
# TAB 1 — SURVEILLANCE OVERVIEW
# ──────────────────────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns(2)

    daily = pd.read_csv("data/daily_global_cases.csv", parse_dates=["report_date"])
    fig1 = px.area(daily, x="report_date", y="new_cases", title="Global Daily New Cases")
    fig1.update_traces(line_color="#14213D", fillcolor="rgba(20,33,61,0.10)")
    fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=15)
    col1.plotly_chart(fig1, use_container_width=True)

    top10 = pd.read_csv("data/top10_countries.csv")
    fig2 = px.bar(
        top10, x="new_cases", y="country_name", orientation="h",
        title="Top 10 Countries by Total Cases", color="new_cases",
        color_continuous_scale=["#C7D1E0", "#14213D"]
    )
    fig2.update_layout(yaxis={"categoryorder": "total ascending"}, plot_bgcolor="white",
                        paper_bgcolor="white", title_font_size=15, coloraxis_showscale=False)
    col2.plotly_chart(fig2, use_container_width=True)

    severity = pd.read_csv("data/severity_distribution.csv")
    fig3 = px.pie(
        severity, names="severity_level", values="count",
        title="Outbreak Severity Distribution", color="severity_level",
        color_discrete_map={"critical": "#B3261E", "high": "#E07A3F",
                             "medium": "#E8C547", "low": "#2E7D4F"}
    )
    fig3.update_layout(paper_bgcolor="white", title_font_size=15)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="section-note">Severity is derived from a statistical anomaly score '
        '(how far daily case counts deviate from each region\'s own recent baseline), not from '
        'raw case volume alone — a small country with an unusual spike is flagged the same way '
        'as a large one.</div>',
        unsafe_allow_html=True
    )

# ──────────────────────────────────────────────────────────────────
# TAB 2 — DECISION SUPPORT
# ──────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Hospital Capacity Planning")
    st.markdown(
        '<div class="section-note">Projected bed demand is derived from case forecasts, '
        'translated into hospitalizations using an assumed hospitalization rate and average '
        'length of stay, then compared against baseline capacity.</div>',
        unsafe_allow_html=True
    )
    capacity = pd.read_csv("data/capacity_planning.csv", parse_dates=["date"])

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=capacity["date"], y=capacity["projected_bed_demand"],
        name="Projected demand", line=dict(color="#14213D", width=2)
    ))
    fig4.add_trace(go.Scatter(
        x=capacity["date"], y=capacity["baseline_capacity"],
        name="Available capacity", line=dict(color="#8A93A2", dash="dash")
    ))
    fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                        title="Projected Bed Demand vs. Baseline Capacity", title_font_size=15)
    st.plotly_chart(fig4, use_container_width=True)

    shortfall_days = int((capacity["capacity_gap"] > 0).sum())
    peak_demand = capacity["projected_bed_demand"].max()
    m1, m2 = st.columns(2)
    m1.metric("Days with Projected Shortfall", shortfall_days)
    m2.metric("Peak Projected Bed Demand", f"{peak_demand:.0f}")

    st.divider()
    st.subheader("Response Budget Allocation")
    st.markdown(
        '<div class="section-note">A fixed response budget is allocated across the '
        'highest-risk regions using constrained optimization — maximizing risk coverage '
        'subject to a total budget limit and a per-region cap, so funding isn\'t concentrated '
        'in a single location.</div>',
        unsafe_allow_html=True
    )
    budget = pd.read_csv("data/budget_allocation.csv")

    fig5 = px.bar(
        budget.sort_values("allocated_budget", ascending=False),
        x="country_name", y="allocated_budget",
        title="Response Budget Allocation by Region",
        color="risk_score", color_continuous_scale=["#C7D1E0", "#B3261E"]
    )
    fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white", title_font_size=15)
    st.plotly_chart(fig5, use_container_width=True)
    st.dataframe(budget, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────
# TAB 3 — RISK ASSESSMENT TOOL
# ──────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Outbreak Risk Assessment")
    st.markdown(
        '<div class="section-note">Enter today\'s case figures for a region to get a risk '
        'assessment from two models trained on historical outbreak patterns. The two models '
        'make different tradeoffs, explained below, so both are shown side by side rather than '
        'collapsed into one number.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)
    new_cases_today = c1.number_input("New cases today", min_value=0, value=5000, step=100)
    new_deaths_today = c2.number_input("New deaths today", min_value=0, value=20, step=1)
    rolling_7d = c3.number_input("7-day rolling average cases", min_value=0, value=4000, step=100)

    if st.button("Run Assessment", type="primary"):
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

        colA, colB = st.columns(2)

        with colA:
            st.markdown(
                f'<div class="model-card">'
                f'<div class="model-card-title">Early-Warning Assessment</div>'
                f'<div class="model-card-sub">Tuned to catch more true outbreaks, at the cost of more false alarms</div>'
                f'<h2 style="margin-top:10px; margin-bottom:0;">{rf_prob:.0%}</h2>'
                f'</div>', unsafe_allow_html=True
            )

        with colB:
            st.markdown(
                f'<div class="model-card">'
                f'<div class="model-card-title">High-Confidence Assessment</div>'
                f'<div class="model-card-sub">Tuned to minimize false alarms, at the cost of missing some true outbreaks</div>'
                f'<h2 style="margin-top:10px; margin-bottom:0;">{xgb_prob:.0%}</h2>'
                f'</div>', unsafe_allow_html=True
            )

        if rf_prob > 0.5 or xgb_prob > 0.5:
            st.markdown(
                '<div class="alert-box alert-high">At least one model flags elevated outbreak risk '
                'for these figures. Recommended: review alongside recent regional trend data.</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="alert-box alert-normal">Both models assess this as within normal range.</div>',
                unsafe_allow_html=True
            )

        with st.expander("Technical details"):
            st.markdown(
                f"- **Early-Warning Assessment** is produced by a Random Forest classifier "
                f"(200 trees, class-balanced). Cross-validated recall: 0.80, precision: 0.56.\n"
                f"- **High-Confidence Assessment** is produced by an XGBoost classifier "
                f"(gradient-boosted, 500 rounds, class-weighted). Cross-validated recall: 0.57, precision: 0.84.\n"
                f"- Both were evaluated via 5-fold stratified cross-validation on a held-out set "
                f"with a 4.45% outbreak-positive base rate.\n"
                f"- Random Forest raw probability: {rf_prob:.4f} · XGBoost raw probability: {xgb_prob:.4f}"
            )

st.divider()
st.markdown(
    '<div class="section-note">Built on a PySpark ETL pipeline, an Azure Synapse lakehouse, '
    'and two independently evaluated classification models, with a linear-programming layer '
    'for constrained resource allocation. Full technical writeup available on request.</div>',
    unsafe_allow_html=True
)
