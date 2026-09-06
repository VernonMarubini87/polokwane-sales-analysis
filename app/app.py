"""
Polokwane Sales — High-Value Transaction Predictor & Revenue Forecast
Polokwane Branch — Meat Wholesale & Retail

Run with:  streamlit run app.py
(Requires: streamlit, pandas, numpy, scikit-learn, xgboost, prophet, joblib, matplotlib, plotly)
"""

import json
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# All file paths below are resolved relative to this script's own location,
# not the process's current working directory -- Streamlit Community Cloud
# always runs with cwd set to the repo root regardless of which file you
# point it at, so a bare relative path like "model.joblib" only works when
# you happen to `cd app` first (as you would running this locally).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def p(*parts):
    return os.path.join(BASE_DIR, *parts)


st.set_page_config(page_title="Polokwane Sales Intelligence", layout="wide")

# ----------------------------------------------------------------------
# Load artifacts (cached so they only load once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(p("best_model_xgboost_tuned.joblib"))
    prophet_model = joblib.load(p("prophet_model.joblib"))
    with open(p("app_reference_values.json")) as f:
        ref = json.load(f)
    with open(p("debtor_name_map.json")) as f:
        debtor_map = json.load(f)
    with open(p("product_desc_map.json")) as f:
        product_map = json.load(f)
    daily_sales = pd.read_csv(p("daily_sales.csv"), parse_dates=["ds"])
    forecast = pd.read_csv(p("forecast_90d.csv"), parse_dates=["ds"])
    top10 = pd.read_csv(p("top10_feature_importance.csv"))
    comparison = pd.read_csv(p("model_comparison_test.csv"), index_col=0)
    with open(p("best_model_params.json")) as f:
        best_params = json.load(f)
    with open(p("analysis_assets", "kpis.json")) as f:
        kpis = json.load(f)
    segments = pd.read_csv(p("analysis_assets", "customer_segments_summary.csv"))
    top10_customers = pd.read_csv(p("analysis_assets", "top10_named_customers.csv"))
    abc_summary = pd.read_csv(p("analysis_assets", "product_abc_summary.csv"))
    abc_full = pd.read_csv(p("analysis_assets", "product_abc_full.csv"))
    yearly = pd.read_csv(p("analysis_assets", "yearly_summary.csv"))
    return (model, prophet_model, ref, debtor_map, product_map, daily_sales, forecast, top10, comparison,
            best_params, kpis, segments, top10_customers, abc_summary, abc_full, yearly)


(model, prophet_model, ref, debtor_map, product_map, daily_sales, forecast, top10, comparison,
 best_params, kpis, segments, top10_customers, abc_summary, abc_full, yearly) = load_artifacts()

st.title("🥩 Polokwane Sales Intelligence")
st.caption("Polokwane Branch — Meat Wholesale & Retail")

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Business Overview", "🔮 Predict Transaction Value", "📈 Revenue Forecast",
    "📊 Model Performance", "👥 Customers", "📦 Products"
])

# ----------------------------------------------------------------------
# TAB 0 — Business overview / KPIs
# ----------------------------------------------------------------------
with tab0:
    st.subheader("Key performance indicators")
    st.caption("Jul 2023 – Aug 2026 · 2023 and 2026 are partial calendar years (see note below)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", f"R{kpis['total_revenue_zar']:,.0f}")
    c2.metric("Transactions", f"{kpis['total_transactions']:,}")
    c3.metric("Avg basket value", f"R{kpis['avg_basket_value_zar']:,.0f}")
    c4.metric("Return rate", f"{kpis['return_rate_pct']:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("YoY growth 2025 vs 2024", f"{kpis['yoy_revenue_growth_2025_pct']:.1f}%")
    c6.metric("Active named customers/mo", f"{kpis['active_named_customers_avg_per_month']:.1f}")
    c7.metric("Revenue from top 9.6% of products", f"{kpis['class_A_products_pct_of_revenue']:.0f}%")
    c8.metric("Products / customer accounts", f"{kpis['unique_products_sold']:,} / {kpis['unique_customer_accounts']}")

    st.info(
        "**2024 vs 2023 YoY growth (80.7%) is inflated** — 2023 only covers Jul–Dec (6 months) "
        "vs a full 12 months in 2024. **2025 vs 2024 (+6.8%) is the fair comparison**, both full years."
    )

    st.subheader("Revenue trend")
    st.image(p("analysis_assets", "monthly_revenue_trend.png"), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.image(p("analysis_assets", "revenue_by_dow.png"), caption="Revenue by day of week", use_container_width=True)
    with col2:
        st.image(p("analysis_assets", "seasonality_month_of_year.png"), caption="Seasonality by month", use_container_width=True)

    st.subheader("Year-over-year comparison")
    st.dataframe(yearly, use_container_width=True)
    st.image(p("analysis_assets", "yoy_comparison.png"), use_container_width=True)

# ----------------------------------------------------------------------
# TAB 1 — Prediction
# ----------------------------------------------------------------------
with tab1:
    st.subheader("Will this transaction be high-value?")
    st.write(
        "Predicts whether a sale falls in the **top 20% by value** "
        "(≥ R221.71), using the tuned XGBoost classifier."
    )

    col1, col2 = st.columns(2)
    with col1:
        debtor_display = [f"{code} — {debtor_map.get(code, code)}" for code in ref["debtor_codes"]]
        debtor_choice = st.selectbox("Customer / Debtor account", debtor_display)
        debtor_code = debtor_choice.split(" — ")[0]

        product_grp = st.selectbox("Product (top 40, or OTHER)", ref["product_groups"])

        qty = st.number_input("Quantity", min_value=0.0, max_value=700.0, value=1.0, step=1.0)
        mass_kg = st.number_input("Mass (kg)", min_value=0.0, max_value=1500.0, value=0.0, step=0.5)

    with col2:
        date_input = st.date_input("Transaction date", value=pd.Timestamp.today())
        doc_type = st.selectbox("Document type", ref["doc_types"])

    if st.button("Predict", type="primary"):
        ts = pd.Timestamp(date_input)
        row = pd.DataFrame([{
            "qty": qty,
            "mass_kg": mass_kg,
            "year": ts.year,
            "month": ts.month,
            "day": ts.day,
            "day_of_week": ts.dayofweek,
            "quarter": ts.quarter,
            "is_weekend": int(ts.dayofweek >= 5),
            "debtor_code": debtor_code,
            "product_code_grp": product_grp,
            "doc_type": doc_type,
        }])

        proba = model.predict_proba(row)[0, 1]
        pred = model.predict(row)[0]

        st.divider()
        if pred == 1:
            st.success(f"**High-value transaction predicted** — probability {proba:.1%}")
        else:
            st.info(f"**Not high-value** — probability of high-value: {proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

# ----------------------------------------------------------------------
# TAB 2 — Forecast
# ----------------------------------------------------------------------
with tab2:
    st.subheader("Daily revenue forecast (Prophet)")
    horizon = st.slider("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=7)

    hist = daily_sales.copy()
    fut = forecast[forecast["ds"] > hist["ds"].max()].head(horizon)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(hist["ds"], hist["y"], label="Actual", color="#334155", linewidth=1, alpha=0.7)
    ax.plot(forecast["ds"], forecast["yhat"], label="Prophet fit / forecast", color="#2563eb", linewidth=1.5)
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"],
                     color="#2563eb", alpha=0.15, label="Confidence interval")
    ax.axvline(hist["ds"].max(), color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(hist["ds"].max() - pd.Timedelta(days=180), fut["ds"].max() if len(fut) else hist["ds"].max())
    ax.set_ylabel("Revenue (ZAR)")
    ax.legend(loc="upper left")
    st.pyplot(fig)

    st.write(f"Projected revenue, next {horizon} days:")
    total = fut["yhat"].sum()
    st.metric("Forecast total revenue", f"R{total:,.0f}")
    st.dataframe(fut.rename(columns={"ds": "date", "yhat": "forecast", "yhat_lower": "low", "yhat_upper": "high"}),
                 use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# TAB 3 — Model performance
# ----------------------------------------------------------------------
with tab3:
    st.subheader("Classification model comparison (test set)")
    st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)

    st.subheader("Best model: Tuned XGBoost")
    st.json(best_params)

    col1, col2 = st.columns(2)
    with col1:
        st.image(p("confusion_matrix.png"), caption="Confusion Matrix")
    with col2:
        st.image(p("roc_curve.png"), caption="ROC Curve")

    st.subheader("Top 10 feature importances")
    st.dataframe(top10, use_container_width=True, hide_index=True)
    st.image(p("feature_importance.png"), caption="Top 10 Feature Importances")

# ----------------------------------------------------------------------
# TAB 4 — Customer analysis
# ----------------------------------------------------------------------
with tab4:
    st.subheader("Customer concentration")
    st.warning(
        "89.5% of total revenue runs through **one pooled retail till account** "
        "(\"Polokwane Cash Account - Butchery\"), not an identifiable customer. "
        "12 of 101 accounts are generic pooled tills/staff/discount accounts across branches. "
        "The segmentation below covers the **89 identifiable wholesale/trade accounts** only "
        "(R5.7M of the R98.3M total)."
    )

    st.image(p("analysis_assets", "top15_customers.png"), use_container_width=True)

    st.subheader("RFM segments (named trade accounts)")
    st.dataframe(segments, use_container_width=True)
    st.image(p("analysis_assets", "customer_segments.png"), use_container_width=True)

    st.subheader("Top 10 named customers")
    st.dataframe(top10_customers, use_container_width=True, hide_index=True)

# ----------------------------------------------------------------------
# TAB 5 — Product analysis
# ----------------------------------------------------------------------
with tab5:
    st.subheader("ABC / Pareto analysis")
    st.write(
        f"**{int(abc_summary.loc[abc_summary['abc_class']=='A','products'].iloc[0])} products** "
        f"({abc_summary.loc[abc_summary['abc_class']=='A','pct_of_products'].iloc[0]:.1f}% of the range) "
        f"drive **80% of revenue** (Class A). Class C — 72% of products — contributes only 5%."
    )
    st.dataframe(abc_summary, use_container_width=True)
    st.image(p("analysis_assets", "product_pareto.png"), use_container_width=True)

    st.subheader("Top 15 products by revenue")
    st.image(p("analysis_assets", "top15_products.png"), use_container_width=True)

    st.subheader("Browse all products")
    class_filter = st.multiselect("ABC class", options=["A", "B", "C"], default=["A"])
    filtered = abc_full[abc_full["abc_class"].isin(class_filter)] if class_filter else abc_full
    st.dataframe(
        filtered[["product_desc", "revenue", "qty", "transactions", "abc_class"]].head(200),
        use_container_width=True, hide_index=True
    )

st.divider()
st.caption("Built as an end-to-end data science portfolio project on Polokwane branch sales data.")
