# Polokwane Branch Sales — Final Business Report
Polokwane Branch Sales · Data: Jul 2023 – Aug 2026

## 1. Executive Summary

Over 1,151 days, the Polokwane branch generated **R98.3 million** in revenue
across 535,920 sales lines. The business is heavily **retail-till-driven**
(89.5% of revenue through one pooled cash account), has a **highly
concentrated product range** (9.6% of products drive 80% of revenue), and
shows **strong weekly and December seasonality**. A tuned XGBoost model can
flag high-value transactions in real time with 97.7% ROC-AUC, and a Prophet
forecast projects near-term revenue to within ~24% average error on daily
figures. All of this is packaged into a Streamlit app for ongoing self-service
use.

## 2. Revenue Performance

- Total revenue: **R98,282,584** (positive-value lines only).
- Average line value: R183; average invoice/basket value: **R11,545**
  across 8,513 unique invoices.
- Return rate: **0.23%** of revenue — low, and largely handled as negative-value
  invoices rather than formal credit notes.
- Fair year-over-year comparison (2025 vs. 2024, both full years): **+6.8%**
  revenue growth. (The headline 2024-vs-2023 figure of +80.7% is a base-period
  artifact — 2023 data only starts in July.)
- Best month on record: **December 2025** (R3.37M); weakest month: **August
  2026** (R1.93M, and only partially recorded).

**Recommendation**: use the 2025-vs-2024 comparison, not raw 2024-vs-2023,
in any future growth reporting — the partial first year otherwise
overstates performance.

## 3. Seasonality & Timing

- **Friday** is the strongest day of the week for revenue; **Sunday** the
  weakest — informs staffing and stock-delivery timing.
- **December** is the strongest month of the year (avg. daily revenue
  R104,238 vs. R77,727–89,977 in other months) — a clear
  holiday-season effect, consistent across all three Decembers in the data.
- **June** is consistently the weakest month.
- Weekly seasonality is strong and stable (seasonal component swings
  roughly ±35,000 to -73,000 ZAR/day around trend) — a dependable pattern
  for rostering and ordering.

**Recommendation**: build December stock and staffing plans around the
confirmed seasonal spike; treat June as the natural low point for
maintenance/quieter operations.

## 4. Customer Base

- Revenue is **extremely concentrated in retail till volume**: one pooled
  account ("Polokwane Cash Account - Butchery") represents 89.5% of all
  revenue. A further 11 generic/pooled accounts (staff, pensioner-discount,
  inter-branch transfer accounts) bring the "non-identifiable" share to
  94.2% of revenue.
- Only **89 of 101 accounts are identifiable wholesale/trade customers**,
  together worth R5.73M (5.8% of total revenue).
- Among those 89: **17 "Champions"** (recent, frequent, high-spend) generate
  65% of named-account revenue; 22 "Loyal" customers add another 26%; 26
  accounts are lapsed/churned.

**Recommendation**: the business's real growth lever among *identifiable*
customers is the 17 Champion accounts (protect them) and the 22 Loyal
accounts (grow them toward Champion status) — full names and RFM scores are
in `customer_rfm.csv`. Because retail till sales dominate overall revenue,
any customer-level loyalty or CRM initiative will only ever address a small
slice of total revenue unless individual retail customers start being
tracked (e.g. via a loyalty card) rather than pooled into one till account.

## 5. Product Performance

- **1,615 distinct products** sold; a classic Pareto pattern holds: **155
  products (9.6%) drive 80% of revenue** (Class A); a long tail of 1,161
  products (72%) contributes only 5%.
- Top single product: **Shoulder Ribs Smoked P/Kg** (R5.06M, 5.1% of
  revenue) — nearly double the next-best product.
- Top 10 products alone account for **28.4%** of total revenue.

**Recommendation**: prioritise stock availability and shelf space for the
155 Class A products; consider a periodic review of the 1,161 Class C
products for rationalisation (slow movers sampled in
`slow_movers_sample.csv`).

## 6. Forecasting

- Daily revenue forecast via Prophet, backtested on the most recent 60 days:
  **MAE R19,485 / RMSE R24,955 / MAPE 24.4%** against an average day of
  ~R93,500.
- The model reliably captures weekly seasonality and the recurring
  December peak, refit on the full series for a rolling 90-day forecast.
- Error is driven mainly by day-to-day noise from individual large orders,
  not by missed structural patterns.

**Recommendation**: use the forecast for medium-term (weekly/monthly)
planning rather than exact daily cash targets, given the ~24% daily MAPE.

## 7. Predictive Model: High-Value Transaction Classifier

- Target: transactions in the top 20% by value (≥R221.71).
- Best model: **tuned XGBoost** — Accuracy 0.929, Precision 0.854, Recall
  0.780, F1 0.815, **ROC-AUC 0.977** — clearly ahead of Random Forest
  (ROC-AUC 0.949) and Logistic Regression (ROC-AUC 0.918).
- Top predictive features: `mass_kg` and specific product categories —
  transaction value is driven mainly by *what* and *how much* is bought,
  more than *who* buys it or *when*.
- Full methodology, confusion matrix, ROC curve, and feature importances:
  see the ML pipeline README.

**Recommendation**: this model is accurate enough to support real-time
flagging (e.g. prioritising large orders for fulfilment, or triggering a
manager approval step) with a manageable false-positive rate (~15% of
non-high-value transactions flagged).

## 8. Deployment

All of the above is available as a Streamlit app (`polokwane_app/`) with six
tabs: Business Overview (KPIs), Transaction Predictor, Revenue Forecast,
Model Performance, Customers, and Products — so these findings can be
explored on an ongoing basis without re-running any code.

## 9. Data Caveats (carried through from earlier phases)

- Analysis is revenue-based; no cost/margin data was available, so this
  report cannot speak to profitability, only sales volume and value.
- 2023 and 2026 are partial calendar years — flagged wherever year-over-year
  figures are shown.
- Findings describe the Polokwane branch only.
- Customer-level findings only cover the 89 identifiable trade accounts;
  the dominant retail till volume (94.2% of revenue) has no
  customer-level granularity in the source data.

## 10. Portfolio Note

This project demonstrates the full data science lifecycle end-to-end:
recovering a broken source export, cleaning and validating it, exploratory
and business analysis (customer segmentation, ABC/Pareto, seasonality),
predictive modelling with proper train/test methodology and hyperparameter
tuning, time-series forecasting, and deployment as an interactive
application — all built and evaluated on a single real dataset.
