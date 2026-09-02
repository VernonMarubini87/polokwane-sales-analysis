# Phase 1–2: Business & Data Understanding

## Phase 1: Business Understanding

**Business context**
Target Meats Pty Ltd, trading as Eskort's Polokwane branch, is a meat/butchery
retail and wholesale operation. Sales are recorded through a Matrix Software
POS system and exported as itemised transaction reports.

**Business problem**
Management currently has raw transaction exports but no structured view of:
- which products and customers drive revenue,
- how sales behave over time (seasonality, trend, day-of-week effects),
- which customers are high-value vs. at-risk of churning,
- what a "big sale" looks like, so staff/systems can flag it early,
- what revenue to expect in the near term for planning and stock ordering.

**Objectives**
1. Turn the raw POS export into a clean, analysis-ready dataset.
2. Quantify sales performance: revenue, volume, and trend over the last ~3 years.
3. Segment customers by value and purchasing behaviour (who to retain, who to grow).
4. Identify which products carry the business (Pareto/ABC analysis).
5. Characterise time-series patterns (weekly/monthly seasonality) and forecast
   near-term revenue.
6. Build a classifier that flags high-value transactions in real time.
7. Package the above into KPIs and a self-service Streamlit dashboard so
   non-technical staff can use the findings without touching code.

**Success criteria**
- A cleaned dataset with documented quality issues and fixes.
- Clear, quantified answers to "who are our best customers/products" and
  "how is revenue trending".
- A forecasting model with a stated, honest error rate (not just a plot).
- A classification model evaluated on held-out data with standard metrics.
- A working, shareable Streamlit app covering both the analytics and the models.

**Constraints**
- Single-branch, single-POS-system dataset — findings describe Polokwane
  only, not the wider Eskort network.
- No cost/margin data — analysis is revenue-based, not profitability-based.
- Modelling was done in a resource-constrained sandbox (1 CPU core); noted
  wherever it affected methodology (see the ML pipeline README).

## Phase 2: Data Understanding

**Source**: `polokwane_sales_clean.csv` — cleaned from a raw Matrix Software
"Sales By Product Listing Detailed" report export (see the earlier data
cleaning phase for how the malformed source file was parsed).

**Grain**: one row per product line per document (an invoice or credit note
can span multiple rows, one per product).

**Size**: 560,912 rows × 11 columns, spanning **2023-07-01 to 2026-08-25**
(1,151 days).

**Columns**

| Column | Type | Description |
|---|---|---|
| `date` | datetime | Transaction date |
| `doc_type` | category | `Invoices` (560,840 rows) or `Credit Notes` (72 rows) |
| `doc_number` | int | Document/invoice number |
| `debtor_code` | string | Customer account code (101 unique) |
| `debtor_name` | string | Customer account name |
| `product_code` | string | Product code (1,617 unique) |
| `product_desc` | string | Product description |
| `qty` | float | Quantity sold (units; negative = return) |
| `mass_kg` | float | Mass in kilograms |
| `value_zar` | float | Line value in ZAR (negative = return) |
| `avg_price_per_kg` | float | Value ÷ mass for that line |

**Headline numbers**
- Total revenue (positive-value lines only): **R98,282,584** over the period.
- 101 customer accounts, 1,617 distinct products.
- 99.99% of rows are `Invoices`; `Credit Notes` are a tiny fraction (72 rows,
  0.01%) — returns are mostly processed as negative-value `Invoices` rather
  than through the `Credit Notes` document type (noted during cleaning).

**Known data characteristics (carried over from the cleaning phase)**
- ~24,500 rows have `value_zar == 0` (largely promotional/free items).
- ~8,400 rows have `qty == 0` (largely till-rounding adjustments,
  `product_code == 'ROUNDING'`).
- 372 `Invoices` rows carry negative qty/value — in-store returns processed
  through the invoice flow rather than as credit notes.
- `debtor_code` values suggest a mix of account types (e.g. `INTC-*` internal
  transfer accounts, `POL*` cash/pensioner discount accounts, named
  wholesale/retail customer accounts) — relevant context for the customer
  analysis phase.

These characteristics are retained as-is in the cleaned file (they're
legitimate business activity, not errors) and are handled explicitly in each
analysis phase below rather than filtered out silently.
