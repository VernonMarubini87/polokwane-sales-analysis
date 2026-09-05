# Polokwane Sales Intelligence

**End-to-end data science project** on 560,912 real POS transactions from
Polokwane branch — July 2023 to August 2026.

Covers the full lifecycle: recovering a broken source export → cleaning →
exploratory & business analysis → customer segmentation → product analysis →
time-series analysis → KPIs → predictive modelling → forecasting → deployment
as a Streamlit app.

**[Live app](#deployment)** · **[Final report](docs/15_final_business_report.md)** · **[Notebooks](notebooks/)**

## Highlights

- Recovered a malformed Matrix Software POS report export (not valid CSV) into
  a clean 560,912-row transaction table.
- Found that **89.5% of revenue runs through one pooled retail till account**
  — a key data-quality insight that reframes what "customer analysis" can
  mean for this business.
- RFM-segmented the 89 identifiable wholesale/trade accounts; ABC/Pareto-
  analysed all 1,615 products (155 products drive 80% of revenue).
- Trained and compared Logistic Regression, Random Forest, and XGBoost to
  flag high-value transactions — **tuned XGBoost: 0.977 ROC-AUC, 0.815 F1**.
- Forecast daily revenue with Prophet — 24.4% MAPE on a 60-day holdout,
  capturing weekly seasonality and a recurring December peak.
- Packaged everything into a 6-tab Streamlit app.

## Project structure

```
├── docs/
│   ├── 01_02_business_data_understanding.md   # Phase 1-2: objectives, data dictionary
│   └── 15_final_business_report.md            # Phase 15: findings & recommendations
├── notebooks/
│   ├── 01_data_loading_cleaning.ipynb          # Phase 3-4
│   ├── 02_exploratory_data_analysis.ipynb      # Phase 5
│   ├── 03_customer_analysis_rfm.ipynb          # Phase 7
│   ├── 04_product_analysis_abc.ipynb           # Phase 8
│   ├── 05_time_series_analysis.ipynb           # Phase 9
│   ├── 06_business_intelligence_kpis.ipynb     # Phase 10
│   ├── 07a_feature_engineering.ipynb           # Phase 6 (ML target/features)
│   ├── 07b_ml_classification_models.ipynb      # Phase 12
│   ├── 07c_hyperparameter_tuning.ipynb         # Phase 13 (GridSearchCV)
│   ├── 07d_model_evaluation_plots.ipynb        # Phase 13 (confusion matrix, ROC, importances)
│   └── 08_sales_forecasting_prophet.ipynb      # Phase 11
├── app/                                        # Phase 14: Streamlit app (self-contained)
│   ├── app.py
│   ├── requirements.txt
│   └── analysis_assets/, *.joblib, *.json, *.csv
├── data/                                       # Interim/processed data (raw CSVs gitignored, see data/README.md)
├── outputs/                                    # Plots, summary tables, metrics from the notebooks
├── models/                                     # Trained model artifacts
├── requirements.txt
└── README.md
```

## The 15-phase lifecycle

| # | Phase | Where |
|---|---|---|
| 1 | Business Understanding | `docs/01_02_business_data_understanding.md` |
| 2 | Data Understanding | `docs/01_02_business_data_understanding.md` |
| 3 | Data Loading | `notebooks/01_data_loading_cleaning.ipynb` |
| 4 | Data Cleaning | `notebooks/01_data_loading_cleaning.ipynb` |
| 5 | Exploratory Data Analysis | `notebooks/02_exploratory_data_analysis.ipynb` |
| 6 | Feature Engineering | `notebooks/07a_feature_engineering.ipynb` |
| 7 | Customer Analysis | `notebooks/03_customer_analysis_rfm.ipynb` |
| 8 | Product Analysis | `notebooks/04_product_analysis_abc.ipynb` |
| 9 | Time-Series Analysis | `notebooks/05_time_series_analysis.ipynb` |
| 10 | Business Intelligence / KPIs | `notebooks/06_business_intelligence_kpis.ipynb` |
| 11 | Sales Forecasting | `notebooks/08_sales_forecasting_prophet.ipynb` |
| 12 | Machine Learning | `notebooks/07b_ml_classification_models.ipynb` |
| 13 | Model Evaluation | `notebooks/07c_hyperparameter_tuning.ipynb`, `07d_model_evaluation_plots.ipynb` |
| 14 | Deployment | `app/app.py` |
| 15 | Final Business Report | `docs/15_final_business_report.md` |

## Key results

**Classification — "will this be a high-value transaction?"** (top 20% by value)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.891 | 0.868 | 0.538 | 0.664 | 0.918 |
| Random Forest | 0.894 | 0.857 | 0.564 | 0.680 | 0.949 |
| **XGBoost (tuned)** | **0.929** | 0.854 | **0.780** | **0.815** | **0.977** |

Top predictive features: `mass_kg` and specific product categories — value is
driven mainly by *what* and *how much* is bought.

**Forecasting — daily revenue (Prophet)**: 60-day holdout MAE R19,485 / MAPE
24.4%, refit on the full series for a 90-day forward forecast.

**Business findings**: see [the final report](docs/15_final_business_report.md)
for customer segmentation, ABC/Pareto product analysis, seasonality, and
recommendations in full.

## Setup

```bash
git clone <this-repo-url>
cd polokwane-sales-analysis
pip install -r requirements.txt
```

Place your own POS export at `data/polokwane_sales.csv` (see
`data/README.md`), then run the notebooks in order (01 → 08). Each notebook
reads what the previous ones produced from `data/`, `outputs/`, and `models/`.

## Running the app

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

The `app/` folder is self-contained (own copies of the trained models and
analysis assets), so it can be deployed on its own — e.g. to Streamlit
Community Cloud — without the rest of the repo.

## Tech stack

Python, pandas, scikit-learn, XGBoost, Prophet, statsmodels, Streamlit,
matplotlib, Jupyter.

## Data & methodology notes

- Revenue-based analysis only — no cost/margin data was available.
- 2023 (Jul–Dec only) and 2026 (Jan–Aug only) are partial calendar years;
  flagged wherever year-over-year figures are shown.
- Classification models were trained on a 120,000-row stratified subsample
  of the 535,920 positive-value transactions (development was done on a
  1-CPU-core sandbox); the full pipeline runs on the complete dataset
  without changes if more compute is available — see `07b`.
- Findings describe the Polokwane branch only.

## Author

Vernon Marubini — [github.com/VernonMarubini87](https://github.com/VernonMarubini87)
Companion project: [retail_sales_analytics](https://github.com/VernonMarubini87/retail_sales_analytics)
