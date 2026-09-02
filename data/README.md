# Data

Raw and cleaned CSVs are excluded from this repo via `.gitignore` (the raw
export is ~340MB, the cleaned file ~70MB — both over what's reasonable to
version in git).

**To run the notebooks from scratch**, place your own POS export at:
```
data/polokwane_sales.csv
```
then run `notebooks/01_data_loading_cleaning.ipynb` — it regenerates
everything else in this folder (`parsed_raw.csv`, `polokwane_sales_clean.csv`,
`model_ready.parquet`, etc.) and in `../outputs/`.

Everything else in this folder **is** committed (all under a few MB):
- `model_ready.parquet` — engineered features + target for the ML pipeline
- `train_test_split.joblib` — the exact 80/20 stratified split used
- `daily_sales.csv`, `forecast_90d.csv` — Prophet forecasting inputs/outputs
