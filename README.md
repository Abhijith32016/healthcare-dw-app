# Healthcare Data Warehouse — Disease Surveillance & Outbreak Prediction

Live business analytics + ML deployment of a healthcare data warehouse project.

**Pipeline:** PySpark ETL → Azure Synapse star schema → ARIMA/Random Forest/XGBoost outbreak models → OCR/NLP clinical-note fusion → linear-programming budget allocation → this Streamlit app.

## What's in this repo

- `app.py` — the deployed Streamlit app (dashboard + business decision support + prediction tool)
- `requirements.txt` — Python dependencies for Streamlit Cloud
- `data/` — small, pre-aggregated CSVs exported from the full pipeline (the full raw/processed data lives in Google Drive / Azure, not here — this repo only ships what the live app needs)
- `models/` — trained model artifacts (`rf_outbreak_model.pkl`, `xgb_outbreak_model.json`)

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Live app

Deployed at: `https://<your-app-name>.streamlit.app` (fill in after deployment)
