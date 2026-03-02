# Project Explanation – NetPulse AI

## Goal
Estimate cellular signal strength (dBm) based on location, network type and multi‑radio measurements, then expose explainable KPIs and a simple prediction workflow for operations teams.

## Architecture

- Web server: Flask app serving HTML/CSS/JS and a JSON API. See [app.py](file:///e:/Data_Scientist_Project/Regression_Projects/Captsone_Project%20-%20Regression/Signal%20Strength%20Prediction/NetPulse_AI/backend/app.py).
- Frontend: Static page + script powering forms, KPIs and charts. See [index.html](file:///e:/Data_Scientist_Project/Regression_Projects/Captsone_Project%20-%20Regression/Signal%20Strength%20Prediction/NetPulse_AI/frontend/templates/index.html) and [script.js](file:///e:/Data_Scientist_Project/Regression_Projects/Captsone_Project%20-%20Regression/Signal%20Strength%20Prediction/NetPulse_AI/frontend/static/script.js).
- Model store: `NetPulse_AI/backend/models/` holding `best_model.pkl` and `meta.json`.
- Data store: `regression.db` (SQLite) with a table carrying the target `Signal Strength (dBm)` and required features.

## Data and Features

- Target: `Signal Strength (dBm)`.
- Features: `locality`, `latitude`, `longitude`, `network_type`, `throughput_mbps`, `latency_ms`, `signal_quality_pct`, `bb60c_dbm`, `srsran_dbm`, `bladerf_dbm`.
- Data preparation handles: missing numeric values via median imputation; categorical encoding with OneHotEncoder; optional `Locality` column.

## Training Pipeline

- Entrypoint: [train_signal_model.py](file:///e:/Data_Scientist_Project/Regression_Projects/Captsone_Project%20-%20Regression/Signal%20Strength%20Prediction/NetPulse_AI/backend/train_signal_model.py).
- Candidates: Ridge, RandomForest, HistGradientBoosting. Best is selected by R² on a hold‑out set.
- Outputs:
  - `best_model.pkl` – serialized estimator and preprocessing.
  - `meta.json` – KPIs (R², MAE, RMSE), feature list, rows trained, version, timestamp, and feature importance.

## API Surface

- `GET /api/metrics` – KPIs and metadata. Use `?refresh=1` to bypass cache.
- `POST /api/train` – Trains and rewrites model + meta.
- `POST /api/predict` – Returns `predicted_dbm`, coverage label, health score, inference time and ±MAE.
- `GET /api/importance` – Feature importance for visualization.

## UI Flow

1. On load, the page calls `/api/metrics` and populates the KPI cards.
2. “Train Model” triggers `/api/train`, then KPIs refresh.
3. Prediction form posts to `/api/predict` and renders:
   - Predicted dBm with animated counter
   - Coverage label (Excellent/Good/Weak/Poor)
   - Health score gauge
   - Optimization suggestions and a small trend chart

## KPIs and Interpretation

- R² – proportion of variance explained (↑ is better).
- MAE/ RMSE (dBm) – average/penalized error magnitude (↓ is better).
- Rows Trained – dataset size used for the final model.
- Model Version – human‑readable tag for the active estimator.

## Operating Notes

- If no database is present, you can bootstrap a baseline using `_bootstrap_model.py` or proceed once the SQLite table is available.
- The training code is tolerant to scikit‑learn version differences and imputes missing numeric values to prevent NaN errors.

