# NetPulse AI – Smart Signal Strength Prediction

NetPulse AI predicts cellular signal strength (dBm) from location, network type and multi‑radio measurements. It exposes a lightweight web UI and REST APIs for training, KPIs and inference.

## Quick Start

1. Install Python 3.10+ and run:

```bash
pip install -r requirements.txt
```

2. Start the backend:

```bash
python NetPulse_AI\backend\app.py
```

3. Open the app:

```
http://127.0.0.1:5000/
```

## Training

- Click “Train Model” in the header, or POST to `/api/train`.
- Requires `regression.db` with a table containing:
  - Target: `Signal Strength (dBm)`
  - Features: Latitude, Longitude, Network Type, Data Throughput (Mbps), Latency (ms), Signal Quality (%), BB60C Measurement (dBm), srsRAN Measurement (dBm), BladeRFxA9 Measurement (dBm), optionally Locality.
- The training pipeline handles missing numeric values and encodes categoricals, selecting the best among Ridge, RandomForest and HistGradientBoosting regressors.
- KPIs and schema metadata are written to `NetPulse_AI/backend/models/meta.json`.

## REST API

- `GET /api/metrics` – Returns KPIs (R², MAE, RMSE, rows, version, last_trained). Use `?refresh=1` to reload cache.
- `POST /api/train` – Trains a new model and updates KPIs.
- `POST /api/predict` – Body (JSON):

```json
{
  "locality": "Patna",
  "latitude": 25.6,
  "longitude": 85.1,
  "network_type": "4G",
  "throughput_mbps": 12.5,
  "latency_ms": 45,
  "signal_quality_pct": 78,
  "bb60c_dbm": -86,
  "srsran_dbm": -88,
  "bladerf_dbm": -87
}
```

Response includes `predicted_dbm`, `coverage`, `health_score`, `inference_ms`, and `ci_dbm` (±MAE).

- `GET /api/importance` – Top features with importance scores.
- `GET /api/table/columns?table=signal_metrics` – Column names from the SQLite table.
- `GET /api/table/distinct?table=signal_metrics&column=Locality&limit=100` – Distinct values for dropdowns.

## KPIs

- R², MAE (dBm), RMSE (dBm), Rows Trained, Model Version/Type, Last Trained.
- Shown in the header and driven by `meta.json`.

## File Map

- Backend app: NetPulse_AI/backend/app.py  
- Training: NetPulse_AI/backend/train_signal_model.py  
- Baseline bootstrap (optional): NetPulse_AI/backend/_bootstrap_model.py  
- Frontend: NetPulse_AI/frontend/templates/index.html, NetPulse_AI/frontend/static/script.js

## Troubleshooting

- If KPIs show “Not Trained”, click “Train Model” or POST `/api/train`.
- If scikit‑learn encoder parameters differ by version, the training code adapts automatically.
- For missing input fields at prediction time, numeric features are imputed and categoricals are encoded; predictions still work.

