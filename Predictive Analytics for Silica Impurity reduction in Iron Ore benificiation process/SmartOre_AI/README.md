# SmartOre AI – Silica Reduction Optimization

Predict % Silica Concentrate from iron‑ore flotation plant signals and provide actionable recommendations to operators with near‑real‑time latency.

## Overview
- Goal: minimize silica variability and off‑spec through live predictions and tips.
- Targets and signals are ingested from `regression.db`; the model is trained offline and loaded by Flask at startup.

## Model
- Algorithm: HistGradientBoostingRegressor
- Parameters: `max_depth=12`, `learning_rate=0.1`, `max_iter=150`, `random_state=42`
- Features (21): % Iron Feed; % Silica Feed; Starch Flow; Amina Flow; Ore Pulp Flow; Ore Pulp pH; Ore Pulp Density; Flotation Column 01–07 Air Flow; Flotation Column 02–07 Level; Avg Air Flow.
- Handling: median imputation, numeric cleaning; optional scaling for linear models.

## Metrics
- R²: ~0.657 (overall)
- Train/Test split:
  - R²_train ≈ 0.664
  - R²_test ≈ 0.623
  - MAE_test ≈ 0.511
  - RMSE_test ≈ 0.690
- SHAP (mean |value|): % Silica Feed, % Iron Feed, Amina Flow, pH, Air Flows dominate.

## Architecture
```
SQLite (regression.db)
    └─ train_model.py → models/model.pkl, scaler.pkl
Flask API (app.py) loads on startup → /api/status, /api/predict
Frontend (templates/static) calls /api/predict and renders results + trend
```

## Quick Start
1) Train once (from `SmartOre_AI/backend`):
```
python train_model.py
```
2) Run server:
```
python app.py
```
3) Open http://127.0.0.1:5000/

## API
- `GET /api/status` → metadata: loaded, version, trained_at, r² metrics, features, shap_top/importances.
- `POST /api/predict` → body with features → prediction JSON `{ silica_concentrate, risk, recommendations, elapsed_ms }`.

## UI
- Status shows version, train time, R²/R²tr/R²te, MAE, RMSE.
- Feature Importance prefers SHAP if available.
- Prediction Trend draws the last ~50 predictions.

## Retrain
```
python train_model.py
# then restart Flask so it reloads artifacts
```

## Troubleshooting
- If the UI warns “Model not available”: stop all processes on port 5000, restart from backend.
- If metrics don’t appear: hard‑refresh (cache busting applied to assets).
