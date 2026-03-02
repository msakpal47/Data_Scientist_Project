# Health Insurance Cost Intelligence

Predicts annual medical insurance charges from demographic and lifestyle inputs. Includes training workflow, dashboard UI, leaderboard with multiple models, feature importance, and CSV report export.

## Overview
- Problem: Estimate insurance premiums (charges) using age, BMI, smoker status, coverage level, medical history and related factors.
- Dataset: SQLite `regression.db` → table `Insurance_Prediction`
- Target column: `charges`
- Artifacts directory: `models/` (model.pkl, scaler.pkl, columns.pkl, feature_importance.json, leaderboard.json)

## Features
- Inputs: age, bmi, children, gender, smoker, region, medical_history, family_medical_history, exercise_frequency, occupation, coverage_level
- Pipeline:
  - Numeric: SimpleImputer(strategy=median) → StandardScaler
  - Categorical: SimpleImputer(strategy=most_frequent) → OneHotEncoder(drop='first')
- Candidate models: RandomForest, ExtraTrees, GradientBoosting, Ridge, Lasso, ElasticNet, SGD
- Selection: Best by validation/Test R²; results stored in `models/leaderboard.json`

## Results
- Best model: ridge
- Key metrics (test):
  - R² ≈ 0.8246
  - MAE ≈ 1463.6
  - RMSE ≈ 1845.5
- Top drivers (feature importance): coverage_level_Premium, smoker_yes, medical_history_Heart disease, exercise_frequency_Never, region_southwest

## How to Run (Dashboard)
1. Activate Python environment
2. Start the app:
   - `python app/app.py`
3. Open in browser:
   - Dashboard: `http://127.0.0.1:5000/`
   - Training: `http://127.0.0.1:5000/setup`

## Train From Database
1. Go to `/setup`
2. Select table `Insurance_Prediction`
3. Target `charges`
4. Click “Train from Database”
5. Artifacts saved under `models/`

## Endpoints
- `GET /` – Dashboard home
- `POST /predict` – Predict from form/JSON
- `POST /predict-json` – Predict from raw JSON
- `GET /feature-importance` – Returns importance dict (with automatic fallback)
- `GET /model/leaderboard` – Returns leaderboard metrics

## Files of Interest
- `app/app.py` – Flask app and endpoints
- `app/templates/index.html` – Dashboard UI (KPI cards, form, importance, leaderboard)
- `training/train_model.py` – Training and model selection, importance computation
- `models/leaderboard.json` – Metrics for comparison
- `models/feature_importance.json` – Cached importance scores
- `project_summary.csv` – One‑row summary (problem → impact)

## Business Impact
- Faster and consistent premium quoting
- Transparent reasoning via feature importance
- Exportable artifacts for audit and communication

## Troubleshooting
- If the dashboard shows “--” for KPIs, ensure `models/leaderboard.json` exists by running training.
- If feature importance appears empty, confirm `feature_importance.json` exists or let `/feature-importance` compute the fallback from the database.
