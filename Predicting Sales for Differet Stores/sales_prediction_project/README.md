# Store Sales Prediction Dashboard

A lightweight Flask application that forecasts daily sales for retail stores using a scikit‑learn pipeline. It includes a modern UI, live feature importance, confidence intervals, and a CSV export of predictions for reporting.

## Overview

- Objective: Provide quick, explainable daily sales forecasts to support staffing and inventory decisions.
- Stack: Flask, SQLite, scikit‑learn (RandomForest), Chart.js.
- Key Features: Web form for inference, model status, training endpoints, feature importance visualization, CSV report download.

## Data & Features

- Sources: Auto-detected from SQLite (`regression.db`) or CSV (`data/train.csv`).
- Core inputs: Store, DayOfWeek, Date, Customers, Promo, StateHoliday, SchoolHoliday, Open.
- Engineered time features: Year, Month, Day, WeekOfYear derived from Date.
- Robust preprocessing: Normalizes holiday codes, fills missing SchoolHoliday/Open, handles date parsing.

## Model

- Pipeline: `FunctionTransformer` (feature engineering) + `RandomForestRegressor`.
- Confidence: Derived from per-tree variance across ensemble predictions.
- Metrics & FI: Written to `models/metrics.json` and `models/feature_importance.json` after training.
- Fallback: If disk model is unavailable, an in-memory model is created so the app remains usable.

## App Structure

- `app/app.py`: Flask server with routes, model loading, training triggers, persistence.
- `app/templates/index.html`: UI layout with metrics and feature importance.
- `app/static/script.js`: Chart/table rendering and client actions.
- `training/train_model.py`: Dataset-backed training pipeline that persists model, metrics, and feature importance.
- `models/`: Model artifacts and metrics.

## API Endpoints

- `GET /`: Dashboard.
- `GET /status`: Model readiness and artifact paths.
- `GET /options`: Dropdown options from DB/CSV.
- `GET /feature-importance`: Sorted feature importance list.
- `POST /predict`: Returns `predicted_sales`, `confidence`, `interval`.
- `POST /train-sync`: Trains immediately (blocking).
- `POST /train`: Starts background training.
- `GET /download`: CSV of predictions (`store, day_of_week, date, customers, promo, holiday, predicted_sales, created_at`).
- `GET /schema`: Inspects DB tables and columns.

## Running Locally

- Start server: `python app/app.py` (uses `PORT` or defaults to 5000).
- Open: `http://127.0.0.1:5000/`
- Train: Click “Train Model” or `POST /train-sync` to generate metrics and feature importance.

## Interview Walkthrough

- Problem framing: Daily sales forecasts drive staffing and inventory; latency and explainability matter.
- Data handling: Date normalization, categorical holiday mapping, missing value strategy; options fed from DB/CSV.
- Modeling: Feature engineering inside pipeline to keep training/inference identical; RF for robustness and quick convergence.
- Evaluation: R2/MAE/RMSE saved to JSON; feature importance ranked for transparency.
- Confidence intervals: Approximation via ensemble dispersion; practical for operational decisions.
- Architecture choices: In-memory fallback guarantees availability; simple endpoints for status/train/predict; export for reporting.
- Trade-offs: RF is strong baseline; future work may include gradient boosting, temporal cross-validation, and holiday calendars.
- Business impact: Faster, explainable decisions; minimizes stockouts and overstaffing; demonstrable UI helps stakeholder buy-in.

## Artifacts

- Summary: [Project_Summary.csv](./Project_Summary.csv)
- Metrics: [models/metrics.json](./models/metrics.json)
- Feature Importance: [models/feature_importance.json](./models/feature_importance.json)
- Model: [models/model.pkl](./models/model.pkl) (may be small if environment restricts disk writes; app uses in-memory fallback)

## Future Enhancements

- Add proper time-aware validation (blocked time-series splits).
- Integrate calendar/holiday datasets for richer seasonality.
- Enable store-level models and global + per-store hybrid predictions.
- Add role-based access and audit logging for enterprise use.
