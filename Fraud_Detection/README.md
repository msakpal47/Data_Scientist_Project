# Fraud Radar Dashboard

## Overview
- Real-time dashboard to inspect transactions, evaluate model performance, and score hypothetical transfers
- Built with Flask + vanilla JS; model trained with scikit-learn; SHAP explanations for local interpretability

## Features
- Portfolio snapshot: class balance, fraud rate, imbalance ratio
- Model performance: Accuracy, Precision, Recall, F1, ROC‑AUC, confusion matrix, last evaluated timestamp
- Threshold tools: PR curve, cost curve, optimal threshold suggestion, live confusion/loss simulation
- Business recommendation: “Prevents X fraud with Y false alerts” at the suggested threshold
- Single transaction scoring: probability, decision, risk level, top contributing feature, local explanation
- SHAP waterfall visualization for a single transaction (fallback explainer if SHAP unavailable)
- Top features: global feature importances from trained model
- Dataset type filter; export filtered CSV
- Probability histogram: fraud vs non‑fraud probability distribution
- Model comparison: HGB vs LogisticRegression vs RandomForest with “Best F1” / “Best ROC‑AUC” badges

## Setup
- Python 3.10+ recommended
- Install dependencies:
  - `pip install -r requirements.txt`

## Quickstart
- Create venv (optional): `python -m venv .venv && .venv\Scripts\activate`
- Install deps: `pip install -r requirements.txt`
- Train: `python -m src.train_model`
- Run: `python -m app.app`
- Open: `http://localhost:8501/`

### Windows (PowerShell) one‑liners
- Create and activate venv:
  - `python -m venv .venv`
  - `.venv\Scripts\Activate.ps1`
- Install & run:
  - `pip install -r requirements.txt`
  - `python -m src.train_model`
  - `python -m app.app`

### Notes on data and models
- Expects a SQLite database at `data/classification.db` with a table containing the transaction schema.
- Training produces artifacts in `models/`: `fraud_model.pkl`, `feature_columns.json`, `feature_importances.json`, `metrics.json`, `pr_curve.json`, `test_eval.json`.
- The app uses these artifacts for metrics, threshold tools, histograms and explanations.

## Train
- Generate model and artifacts:
  - `python -m src.train_model`
- Outputs in `models/`: `fraud_model.pkl`, `feature_columns.json`, `feature_importances.json`, `metrics.json`, `pr_curve.json`

## Run
- Start the app:
  - `python -m app.app`
- Open: `http://localhost:8501/`

## API
- `GET /api/summary?tx_type=...`
- `GET /api/metrics`
- `GET /api/feature_importances`
- `GET /api/pr_curve`
- `GET /api/threshold_suggestion`
- `GET /api/confusion_sim?threshold=...`
- `GET /api/cost_sim?threshold=...&cost_fp=...&cost_fn=...`
- `GET /api/optimal_threshold?cost_fp=...&cost_fn=...`
- `GET /api/cost_curve?cost_fp=...&cost_fn=...`
- `GET /api/rate_sim?threshold=...`
- `GET /api/model_compare`
- `GET /api/prob_histogram`
- `POST /api/predict`
- `POST /api/explain`

## Notes
- Metrics reflect held‑out validation data; they update only after retraining
- Threshold slider drives simulation; use optimal suggestion for cost‑aware decisions
- SHAP requires background data; robust fallback uses baseline perturbation; API guarantees valid JSON outputs
