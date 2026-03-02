# Loan Eligibility Prediction – Microfinance

## Overview
- Supervised binary classification to predict if a telecom customer takes a short-term micro-loan (label=1) or not (label=0).
- End-to-end solution: data in SQLite, training pipeline, Flask API, and web UI with live scoring.
- Business goals: improve targeting precision, reduce credit risk, and optimize campaign ROI.

## Quick Start
1) Run the web app

```bash
python app.py
# App runs at http://127.0.0.1:5000/
```

2) Train a model (XGBoost recommended)

```bash
# F1-tuned threshold
python train_model.py --model-type xgb --train-rows 60000 --eval-rows 20000 --random-state 42

# Business policy: maximize recall at FPR ≤ 10%
python train_model.py --model-type xgb --train-rows 60000 --eval-rows 20000 --tune-policy recall_at_fpr --target-fpr 0.1 --random-state 42
```

3) Explainability (SHAP)

```bash
python shap_analysis.py --sample-rows 5000
# Writes artifacts/shap_top10.json
```

## Data & Artifacts
- Source DB: `classification.db`, table: `Telecom_microservices_loan`.
- Artifacts:
  - `artifacts/loan_eligibility_model.joblib`
  - `artifacts/train_metadata.json` (metrics, threshold, top features, versioning)
  - Optional plots generated on demand: `artifacts/roc_curve.png`, `artifacts/confusion_matrix.png`, `artifacts/calibration_curve.png`

## Feature Engineering
- Drops identifiers/temporal leakage: `msisdn` removed during training and prediction, `pdate` dropped.
- Categorical: `pcircle` one-hot encoded.
- Numerics: coercion to numeric, median imputation, scaling.
- Derived features:
  - `payback_rate30/90`, `avg_loan_amt30/90`, `avg_rech_amt30/90`
  - `loan_to_recharge_ratio_30/90`, `repayment_ratio_30/90`, `recharge_growth`
  - `last_rech_dayofyear` from last recharge date parts

## Models
- Auto model selection: XGBoost → RandomForest → SGD (fallback) inside a scikit-learn Pipeline.
- Class imbalance: `scale_pos_weight` computed from training labels (used by XGBoost).
- Threshold tuning:
  - `--tune-policy f1` (default) or `--tune-policy recall_at_fpr --target-fpr 0.1`
  - Tuned threshold stored in `train_metadata.json` and used by the API.

## API Endpoints
- `GET /` – Web UI.
- `GET /api/schema` – Expected feature fields.
- `POST /api/predict` – JSON payload with feature fields; responds with `prediction`, `probability`, `threshold`. Input is schema-validated; `msisdn` is ignored if provided.
- `GET /api/predict-live?n=10&offset=<int>` – Batch live predictions from SQLite.
- `GET /api/metrics` – Rounded metrics, tuned threshold, top features, confusion matrix, and plot URLs.
- `GET /metrics` – Metrics UI (lists metrics, confusion matrix table, ROC & calibration plots).

## UI Behavior
- Displays probability rounded to 4 decimals and the decision threshold.
- Risk bands:
  - ≥ 0.80: High Eligible
  - ≥ 0.65: Medium Eligible
  - ≥ 0.45: Borderline
  - < 0.45: Not Eligible
- Live table sorted by probability (desc) and color-coded by risk band.

## Notes
- Training/Inference consistency is enforced by the single Pipeline.
- `msisdn` is never used as a model feature; it can be displayed in the UI only.
- Retraining overwrites artifacts; the running API picks up the latest threshold and model.

## Project Structure (key files)
- `loan_pipeline.py` – Feature engineering and Pipeline construction.
- `train_model.py` – Training CLI, threshold tuning, and artifact saving.
- `predict_loan_eligibility.py` – Batch inference to CSV from live data.
- `app.py` – Flask app and API routes.
- `templates/` – HTML templates (UI, metrics page).
- `static/` – Front-end JS and styles.
