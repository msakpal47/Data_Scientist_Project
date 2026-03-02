# Fault Detection in Manufacturing – Dashboard and Pipeline

## Overview
- Predicts fault_occurred from manufacturing sensor data at scale (810,643 rows).
- Web dashboard (Flask + HTML/CSS/JS) integrates: connect, preview, train, evaluate, infer, and export.
- Handles class imbalance, removes TTF leakage, and provides robust evaluation visuals.

## Key Features
- Training modes:
  - Fast streaming (SGDClassifier with partial_fit and balanced sample weights)
  - RandomForest (class_weight=balanced, optional oversampling)
- Controls:
  - fast_mode, chunk_size
  - eval_offset, eval_size (choose eval segment to ensure positives)
  - oversample (non-stream path)
- Evaluation:
  - Accuracy, Precision, Recall, F1, ROC AUC
  - Confusion Matrix and PR Curve
  - Eval Label Distribution
- Inference:
  - Live row prediction
  - Live simulation summary: total rows, predicted faults, fault rate
- Persistence:
  - Download model: fault_detection_model.pkl
  - Load saved model for inference without retraining

## Leakage and Data Integrity
- Excludes any TTF-derived columns to avoid using future information.
- Median imputation fit on train; reused for eval/live to keep consistency.

## Run
1. Open PowerShell in the project folder:
   - `python app.py`
2. In the browser, go to:
   - `http://127.0.0.1:5000/`
3. Steps:
   - Connect (defaults point to `classification.db` and `Data Dictionary.txt`)
   - Train panel:
     - Target: `fault_occurred`
     - Select features (numeric-only recommended)
     - Configure `fast_mode`, `chunk_size`, `eval_offset`, `eval_size`, `oversample`
     - Click Train; check Metrics tab (cards, confusion matrix, PR curve, label distribution)
   - Inference:
     - Predict a single live row
     - Simulate Live for summary stats
   - Download Model or Load Saved Model

## Files
- `app.py` – backend APIs, training, evaluation, inference, artifacts
- `templates/index.html` – UI
- `static/app.js`, `static/app.css` – frontend logic and styles
- `classification.db` – SQLite data (ignored by git)
- `fault_detection_model.pkl` – saved model payload
- `predictions_live.csv` – batch predictions (via `predict_batch.py`)
- `final_report.txt`, `project_summery.csv` – documentation and summary

## Notes
- If eval shows 0 positives, adjust `eval_offset`/`eval_size` or use oversampling in the non-stream path.
- Streaming path auto-searches earlier eval windows when necessary.
