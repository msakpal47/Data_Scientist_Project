Metro Train Maintenance – Fault Detection Web App
=================================================

Overview
--------
- Predicts whether a fault will occur using sensor signals from the manufacturing/metro domain.
- Provides a lightweight web UI for training, evaluation, and single‑record prediction.
- Stores models and schema for reuse; exposes a simple REST API.

Run
---
- Start the backend:
  - python backend\\app.py
- Open the UI:
  - http://127.0.0.1:8000

Data
----
- Source: backend/classification.db, table fault_detection_manufacturing.
- Target: fault_occurred (0/1).
- Features: numeric sensor fields; schema is persisted in backend/models/schema.json.

Training
--------
- Models: Logistic Regression (balanced) and XGBoost supported.
- Split Modes:
  - Random: stratified 80/20.
  - Time‑Based (production‑safe): requires a timestamp column (e.g., time). Data are sorted by timestamp. Train = rows strictly before the last positive fault timestamp; Test = rows at/after the last positive timestamp. Ensures test contains real future faults when they exist. No shuffle, no stratify, no random_state.
- Speed Mode:
  - Downsamples negative examples in TRAIN only to accelerate fitting (temporal test remains intact).
- Calibration:
  - Optional probability calibration for Logistic Regression (disabled in speed mode).
- Cross‑Validation:
  - Optional stratified K‑fold (client sets number of folds).

Promotion Guardrails
--------------------
- Production promotion (/api/model_promote) requires:
  - Time‑Based split (split = temporal).
  - SMOTE off.
- Temporal holdout must include at least one positive event; promotion is blocked if test positives == 0.
- The UI mirrors these requirements and blocks promotion when conditions are not met, surfacing clear messages.

UI Highlights
-------------
- Train/Retrain with model type, split mode, and optional timestamp column.
- Metrics: Accuracy, Precision, Recall, F1, ROC AUC, PR AUC.
- Threshold helpers: best F1 and best threshold at a recall target.
- Model health panel shows split strategy, dataset sizes, and imbalance ratio.

API
---
- POST /api/train
- POST /api/train_async and GET /api/train_status
- POST /api/predict
- POST /api/model_promote
- GET  /api/model_info
- GET  /api/schema and POST /api/schema_refresh
- GET  /api/time_columns

Notes
-----
- When Time‑Based split is selected, any failure to perform a proper temporal split results in a clear error rather than silently falling back to random. Responses include split_warning and invalid_temporal_test flags for transparency.
- Default threshold is persisted using the best@recall or best F1 strategy returned by training.
