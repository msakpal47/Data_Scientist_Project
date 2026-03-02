# Interview Explainer – Business Perspective

## 1) Problem & Value
- Objective: reduce silica variability and off‑spec in iron‑ore concentrate by predicting % Silica Concentrate from live process signals and surfacing actionable tips.
- Business value: fewer quality penalties and reprocessing loops; optimized reagent usage; better operator confidence; improved throughput stability.

## 2) Stakeholders
- Plant operations: receive near‑real‑time predictions, risk labels, and recommendations.
- Process engineers: analyze top drivers, iterate set points, validate improvements.
- Management: track KPIs (R²_test, MAE, RMSE), reduction in off‑spec tonnage, reagent cost savings.

## 3) Data & Reliability
- Signals: 21 inputs including % Silica Feed, % Iron Feed, reagent flows, pH, density, and air flows.
- Data hygiene: numeric cleaning, median imputation, robust to target naming variants.
- Explainability: SHAP top features highlight drivers (e.g., % Silica Feed, Amina, pH, air flows).

## 4) Model & Metrics
- Algorithm: HistGradientBoostingRegressor (tree‑based, efficient for tabular).
- Performance: R²_test ≈ 0.623; MAE_test ≈ 0.511; RMSE_test ≈ 0.690 on a simple 80/20 split.
- Latency: < 50 ms due to in‑memory model cache.

## 5) System Design
- Offline: train from SQLite → save artifacts to backend/models.
- Online: Flask API loads once on startup, exposes `/api/status` and `/api/predict` to the UI.
- Resilience: safe fallback so UI stays responsive even on transient errors.

## 6) Operational Playbook
- Retraining cadence: weekly or after process changes; restart server to load new artifacts.
- Monitoring: compare predicted vs. lab results; watch drift in inputs and residuals.
- Guardrails: only one server on port 5000 to avoid stale instances.

## 7) ROI Levers
- Cut off‑spec by proactive reagent/air adjustments → fewer re‑grinds, lower penalties.
- Reduce reagent over‑dosage by quantifying driver importance (Amina, pH, air flows).
- Operator empowerment: risk flags + recommendations speed up decision‑making.

## 8) Roadmap
- Cross‑validation, hyperparameter tuning at scale.
- Production WSGI (waitress/gunicorn) and observability (logs/metrics).
- Broader dashboards and shift‑level reporting.

## 9) Sample Interview Answers
- Why HGBR? “Handles nonlinear tabular patterns efficiently, robust to noise, fast inference.”
- How do we trust it? “We surface SHAP drivers and maintain a consistent offline/online pipeline.”
- What if data drifts? “Monitor residuals and retrain; fallback keeps UX stable while we fix.”
