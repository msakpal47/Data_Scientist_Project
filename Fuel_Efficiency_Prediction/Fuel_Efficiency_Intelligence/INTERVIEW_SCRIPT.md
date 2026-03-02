Title: Fuel Efficiency Intelligence & Sustainability Optimization Platform

Problem
- Estimate vehicle fuel consumption using design and operational features
- Provide confidence ranges and explainability
- Simulate design changes and quantify business impact (cost, CO2)

Data & Features
- Source: SQLite (regression.db), table: fuel_efficiency_automobile
- Core features: Mass (kg), Engine Power (kW), Engine Capacity (cc), WLTP CO2 (g/km), Energy Use (Wh/km), Fuel Type/Mix, Electric Range (km)
- Target: Fuel consumption (L/100km)

ML Approach
- Models compared: Linear Regression, RandomForest, XGBoost
- Preprocessing: sklearn ColumnTransformer (StandardScaler for numeric, OneHotEncoder for categoricals)
- Unified Pipeline persisted for consistent training/inference
- Metrics: MAE, RMSE, R² on holdout; optional CV R²
- Explainability: SHAP-based local explanations (fallback to perturbation)

Architecture
- Training: /training (train_model.py, train_compare.py)
- Serving: Flask app with endpoints (/predict, /simulate, /explain, /model-metrics, /feature-importance, /api-docs)
- UI: Dark enterprise theme with KPI cards, confidence range, feature importance, simulation panel
- Models: Serialized pipeline + feature schema
- Deployment: Dockerfile, requirements

Key Decisions
- Persist full pipeline: eliminates feature mismatch in production
- Schema guard at inference: fills missing features, enforces ordering
- Confidence interval: simple ±5% (bootstrapped intervals can be added)
- Simulation: ML-driven re-prediction after feature perturbation

Business Value
- Accelerates R&D via virtual experiments before prototyping
- Quantifies cost and CO2 impact for design choices
- Enables procurement and consumer comparison workflows (A/B scenarios)

Results Snapshot (example)
- MAE ≈ 0.06, RMSE ≈ 0.18, R² ≈ 0.99 (synthetic/low-noise or tightly coupled features)
- CV R² ≈ 0.98 on a sampled subset

Demo Flow (Interview)
1) Show the UI: enter realistic vehicle specs → Predict
2) Point out KPI cards, gauge, confidence range
3) Run a simulation: increase mass or WLTP CO2 by 10% → observe cost/CO2 deltas
4) Explain feature importance and SHAP contributions for the prediction
5) Show /model-metrics and /feature-importance endpoints for auditability
6) Mention Dockerfile and OpenAPI for deployment-readiness

Future Enhancements
- Bootstrapped uncertainty estimates
- SHAP global plots and per-feature audits
- Scenario comparison (Vehicle A vs B) in UI
- API validation schemas and CI pipeline

One-Liner
- “A pipeline-driven, explainable fuel efficiency platform that quantifies design trade-offs in real time for cost and sustainability.” 
