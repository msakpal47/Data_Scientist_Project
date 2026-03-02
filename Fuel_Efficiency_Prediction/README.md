Fuel Efficiency Intelligence Platform

Overview
- Predicts fuel consumption (L/100km) from automotive features
- Confidence range, explainability, and business simulation
- Flask API with a dark-themed web UI
- Trains from SQLite (regression.db), persists full sklearn pipeline

Quick Start
- Python
  - pip install -r Fuel_Efficiency_Intelligence/requirements.txt
  - python Fuel_Efficiency_Intelligence/training/train_model.py --limit 80000
  - python Fuel_Efficiency_Intelligence/app/app.py
  - Open http://localhost:8000
- Docker
  - docker build -t fuel-efficiency .
  - docker run -p 8000:8000 fuel-efficiency

Key Endpoints
- POST /predict  → returns prediction and interval
- POST /simulate → re-predict after feature change (%)
- POST /explain  → SHAP-based or perturbation explanation
- GET /model-metrics → latest MAE, RMSE, R², rows, CV
- GET /feature-importance → aggregated importances
- GET /api-docs → OpenAPI JSON

Training
- Single model: Fuel_Efficiency_Intelligence/training/train_model.py
- Model comparison: Fuel_Efficiency_Intelligence/training/train_compare.py
- Artifacts: Fuel_Efficiency_Intelligence/models/

Project Structure
- Fuel_Efficiency_Intelligence/
  - app/         → Flask app, templates, static, openapi
  - training/    → Training scripts
  - models/      → Persisted pipelines and logs

Notes
- If SHAP is unavailable, /explain falls back to perturbation contributions
- Ensure realistic inputs for credible outcomes (UI guides with ranges)
