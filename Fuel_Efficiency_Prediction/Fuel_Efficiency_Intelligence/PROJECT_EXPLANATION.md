Project Explanation

Goal
- Estimate vehicle fuel consumption using design and operational features
- Provide explainable, business-oriented insights and simulations

Data
- SQLite database regression.db
- Table fuel_efficiency_automobile with numeric and categorical features
- Target column: Fuel consumption  (note the trailing space)

ML
- Preprocessing: ColumnTransformer (StandardScaler for numeric, OneHotEncoder for categoricals)
- Models: Linear, RandomForest, XGBoost (comparison script)
- Pipeline persists preprocessing + model for inference consistency
- Metrics: MAE, RMSE, R² on holdout; optional CV R²

App
- Flask API: /predict, /simulate, /explain, /model-metrics, /feature-importance, /api-docs
- UI: Dark theme with KPI cards, confidence range, feature importance bars, simulation panel
- Simulation: Re-predicts after adjusting a selected feature by a percentage

Outputs
- Models and logs in Fuel_Efficiency_Intelligence/models
- Smoke test JSONs for predictions, metrics, importances, simulation

Run
- pip install -r Fuel_Efficiency_Intelligence/requirements.txt
- python Fuel_Efficiency_Intelligence/training/train_model.py --limit 80000
- python Fuel_Efficiency_Intelligence/app/app.py
- Open http://localhost:8000

Next Steps
- SHAP plots and A/B scenario comparison in UI
- API schema validation and Docker Compose
