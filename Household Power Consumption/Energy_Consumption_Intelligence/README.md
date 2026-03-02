# Energy Consumption Intelligence

Production-ready web application that forecasts daily household energy consumption and translates it into business KPIs (cost, CO₂ impact, and efficiency grade). Built with a clean architecture separating UI, API, ML inference, and business logic.

## Why It Matters
- Converts model output into actionable metrics: estimated monthly/annual cost and annual CO₂ emissions.
- Highlights usage efficiency through a clear grade with color-coded badge for quick decisions.
- Architecture and code quality are portfolio- and deployment-ready.

## Architecture
```
Energy_Consumption_Intelligence/
├── app/
│   ├── __init__.py                 # app factory
│   ├── app.py                      # creates Flask app and registers blueprint
│   ├── routes.py                   # HTTP routes (index, predict)
│   ├── services/
│   │   ├── prediction_service.py   # model/scaler loading, validation, inference
│   │   ├── business_service.py     # cost/CO₂ + grading logic
│   │   └── explain_service.py      # placeholder for SHAP/feature impact
│   ├── static/
│   │   ├── css/style.css
│   │   └── js/script.js
│   └── templates/index.html
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── training/
│   ├── preprocess.py
│   ├── train_model.py
│   └── make_dummy_model.py         # helper to create a small test model
├── config.py
└── run.py
```

## Endpoints
- `GET /` – Renders the UI (dynamic inputs driven by `feature_columns.pkl`).
- `POST /predict` – Accepts JSON and returns:
  ```json
  {
    "prediction": 25.0,
    "daily_cost": 90.0,
    "monthly_cost": 2700.0,
    "annual_cost": 32400.0,
    "annual_co2_kg": 7482.0,
    "efficiency_grade": "D",
    "grade_legend": "A ≤ 10, B ≤ 15, C ≤ 20, D > 20",
    "grade_descriptor": "High Consumption"
  }
  ```

## Prediction Units and Business Logic
- Prediction unit: `kWh/day`.
- Default rate: `₹3.6` per kWh.
- Default emission factor: `0.82 kg CO₂ / kWh` (typical grid factor; configurable).
- Costs:
  - `daily_cost = prediction × rate`
  - `monthly_cost = daily_cost × 30`
  - `annual_cost = monthly_cost × 12`
- CO₂:
  - `annual_co2_kg = prediction × 365 × emission_factor`
- Efficiency grade:
  - A ≤ 10, B ≤ 15, C ≤ 20, D > 20
  - Descriptor: A “Low Consumption”, B “Moderate”, C “Above Average”, D “High Consumption”

## Running the App
```
python .\Energy_Consumption_Intelligence\run.py
```
Open `http://127.0.0.1:5000/` in your browser.

## Training a Real Model
1. Prepare your dataset CSV and identify the target column (e.g., daily_kwh).
2. From the project root:
   ```bash
   cd Energy_Consumption_Intelligence\training
   set DATA_CSV=path\to\your.csv
   set TARGET_COLUMN=daily_kwh
   python train_model.py
   ```
3. Confirm that `models/` now contains:
   - `model.pkl`, `scaler.pkl`, `feature_columns.pkl`
4. Refresh the UI – named inputs will appear for each feature.

## Validation & Performance
- Prior run (example): MAE 0.062, RMSE 0.179, R² 0.990, rows ≈ 56,277.
- Ensure proper train/test split and check for data leakage when using real data.

## Explainability (Roadmap)
- Integrate SHAP in `explain_service.py` and render a top-features chart on the UI.

## Model Comparison (Roadmap)
- Benchmark Linear Regression vs Random Forest vs XGBoost and surface results in a small dashboard.

## CSV Upload & Batch Prediction (Roadmap)
- Add an upload endpoint for CSVs and return batched predictions with costs and CO₂.

## License
For portfolio/educational use; adapt cost and emission factors to your region/provider.

