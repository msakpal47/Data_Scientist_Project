# Health Insurance Cost Intelligence

Predicts annual medical insurance charges from demographic and lifestyle inputs. Includes training UI, prediction UI, leaderboard with multiple models, feature importance, and CSV report download.

## How to Run
- Ensure Python environment is active.
- Start server:
  - `python app/app.py`
- Open browser:
  - Main dashboard: `http://127.0.0.1:5000/`
  - Training page: `http://127.0.0.1:5000/setup`

## Train a Model
1. Go to `/setup`
2. Select table: `Insurance_Prediction`
3. Target: `charges`
4. Click “Train from Database”
5. Artifacts saved to `models/` (model.pkl, scaler.pkl, columns.pkl, feature_importance.json, leaderboard.json)

## Predict
- Go to `/`
- Fill inputs and click “Predict”
- Choose “Detailed” to see a table of all inputs
- Click “Download Prediction Report” to export a CSV with inputs + prediction

## Pipeline
- Numeric: median imputation → StandardScaler
- Categorical: most-frequent imputation → OneHotEncoder(drop='first')
- Models compared: RandomForest, ExtraTrees, GradientBoosting, Ridge, Lasso, ElasticNet, SGD (XGBoost optional)
- Selection by test R²; leaderboard stored in `models/leaderboard.json`

## Feature Importance
- Tree models: `feature_importances_`
- Linear models: `|coef_|`
- Fallback: permutation importance
- Fetched from `/feature-importance`; chart shows Top 20; text fallback if Chart.js unavailable

## Notes
- Dataset: `Insurance_Prediction` in `regression.db`
- Target: `charges`
- No risk/fraud logic included; focus is premium prediction

## Files of Interest
- `app/app.py` – Flask app and endpoints
- `app/templates/index.html` – Dashboard UI
- `app/templates/setup.html` – Training UI
- `training/train_model.py` – Training and model selection
- `models/leaderboard.json` – Metrics for comparison
- `models/feature_importance.json` – Cached importance scores

## Business Impact
- Faster, consistent premium quoting
- Transparent reasoning via feature importance
- Exportable artifacts for audits and presentations
