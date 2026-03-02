import json
import joblib
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data

df = load_data()
if len(df) < 10:
    raise RuntimeError("Not enough rows for training.")
if len(df) <= 100000:
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
else:
    n = len(df)
    train_end = int(n * 0.8)
    train_df = df.iloc[:train_end]
    test_df = df.iloc[train_end:]
    max_train = min(len(train_df), 200000)
    max_test = min(len(test_df), 50000)
    train_df = train_df.iloc[:max_train]
    test_df = test_df.iloc[:max_test]

X_train, y_train, features = preprocess_data(train_df, training=True)
X_test, y_test, _ = preprocess_data(test_df, training=False)

xgb_model = XGBRegressor(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)
rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
)

xgb_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
rf_preds = rf_model.predict(X_test)

def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"r2": float(r2), "mae": float(mae), "rmse": rmse}

metrics_dict = {
    "xgb": metrics(y_test, xgb_preds),
    "rf": metrics(y_test, rf_preds),
}

best = max(metrics_dict.keys(), key=lambda k: metrics_dict[k]["r2"])
print("XGB:", metrics_dict["xgb"])
print("RF:", metrics_dict["rf"])
print("Primary model:", best)

joblib.dump(xgb_model, "models/model.pkl")

importance = xgb_model.feature_importances_
feature_dict = dict(zip(features, importance.tolist()))
with open("models/feature_importance.json", "w") as f:
    json.dump(feature_dict, f)

res_points = []
n = min(len(y_test), 1000)
for i in range(n):
    res_points.append({"pred": float(xgb_preds[i]), "resid": float(y_test[i] - xgb_preds[i])})
with open("models/residuals_sample.json", "w") as f:
    json.dump({"points": res_points}, f)

booster = xgb_model.get_booster()
dm = xgb.DMatrix(X_test)
contribs = booster.predict(dm, pred_contribs=True)
contribs = np.array(contribs)
contribs = contribs[:, :-1]
shap_mean = np.mean(np.abs(contribs), axis=0)
shap_dict = dict(zip(features, shap_mean.tolist()))
with open("models/shap_importance.json", "w") as f:
    json.dump(shap_dict, f)

with open("models/model_metrics.json", "w") as f:
    json.dump({"models": metrics_dict, "primary": "xgb"}, f)
