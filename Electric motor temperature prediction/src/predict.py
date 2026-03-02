import os
import json
import joblib
import pandas as pd
from src.data_preprocessing import preprocess_data, load_data


def predict_temperature(input_dict: dict) -> float:
    if not os.path.exists("models/model.pkl"):
        os.makedirs("models", exist_ok=True)
        df = load_data()
        if len(df) < 1000:
            raise FileNotFoundError("Model not found and insufficient data to train.")
        df_small = df.iloc[: min(20000, len(df))]
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df_small, test_size=0.2, random_state=42)
        X_train, y_train, features = preprocess_data(train_df, training=True)
        X_test, y_test, _ = preprocess_data(test_df, training=False)
        from xgboost import XGBRegressor
        model_small = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        model_small.fit(X_train, y_train)
        preds = model_small.predict(X_test)
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        import numpy as np
        r2 = float(r2_score(y_test, preds))
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        joblib.dump(model_small, "models/model.pkl")
        imp = model_small.feature_importances_
        with open("models/feature_importance.json", "w") as f:
            json.dump(dict(zip(features, imp.tolist())), f)
        with open("models/model_metrics.json", "w") as f:
            json.dump({"r2": r2, "mae": mae, "rmse": rmse}, f)
        # residuals sample for chart
        res_points = []
        n = min(len(y_test), 500)
        for i in range(n):
            res_points.append({"pred": float(preds[i]), "resid": float(y_test.iloc[i] - preds[i])})
        with open("models/residuals_sample.json", "w") as f:
            json.dump({"points": res_points}, f)
        # SHAP-like contributions using booster pred_contribs
        try:
            booster = model_small.get_booster()
            import xgboost as xgb  # local import
            dm = xgb.DMatrix(X_test)
            contribs = booster.predict(dm, pred_contribs=True)
            import numpy as np
            contribs = np.array(contribs)[:, :-1]
            shap_mean = np.mean(np.abs(contribs), axis=0)
            with open("models/shap_importance.json", "w") as f:
                json.dump(dict(zip(features, shap_mean.tolist())), f)
        except Exception:
            pass
    model = joblib.load("models/model.pkl")
    df = pd.DataFrame([input_dict])
    X, _, _ = preprocess_data(df, training=False)
    prediction = model.predict(X)
    return float(prediction[0])
