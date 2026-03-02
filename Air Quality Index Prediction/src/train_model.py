import os
import json
import argparse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from xgboost import XGBRegressor
from src.data_preprocessing import load_data, preprocess_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Enable RandomizedSearchCV tuning")
    args = parser.parse_args()

    df = load_data()
    # Ensure proper datetime and sort before split
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")

    # Lag features (1 hour)
    for col in ["PM10", "PM2.5", "NO2"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    # Rolling features
    if "PM10" in df.columns:
        df["PM10_rolling_3"] = df["PM10"].rolling(3).mean()
    if "PM2.5" in df.columns:
        df["PM25_rolling_3"] = df["PM2.5"].rolling(3).mean()

    df.fillna(method="bfill", inplace=True)

    # Print correlation for quick diagnostics
    try:
        corr = df[["AQI"] + [c for c in ["CO","CO2","NO2","SO2","O3","PM2.5","PM10","PM10_lag1","PM2.5_lag1","NO2_lag1"] if c in df.columns]].corr()["AQI"].sort_values(ascending=False)
        print("Correlation with AQI:\n", corr)
    except Exception:
        pass

    # Time-based split
    split_idx = int(len(df) * 0.7)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    features = ["CO","CO2","NO2","SO2","O3","PM2.5","PM10","Hour","PM10_lag1","PM2.5_lag1","NO2_lag1","PM10_rolling_3","PM25_rolling_3"]

    X_train, y_train = preprocess_data(train_df, training=True, feature_list=features, use_scaler=False)
    X_test, y_test = preprocess_data(test_df, training=False)  # uses saved feature spec

    base_model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    if args.tune:
        param_dist = {
            "n_estimators": [400, 600, 800, 1000],
            "max_depth": [4, 6, 8, 10],
            "learning_rate": [0.01, 0.02, 0.03, 0.05],
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [1.0, 1.5, 2.0],
        }
        tscv = TimeSeriesSplit(n_splits=3)
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=20,
            scoring="r2",
            cv=tscv,
            verbose=1,
            n_jobs=-1,
            random_state=42,
        )
        model = search.fit(X_train, y_train).best_estimator_
        print("Best params:", getattr(search, "best_params_", None))
    else:
        model = base_model.fit(X_train, y_train)

    # Model comparison with TimeSeriesSplit on training window
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    import numpy as np

    models = {
        "Linear": Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())]),
        "RF": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1),
        "XGB": model,
    }

    tscv_eval = TimeSeriesSplit(n_splits=3)
    comparison = {}
    for name, mdl in models.items():
        if name != "XGB":
            mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        mae_m = mean_absolute_error(y_test, y_pred)
        rmse_m = float(np.sqrt(((y_test - y_pred) ** 2).mean()))
        r2_m = r2_score(y_test, y_pred)
        comparison[name] = {"MAE": float(mae_m), "RMSE": float(rmse_m), "R2": float(r2_m)}

    preds = models["XGB"].predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print("MAE:", mae)
    print("R2 Score:", r2)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, os.path.join("models", "model.pkl"))

    # Save feature importance and model confidence
    importance = model.feature_importances_
    feature_dict = dict(zip(features, importance.tolist()))

    with open(os.path.join("models", "feature_importance.json"), "w") as f:
        json.dump(feature_dict, f)

    with open(os.path.join("models", "model_confidence.json"), "w") as f:
        json.dump({"r2_score": float(r2), "mae": float(mae)}, f)

    # Save comparison
    with open(os.path.join("models", "model_comparison.json"), "w") as f:
        json.dump(comparison, f, indent=2)

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(os.path.join("static", "analytics"), exist_ok=True)
        n_sample = min(500, X_test.shape[0])
        if n_sample > 0:
            Xs = X_test[:n_sample]
            try:
                explainer = shap.Explainer(model)
                ex = explainer(Xs)
                plt.figure()
                try:
                    shap.summary_plot(ex, show=False)
                except Exception:
                    shap.plots.beeswarm(ex, show=False)
            except Exception:
                try:
                    explainer = shap.TreeExplainer(model)
                    sv = explainer.shap_values(Xs)
                    plt.figure()
                    shap.summary_plot(sv, Xs, feature_names=features, show=False)
                except Exception as e2:
                    print("Secondary SHAP error:", repr(e2))
            plt.tight_layout()
            plt.savefig(os.path.join("static", "analytics", "shap_summary.png"), dpi=150)
            plt.close()
    except Exception as e:
        print("SHAP generation failed:", repr(e))

    # Residuals plot
    try:
        import matplotlib.pyplot as plt
        res = y_test - preds
        plt.figure()
        plt.scatter(preds, res, s=8, alpha=0.6)
        plt.axhline(0, color="red", linewidth=1)
        plt.xlabel("Predicted AQI")
        plt.ylabel("Residuals")
        plt.title("Residual Error Plot")
        plt.tight_layout()
        os.makedirs(os.path.join("static", "analytics"), exist_ok=True)
        plt.savefig(os.path.join("static", "analytics", "residuals.png"), dpi=150)
        plt.close()
    except Exception:
        pass

    # Correlation heatmap
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        corr = df.select_dtypes(include=[np.number]).corr()
        plt.figure(figsize=(6, 4))
        sns.heatmap(corr, cmap="coolwarm", center=0)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        os.makedirs(os.path.join("static", "analytics"), exist_ok=True)
        plt.savefig(os.path.join("static", "analytics", "corr_heatmap.png"), dpi=150)
        plt.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
