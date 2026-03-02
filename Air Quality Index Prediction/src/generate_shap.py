import os
import numpy as np
import pandas as pd
import joblib
from src.data_preprocessing import load_data, preprocess_data

def main():
    model_path = os.path.join("models", "model.pkl")
    if not os.path.exists(model_path):
        print("Missing model.pkl")
        return
    model = joblib.load(model_path)

    df = load_data()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date")

    for col in ["PM10", "PM2.5", "NO2"]:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)
    if "PM10" in df.columns:
        df["PM10_rolling_3"] = df["PM10"].rolling(3).mean()
    if "PM2.5" in df.columns:
        df["PM25_rolling_3"] = df["PM2.5"].rolling(3).mean()
    df.bfill(inplace=True)

    holdout = df.tail(500)

    X, y = preprocess_data(holdout, training=False)
    n = min(300, X.shape[0])
    if n == 0:
        print("No rows for SHAP")
        return
    Xs = X[-n:]

    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(os.path.join("static", "analytics"), exist_ok=True)
        try:
            explainer = shap.Explainer(model)
            ex = explainer(Xs)
            plt.figure()
            try:
                shap.summary_plot(ex, show=False)
            except Exception:
                shap.plots.beeswarm(ex, show=False)
        except Exception:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(Xs)
            plt.figure()
            shap.summary_plot(sv, Xs, show=False)
        plt.tight_layout()
        out = os.path.join("static", "analytics", "shap_summary.png")
        plt.savefig(out, dpi=150)
        plt.close()
        print("Saved:", out)
    except Exception as e:
        try:
            with open(os.path.join("static","analytics","shap_error.txt"), "w") as f:
                f.write(repr(e))
        except Exception:
            pass
        print("SHAP failed:", repr(e))

if __name__ == "__main__":
    main()
