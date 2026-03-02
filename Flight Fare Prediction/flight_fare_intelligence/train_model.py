import os
import sqlite3
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False


def ensure_models_dir():
    os.makedirs("models", exist_ok=True)


def load_data(db_path="regression.db", table="flight_fare"):
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    # Fallback synthetic dataset if DB not present
    rng = np.random.default_rng(42)
    airlines = ["IndiGo", "Air India", "Vistara", "SpiceJet", "Akasa Air"]
    cities = ["Delhi", "Mumbai", "Bangalore", "Hyderabad", "Chennai", "Kolkata", "Goa"]
    times = ["Early_Morning", "Morning", "Afternoon", "Evening", "Night", "Late_Night"]
    n = 4000
    rows = []
    for _ in range(n):
        src, dst = rng.choice(cities, 2, replace=False)
        airline = rng.choice(airlines)
        dep = rng.choice(times)
        arr = rng.choice(times)
        stops = int(rng.choice([0, 1, 2], p=[0.6, 0.3, 0.1]))
        duration = int(rng.normal(120, 40))
        duration = int(max(45, min(420, duration)))
        days_left = int(rng.integers(0, 120))
        cls = int(rng.choice([0, 1], p=[0.8, 0.2]))  # 0 Economy, 1 Business
        base = 55.0 + (0 if cls == 0 else 10.0)
        price = (
            base
            + 0.03 * duration
            + 2.5 * stops
            - 0.04 * days_left
            + rng.normal(0, 2.0)
        )
        rows.append(
            dict(
                airline=airline,
                source_city=src,
                departure_time=dep,
                stops=stops,
                arrival_time=arr,
                destination_city=dst,
                duration=duration,
                days_left=days_left,
                price=float(max(30.0, price)),
                **{"class": cls},
            )
        )
    df = pd.DataFrame(rows)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["route"] = (df["source_city"].astype(str) + "_" + df["destination_city"].astype(str))
    df["is_weekend_departure"] = df["departure_time"].astype(str).isin(["Evening", "Night"]).astype(int)
    df["duration_hours"] = df["duration"].astype(float) / 60.0
    return df


def encode_frame(df: pd.DataFrame):
    df = df.drop(columns=["flight"], errors="ignore")
    label_encoders = {}
    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    return df, label_encoders


def train_and_save(df: pd.DataFrame, name: str, feature_order):
    X = df[feature_order].copy()
    y = df["price"].astype(float).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if HAS_XGB:
        model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    resid = y_test - preds
    with open(os.path.join("models", f"{name}_model.pkl"), "wb") as f:
    with open(os.path.join("models", f"{name}_model.pkl"), "wb") as f:
    try:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores))
    except Exception:
        cv_mean = None
        cv_std = None
    with open(os.path.join("models", f"{name}_meta.pkl"), "wb") as f:
        pickle.dump({
            "r2": float(r2),
            "mae": float(mae),
            "residual_std": resid_std,
            "feature_order": feature_order,
            "cv_r2_mean": cv_mean,
            "cv_r2_std": cv_std,
            "model_version": "v1",
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_rows": int(df.shape[0]),
            "train_size": int(X_train.shape[0]),
            "test_size": int(X_test.shape[0]),
            "split_random_state": 42
        }, f)
        pickle.dump({"r2": float(r2), "mae": float(mae), "residual_std": resid_std, "feature_order": feature_order}, f)


def main():
    ensure_models_dir()
    df = load_data()
    print("Dataset shape:", df.shape)
    print(df.head())
    price_min, price_max = float(df["price"].min()), float(df["price"].max())
    print("Raw price range before scaling:", price_min, price_max)
    if 45 <= price_min <= 80 and 50 <= price_max <= 100:
        df["price"] = df["price"] * 100.0
        print("Applied scaling x100. New price range:", float(df["price"].min()), float(df["price"].max()))
    df = engineer_features(df)
    df, encoders = encode_frame(df)
    with open(os.path.join("models", "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
    feature_order = [
        "airline",
        "source_city",
        "departure_time",
        "stops",
        "arrival_time",
        "destination_city",
        "duration",
        "days_left",
        "route",
        "is_weekend_departure",
        "duration_hours",
    ]
    economy_df = df[df["class"] == df["class"].min()].copy()
    business_df = df[df["class"] == df["class"].max()].copy()
    r2_e, mae_e = train_and_save(economy_df, "economy", feature_order)
    r2_b, mae_b = train_and_save(business_df, "business", feature_order)
    print("Economy R2:", r2_e, "MAE:", mae_e)
    print("Business R2:", r2_b, "MAE:", mae_b)
    print(df["price"].min(), df["price"].max())


if __name__ == "__main__":
    main()
