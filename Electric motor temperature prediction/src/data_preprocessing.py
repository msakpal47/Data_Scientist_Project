import os
import sqlite3
import joblib
import pandas as pd

FEATURES = [
    "u_q",
    "u_d",
    "i_d",
    "i_q",
    "coolant",
    "motor_speed",
    "ambient",
]

TARGET = "pm"


def load_data():
    db_candidates = [
        "data/Regression.db",
        "data/regression.db",
        "Regression.db",
        "regression.db",
    ]
    db_path = next((p for p in db_candidates if os.path.exists(p)), None)
    if db_path is None:
        raise FileNotFoundError("Database not found. Expected one of: " + ", ".join(db_candidates))
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM electric_motor_temperature", conn)
    conn.close()
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["current_magnitude"] = (df["i_d"] ** 2 + df["i_q"] ** 2) ** 0.5
    df["voltage_magnitude"] = (df["u_d"] ** 2 + df["u_q"] ** 2) ** 0.5
    df["power_estimate"] = df["current_magnitude"] * df["voltage_magnitude"]
    return df


def preprocess_data(df: pd.DataFrame, training: bool = True):
    from sklearn.preprocessing import StandardScaler

    df = feature_engineering(df)
    features = FEATURES + ["current_magnitude", "voltage_magnitude", "power_estimate"]
    X = df[features]
    y = df[TARGET] if TARGET in df.columns else None

    if training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
    else:
        scaler = joblib.load("models/scaler.pkl")
        X_scaled = scaler.transform(X)

    return X_scaled, y, features
