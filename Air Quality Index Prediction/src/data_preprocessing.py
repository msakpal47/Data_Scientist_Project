import os
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib


def _resolve_db_path() -> str:
    candidates = [
        os.path.join("data", "Regression.db"),
        os.path.join("data", "regression.db"),
        "Regression.db",
        "regression.db",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    # Fallback to default location under data/
    os.makedirs("data", exist_ok=True)
    return os.path.join("data", "Regression.db")


def load_data():
    db_path = _resolve_db_path()
    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql("SELECT * FROM air_quality_index", conn)
    finally:
        conn.close()
    return df


def _load_feature_spec(default_features):
    spec_path = os.path.join("models", "feature_spec.json")
    if os.path.exists(spec_path):
        try:
            import json
            with open(spec_path, "r") as f:
                spec = json.load(f)
            feats = spec.get("features")
            if isinstance(feats, list) and feats:
                return feats
        except Exception:
            pass
    return default_features


def preprocess_data(df: pd.DataFrame, training: bool = True, feature_list=None, use_scaler: bool = True):
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Hour"] = df["Date"].dt.hour
        df["Day"] = df["Date"].dt.day
        df["Month"] = df["Date"].dt.month
        df["DayOfWeek"] = df["Date"].dt.dayofweek
    else:
        # If Date is missing during inference, create empty time features
        for col in ["Hour", "Day", "Month", "DayOfWeek"]:
            if col not in df.columns:
                df[col] = pd.NA

    default_features = [
        "CO",
        "CO2",
        "NO2",
        "SO2",
        "O3",
        "PM2.5",
        "PM10",
        "Hour",
    ]

    if feature_list is None:
        # In inference, prefer the trained feature spec for consistency
        # If missing, fall back to default_features
        features = _load_feature_spec(default_features)
    else:
        features = feature_list

    # Ensure all feature columns exist for both training and inference
    for col in features:
        if col not in df.columns:
            df[col] = pd.NA

    X = df[features]
    try:
        X = X.apply(pd.to_numeric, errors="coerce")
    except Exception:
        pass
    y = df["AQI"] if "AQI" in df.columns else None

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    imputer_path = os.path.join(models_dir, "imputer.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    spec_path = os.path.join(models_dir, "feature_spec.json")

    if training:
        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)
        joblib.dump(imputer, imputer_path)

        if use_scaler:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            joblib.dump(scaler, scaler_path)

        # Persist the feature order and scaler usage used during training
        try:
            import json
            with open(spec_path, "w") as f:
                json.dump({"features": features, "use_scaler": use_scaler}, f)
        except Exception:
            pass
    else:
        if not os.path.exists(imputer_path):
            raise FileNotFoundError("Preprocessing artifacts not found. Train the model first to generate imputer.")

        # Read scaler usage flag if available
        if os.path.exists(spec_path):
            try:
                import json
                with open(spec_path, "r") as f:
                    spec = json.load(f)
                use_scaler = bool(spec.get("use_scaler", use_scaler))
            except Exception:
                pass

        imputer = joblib.load(imputer_path)
        X = imputer.transform(X)

        if use_scaler and os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X = scaler.transform(X)

    return X, y
