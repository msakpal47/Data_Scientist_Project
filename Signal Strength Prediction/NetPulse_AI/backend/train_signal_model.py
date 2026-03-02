import os
import json
import sqlite3
import joblib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "regression.db"))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "meta.json")

REQUIRED_TARGET = "Signal Strength (dBm)"
REQUIRED_FEATURES = [
    "Latitude",
    "Longitude",
    "Network Type",
    "Data Throughput (Mbps)",
    "Latency (ms)",
    "Signal Quality (%)",
    "BB60C Measurement (dBm)",
    "srsRAN Measurement (dBm)",
    "BladeRFxA9 Measurement (dBm)",
]

RENAME_MAP = {
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Network Type": "network_type",
    "Data Throughput (Mbps)": "throughput_mbps",
    "Latency (ms)": "latency_ms",
    "Signal Quality (%)": "signal_quality_pct",
    "BB60C Measurement (dBm)": "bb60c_dbm",
    "srsRAN Measurement (dBm)": "srsran_dbm",
    "BladeRFxA9 Measurement (dBm)": "bladerf_dbm",
    "Locality": "locality",
    REQUIRED_TARGET: "target_dbm",
}


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def find_table(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [r[1] for r in cur.fetchall()]
        if REQUIRED_TARGET in cols and all(f in cols for f in REQUIRED_FEATURES):
            return t
    return None


def load_dataframe():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("regression.db not found")
    conn = sqlite3.connect(DB_PATH)
    try:
        table = find_table(conn)
        if table is None:
            raise RuntimeError("No table with required schema found")
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        return df
    finally:
        conn.close()


def prepare_data(df):
    cols = [*REQUIRED_FEATURES, REQUIRED_TARGET]
    optional = ["Locality"]
    keep = [c for c in cols if c in df.columns]
    for o in optional:
        if o in df.columns:
            keep.append(o)
    df = df[keep].copy()
    df = df.rename(columns=RENAME_MAP)
    df = df.dropna()
    features = [
        "latitude",
        "longitude",
        "network_type",
        "throughput_mbps",
        "latency_ms",
        "signal_quality_pct",
        "bb60c_dbm",
        "srsran_dbm",
        "bladerf_dbm",
    ]
    if "locality" in df.columns:
        features = ["locality"] + features
    X = df[features]
    y = df["target_dbm"]
    return X, y, features


def build_pipelines(features, cat_features):
    num_features = [f for f in features if f not in cat_features]
    ohe = _make_ohe()
    pre_trees = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_features),
            ("num", SimpleImputer(strategy="median"), num_features),
        ],
        remainder="drop",
    )
    pre_ridge = ColumnTransformer(
        transformers=[
            ("cat", ohe, cat_features),
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),]), num_features),
        ],
        remainder="drop",
    )
    rf = Pipeline([
        ("prep", pre_trees),
        ("model", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    hgb = Pipeline([
        ("prep", pre_trees),
        ("model", HistGradientBoostingRegressor(random_state=42, max_depth=None)),
    ])
    ridge = Pipeline([
        ("prep", pre_ridge),
        ("scale", StandardScaler(with_mean=True)),
        ("model", Ridge(alpha=1.0, random_state=42)),
    ])
    return [("HistGBR", hgb), ("RandomForest", rf), ("Ridge", ridge)]


def evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    return r2, mae, rmse


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataframe()
    X, y, features = prepare_data(df)
    cat_features = [f for f in ["network_type", "locality"] if f in features]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    candidates = build_pipelines(features, cat_features)
    best = None
    best_metrics = None
    all_metrics = []
    for name, pipe in candidates:
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        r2, mae, rmse = evaluate(y_test, y_pred)
        all_metrics.append({"name": name, "r2": r2, "mae": mae, "rmse": rmse})
        if best is None or r2 > best_metrics["r2"]:
            best = (name, pipe)
            best_metrics = {"r2": r2, "mae": mae, "rmse": rmse}
    name, pipe = best
    joblib.dump(pipe, MODEL_PATH)
    pi = permutation_importance(pipe, X_test, y_test, n_repeats=10, random_state=42)
    importances = []
    for i, f in enumerate(X_test.columns.tolist()):
        importances.append({"feature": f, "importance": float(pi.importances_mean[i])})
    meta = {
        "best_model": name,
        "model_type": name,
        "r2": float(best_metrics["r2"]),
        "mae": float(best_metrics["mae"]),
        "rmse": float(best_metrics["rmse"]),
        "feature_importance": importances,
        "features": X_test.columns.tolist(),
        "rows_trained": int(len(X_train)),
        "model_version": f"{name}-v1",
        "last_trained_iso": datetime.now(timezone.utc).isoformat(),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({"selected": name, "metrics": best_metrics}, indent=2))


if __name__ == "__main__":
    main()
