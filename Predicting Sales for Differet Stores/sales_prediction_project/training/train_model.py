import os
import json
import sqlite3
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_CANDIDATES = [
    os.path.abspath(os.path.join(BASE_DIR, "regression.db")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "regression.db")),
]

def locate_data():
    candidates = [
        os.path.join(BASE_DIR, "data", "train.csv"),
        os.path.join(BASE_DIR, "..", "train.csv"),
        os.path.join(BASE_DIR, "..", "data", "train.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return ("csv", p)
    for db in DB_CANDIDATES:
        if os.path.exists(db):
            return ("db", db)
    return (None, None)

def load_from_db(path):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    # prefer typical names and explicit project table
    prefer = ["predicting_sales", "store_sales", "sales", "rossmann", "data"]
    table = next((t for t in prefer if t in tables), None)
    if table is None:
        # heuristic: pick a table that has 'Sales' and at least one of core columns
        best = None
        for t in tables:
            if t in ("sqlite_sequence", "predictions"):
                continue
            try:
                cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
                low = [str(x).lower() for x in cols]
                has_sales = any(x == "sales" for x in low)
                core = any(x in low for x in ["store","dayofweek","date","customers"])
                if has_sales and core:
                    best = t
                    break
            except Exception:
                continue
        table = best
    if table is None:
        conn.close()
        raise RuntimeError("No sales table found in database")
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()
    return df

def prepare(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = df["Date"].dt.year.fillna(0).astype(int)
    df["Month"] = df["Date"].dt.month.fillna(0).astype(int)
    df["Day"] = df["Date"].dt.day.fillna(0).astype(int)
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["StateHoliday"] = df["StateHoliday"].apply(lambda x: 0 if str(x) in ["0", "0.0", "nan", "None"] else 1).astype(int)
    if "SchoolHoliday" not in df.columns:
        df["SchoolHoliday"] = 0
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(int)
    if "Open" not in df.columns:
        df["Open"] = 1
    df["Open"] = df["Open"].fillna(1).astype(int)
    cols = ["Store","DayOfWeek","Customers","Promo","StateHoliday","SchoolHoliday","Open","Year","Month","Day","WeekOfYear"]
    return df[cols]

def fallback_dataframe():
    data = {
        "Store": [1,1,2,2,3,3,4,4,5,5],
        "DayOfWeek": [1,2,3,4,5,6,0,1,2,3],
        "Date": pd.date_range("2015-01-01", periods=10, freq="D").astype(str).tolist(),
        "Customers": [100,120,130,140,150,160,170,180,190,200],
        "Promo": [0,1,0,1,0,1,0,1,0,1],
        "StateHoliday": [0]*10,
        "SchoolHoliday": [0]*10,
        "Open": [1]*10,
        "Sales": [1000,1200,1100,1300,1250,1350,1400,1500,1450,1550],
    }
    return pd.DataFrame(data)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("BASE_DIR:", BASE_DIR)
    print("MODELS_DIR:", MODELS_DIR)
    source_type, source_path = locate_data()
    print("Training: source_type=", source_type, "path=", source_path)
    if source_type == "csv":
        df = pd.read_csv(source_path)
    elif source_type == "db":
        df = load_from_db(source_path)
    else:
        df = fallback_dataframe()
    if "Sales" not in df.columns:
        df = fallback_dataframe()
    df = df.dropna(subset=["Sales"])
    if len(df) < 10:
        df = fallback_dataframe()
    X = df[["Store","DayOfWeek","Date","Customers","Promo","StateHoliday","SchoolHoliday","Open"]]
    y = df["Sales"].astype(float).values
    pipeline = Pipeline([
        ("prep", FunctionTransformer(prepare, validate=False)),
        ("model", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
    ])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(((y_test - y_pred) ** 2).mean()))
    model_path = os.path.abspath(os.path.join(MODELS_DIR, "model.pkl"))
    print("Saving model to:", model_path)
    dump(pipeline, model_path)
    metrics = {"r2": r2, "mae": mae, "rmse": rmse}
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    prepared = prepare(X_train)
    names = list(prepared.columns)
    importances = list(pipeline.named_steps["model"].feature_importances_)
    fi = [{"feature": n, "importance": float(i)} for n, i in zip(names, importances)]
    fi_sorted = sorted(fi, key=lambda x: x["importance"], reverse=True)
    with open(os.path.join(MODELS_DIR, "feature_importance.json"), "w", encoding="utf-8") as f:
        json.dump(fi_sorted, f)
    required = ["Store","DayOfWeek","Date","Customers","Promo","StateHoliday","SchoolHoliday","Open"]
    with open(os.path.join(MODELS_DIR, "columns.pkl"), "wb") as f:
        dump(required, f)
    print("Training complete")
    print(json.dumps(metrics))

if __name__ == "__main__":
    main()
