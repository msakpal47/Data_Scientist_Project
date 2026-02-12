import json
import os
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "classification.db")
TABLE = "fault_detection_manufacturing"

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fault_model.pkl")
SCHEMA_PATH = os.path.join(MODEL_DIR, "schema.json")

WEB_DIR = os.path.join(BASE_DIR, "web")

FEATURES = [
    "IONGAUGEPRESSURE",
    "ETCHBEAMVOLTAGE",
    "ETCHBEAMCURRENT",
    "ETCHSUPPRESSORVOLTAGE",
    "ETCHSUPPRESSORCURRENT",
    "FLOWCOOLFLOWRATE",
    "FLOWCOOLPRESSURE",
    "ETCHGASCHANNEL1READBACK",
    "ETCHPBNGASREADBACK",
    "FIXTURETILTANGLE",
    "ROTATIONSPEED",
    "ACTUALROTATIONANGLE",
    "FIXTURESHUTTERPOSITION",
    "ETCHSOURCEUSAGE",
    "ETCHAUXSOURCETIMER",
    "ETCHAUX2SOURCETIMER",
    "ACTUALSTEPDURATION",
    "TTF_FlowCool Pressure Dropped Below Limit",
    "TTF_Flowcool Pressure Too High Check Flowcool Pump",
    "TTF_Flowcool leak",
]
TARGET = "fault_occurred"

app = Flask(__name__, static_folder=None)

def _load_df(columns: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    if columns:
        col_sql = ", ".join([f'"{c}"' for c in columns])
    else:
        col_sql = "*"
    sql = f'SELECT {col_sql} FROM "{TABLE}"'
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    df = pd.read_sql_query(sql, con)
    con.close()
    return df

def _load_training_df(columns: list[str], max_negative_rows: int) -> pd.DataFrame:
    con = sqlite3.connect(DB_PATH)
    col_sql = ", ".join([f'"{c}"' for c in columns])
    pos_sql = f'SELECT {col_sql} FROM "{TABLE}" WHERE "{TARGET}" = 1'
    neg_sql = (
        f'SELECT {col_sql} FROM "{TABLE}" WHERE "{TARGET}" = 0 '
        f"ORDER BY RANDOM() LIMIT {int(max_negative_rows)}"
    )
    df_pos = pd.read_sql_query(pos_sql, con)
    df_neg = pd.read_sql_query(neg_sql, con)
    con.close()
    return pd.concat([df_pos, df_neg], ignore_index=True)

def _clean_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    for c in FEATURES:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET])
    y = df[TARGET].astype(int)
    X = df[FEATURES]
    return X, y

def _build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    n_jobs=-1,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )

def _compute_schema(df: pd.DataFrame) -> dict:
    schema = {"table": TABLE, "target": TARGET, "features": []}
    for f in FEATURES:
        s = pd.to_numeric(df[f], errors="coerce")
        schema["features"].append(
            {
                "name": f,
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
                "median": float(np.nanmedian(s)),
            }
        )
    schema["generated_at"] = datetime.now(timezone.utc).isoformat()
    return schema

def _load_schema() -> dict | None:
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def _save_schema(schema: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCHEMA_PATH, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

def _load_model_bundle() -> dict | None:
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def _save_model_bundle(bundle: dict) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

@app.get("/")
def index():
    return send_from_directory(WEB_DIR, "index.html")

@app.get("/styles.css")
def styles():
    return send_from_directory(WEB_DIR, "styles.css")

@app.get("/app.js")
def app_js():
    return send_from_directory(WEB_DIR, "app.js")

@app.get("/@vite/client")
def vite_client_stub():
    return ("", 204)

@app.get("/api/health")
def health():
    return jsonify({"ok": True})

@app.get("/api/schema")
def api_schema():
    schema = _load_schema()
    if schema is None:
        df = _load_df(columns=FEATURES + [TARGET], limit=200000)
        schema = _compute_schema(df)
        _save_schema(schema)
    return jsonify(schema)

@app.post("/api/train")
def api_train():
    payload = request.get_json(silent=True) or {}
    max_negative_rows = payload.get("max_negative_rows", payload.get("max_rows", 200000))
    max_negative_rows = int(max_negative_rows) if max_negative_rows else 200000

    df = _load_training_df(columns=FEATURES + [TARGET], max_negative_rows=max_negative_rows)
    schema = _compute_schema(df)
    _save_schema(schema)

    X, y = _clean_xy(df)
    if y.nunique() < 2:
        return jsonify({"ok": False, "error": "Target has <2 classes after cleaning."}), 400

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = _build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    bundle = {"model": pipe, "features": FEATURES, "target": TARGET, "accuracy": acc}
    _save_model_bundle(bundle)
    return jsonify({"ok": True, "accuracy": acc})

@app.post("/api/predict")
def api_predict():
    bundle = _load_model_bundle()
    if bundle is None:
        return jsonify({"ok": False, "error": "Model not trained. Call /api/train first."}), 400

    payload = request.get_json(silent=True) or {}
    x = []
    for f in bundle["features"]:
        v = payload.get(f, None)
        x.append(np.nan if v is None else float(v))
    X = pd.DataFrame([x], columns=bundle["features"])
    model = bundle["model"]

    pred = int(model.predict(X)[0])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
    return jsonify({"ok": True, "prediction": pred, "probability": proba})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="127.0.0.1", port=port, debug=False)
