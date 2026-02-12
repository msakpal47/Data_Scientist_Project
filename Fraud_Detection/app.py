import os
import sqlite3
import pickle
import numpy as np
import pandas as pd
import json
from flask import Flask, jsonify, request, render_template, Response
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight


DB_PATH = os.path.join(os.getcwd(), "classification.db")
MODEL_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "fraud_model.pkl")


def list_tables(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cur.fetchall()
    return [r[0] for r in rows]


def load_data(limit_rows: int | None = 200000) -> pd.DataFrame:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        tables = list_tables(conn)
        if not tables:
            raise RuntimeError("No tables found in classification.db")
        table = tables[0]
        query = f"SELECT * FROM {table}"
        if limit_rows is not None:
            query += f" LIMIT {int(limit_rows)}"
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def build_pipeline(df: pd.DataFrame) -> Pipeline:
    numeric_features = [
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    categorical_features = ["type"]

    drop_features = [c for c in ["nameOrig", "nameDest"] if c in df.columns]

    num_cols = [c for c in numeric_features if c in df.columns]
    cat_cols = [c for c in categorical_features if c in df.columns]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[("impute", SimpleImputer(strategy="median"))]), num_cols),
            ("cat", Pipeline(steps=[("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ],
        remainder="drop",
    )

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )
    return pipe, drop_features


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target_col = "isFraud"
    if target_col not in df.columns:
        raise RuntimeError("Expected target column 'isFraud' not found in data")
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def train_model(df: pd.DataFrame) -> Pipeline:
    pipe, drop_cols = build_pipeline(df)
    X, y = prepare_data(df)
    if drop_cols:
        for c in drop_cols:
            if c in X.columns:
                X = X.drop(columns=[c])
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1)
    X = X[mask]
    y = y[mask]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    classes = np.array([0, 1])
    cls_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_map = {cls: w for cls, w in zip(classes, cls_weights)}
    sample_weight = y_train.map(weight_map).to_numpy()
    pipe.fit(X_train, y_train, clf__sample_weight=sample_weight)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)
    return pipe


def load_model() -> Pipeline:
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    df_local = load_data()
    return train_model(df_local)


app = Flask(__name__)
data_frame: pd.DataFrame = load_data()
model = load_model()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    if "isFraud" not in data_frame.columns:
        return jsonify({"non_fraud": 0, "fraud": 0, "fraud_rate": 0.0, "flagged": 0, "flagged_rate": 0.0})
    is_fraud_num = pd.to_numeric(data_frame["isFraud"], errors="coerce").fillna(0).astype(int)
    non_fraud = int((is_fraud_num == 0).sum())
    fraud = int((is_fraud_num == 1).sum())
    total = non_fraud + fraud
    rate = float(fraud / total) if total > 0 else 0.0
    flagged = 0
    flagged_rate = 0.0
    if "isFlaggedFraud" in data_frame.columns:
        flagged_num = pd.to_numeric(data_frame["isFlaggedFraud"], errors="coerce").fillna(0).astype(int)
        flagged = int((flagged_num == 1).sum())
        flagged_rate = float(flagged / len(data_frame)) if len(data_frame) > 0 else 0.0
    return jsonify({"non_fraud": non_fraud, "fraud": fraud, "fraud_rate": rate, "flagged": flagged, "flagged_rate": flagged_rate})


@app.route("/api/transactions")
def api_transactions():
    filter_value = request.args.get("filter", "all")
    limit = request.args.get("limit", type=int)
    page = request.args.get("page", type=int, default=1)
    df = data_frame
    if "isFraud" in df.columns:
        is_fraud_num = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
        if filter_value == "fraud":
            df = df[is_fraud_num == 1]
        elif filter_value == "nonfraud":
            df = df[is_fraud_num == 0]
    if filter_value == "flagged" and "isFlaggedFraud" in df.columns:
        flagged_num = pd.to_numeric(df["isFlaggedFraud"], errors="coerce").fillna(0).astype(int)
        df = df[flagged_num == 1]
    total_rows = int(df.shape[0])
    if limit and limit > 0:
        start = max(0, (page - 1) * limit)
        end = start + limit
        df = df.iloc[start:end]
    cols = list(df.columns)
    df_safe = df.replace([np.inf, -np.inf], np.nan)
    rows = json.loads(df_safe.to_json(orient="records"))
    return jsonify({"columns": cols, "rows": rows, "total_rows": total_rows})

@app.route("/api/export")
def api_export():
    filter_value = request.args.get("filter", "all")
    df = data_frame
    if "isFraud" in df.columns:
        is_fraud_num = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
        if filter_value == "fraud":
            df = df[is_fraud_num == 1]
        elif filter_value == "nonfraud":
            df = df[is_fraud_num == 0]
    if filter_value == "flagged" and "isFlaggedFraud" in df.columns:
        flagged_num = pd.to_numeric(df["isFlaggedFraud"], errors="coerce").fillna(0).astype(int)
        df = df[flagged_num == 1]
    df_safe = df.replace([np.inf, -np.inf], np.nan)
    csv_data = df_safe.to_csv(index=False)
    filename = f"{filter_value}_transactions.csv"
    return Response(csv_data, mimetype="text/csv", headers={"Content-Disposition": f"attachment; filename=\"{filename}\""})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    input_df = pd.DataFrame(
        [
            {
                "type": payload.get("type", "TRANSFER"),
                "amount": float(payload.get("amount", 0.0)),
                "oldbalanceOrg": float(payload.get("oldbalanceOrg", 0.0)),
                "newbalanceOrig": float(payload.get("newbalanceOrig", 0.0)),
                "oldbalanceDest": float(payload.get("oldbalanceDest", 0.0)),
                "newbalanceDest": float(payload.get("newbalanceDest", 0.0)),
            }
        ]
    )
    prob = float(model.predict_proba(input_df)[0, 1])
    label = int(prob >= 0.5)
    return jsonify({"probability": prob, "label": label})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True)
