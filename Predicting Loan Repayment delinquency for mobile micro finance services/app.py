import os

import pandas as pd
from flask import Flask, jsonify, render_template, request

from loan_pipeline import TABLE_NAME, TARGET_COLUMN, load_model, read_rows_from_sqlite


BASE_DIR = os.path.dirname(__file__)
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "classification.db")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "loan_eligibility_model.joblib")
DEFAULT_METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "train_metadata.json")


app = Flask(__name__)
_MODEL = None
_LIVE_OFFSET = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_model(DEFAULT_MODEL_PATH)
    return _MODEL


def get_live_offset() -> int:
    global _LIVE_OFFSET
    if _LIVE_OFFSET is not None:
        return _LIVE_OFFSET

    if os.path.exists(DEFAULT_METADATA_PATH):
        import json

        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        _LIVE_OFFSET = int(meta.get("live_offset", 0))
    else:
        _LIVE_OFFSET = 0
    return _LIVE_OFFSET


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/schema")
def schema():
    model = get_model()
    columns = getattr(model.named_steps["features"], "input_columns_", [])
    return jsonify({"expected_fields": columns})


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send JSON object with feature fields."}), 400

    model = get_model()
    df = pd.DataFrame([payload])
    proba = float(model.predict_proba(df)[:, 1][0])
    pred = int(proba >= 0.5)
    return jsonify({"prediction": pred, "probability": proba})


@app.get("/api/predict-live")
def predict_live():
    try:
        offset = int(request.args.get("offset", str(get_live_offset())))
        n = int(request.args.get("n", "10"))
    except ValueError:
        return jsonify({"error": "offset and n must be integers"}), 400

    n = max(1, min(n, 100))
    offset = max(0, offset)

    df = read_rows_from_sqlite(
        db_path=DEFAULT_DB_PATH,
        table_name=TABLE_NAME,
        limit=n,
        offset=offset,
    )
    if TARGET_COLUMN in df.columns:
        df = df.drop(columns=[TARGET_COLUMN])

    model = get_model()
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    out = df.copy()
    out["prediction"] = pred
    out["probability"] = proba
    return jsonify({"rows": out.to_dict(orient="records")})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

