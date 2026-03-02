import os
import io

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from loan_pipeline import TABLE_NAME, TARGET_COLUMN, load_model, read_rows_from_sqlite


BASE_DIR = os.path.dirname(__file__)
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "classification.db")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "loan_eligibility_model.joblib")
DEFAULT_METADATA_PATH = os.path.join(BASE_DIR, "artifacts", "train_metadata.json")
DEFAULT_ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")


app = Flask(__name__)
_MODEL = None
_LIVE_OFFSET = None
_THRESHOLD = None


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


def get_threshold() -> float:
    global _THRESHOLD
    if _THRESHOLD is not None:
        return float(_THRESHOLD)
    if os.path.exists(DEFAULT_METADATA_PATH):
        import json

        with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        _THRESHOLD = float(meta.get("threshold", 0.5))
    else:
        _THRESHOLD = 0.5
    return float(_THRESHOLD)

def _load_eval_data():
    import json
    from loan_pipeline import read_rows_from_sqlite
    if not os.path.exists(DEFAULT_METADATA_PATH):
        return None, None, None
    with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    train_rows = int(meta.get("train_rows", 0))
    eval_rows = int(meta.get("eval_rows", 0))
    if eval_rows <= 0:
        return None, None, None
    df = read_rows_from_sqlite(
        db_path=DEFAULT_DB_PATH,
        table_name=TABLE_NAME,
        limit=eval_rows,
        offset=train_rows,
    )
    if TARGET_COLUMN not in df.columns:
        return None, None, None
    model = get_model()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int).to_numpy()
    proba = model.predict_proba(X)[:, 1]
    return y, proba, meta

def _ensure_plots():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, RocCurveDisplay, confusion_matrix
        from sklearn.calibration import calibration_curve
    except Exception:
        return
    y, proba, _ = _load_eval_data()
    if y is None or proba is None:
        return
    os.makedirs(DEFAULT_ARTIFACTS_DIR, exist_ok=True)
    # ROC curve
    try:
        fpr, tpr, _ = roc_curve(y, proba)
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
        disp.plot(ax=ax)
        ax.set_title("ROC Curve")
        fig.tight_layout()
        fig.savefig(os.path.join(DEFAULT_ARTIFACTS_DIR, "roc_curve.png"))
        plt.close(fig)
    except Exception:
        pass
    # Confusion matrix at tuned threshold
    try:
        thr = get_threshold()
        y_pred = (proba >= thr).astype(int)
        cm = confusion_matrix(y, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1], ["0", "1"])
        ax.set_yticks([0, 1], ["0", "1"])
        for (i, j), v in zip([(0,0),(0,1),(1,0),(1,1)], cm.flatten()):
            ax.text(j, i, str(int(v)), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(os.path.join(DEFAULT_ARTIFACTS_DIR, "confusion_matrix.png"))
        plt.close(fig)
    except Exception:
        pass
    # Calibration curve
    try:
        frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="uniform")
        fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
        ax.plot(mean_pred, frac_pos, "s-", label="Model")
        ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title("Calibration Curve")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(DEFAULT_ARTIFACTS_DIR, "calibration_curve.png"))
        plt.close(fig)
    except Exception:
        pass


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
    expected = getattr(model.named_steps["features"], "input_columns_", [])
    if expected:
        missing = [c for c in expected if c not in payload]
        if missing:
            return jsonify({"error": "Missing required fields", "missing": missing}), 400
    clean = dict(payload)
    if "msisdn" in clean:
        clean.pop("msisdn", None)
    df = pd.DataFrame([clean])
    proba = float(model.predict_proba(df)[:, 1][0])
    thr = get_threshold()
    pred = int(proba >= thr)
    return jsonify({"prediction": pred, "probability": proba, "threshold": thr})


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
    thr = get_threshold()
    pred = (proba >= thr).astype(int)

    out = df.copy()
    out["prediction"] = pred
    out["probability"] = proba
    return jsonify({"rows": out.to_dict(orient="records")})


@app.get("/api/metrics")
def metrics():
    if not os.path.exists(DEFAULT_METADATA_PATH):
        return jsonify({"error": "metadata not found"}), 404
    import json

    with open(DEFAULT_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    m = meta.get("metrics", {})
    def r4(v):
        try:
            return round(float(v), 4)
        except Exception:
            return v
    _ensure_plots()
    cm = m.get("confusion_matrix")
    y, proba, _ = _load_eval_data()
    support_count = int(len(y)) if y is not None else int(meta.get("eval_rows", 0))
    pos_rate = float(y.mean()) if y is not None else None
    return jsonify({
        "accuracy": r4(m.get("accuracy")),
        "precision": r4(m.get("precision")),
        "recall": r4(m.get("recall")),
        "f1": r4(m.get("f1")),
        "roc_auc": r4(m.get("roc_auc")),
        "threshold": r4(meta.get("threshold", 0.5)),
        "threshold_policy": meta.get("threshold_policy"),
        "target_fpr": r4(meta.get("target_fpr")),
        "recall_at_target_fpr": r4(meta.get("recall_at_target_fpr")),
        "fpr_at_threshold": r4(meta.get("fpr_at_threshold")),
        "validation_samples": support_count,
        "positive_rate_pct": r4((pos_rate * 100.0) if pos_rate is not None else None),
        "trained_at": meta.get("trained_at"),
        "table_name": meta.get("table_name"),
        "model_version": meta.get("model_version", "v1.0"),
        "confusion_matrix": cm,
        "top_features": meta.get("top_features", [])[:10],
        "plots": {
            "roc": "/plots/roc_curve.png",
            "confusion_matrix": "/plots/confusion_matrix.png",
            "calibration": "/plots/calibration_curve.png",
        },
    })

@app.get("/metrics")
def metrics_page():
    return render_template("metrics.html")

@app.get("/plots/<path:filename>")
def plots(filename: str):
    return send_from_directory(DEFAULT_ARTIFACTS_DIR, filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

