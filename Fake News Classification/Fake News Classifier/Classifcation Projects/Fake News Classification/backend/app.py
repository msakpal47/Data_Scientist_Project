import json
import os
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, request, render_template, redirect, url_for
from joblib import load

from .preprocess import clean_text
import json


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def artifacts_dir() -> Path:
    m = project_root() / "models"
    if (m / "model.pkl").exists() and (m / "vectorizer.pkl").exists():
        return m
    return Path(__file__).resolve().parent


def load_artifacts():
    model_path = artifacts_dir() / "model.pkl"
    vec_path = artifacts_dir() / "vectorizer.pkl"
    if not model_path.exists() or not vec_path.exists():
        return None, None
    model = load(model_path)
    vectorizer = load(vec_path)
    return model, vectorizer


def label_map() -> dict:
    lm = artifacts_dir() / "label_map.json"
    if lm.exists():
        try:
            return json.loads(lm.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"0": "FAKE", "1": "TRUE"}

def get_accuracy() -> float | None:
    mf = artifacts_dir() / "metrics.json"
    if mf.exists():
        try:
            return float(json.loads(mf.read_text(encoding="utf-8")).get("accuracy"))
        except Exception:
            return None
    return None

app = Flask(
    __name__,
    template_folder=str(project_root() / "templates"),
    static_folder=str(project_root() / "static"),
)


@app.route("/")
def index():
    return render_template("index.html", accuracy=get_accuracy())


@app.route("/predict", methods=["GET", "POST"])
def predict_form():
    if request.method == "GET":
        return redirect(url_for("index"))
    text = request.form.get("news", "").strip()
    if not text:
        return render_template("index.html", prediction="No input provided", input_text="", accuracy=get_accuracy())
    if len(text) < 20:
        return render_template("index.html", prediction="Text too short. Please provide more context.", input_text=text, accuracy=get_accuracy())
    model, vectorizer = load_artifacts()
    if model is None or vectorizer is None:
        return render_template("index.html", prediction="Model artifacts not found", input_text=text)
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    proba = float(model.predict_proba(X)[:, 1][0])
    # Use learned threshold if available
    tfile = artifacts_dir() / "threshold.txt"
    thresh = 0.5
    if tfile.exists():
        pass
    pred = int(proba >= thresh)
    lm = label_map()
    pos_name = lm.get("1", "REAL")
    neg_name = lm.get("0", "FAKE")
    label = pos_name if pred == 1 else neg_name
    return render_template("index.html", prediction=f"{label} (P({pos_name})={proba:.3f}, threshold={thresh:.2f})", input_text=text, accuracy=get_accuracy())


@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    if request.method == "GET":
        return jsonify({"usage": "POST JSON: { 'texts': ['...'] }"}), 200
    body = request.get_json(force=True, silent=True) or {}
    texts = body.get("texts") or []
    if isinstance(texts, str):
        texts = [texts]
    texts = [t for t in texts if isinstance(t, str) and len(t.strip()) >= 20]

    model, vectorizer = load_artifacts()
    if model is None or vectorizer is None:
        return jsonify({"error": "Model artifacts not found. Run backend/train_model.py first."}), 400

    cleaned = [clean_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    probs = model.predict_proba(X)[:, 1]
    thresh = 0.5
    tfile = artifacts_dir() / "threshold.txt"
    if tfile.exists():
        try:
            thresh = float(tfile.read_text(encoding="utf-8").strip())
        except Exception:
            pass
    preds = (probs >= thresh).astype(int).tolist()
    feats = vectorizer.get_feature_names_out()
    coefs = getattr(model, "coef_", None)
    coefs = coefs[0] if coefs is not None else None
    explanations = []
    for i in range(X.shape[0]):
        row = X[i]
        idx = row.indices
        dat = row.data
        items = []
        if coefs is not None:
            for k, j in enumerate(idx):
                items.append({"token": feats[j], "weight": float(coefs[j] * dat[k])})
            items = sorted(items, key=lambda x: abs(x["weight"]), reverse=True)[:6]
        explanations.append(items)
    return jsonify(
        {
            "results": [
                {
                    "text": t,
                    "pred": int(p),
                    "label": bool(p),
                    "class_name": label_map().get("1", "TRUE") if int(p) == 1 else label_map().get("0", "FAKE"),
                    "proba_true": float(s),
                    "threshold": float(thresh),
                    "confidence": ("High" if abs(float(s) - float(thresh)) >= 0.35 else ("Medium" if abs(float(s) - float(thresh)) >= 0.15 else "Low")),
                    "top_words": ex,
                }
                for t, p, s, ex in zip(texts, preds, probs, explanations)
            ]
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
