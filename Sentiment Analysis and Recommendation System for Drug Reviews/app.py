import os
import html
import sqlite3
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from recommendation_engine import recommend_for_condition

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "sentiment_model.joblib"),
    os.path.join(MODELS_DIR, "sentiment_model.pkl"),
    os.path.join(BASE_DIR, "sentiment_model.joblib"),
    os.path.join(BASE_DIR, "sentiment_model.pkl"),
]
STATIC_DIR = os.path.join(BASE_DIR, "static")
DB_PATH = os.path.join(BASE_DIR, "classification.db")
TABLE_NAME = "drug_reviews_sentiment_analysis"

app = Flask(__name__, static_folder=STATIC_DIR)
_model_path = None
for _p in MODEL_CANDIDATES:
    if os.path.exists(_p):
        _model_path = _p
        break
if _model_path is None:
    raise FileNotFoundError("Model file not found")
model = joblib.load(_model_path)
_metrics_cache = None

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/metrics")
def metrics_page():
    return send_from_directory(BASE_DIR, "metrics.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True) or {}
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "text is required"}), 400
    text = html.unescape(text)
    pred = int(model.predict([text])[0])
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba([text])[0][pred])
    else:
        prob = None
    return jsonify({"sentiment": "positive" if pred == 1 else "negative", "label": pred, "confidence": prob})

@app.route("/predict_sentiment", methods=["POST"])
def predict_sentiment():
    return predict()

@app.route("/recommend_drug", methods=["POST", "GET"])
def recommend_drug():
    if request.method == "POST":
        data = request.get_json(force=True) or {}
        condition = data.get("condition", "")
        top_k = int(data.get("top_k", 3))
        min_reviews = int(data.get("min_reviews", 5))
    else:
        condition = request.args.get("condition", "")
        top_k = int(request.args.get("top_k", 3))
        min_reviews = int(request.args.get("min_reviews", 5))
    if not condition.strip():
        return jsonify({"error": "condition is required"}), 400
    recs = recommend_for_condition(condition.strip(), model, top_k=top_k, min_reviews=min_reviews)
    return jsonify({"condition": condition.strip(), "count": len(recs), "results": recs})

def _jsonify_float(x):
    try:
        return float(x)
    except Exception:
        try:
            return int(x)
        except Exception:
            return x

def _compute_metrics():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE review IS NOT NULL AND (rating <= 4 OR rating >= 7)"
    )
    total = int(cur.fetchone()[0])
    n_train = 112908
    n_val = 32259
    n_prod = 16130
    def fetch_slice(offset, limit):
        cur.execute(
            f"""
SELECT uniqueID, review, rating
FROM {TABLE_NAME}
WHERE review IS NOT NULL AND (rating <= 4 OR rating >= 7)
ORDER BY uniqueID ASC
LIMIT ? OFFSET ?
""",
            (limit, offset),
        )
        rows = cur.fetchall()
        return rows
    def label_from_rating(r):
        r = int(r)
        if r >= 7:
            return 1
        if r <= 4:
            return 0
        return -1
    def evaluate(rows):
        texts = []
        y_true = []
        for _, review, rating in rows:
            if review is None:
                continue
            t = html.unescape(str(review)).strip()
            if not t:
                continue
            y = label_from_rating(rating)
            if y == -1:
                continue
            texts.append(t)
            y_true.append(y)
        if not texts:
            return None
        y_pred = model.predict(texts)
        rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred).tolist()
        y_proba = None
        roc = None
        cal = None
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(texts)[:, 1]
                fpr, tpr, _ = roc_curve(y_true, y_proba)
                auc = roc_auc_score(y_true, y_proba)
                roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": float(auc)}
                prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
                cal = {"prob_pred_mean": prob_pred.tolist(), "frac_pos": prob_true.tolist()}
            except Exception:
                pass
        pos = int(sum(1 for y in y_true if y == 1))
        neg = int(sum(1 for y in y_true if y == 0))
        return {"report": rep, "confusion_matrix": cm, "n_samples": len(y_true), "class_balance": {"pos": pos, "neg": neg}, "roc": roc, "calibration": cal}
    val_rows = fetch_slice(n_train, n_val) if total >= n_train else []
    prod_rows = fetch_slice(n_train + n_val, n_prod) if total >= (n_train + n_val) else []
    val_metrics = evaluate(val_rows) if val_rows else None
    prod_metrics = evaluate(prod_rows) if prod_rows else None
    vec = None
    clf = None
    try:
        vec = model.named_steps.get("tfidf")
        clf = model.named_steps.get("clf")
    except Exception:
        vec = None
        clf = None
    top_pos = []
    top_neg = []
    if vec is not None and clf is not None and hasattr(clf, "coef_"):
        feats = vec.get_feature_names_out()
        w = clf.coef_[0]
        pos_idx = w.argsort()[-20:][::-1]
        neg_idx = w.argsort()[:20]
        top_pos = [str(feats[i]) for i in pos_idx]
        top_neg = [str(feats[i]) for i in neg_idx]
    # Sentiment distribution per condition (top 20 by count)
    cur.execute(
        f"""
SELECT "condition",
SUM(CASE WHEN rating >= 7 THEN 1 ELSE 0 END) AS pos,
SUM(CASE WHEN rating <= 4 THEN 1 ELSE 0 END) AS neg,
SUM(CASE WHEN rating >= 7 OR rating <= 4 THEN 1 ELSE 0 END) AS total
FROM {TABLE_NAME}
WHERE "condition" IS NOT NULL
GROUP BY "condition"
ORDER BY total DESC
LIMIT 20
"""
    )
    dist_rows = cur.fetchall()
    sentiment_dist = []
    for cond, pos, neg, tot in dist_rows:
        ratio = float(pos) / float(tot) if tot else 0.0
        sentiment_dist.append({"condition": str(cond), "pos": int(pos), "neg": int(neg), "total": int(tot), "pos_ratio": ratio})
    meta = {
        "model": "LogisticRegression",
        "vectorizer": "TfidfVectorizer",
        "ngram_range": [1, 2],
        "max_features": 50000,
        "model_path": _model_path,
        "model_version": datetime.fromtimestamp(os.path.getmtime(_model_path)).strftime("%Y%m%d-%H%M"),
        "training_date": datetime.fromtimestamp(os.path.getmtime(_model_path)).isoformat(),
    }
    return {
        "meta": meta,
        "splits": {"train": n_train, "val": n_val, "prod": n_prod, "eligible_total": total},
        "validation": val_metrics,
        "production": prod_metrics,
        "feature_importance": {"top_positive": top_pos, "top_negative": top_neg},
        "sentiment_distribution": sentiment_dist,
    }

@app.route("/metrics_data")
def metrics_data():
    global _metrics_cache
    if _metrics_cache is None:
        _metrics_cache = _compute_metrics()
    def normalize(obj):
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        return _jsonify_float(obj)
    return jsonify(normalize(_metrics_cache))

@app.route("/conditions")
def conditions():
    try:
        con = sqlite3.connect(DB_PATH)
        cur = con.cursor()
        cur.execute(
            f'SELECT "condition", COUNT(*) FROM {TABLE_NAME} WHERE "condition" IS NOT NULL GROUP BY "condition" ORDER BY "condition" ASC'
        )
        rows = cur.fetchall()
        def valid_cond(s: str) -> bool:
            if not s:
                return False
            t = s.strip()
            if "<" in t or ">" in t or "users found this comment helpful" in t:
                return False
            if len(t) < 2:
                return False
            return True
        filtered = [(str(c).strip(), int(n)) for c, n in rows if valid_cond(str(c) if c is not None else "")]
        filtered.sort(key=lambda x: (-x[1], x[0]))
        out = [{"condition": c, "count": n} for c, n in filtered]
        return jsonify({"count": len(out), "results": out})
    except Exception as e:
        return jsonify({"error": f"database error: {e}"}), 500
    finally:
        try:
            con.close()
        except Exception:
            pass

@app.route("/__routes")
def list_routes():
    rules = []
    for r in app.url_map.iter_rules():
        rules.append({"rule": r.rule, "methods": sorted(list(r.methods))})
    return jsonify({"routes": sorted(rules, key=lambda x: x["rule"])})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
