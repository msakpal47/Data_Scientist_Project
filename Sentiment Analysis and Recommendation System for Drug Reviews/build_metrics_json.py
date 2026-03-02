import os
import json
import html
import sqlite3
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from datetime import datetime

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
DB_PATH = os.path.join(BASE_DIR, "classification.db")
TABLE = "drug_reviews_sentiment_analysis"
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_CANDIDATES = [
    os.path.join(MODELS_DIR, "sentiment_model.joblib"),
    os.path.join(MODELS_DIR, "sentiment_model.pkl"),
    os.path.join(BASE_DIR, "sentiment_model.joblib"),
    os.path.join(BASE_DIR, "sentiment_model.pkl"),
]
OUT_PATH = os.path.join(BASE_DIR, "static", "metrics.json")

def load_model():
    for p in MODEL_CANDIDATES:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError("Model file not found")

def label_from_rating(r):
    r = int(r)
    if r >= 7:
        return 1
    if r <= 4:
        return 0
    return -1

def main():
    model = load_model()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE review IS NOT NULL AND (rating <= 4 OR rating >= 7)")
    total = int(cur.fetchone()[0])
    n_train = 112908
    n_val = 32259
    n_prod = 16130
    def fetch_slice(offset, limit):
        cur.execute(
            f"""
SELECT uniqueID, review, rating
FROM {TABLE}
WHERE review IS NOT NULL AND (rating <= 4 OR rating >= 7)
ORDER BY uniqueID ASC
LIMIT ? OFFSET ?
""",
            (limit, offset),
        )
        return cur.fetchall()
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
        pass
    top_pos = []
    top_neg = []
    if vec is not None and clf is not None and hasattr(clf, "coef_"):
        feats = vec.get_feature_names_out()
        w = clf.coef_[0]
        pos_idx = w.argsort()[-20:][::-1]
        neg_idx = w.argsort()[:20]
        top_pos = [str(feats[i]) for i in pos_idx]
        top_neg = [str(feats[i]) for i in neg_idx]
    # Sentiment distribution per condition (top 20)
    cur.execute(
        f"""
SELECT "condition",
SUM(CASE WHEN rating >= 7 THEN 1 ELSE 0 END) AS pos,
SUM(CASE WHEN rating <= 4 THEN 1 ELSE 0 END) AS neg,
SUM(CASE WHEN rating >= 7 OR rating <= 4 THEN 1 ELSE 0 END) AS total
FROM {TABLE}
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
        "model_version": datetime.fromtimestamp(os.path.getmtime(MODEL_CANDIDATES[0] if os.path.exists(MODEL_CANDIDATES[0]) else MODEL_CANDIDATES[2])).strftime("%Y%m%d-%H%M"),
        "training_date": datetime.fromtimestamp(os.path.getmtime(MODEL_CANDIDATES[0] if os.path.exists(MODEL_CANDIDATES[0]) else MODEL_CANDIDATES[2])).isoformat(),
    }
    payload = {
        "meta": meta,
        "splits": {"train": n_train, "val": n_val, "prod": n_prod, "eligible_total": total},
        "validation": val_metrics,
        "production": prod_metrics,
        "feature_importance": {"top_positive": top_pos, "top_negative": top_neg},
        "sentiment_distribution": sentiment_dist,
    }
    os.makedirs(os.path.join(BASE_DIR, "static"), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("WROTE", OUT_PATH)

if __name__ == "__main__":
    main()
