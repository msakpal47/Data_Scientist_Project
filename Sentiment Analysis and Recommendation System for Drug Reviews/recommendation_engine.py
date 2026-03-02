import os
import html
import sqlite3
from typing import List, Dict, Any, Tuple

BASE_DIR = r"e:\Data_Scientist_Project\Classifcation Projects\Sentiment Analysis and Recommendation System for Drug Reviews"
DB_PATH = os.path.join(BASE_DIR, "classification.db")
TABLE_NAME = "drug_reviews_sentiment_analysis"


def _fetch_reviews_for_condition(con: sqlite3.Connection, condition: str) -> List[Tuple[str, str, float, int]]:
    cur = con.cursor()
    cur.execute(
        f"SELECT drugName, review, rating, usefulCount FROM {TABLE_NAME} WHERE condition = ?",
        (condition,),
    )
    rows = cur.fetchall()
    return rows


def recommend_for_condition(condition: str, model, top_k: int = 3, min_reviews: int = 5) -> List[Dict[str, Any]]:
    if not condition or not condition.strip():
        return []
    if top_k <= 0:
        top_k = 3
    con = sqlite3.connect(DB_PATH)
    try:
        rows = _fetch_reviews_for_condition(con, condition.strip())
    finally:
        con.close()
    if not rows:
        return []
    texts = []
    meta = []
    for drug, review, rating, useful in rows:
        if review is None:
            continue
        t = html.unescape(str(review)).strip()
        if not t:
            continue
        try:
            r = float(rating) if rating is not None else 0.0
        except Exception:
            r = 0.0
        try:
            u = int(useful) if useful is not None else 0
        except Exception:
            u = 0
        texts.append(t)
        meta.append((drug or "", r, u))
    if not texts:
        return []
    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(texts)
        proba = [float(x[1]) for x in p]
        preds = [1 if pr >= 0.5 else 0 for pr in proba]
    else:
        preds = [int(x) for x in model.predict(texts)]
        proba = [1.0] * len(preds)
    pos_items = []
    for (drug, r, u), pr, label in zip(meta, proba, preds):
        if label == 1:
            pos_items.append((drug, r, u, pr))
    if not pos_items:
        return []
    useful_vals = [u for _, _, u, _ in pos_items]
    umax = max(useful_vals) if useful_vals else 0
    agg = {}
    for drug, r, u, pr in pos_items:
        if not drug:
            continue
        v = agg.get(drug)
        if v is None:
            agg[drug] = [0.0, 0.0, 0.0, 0]
            v = agg[drug]
        v[0] += r
        v[1] += pr
        v[2] += u
        v[3] += 1
    scored = []
    for drug, (sum_r, sum_p, sum_u, cnt) in agg.items():
        if cnt < min_reviews:
            continue
        avg_r = (sum_r / cnt) if cnt else 0.0
        avg_p = (sum_p / cnt) if cnt else 0.0
        avg_u = (sum_u / cnt) if cnt else 0.0
        r_norm = max(0.0, min(1.0, avg_r / 10.0))
        u_norm = (avg_u / umax) if umax > 0 else 0.0
        score = 0.5 * r_norm + 0.3 * avg_p + 0.2 * u_norm
        scored.append(
            {
                "drugName": drug,
                "score": float(score),
                "avg_rating": float(avg_r),
                "avg_sentiment_prob": float(avg_p),
                "avg_useful_norm": float(u_norm),
                "num_reviews": int(cnt),
            }
        )
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]
