import os
import logging
import argparse
import json
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify
import sys
VERSION = "0.3.1"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)
from src.text_utils import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


MODEL_DIR = os.path.join(BASE_DIR, "models")


clusterer_path = os.path.join(MODEL_DIR, "clusterer_bert_kmeans.pkl")
labels_path = os.path.join(MODEL_DIR, "cluster_labels.json")
st_name_path = os.path.join(MODEL_DIR, "st_model_name.txt")
tfidf_model_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
tfidf_vec_path = os.path.join(MODEL_DIR, "tfidf.pkl")


use_bert = False
clusterer = None
labels_map = {}
bert_model = None
st_model_name = None
tfidf_model = None
tfidf_vectorizer = None

def ensure_tfidf_models():
    global tfidf_model, tfidf_vectorizer
    if tfidf_model is not None and tfidf_vectorizer is not None:
        return
    try:
        import pandas as pd
        import sqlite3
        data_paths = [
            os.path.join(BASE_DIR, "data", "clustering.db"),
            os.path.join(BASE_DIR, "clustering.db"),
            os.path.join(BASE_DIR, "data", "sample_reviews_30.csv"),
        ]
        texts = []
        db_used = False
        for p in data_paths:
            if os.path.exists(p) and p.endswith(".db"):
                try:
                    conn = sqlite3.connect(p)
                    try:
                        df = pd.read_sql("SELECT review_full FROM tripadvisor_reviews", conn)
                    finally:
                        conn.close()
                    texts = df["review_full"].astype(str).tolist()
                    db_used = True
                    break
                except Exception:
                    pass
        if not texts:
            for p in data_paths:
                if os.path.exists(p) and p.endswith(".csv"):
                    try:
                        df = pd.read_csv(p)
                        col = "review_full" if "review_full" in df.columns else df.columns[0]
                        texts = df[col].astype(str).tolist()
                        break
                    except Exception:
                        pass
        if not texts:
            texts = [
                "great staff and friendly service",
                "location is convenient near city center",
                "dirty room and poor cleanliness",
                "delicious breakfast and good restaurant",
                "price is expensive not value for money",
                "comfortable bed and nice room",
            ]
        cleaned = [clean_text(t) for t in texts]
        max_features = 2000 if db_used else 1000
        k = 5 if len(cleaned) >= 100 else min(5, max(2, len(cleaned)//10 or 2))
        vectorizer = TfidfVectorizer(max_features=max_features)
        X = vectorizer.fit_transform(cleaned).toarray()
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(km, tfidf_model_path)
        joblib.dump(vectorizer, tfidf_vec_path)
        tfidf_model = km
        tfidf_vectorizer = vectorizer
    except Exception as e:
        logging.exception("Failed to auto-build TF-IDF models: %s", e)

try:
    if SentenceTransformer is not None and os.path.exists(clusterer_path) and os.path.exists(st_name_path):
        clusterer = joblib.load(clusterer_path)
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_map = json.load(f)
        with open(st_name_path, "r", encoding="utf-8") as f:
            st_model_name = f.read().strip()
        use_bert = True
except Exception:
    use_bert = False

if not use_bert:
    if os.path.exists(tfidf_model_path) and os.path.exists(tfidf_vec_path):
        tfidf_model = joblib.load(tfidf_model_path)
        tfidf_vectorizer = joblib.load(tfidf_vec_path)
    else:
        ensure_tfidf_models()


app = Flask(__name__, template_folder="templates", static_folder="static")

def _get_bert_model():
    global bert_model
    if bert_model is None and SentenceTransformer is not None and st_model_name:
        try:
            bert_model = SentenceTransformer(st_model_name)
        except Exception:
            bert_model = None
    return bert_model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/health")
def health():
    return "ok"

@app.route("/version")
def version():
    return jsonify({"version": VERSION})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data.get("review", "")
    cleaned = clean_text(review)
    bm = _get_bert_model() if use_bert else None
    print("predict called; use_bert=", use_bert, "bert_loaded=", bm is not None, "tfidf_loaded=", tfidf_model is not None)
    if use_bert and bm is not None and clusterer is not None:
        embedding = bm.encode([cleaned])
        cluster = int(clusterer.predict(embedding)[0])
        theme = labels_map.get(str(cluster), "General Feedback")
    elif tfidf_model is not None and tfidf_vectorizer is not None:
        vec = tfidf_vectorizer.transform([cleaned]).toarray()
        cluster = int(tfidf_model.predict(vec)[0])
        theme = "General Feedback"
    else:
        cluster = 0
        words = set(cleaned.split())
        if any(w in words for w in ["staff", "service", "friendly", "helpful"]):
            theme = "Customer Service"
        elif any(w in words for w in ["location", "near", "central"]):
            theme = "Location Advantage"
        elif any(w in words for w in ["dirty", "clean", "cleanliness"]):
            theme = "Cleanliness Issues"
        elif any(w in words for w in ["food", "breakfast", "pizza", "restaurant"]):
            theme = "Food Experience"
        elif any(w in words for w in ["price", "expensive", "overpriced", "value"]):
            theme = "Pricing"
        elif any(w in words for w in ["room", "bed", "bathroom", "shower"]):
            theme = "Room Quality"
        else:
            theme = "General Feedback"
    tokens = cleaned.split()
    pos_words = {
        "good","great","excellent","amazing","nice","friendly","clean","comfortable","love","like","fantastic",
        "wonderful","awesome","tasty","delicious","affordable","value","recommend","helpful","best","better",
        "enjoy","pleasant","superb","perfect","happy","satisfied","sweet","lovely","spacious","cozy","quiet"
    }
    neg_words = {
        "bad","terrible","poor","awful","dirty","rude","slow","noisy","broken","worst","disappointing",
        "expensive","overpriced","uncomfortable","hate","smelly","cold","hot","soggy","bland","stale","issue","problem",
        "sad","angry","delay","late","cancel","smell","crowded","tiny","hard","rude","leak","bugs","insect"
    }
    negators = {"not","no","never","nor","hardly","barely","scarcely"}
    intensifiers = {"very","extremely","really","so","too","quite","super","highly","extremly"}
    diminishers = {"slightly","somewhat","a_bit","bit","fairly","rather","kinda","kind_of"}
    if "kind of" in review.lower():
        tokens.append("kind_of")
    score = 0.0
    i = 0
    while i < len(tokens):
        w = tokens[i]
        prev1 = tokens[i-1] if i > 0 else ""
        prev2 = tokens[i-2] if i > 1 else ""
        prev_neg = prev1 in negators or prev2 in negators
        weight = 1.0
        if prev1 in intensifiers or prev2 in intensifiers:
            weight = 1.5
        if prev1 in diminishers or prev2 in diminishers:
            weight = 0.5
        if w in pos_words:
            score += (-1 if prev_neg else 1) * weight
        elif w in neg_words:
            score += (1 if prev_neg else -1) * weight
        i += 1
    if "but" in review.lower():
        parts = review.lower().split("but")
        if len(parts) >= 2:
            after = clean_text(parts[-1]).split()
            adjust = 0.0
            for w in after:
                if w in pos_words:
                    adjust += 1
                elif w in neg_words:
                    adjust -= 1
            score = score * 0.5 + adjust
    if score > 0:
        sentiment = "Positive"
    elif score < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
    resp = {"cluster": cluster, "theme": theme, "sentiment": sentiment}
    print("response", resp)
    return jsonify(resp)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("server.log", encoding="utf-8"), logging.StreamHandler()]
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("HOST", os.environ.get("FLASK_RUN_HOST", "0.0.0.0")))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", os.environ.get("FLASK_RUN_PORT", "3000"))))
    args = parser.parse_args()
    host = args.host
    port = args.port
    logging.info(f"Starting Flask app on {host}:{port}")
    candidates = [int(port)]
    for p in (5000, 8000):
        if p not in candidates:
            candidates.append(p)
    started = False
    for p in candidates:
        logging.info(f"Attempting to start on {host}:{p}")
        try:
            try:
                from waitress import serve
                logging.info("Using waitress WSGI server")
                serve(app, host=host, port=p, threads=4)
            except Exception:
                logging.info("Falling back to Flask's built-in server")
                app.run(host=host, port=p, debug=False, use_reloader=False, threaded=True)
            started = True
            break
        except OSError as e:
            logging.error(f"Port {p} unavailable: {e}")
            continue
        except Exception as e:
            logging.exception(f"Failed to start on {host}:{p}: {e}")
            continue
    if not started:
        raise SystemExit("Could not start server on any candidate port")
