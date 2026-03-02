import os
import sys
import sqlite3
import joblib
import pandas as pd
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.text_utils import clean_text
import argparse


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DB_DEFAULT = os.path.join(DATA_DIR, "clustering.db")
DB_FALLBACK = os.path.join(ROOT_DIR, "clustering.db")
TABLE = "tripadvisor_reviews"
TEXT_COL = "review_full"


def db_path():
    return DB_DEFAULT if os.path.exists(DB_DEFAULT) else DB_FALLBACK


def load_df(limit):
    conn = sqlite3.connect(db_path())
    try:
        df = pd.read_sql(f"SELECT {TEXT_COL} FROM {TABLE} LIMIT {limit}", conn)
    finally:
        conn.close()
    return df


def try_load_bert():
    clusterer_path = os.path.join(MODELS_DIR, "clusterer_bert_kmeans.pkl")
    st_name_path = os.path.join(MODELS_DIR, "st_model_name.txt")
    labels_path = os.path.join(MODELS_DIR, "cluster_labels.json")
    if os.path.exists(clusterer_path) and os.path.exists(st_name_path):
        clusterer = joblib.load(clusterer_path)
        with open(st_name_path, "r", encoding="utf-8") as f:
            model_name = f.read().strip()
        st_model = SentenceTransformer(model_name)
        labels_map = {}
        if os.path.exists(labels_path):
            import json
            with open(labels_path, "r", encoding="utf-8") as f:
                labels_map = json.load(f)
        return clusterer, st_model, labels_map
    return None, None, {}


def try_load_tfidf():
    model_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    vec_path = os.path.join(MODELS_DIR, "tfidf.pkl")
    if os.path.exists(model_path) and os.path.exists(vec_path):
        return joblib.load(model_path), joblib.load(vec_path)
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(DATA_DIR, "sample_reviews_30.csv"))
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_df(args.limit)
    df["cleaned_review"] = df[TEXT_COL].apply(clean_text)

    bert_clusterer, st_model, labels_map = (None, None, {})
    if SentenceTransformer is not None:
        try:
            bert_clusterer, st_model, labels_map = try_load_bert()
        except Exception:
            bert_clusterer, st_model, labels_map = (None, None, {})
    tfidf_model, tfidf_vec = try_load_tfidf()

    clusters = []
    themes = []
    model_types = []
    for text in df["cleaned_review"]:
        if bert_clusterer is not None and st_model is not None:
            emb = st_model.encode([text])
            c = int(bert_clusterer.predict(emb)[0])
            t = labels_map.get(str(c), "General Feedback")
            clusters.append(c)
            themes.append(t)
            model_types.append("bert")
        elif tfidf_model is not None and tfidf_vec is not None:
            vec = tfidf_vec.transform([text]).toarray()
            c = int(tfidf_model.predict(vec)[0])
            clusters.append(c)
            themes.append("General Feedback")
            model_types.append("tfidf")
        else:
            clusters.append(None)
            themes.append(None)
            model_types.append("none")

    df["cluster"] = clusters
    df["theme"] = themes
    df["model_type"] = model_types
    df.to_csv(args.out, index=False)
    print(args.out)


if __name__ == "__main__":
    main()
