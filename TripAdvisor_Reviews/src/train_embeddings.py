import argparse
import json
import os
import sys
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.text_utils import clean_text


DATA_DIR = "data"
MODELS_DIR = "models"
DEFAULT_DB = os.path.join(DATA_DIR, "clustering.db")
FALLBACK_DB = "clustering.db"
TABLE = "tripadvisor_reviews"
TEXT_COL = "review_full"
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_FEATURES = 5000
KMEANS_CLUSTERS = 5
TOP_WORDS = 12


def db_path():
    if os.path.exists(DEFAULT_DB):
        return DEFAULT_DB
    return FALLBACK_DB


def load_data():
    conn = sqlite3.connect(db_path())
    try:
        try:
            df = pd.read_sql(f"SELECT {TEXT_COL}, rating_review FROM {TABLE}", conn)
        except Exception:
            df = pd.read_sql(f"SELECT {TEXT_COL} FROM {TABLE}", conn)
    finally:
        conn.close()
    return df


def compute_embeddings(texts, model_name):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, show_progress_bar=True, normalize_embeddings=False)
    return emb, model_name


def compute_top_terms(cleaned_series, labels, n_clusters, top_n):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(cleaned_series)
    terms = vectorizer.get_feature_names_out()
    result = {}
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        if len(idx) == 0:
            result[str(c)] = []
            continue
        sub = X[idx].toarray()
        mean_vec = sub.mean(axis=0)
        top_idx = np.argsort(mean_vec)[-top_n:][::-1]
        top_terms = [terms[i] for i in top_idx]
        result[str(c)] = top_terms
    return result


def auto_label(top_words):
    s = set(top_words)
    if any(w in s for w in ["staff", "service", "friendly", "helpful"]):
        return "Customer Service"
    if any(w in s for w in ["location", "near", "close", "central"]):
        return "Location Advantage"
    if any(w in s for w in ["dirty", "clean", "cleanliness"]):
        return "Cleanliness Issues"
    if any(w in s for w in ["food", "breakfast", "restaurant", "dinner"]):
        return "Food Experience"
    if any(w in s for w in ["price", "expensive", "overpriced", "value"]):
        return "Pricing"
    if any(w in s for w in ["check", "checkin", "check-in", "queue", "slow", "wait"]):
        return "Check-in Experience"
    if any(w in s for w in ["room", "bed", "bathroom", "shower"]):
        return "Room Quality"
    if any(w in s for w in ["noise", "noisy", "loud"]):
        return "Noise Issues"
    if any(w in s for w in ["wifi", "internet"]):
        return "Connectivity"
    if any(w in s for w in ["pool", "spa", "gym"]):
        return "Amenities"
    return "General Feedback"


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def run_kmeans(embeddings, n_clusters):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return km, labels


def run_agglomerative(embeddings, n_clusters):
    hc = AgglomerativeClustering(n_clusters=n_clusters)
    labels = hc.fit_predict(embeddings)
    return hc, labels


def run_dbscan(embeddings, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    labels = db.fit_predict(embeddings)
    return db, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--algo", choices=["kmeans", "agglomerative", "dbscan"], default="kmeans")
    parser.add_argument("--clusters", type=int, default=KMEANS_CLUSTERS)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_samples", type=int, default=5)
    parser.add_argument("--export_prefix", default="bert")
    parser.add_argument("--lda", action="store_true")
    parser.add_argument("--lda_topics", type=int, default=5)
    parser.add_argument("--dendrogram_path", default="")
    parser.add_argument("--dendrogram_limit", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    df = load_data()
    df["cleaned_review"] = df[TEXT_COL].apply(clean_text)
    texts = df["cleaned_review"].fillna("").tolist()
    embeddings, model_name = compute_embeddings(texts, args.model_name)

    if args.algo == "kmeans":
        clusterer, labels = run_kmeans(embeddings, args.clusters)
        clusterer_path = os.path.join(MODELS_DIR, f"clusterer_{args.export_prefix}_kmeans.pkl")
    elif args.algo == "agglomerative":
        clusterer, labels = run_agglomerative(embeddings, args.clusters)
        clusterer_path = os.path.join(MODELS_DIR, f"clusterer_{args.export_prefix}_agglomerative.pkl")
    else:
        clusterer, labels = run_dbscan(embeddings, args.eps, args.min_samples)
        clusterer_path = os.path.join(MODELS_DIR, f"clusterer_{args.export_prefix}_dbscan.pkl")

    df["cluster"] = labels

    n_clusters = len(set([l for l in labels if l != -1])) if args.algo == "dbscan" else args.clusters
    top_terms = compute_top_terms(df["cleaned_review"], labels, n_clusters, TOP_WORDS)
    labels_map = {}
    for k, v in top_terms.items():
        labels_map[k] = auto_label(v)

    joblib.dump(clusterer, clusterer_path)
    with open(os.path.join(MODELS_DIR, "st_model_name.txt"), "w", encoding="utf-8") as f:
        f.write(model_name)
    save_json(os.path.join(MODELS_DIR, "cluster_top_terms.json"), top_terms)
    save_json(os.path.join(MODELS_DIR, "cluster_labels.json"), labels_map)

    export_csv = os.path.join(DATA_DIR, f"clustered_reviews_{args.export_prefix}_{args.algo}.csv")
    if "rating_review" in df.columns:
        try:
            df["sentiment"] = df["rating_review"].apply(lambda x: "Positive" if float(x) >= 4 else "Negative")
        except Exception:
            pass
    df.to_csv(export_csv, index=False)

    if args.lda:
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation
        cv = CountVectorizer(max_df=0.9, min_df=10, stop_words="english")
        dtm = cv.fit_transform(df["cleaned_review"])
        lda = LatentDirichletAllocation(n_components=args.lda_topics, random_state=42)
        lda.fit(dtm)
        feats = cv.get_feature_names_out()
        topics = {}
        for t in range(args.lda_topics):
            comp = lda.components_[t]
            idx = np.argsort(comp)[-TOP_WORDS:][::-1]
            topics[str(t)] = [feats[i] for i in idx]
        save_json(os.path.join(MODELS_DIR, "lda_topics.json"), topics)

    if args.dendrogram_path:
        try:
            import scipy.cluster.hierarchy as sch
            import matplotlib.pyplot as plt
            lim = min(len(embeddings), args.dendrogram_limit)
            Z = sch.linkage(embeddings[:lim], method="ward")
            plt.figure(figsize=(10, 5))
            sch.dendrogram(Z)
            plt.tight_layout()
            plt.savefig(args.dendrogram_path)
            plt.close()
        except Exception:
            pass

    print(clusterer_path)
    print(os.path.join(MODELS_DIR, "st_model_name.txt"))
    print(os.path.join(MODELS_DIR, "cluster_top_terms.json"))
    print(os.path.join(MODELS_DIR, "cluster_labels.json"))
    print(export_csv)


if __name__ == "__main__":
    main()

