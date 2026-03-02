import os
import sqlite3
import joblib
import time
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from preprocess import build_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "clustering.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "reviews_cluster.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "final_report.txt")

os.makedirs(MODEL_DIR, exist_ok=True)


def load_rows(limit=100000):
    print(f"Loading {limit} rows from DB...")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    try:
        cur.execute(f"SELECT * FROM amazon_book_reviews LIMIT {limit}")
    except Exception as e:
        raise Exception(f"Table 'amazon_book_reviews' not found → {e}")

    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    print(f"Loaded rows: {len(rows)}")
    return rows


def main():
    start = time.time()

    print("====================================")
    print("🚀 TRAINING STARTED")
    print("BASE_DIR:", BASE_DIR)
    print("MODEL_PATH:", MODEL_PATH)
    print("====================================")

    rows = load_rows(limit=100000)

    print("Creating TF-IDF + SVD + Scaler...")

    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        stop_words="english",
        ngram_range=(1, 2),
    )

    scaler = StandardScaler()

    svd = TruncatedSVD(
        n_components=128,
        random_state=42,
    )

    print("Building features...")
    X = build_features(rows, vectorizer, scaler, svd, fit=True)

    print("Feature shape:", X.shape)

    print("Training KMeans...")
    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(X)

    print("Evaluating model...")
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    inertia = model.inertia_

    print("Silhouette:", sil)
    print("Davies–Bouldin:", dbi)
    print("Inertia:", inertia)

    print("Saving model...")

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "scaler": scaler,
            "svd": svd,
            "cluster": model,
        },
        MODEL_PATH,
    )

    print("✅ Model saved successfully!")

    counts = Counter(labels)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Amazon Books Reviews Clustering Report\n\n")
        f.write("Model: KMeans\n")
        f.write("Clusters: 5\n\n")
        f.write("Features Used:\n")
        f.write("- TF-IDF text vectors\n")
        f.write("- Rating score\n")
        f.write("- Helpfulness ratio\n")
        f.write("- Log(price)\n")
        f.write("- Review length\n\n")
        f.write(f"Silhouette Score: {sil:.4f}\n")
        f.write(f"Davies–Bouldin: {dbi:.4f}\n")
        f.write(f"Inertia: {inertia:.2f}\n\n")
        f.write("Cluster Sizes:\n")
        for k, v in counts.items():
            f.write(f"Cluster {k}: {v}\n")

    end = time.time()

    print("====================================")
    print("🎉 TRAINING COMPLETED")
    print("Model Path:", MODEL_PATH)
    print("Report Path:", REPORT_PATH)
    print(f"Time taken: {end-start:.2f} sec")
    print("====================================")


if __name__ == "__main__":
    main()
