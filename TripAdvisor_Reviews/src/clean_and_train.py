import os
import sys
import sqlite3
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
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
N_CLUSTERS = 5
MAX_FEATURES = 5000
TOP_WORDS_PER_CLUSTER = 12


def db_path():
    if os.path.exists(DEFAULT_DB):
        return DEFAULT_DB
    return FALLBACK_DB


def load_data():
    conn = sqlite3.connect(db_path())
    try:
        df = pd.read_sql(f"SELECT {TEXT_COL} FROM {TABLE}", conn)
    finally:
        conn.close()
    return df


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_data()
    df["cleaned_review"] = df[TEXT_COL].apply(clean_text)
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
    X = vectorizer.fit_transform(df["cleaned_review"])
    X_dense = X.toarray()
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_dense)
    df["cluster"] = labels
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf.pkl"))
    terms = vectorizer.get_feature_names_out()
    centers = kmeans.cluster_centers_
    for i in range(N_CLUSTERS):
        center_terms = centers[i]
        top_idx = np.argsort(center_terms)[-TOP_WORDS_PER_CLUSTER:][::-1]
        top_terms = [terms[j] for j in top_idx]
        print(f"Cluster {i} Top Words: {top_terms}")
    df.to_csv(os.path.join(DATA_DIR, "clustered_reviews_tfidf_kmeans.csv"), index=False)
    print("Saved models and data")


if __name__ == "__main__":
    train()
