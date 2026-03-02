import os
import sys
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from src.text_utils import clean_text

DATA_CSV = os.path.join(ROOT_DIR, "data", "sample_reviews_30.csv")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
KMEANS_PATH = os.path.join(MODELS_DIR, "kmeans_model.pkl")
VEC_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")


def main():
    if not os.path.exists(DATA_CSV):
        raise SystemExit(f"Missing dataset: {DATA_CSV}")
    df = pd.read_csv(DATA_CSV)
    col = "review_full" if "review_full" in df.columns else df.columns[0]
    texts = df[col].astype(str).tolist()
    cleaned = [clean_text(t) for t in texts]
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(cleaned).toarray()
    k = min(5, max(2, len(cleaned)//10 or 2))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(km, KMEANS_PATH)
    joblib.dump(vectorizer, VEC_PATH)
    print(KMEANS_PATH)
    print(VEC_PATH)


if __name__ == "__main__":
    main()
