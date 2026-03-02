import sqlite3, joblib, os
from preprocess import build_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "clustering.db")
MODEL_PATH = os.path.join(BASE_DIR, "models", "reviews_cluster.pkl")


def main(limit=1000):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM amazon_book_reviews LIMIT {limit}")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    model = joblib.load(MODEL_PATH)

    X = build_features(
        rows,
        model["vectorizer"],
        model["scaler"],
        model["svd"],
        fit=False,
    )

    labels = model["cluster"].predict(X)

    for r, l in zip(rows[:10], labels[:10]):
        print(r["Id"], "→ Cluster", l)


if __name__ == "__main__":
    main()
