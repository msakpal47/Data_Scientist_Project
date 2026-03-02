import os
import sqlite3

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from preprocess import build_user_features


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "clustering.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ecommerce_cluster.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def main() -> None:
    print("Loading DB...")
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(DB_PATH)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql("SELECT * FROM ecommerce_behavior", conn)
    except Exception as e:
        raise RuntimeError("Failed to load table 'ecommerce_behavior'") from e

    print("Building features...")
    features = build_user_features(df)

    X = features.drop(columns=["user_id"])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Training model...")
    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)
    print("Silhouette:", score)

    print("Computing segment labels...")
    feat_with_labels = features.copy()
    feat_with_labels["cluster"] = labels
    stats = (
        feat_with_labels.groupby("cluster")
        .agg(
            count=("user_id", "count"),
            avg_spent=("total_spent", "mean"),
            avg_conv=("conversion_rate", "mean"),
        )
        .reset_index()
    )
    # Identify clusters
    low_idx = int(stats.sort_values("avg_spent").iloc[0]["cluster"])
    high_idx = int(stats.sort_values("avg_spent").iloc[-1]["cluster"])
    remaining = [int(c) for c in stats["cluster"].tolist() if c not in (low_idx, high_idx)]
    # Among remaining, lowest conversion -> churn risk, next -> at risk, last -> medium buyers
    rem_stats = stats[stats["cluster"].isin(remaining)].sort_values("avg_conv")
    churn_idx = int(rem_stats.iloc[0]["cluster"])
    at_risk_idx = int(rem_stats.iloc[1]["cluster"]) if len(rem_stats) > 1 else churn_idx
    # Whatever left is medium buyers
    medium_idx = [r for r in remaining if r not in (churn_idx, at_risk_idx)]
    medium_idx = int(medium_idx[0]) if medium_idx else low_idx
    segments = {
        low_idx: "Low Value",
        medium_idx: "Medium Buyers",
        high_idx: "High Value",
        at_risk_idx: "At Risk",
        churn_idx: "Churn Risk",
    }

    joblib.dump(
        {
            "scaler": scaler,
            "model": model,
            "columns": X.columns.tolist(),
            "users": features["user_id"].tolist(),
            "segments": segments,
        },
        MODEL_PATH,
    )

    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
