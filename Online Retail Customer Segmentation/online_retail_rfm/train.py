import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from preprocess import load_and_create_rfm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "clustering.db")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rfm_cluster.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    print("Creating RFM table...")
    rfm = load_and_create_rfm(DB_PATH)

    features = rfm[['Recency', 'Frequency', 'Monetary']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    print("Training KMeans...")
    model = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    rfm['Cluster'] = labels
    cluster_map = {
        0: "Low Value",
        1: "Medium",
        2: "High Value",
        3: "Loyal",
        4: "At Risk"
    }
    rfm['Segment'] = rfm['Cluster'].map(cluster_map)

    sil = silhouette_score(X_scaled, labels)
    print("Silhouette Score:", sil)

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "rfm_data": rfm
    }, MODEL_PATH)

    print("Model saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
