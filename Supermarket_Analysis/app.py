from flask import Flask, render_template, render_template_string, Response
import pickle
from sklearn.metrics import silhouette_score

app = Flask(__name__)

# ----------------------------
# Load Saved Model Artifacts
# ----------------------------
def load_artifacts():
    try:
        artifacts = pickle.load(open("artifacts.pkl", "rb"))
        customer_df = artifacts["customer_df"]
        scaler = artifacts["scaler"]
        model = artifacts["model"]
        return customer_df, scaler, model
    except FileNotFoundError:
        customer_df = pickle.load(open("customer_data.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        model = pickle.load(open("model.pkl", "rb"))
        return customer_df, scaler, model


# ----------------------------
# Dashboard Route
# ----------------------------
@app.route("/")
def home():
    customer_df, scaler, model = load_artifacts()

    # Prepare features
    features = customer_df.drop(["user_id", "cluster"], axis=1)
    X_scaled = scaler.transform(features)

    score = silhouette_score(X_scaled, customer_df["cluster"])

    # Cluster distribution table
    cluster_counts_df = (
        customer_df["cluster"]
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_counts_df.columns = ["Cluster", "Customers"]
    cluster_counts_rows = cluster_counts_df.to_dict(orient="records")

    # Cluster insights table (NO user_id here)
    cluster_summary_df = customer_df.groupby("cluster").agg({
        "total_orders": "mean",
        "total_products": "mean",
        "avg_days_between_orders": "mean",
        "reorder_ratio": "mean",
        "avg_cart_size": "mean",
        "preferred_hour": "mean"
    }).round(2).reset_index()

    cluster_summary = cluster_summary_df.to_dict(orient="records")

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Segmentation Dashboard</title>
        <link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
    </head>
    <body>
    <div class="container">
        <h1>📊 Supermarket Customer Clustering Dashboard</h1>
        <div class="table-card">
            <h2>Model Evaluation</h2>
            <div class="table-container">
                <table>
                    <tr><th>Total Customers</th><th>Clusters</th><th>Silhouette Score</th></tr>
                    <tr><td>{{ total_customers }}</td><td>{{ num_clusters }}</td><td>{{ silhouette_score }}</td></tr>
                </table>
            </div>
        </div>
        <div class="table-card">
            <h2>Cluster Distribution</h2>
            <div class="table-container">
                <table>
                    <tr><th>Cluster</th><th>Customers</th><th>Actions</th></tr>
                    {% for row in cluster_counts_rows %}
                    <tr><td>{{ row.Cluster }}</td><td>{{ row.Customers }}</td><td><a href="/cluster/{{ row.Cluster }}">View</a></td></tr>
                    {% endfor %}
                </table>
            </div>
        </div>
        <div class="table-card">
            <h2>Cluster Insights</h2>
            <div class="table-container">
                <table>
                    <tr>
                        <th>Cluster</th><th>Total Orders</th><th>Total Products</th>
                        <th>Avg Days</th><th>Reorder Ratio</th><th>Cart Size</th><th>Preferred Hour</th>
                    </tr>
                    {% for row in cluster_summary %}
                    <tr>
                        <td>{{ row.cluster }}</td><td>{{ row.total_orders }}</td><td>{{ row.total_products }}</td>
                        <td>{{ row.avg_days_between_orders }}</td><td>{{ row.reorder_ratio }}</td>
                        <td>{{ row.avg_cart_size }}</td><td>{{ row.preferred_hour }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </div>
    </div>
    </body>
    </html>
    """
    return render_template_string(
        html,
        silhouette_score=round(score, 3),
        total_customers=len(customer_df),
        num_clusters=len(cluster_counts_rows),
        cluster_counts_rows=cluster_counts_rows,
        cluster_summary=cluster_summary,
    )


# ----------------------------
# Cluster Detail Page
# ----------------------------
@app.route("/cluster/<int:cid>")
def cluster_details(cid):
    customer_df, _, _ = load_artifacts()

    cols = [
        "user_id",
        "total_orders",
        "total_products",
        "avg_days_between_orders",
        "reorder_ratio",
        "avg_cart_size",
        "preferred_hour",
    ]

    rows = (
        customer_df[customer_df["cluster"] == cid][cols]
        .round(2)
        .head(100)
        .to_dict(orient="records")
    )

    return render_template(
        "cluster.html",
        cluster_id=cid,
        customers=len(rows),
        rows=rows,
    )


# ----------------------------
# Download CSV
# ----------------------------
@app.route("/download/cluster/<int:cid>.csv")
def download_cluster_csv(cid):
    customer_df, _, _ = load_artifacts()

    cols = [
        "user_id",
        "total_orders",
        "total_products",
        "avg_days_between_orders",
        "reorder_ratio",
        "avg_cart_size",
        "preferred_hour",
    ]

    rows = customer_df[customer_df["cluster"] == cid][cols].round(2)
    csv_data = rows.to_csv(index=False)

    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=cluster_{cid}.csv"},
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
