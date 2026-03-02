import sqlite3
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

conn = sqlite3.connect("clustering.db")
query = """
SELECT user_id,
       order_number,
       product_id,
       days_since_prior_order,
       reordered,
       add_to_cart_order,
       order_hour_of_day
FROM supermarket_data
"""
df = pd.read_sql(query, conn)
df.fillna(0, inplace=True)
customer_df = df.groupby("user_id").agg(
    total_orders=("order_number", "max"),
    total_products=("product_id", "count"),
    avg_days_between_orders=("days_since_prior_order", "mean"),
    reorder_ratio=("reordered", "mean"),
    avg_cart_size=("add_to_cart_order", "mean"),
    preferred_hour=("order_hour_of_day", "mean"),
).reset_index()
if len(customer_df) > 1324:
    customer_df = customer_df.sort_values("total_orders", ascending=False).head(1324)
features = customer_df.drop("user_id", axis=1)
print("Training on shape:", features.shape)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
customer_df["cluster"] = kmeans.fit_predict(X_scaled)
artifacts = {
    "model": kmeans,
    "scaler": scaler,
    "customer_df": customer_df,
}
pickle.dump(artifacts, open("artifacts.pkl", "wb"))
print("Model Saved Successfully")
print("Customers:", len(customer_df), "Unique users:", customer_df["user_id"].nunique())
print("Cluster counts:", customer_df["cluster"].value_counts().to_dict())
