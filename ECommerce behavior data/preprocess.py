import pandas as pd
import numpy as np


def build_user_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.groupby("user_id").agg(
        total_events=("event_type", "count"),
        total_views=("event_type", lambda x: (x == "view").sum()),
        total_purchases=("event_type", lambda x: (x == "purchase").sum()),
        avg_price=("price", "mean"),
        total_spent=("price", "sum"),
        unique_products=("product_id", "nunique"),
        unique_categories=("category_id", "nunique"),
        session_count=("user_session", "nunique"),
    ).reset_index()

    te = features["total_events"].replace(0, np.nan)
    features["conversion_rate"] = (features["total_purchases"] / te).fillna(0.0)

    return features.fillna(0)
