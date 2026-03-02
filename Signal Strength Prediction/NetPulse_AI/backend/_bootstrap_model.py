import os, json, joblib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
META_PATH = os.path.join(MODELS_DIR, "meta.json")

features = [
    "locality",
    "latitude",
    "longitude",
    "network_type",
    "throughput_mbps",
    "latency_ms",
    "signal_quality_pct",
    "bb60c_dbm",
    "srsran_dbm",
    "bladerf_dbm",
]

cat_features = ["network_type", "locality"]
num_features = [f for f in features if f not in cat_features]

pre = ColumnTransformer(
    transformers=[
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_features),
        ("num", SimpleImputer(strategy="median"), num_features),
    ],
    remainder="drop",
)

model = Pipeline([
    ("prep", pre),
    ("reg", DummyRegressor(strategy="constant", constant=-85.0)),
])

df = pd.DataFrame([
    {"locality":"Patna","latitude":25.6,"longitude":85.1,"network_type":"4G","throughput_mbps":12.0,"latency_ms":45,"signal_quality_pct":78,"bb60c_dbm":-86,"srsran_dbm":-88,"bladerf_dbm":-87},
    {"locality":"Gaya","latitude":24.8,"longitude":84.9,"network_type":"LTE","throughput_mbps":8.0,"latency_ms":70,"signal_quality_pct":65,"bb60c_dbm":-92,"srsran_dbm":-91,"bladerf_dbm":-90},
])
y = np.array([-85.0, -85.0])
model.fit(df, y)

os.makedirs(MODELS_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump({
        "best_model": "DummyRegressor",
        "model_type": "DummyRegressor",
        "r2": None,
        "mae": None,
        "rmse": None,
        "feature_importance": [],
        "features": features,
        "rows_trained": int(len(df)),
        "model_version": "DummyRegressor-v0",
        "last_trained_iso": datetime.now(timezone.utc).isoformat(),
    }, f, indent=2)
