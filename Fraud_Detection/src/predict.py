import os
import json
import pickle
import pandas as pd
from typing import Dict, Any
from .preprocessing import preprocess

def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def models_dir() -> str:
    return os.path.join(project_root(), "models")

def load_model():
    path = os.path.join(models_dir(), "fraud_model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

def load_feature_columns() -> list[str]:
    cols_path = os.path.join(models_dir(), "feature_columns.json")
    if os.path.exists(cols_path):
        with open(cols_path, "r", encoding="utf-8") as f:
            return list((json.load(f) or {}).get("columns", []))
    return []

_model = None
_feature_cols = None

def _ensure_loaded():
    global _model, _feature_cols
    if _model is None:
        _model = load_model()
    if _feature_cols is None:
        _feature_cols = load_feature_columns()

def predict_transaction(payload: Dict[str, Any], threshold: float = 0.5) -> Dict[str, Any]:
    _ensure_loaded()
    df = pd.DataFrame([{
        "type": payload.get("type", "TRANSFER"),
        "amount": float(payload.get("amount", 0.0)),
        "oldbalanceOrg": float(payload.get("oldbalanceOrg", 0.0)),
        "newbalanceOrig": float(payload.get("newbalanceOrig", 0.0)),
        "oldbalanceDest": float(payload.get("oldbalanceDest", 0.0)),
        "newbalanceDest": float(payload.get("newbalanceDest", 0.0)),
    }])
    X_live = preprocess(df, expected_cols=_feature_cols if _feature_cols else None)
    prob = float(_model.predict_proba(X_live)[0, 1])
    label = int(prob >= threshold)
    return {"probability": prob, "label": label, "threshold": float(threshold)}
