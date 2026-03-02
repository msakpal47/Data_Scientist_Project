import os
import pickle
import numpy as np
import pandas as pd
try:
    import joblib  # type: ignore
except Exception:
    joblib = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")

_MODEL = None
_SCALER = None
_COLUMNS = None


def _load(path):
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass
    with open(path, "rb") as f:
        return pickle.load(f)


def _ensure_loaded():
    global _MODEL, _SCALER, _COLUMNS
    if _MODEL is not None or _COLUMNS is not None:
        return
    model_path = os.path.join(MODELS_DIR, "model.pkl")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(model_path):
        _MODEL = _load(model_path)
    if os.path.exists(cols_path):
        _COLUMNS = _load(cols_path)
    if os.path.exists(scaler_path):
        _SCALER = _load(scaler_path)


def get_feature_columns():
    _ensure_loaded()
    return _COLUMNS if isinstance(_COLUMNS, list) else None


def _prep_dataframe(values_dict, values_list):
    _ensure_loaded()
    if isinstance(_COLUMNS, list):
        if values_dict:
            row = []
            for c in _COLUMNS:
                v = values_dict.get(c)
                if v is None:
                    raise ValueError(f"missing value for {c}")
                f = float(v)
                if f < 0:
                    raise ValueError(f"{c} must be positive")
                row.append(f)
        else:
            if len(values_list) != len(_COLUMNS):
                raise ValueError(f"expected {len(_COLUMNS)} features, got {len(values_list)}")
            row = []
            for x in values_list:
                f = float(x)
                if f < 0:
                    raise ValueError("inputs must be positive")
                row.append(f)
        X = pd.DataFrame([row], columns=_COLUMNS)
    else:
        arr = []
        for x in (values_list or list(values_dict.values())):
            f = float(x)
            if f < 0:
                raise ValueError("inputs must be positive")
            arr.append(f)
        cols = [f"f{i}" for i in range(len(arr))]
        X = pd.DataFrame([arr], columns=cols)
    if _SCALER is not None and isinstance(_COLUMNS, list):
        X = pd.DataFrame(_SCALER.transform(X), columns=X.columns)
    return X


def predict_from_payload(payload):
    _ensure_loaded()
    if _MODEL is None:
        raise RuntimeError("model not available")
    values_dict = {}
    values_list = None
    if isinstance(payload, dict) and payload:
        if "features" in payload:
            if isinstance(payload["features"], dict):
                values_dict = payload["features"]
            else:
                values_list = payload["features"]
        else:
            values_dict = payload
    X = _prep_dataframe(values_dict, values_list)
    y = _MODEL.predict(X)
    return float(np.asarray(y)[0])
