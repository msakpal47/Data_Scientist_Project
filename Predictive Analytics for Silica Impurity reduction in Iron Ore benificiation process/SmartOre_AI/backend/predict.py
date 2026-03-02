import os
import time
import numpy as np
import joblib


_CACHE = {
    "model": None,
    "scaler": None,
    "features": None,
    "medians": None,
    "meta": None,
    "importances": None,
}


def _paths():
    backend_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.abspath(os.path.join(backend_dir, "models"))
    model_path = os.path.abspath(os.path.join(models_dir, "model.pkl"))
    scaler_path = os.path.abspath(os.path.join(models_dir, "scaler.pkl"))
    print("MODEL PATH:", model_path)
    print("SCALER PATH:", scaler_path)
    return model_path, scaler_path


def init_cache():
    model_path, scaler_path = _paths()
    if not os.path.exists(model_path):
        _CACHE.update(
            {"model": None, "scaler": None, "features": None, "medians": None, "meta": None, "importances": None}
        )
        return False
    payload = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    _CACHE["model"] = payload["model"]
    _CACHE["features"] = payload["features"]
    _CACHE["medians"] = payload.get("medians")
    _CACHE["meta"] = payload.get("meta") or {}
    _CACHE["importances"] = payload.get("importances")
    _CACHE["scaler"] = scaler
    if "shap_top" in payload:
        _CACHE["shap_top"] = payload["shap_top"]
    return True


def status():
    ok = _CACHE["model"] is not None
    m = _CACHE.get("meta") or {}
    return {
        "loaded": bool(ok),
        "version": m.get("version"),
        "trained_at": m.get("trained_at"),
        "r2": m.get("r2"),
        "r2_train": m.get("r2_train"),
        "r2_test": m.get("r2_test"),
        "mae_train": m.get("mae_train"),
        "mae_test": m.get("mae_test"),
        "rmse_train": m.get("rmse_train"),
        "rmse_test": m.get("rmse_test"),
        "features": _CACHE.get("features") or [],
        "importances": _CACHE.get("importances") or [],
        "shap_top": _CACHE.get("shap_top") or [],
    }


def _risk_category(value):
    if value <= 1.5:
        return "Low"
    if value <= 3.0:
        return "Medium"
    return "High"


def _float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _advise(inp, pred):
    tips = []
    ph = _float(inp.get("Ore Pulp pH"))
    silica_feed = _float(inp.get("% Silica Feed"))
    amina = _float(inp.get("Amina Flow"))
    starch = _float(inp.get("Starch Flow"))
    air = _float(inp.get("Avg Air Flow"))
    if pred > 3.0 and ph is not None and not np.isnan(ph) and ph < 10.0:
        tips.append("Increase pH slightly to improve silica rejection")
    if pred > 2.0 and amina is not None and not np.isnan(amina):
        tips.append("Increase amina flow to enhance silica collection")
    if pred > 2.5 and starch is not None and not np.isnan(starch):
        tips.append("Increase starch dosage to depress iron and aid separation")
    if pred > 2.0 and air is not None and not np.isnan(air) and air < 100:
        tips.append("Increase air flow to improve bubble carryover")
    if silica_feed is not None and not np.isnan(silica_feed) and silica_feed > 5.0:
        tips.append("High silica feed detected; adjust reagents proactively")
    if not tips:
        tips.append("Keep current operating conditions; monitor trends")
    return tips


def predict_payload(payload):
    if _CACHE["model"] is None:
        init_cache()
        if _CACHE["model"] is None:
            try:
                from train_model import train_and_save
                train_and_save()
                init_cache()
            except Exception as e:
                print("auto-train failed:", e)
            if _CACHE["model"] is None:
                sf = _float(payload.get("% Silica Feed"), 0.0)
                st = _float(payload.get("Starch Flow"), 0.0)
                am = _float(payload.get("Amina Flow"), 0.0)
                ph = _float(payload.get("Ore Pulp pH"), 10.0)
                y_fb = max(0.0, 0.35 * sf - 0.03 * st + 0.02 * am - 0.1 * (ph - 10.0))
                risk = _risk_category(float(y_fb))
                recs = _advise(payload, float(y_fb))
                return {
                    "silica_concentrate": float(y_fb),
                    "risk": risk,
                    "recommendations": recs,
                    "fallback": True,
                }
    model = _CACHE["model"]
    scaler = _CACHE["scaler"]
    features = _CACHE["features"]
    medians = _CACHE["medians"]
    row = []
    for f in features:
        v = payload.get(f)
        row.append(_float(v, np.nan))
    X = np.array([row], dtype=float)
    if medians is not None and len(medians) == len(features):
        filled = []
        for i, val in enumerate(X[0]):
            if np.isnan(val):
                filled.append(float(medians[i]))
            else:
                filled.append(val)
        X_filled = np.array([filled], dtype=float)
    else:
        X_filled = np.nan_to_num(X, nan=0.0)
    if scaler is not None:
        X_scaled = scaler.transform(X_filled)
    else:
        X_scaled = X_filled
    t1 = time.time()
    y = model.predict(X_scaled)[0]
    t2 = time.time()
    risk = _risk_category(float(y))
    recs = _advise(payload, float(y))
    return {
        "silica_concentrate": float(y),
        "risk": risk,
        "recommendations": recs,
    }


def warmup():
    if _CACHE["model"] is None:
        return False
    feats = _CACHE.get("features") or []
    meds = _CACHE.get("medians") or []
    payload = {}
    for i, f in enumerate(feats):
        v = float(meds[i]) if i < len(meds) else 0.0
        payload[f] = v
    try:
        predict_payload(payload)
        return True
    except Exception:
        return False


def fallback_predict(payload):
    sf = _float(payload.get("% Silica Feed"), 0.0)
    st = _float(payload.get("Starch Flow"), 0.0)
    am = _float(payload.get("Amina Flow"), 0.0)
    ph = _float(payload.get("Ore Pulp pH"), 10.0)
    y_fb = max(0.0, 0.35 * sf - 0.03 * st + 0.02 * am - 0.1 * (ph - 10.0))
    risk = _risk_category(float(y_fb))
    recs = _advise(payload, float(y_fb))
    return {
        "silica_concentrate": float(y_fb),
        "risk": risk,
        "recommendations": recs,
        "fallback": True,
    }

if __name__ == "__main__":
    ok = init_cache()
    print(status())
